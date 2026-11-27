"""P206 的模型接口（框架 2026-11-27）。每个模型包成一个 Encoder：

    enc.sfreq            输入采样率（框架负责把观测段降采样到这个采样率）
    enc.embed(x, ctx)    x：(n_ch, n_samp) float32 µV，一个观测段里的可用通道；ctx = {names: 通道名, xyz: (n_ch, 3) MNI}
                         → (n_ch, D)；某个通道算不出时那一行给 NaN（框架当作这个通道这一段不可用）

读法（采样率、片段长度、层、池化、归一化）全部在 configs/models/p206_encoders.yaml，**C 之前锁定**（docs/06 第 3 节第 6 条）。
名字带 “-rand” 的是同结构随机初始化版本（00 第 4 节 P206，C2）：结构与预训练相同，权重按 seed 重新初始化。

状态（2026-11-27）：
  spec_test       只给测试和本地冒烟用（各频段对数功率，D = 频段数），**不是 P206 的模型**
  chronos         接口已在 pvb-p206a-env-1 冒烟中跑通（ChronosPipeline.embed）；读法是草案
  moment          接口按 p206a_smoke 的写法（MOMENTPipeline，embedding 任务）；A2 冒烟后核对
  brainbert、popt、mvpformer_s、mvpformer_m：**等 A2 的冒烟报告**（类说明、__init__ 参数、前向形状）后实现
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Encoder:
    name = ""
    sfreq = 512.0

    def embed(self, x: np.ndarray, ctx: dict) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError


class SpecTest(Encoder):
    """测试用：每个通道各频段的对数平均功率（Hanning 窗 FFT）。只用段内数据，确定性。"""

    def __init__(self, sfreq: float = 256.0, bands: dict | None = None, **_):
        self.sfreq = float(sfreq)
        self.bands = bands or {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 70]}

    def embed(self, x, ctx):
        xd = x - x.mean(axis=1, keepdims=True)
        P = np.abs(np.fft.rfft(xd * np.hanning(xd.shape[1])[None], axis=1)) ** 2
        f = np.fft.rfftfreq(xd.shape[1], 1 / self.sfreq)
        cols = [P[:, (f >= lo) & (f < hi)].mean(axis=1) for lo, hi in self.bands.values()]
        return np.log(np.stack(cols, axis=1) + 1e-12)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def randomize(module, seed: int) -> int:
    """同结构随机初始化：对每个有 reset_parameters 的子模块重新初始化。返回重置的子模块数（C 步核对参数确实变了）。"""
    import torch
    torch.manual_seed(seed)
    n = 0
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters(); n += 1
    return n


class _Univariate(Encoder):
    """通用时间序列模型（单变量）：每个通道各自一条序列，按 batch 个通道一批送进模型。"""

    def __init__(self, sfreq, zscore=True, batch=128, **_):
        self.sfreq, self.zscore, self.batch = float(sfreq), bool(zscore), int(batch)

    def _series(self, x):
        return _zscore(x) if self.zscore else x

    def embed(self, x, ctx):
        xs = self._series(np.asarray(x, np.float32))
        return np.vstack([self._embed_batch(xs[i:i + self.batch]) for i in range(0, len(xs), self.batch)])


class ChronosEnc(_Univariate):
    """chronos-t5-large：ChronosPipeline.embed(context (B, T)) → (B, T', d_model)，时间上取平均（编码器末层）。"""

    def __init__(self, path, device, random_init=False, seed=0, **kw):
        super().__init__(**kw)
        import torch
        from chronos import ChronosPipeline
        self.torch = torch
        self.p = ChronosPipeline.from_pretrained(str(path), device_map=device, torch_dtype=torch.float32)
        self.n_reset = randomize(self.p.model, seed) if random_init else 0

    def _embed_batch(self, xb):
        with self.torch.no_grad():
            emb, _ = self.p.embed(self.torch.from_numpy(np.ascontiguousarray(xb)))
        return emb.float().mean(dim=1).cpu().numpy()


class MomentEnc(_Univariate):
    """MOMENT-1-large（embedding 任务）：输入固定 n_times（512）个点；短的左侧补零并用 input_mask 标出，长的取最后 n_times 个。"""

    def __init__(self, path, device, random_init=False, seed=0, n_times=512, **kw):
        super().__init__(**kw)
        import torch
        from momentfm import MOMENTPipeline
        self.torch, self.dev, self.n_times = torch, device, int(n_times)
        m = MOMENTPipeline.from_pretrained(str(path), model_kwargs={"task_name": "embedding"})
        m.init()
        self.n_reset = randomize(m, seed) if random_init else 0
        self.m = m.to(device).eval()

    def _embed_batch(self, xb):
        T = self.n_times
        B, n = xb.shape
        x = np.zeros((B, 1, T), np.float32); mask = np.zeros((B, T), np.float32)
        k = min(n, T)
        x[:, 0, T - k:] = xb[:, n - k:]; mask[:, T - k:] = 1
        t = self.torch
        with t.no_grad():
            o = self.m(x_enc=t.from_numpy(x).to(self.dev), input_mask=t.from_numpy(mask).to(self.dev))
        return o.embeddings.float().cpu().numpy()


class _Pending(Encoder):
    def __init__(self, name, **_):
        raise NotImplementedError(f"{name}：等 A2（pvb-p206a2-env-1）的冒烟报告核实接口后再实现（docs/06 第 2 节）")


REGISTRY = {
    "spec_test": lambda c, path, dev, rand, seed: SpecTest(**c),
    "chronos": lambda c, path, dev, rand, seed: ChronosEnc(path, dev, rand, seed, **c),
    "moment": lambda c, path, dev, rand, seed: MomentEnc(path, dev, rand, seed, **c),
    "brainbert": lambda c, path, dev, rand, seed: _Pending("brainbert"),
    "popt": lambda c, path, dev, rand, seed: _Pending("popt"),
    "mvpformer_s": lambda c, path, dev, rand, seed: _Pending("mvpformer_s"),
    "mvpformer_m": lambda c, path, dev, rand, seed: _Pending("mvpformer_m"),
}


def make(name: str, enc_cfg: dict, models_dir: str | Path, device: str) -> Encoder:
    """name = 配置里的模型名，可加 “-rand”。enc_cfg = p206_encoders.yaml 全文。"""
    base, rand = (name[:-5], True) if name.endswith("-rand") else (name, False)
    if base not in REGISTRY:
        raise KeyError(f"未知模型 {name}；可用：{sorted(REGISTRY)}（加 -rand 为随机初始化）")
    c = dict(enc_cfg.get(base, {}))
    path = Path(models_dir) / c.pop("path", base)
    seed = int(c.pop("rand_seed", enc_cfg.get("rand_seed", 0)))
    enc = REGISTRY[base](c, path, device, rand, seed)
    enc.name = name
    return enc
