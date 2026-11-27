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
  brainbert、mvpformer_s、mvpformer_m：按 pvb-p206a3-smoke-4（A4）的前向形状实现（2026-11-27）；读法是草案
  popt：按类说明实现，前向形状待 C 的冒烟核对
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


# ---------------- braindecode 的 iEEG 模型（接口按 pvb-p206a3-smoke-4 的冒烟报告，2026-11-27） ----------------

def real_model_dir(path) -> Path:
    """容器里 data/models 是软链接；braindecode 2.1 要求权重文件的真实路径在模型目录内 → 用软链接指向的真实目录。"""
    import os
    path = Path(path)
    for n in ("model.safetensors", "pytorch_model.bin", "config.json"):
        if (path / n).exists():
            return Path(os.path.realpath(path / n)).parent
    return path


def load_braindecode(cls, path, device, random_init=False, seed=0, **kw):
    """预训练权重；random_init = 用同一份完整配置重建（from_config(get_config())，A4 核对过张量形状全部一致）。"""
    import torch
    m = cls.from_pretrained(str(real_model_dir(path)), **kw)
    n_reset = 0
    if random_init:
        torch.manual_seed(seed)
        m = cls.from_config(m.get_config())
        n_reset = -1                                    # 整个模型重建
    return m.to(device).eval(), n_reset


def laplacian_nearest2(x: np.ndarray, xyz: np.ndarray | None) -> np.ndarray:
    """BrainBERT 预训练用的 Laplacian 重参考：每个通道减去 MNI 距离最近的两个可用通道的均值（ECoG 栅格 / 条上近似为
    相邻触点；docs/06 第 5 节）。只用同一段内的通道，不跨段。没有坐标的通道 → 那一行 NaN（该段不可用）。"""
    if xyz is None:
        return np.full_like(x, np.nan)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(xyz).all(axis=1)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return out
    d = np.linalg.norm(xyz[idx, None, :] - xyz[None, idx, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :2]
    out[idx] = x[idx] - 0.5 * (x[idx[nn[:, 0]]] + x[idx[nn[:, 1]]])
    return out


class BrainBERTEnc(Encoder):
    """BrainBERT（42.6 M；预训练 2048 Hz、Laplacian 重参考）。每个电极单独一个样本 (B, 1, T)；
    return_features 的 tokens = (B, 1, 帧, 768)（2.99 s → 104 帧）。readout：all_frames = 全部帧平均；center10 = 模型自带的
    embedding（中间 10 帧，上游协议，原为 5 s 窗设计）。"""

    def __init__(self, path, device, random_init=False, seed=0, sfreq=2048, readout="all_frames", reref="laplacian",
                 batch=256, **_):
        import torch
        from braindecode.models import BrainBERT
        self.torch, self.dev = torch, device
        self.sfreq, self.readout, self.reref, self.batch = float(sfreq), readout, reref, int(batch)
        self.m, self.n_reset = load_braindecode(BrainBERT, path, device, random_init, seed)

    def features(self, x, ctx):
        """(n_ch, T) → (n_ch, 768)；PopT 也用它。"""
        if self.reref == "laplacian":
            x = laplacian_nearest2(x, ctx.get("xyz"))
        out = np.full((len(x), 768), np.nan, np.float32)
        ok = np.isfinite(x).all(axis=1)
        xo = np.ascontiguousarray(x[ok], np.float32)
        res = []
        t = self.torch
        for i in range(0, len(xo), self.batch):
            with t.no_grad():
                f = self.m(t.from_numpy(xo[i:i + self.batch, None, :]).to(self.dev), return_features=True)
            e = f["tokens"][:, 0].mean(dim=1) if self.readout == "all_frames" else f["embedding"]
            res.append(e.float().cpu().numpy())
        if res:
            out[ok] = np.vstack(res)
        return out

    def embed(self, x, ctx):
        return self.features(x, ctx)


class MVPFormerEnc(Encoder):
    """MVPFormer-S / -M（512 Hz；每个 token = 5 s = 2560 点；多通道一起输入，通道之间有相对位置注意力，最多 128 通道）。
    return_features 的 tokens = (1, C, 段数, D)；2.99 s 的观测段（1531 点）由模型补成 1 段（A4 核对：(1, 8, 1531) → (1, 8, 1, D)）。
    readout：tokens 在段上取平均（只有 1 段时就是这一段）。通道顺序 = 可用通道按 channels.tsv 的顺序。"""

    def __init__(self, path, device, random_init=False, seed=0, sfreq=512, max_channels=128, **_):
        import torch
        from braindecode.models import MVPFormer
        self.torch, self.dev, self.sfreq, self.max_ch = torch, device, float(sfreq), int(max_channels)
        self.m, self.n_reset = load_braindecode(MVPFormer, path, device, random_init, seed)

    def embed(self, x, ctx):
        if len(x) > self.max_ch:
            raise ValueError(f"通道数 {len(x)} > 模型上限 {self.max_ch}")
        t = self.torch
        with t.no_grad():
            f = self.m(t.from_numpy(np.ascontiguousarray(x[None], np.float32)).to(self.dev), return_features=True)
        return f["tokens"][0].mean(dim=1).float().cpu().numpy()        # (C, D)


class PopTEnc(Encoder):
    """PopT（20 M）：输入 = 每个电极的 BrainBERT 特征（768）+ 电极坐标；预训练配置 n_chans = 64，换电极数要重建
    （from_pretrained(n_chans=N)，按 N 缓存）。逐电极表征 = tokens 里去掉 CLS 的部分。coords：mni（MNI mm 直接给）/ none
    （不给坐标，模型退回顺序编号）。**接口按类说明写，前向形状待 C 的冒烟核对**（A4 只核对到 n_chans 不匹配的报错）。"""

    def __init__(self, path, device, random_init=False, seed=0, brainbert=None, coords="mni", **_):
        import torch
        from braindecode.models import PopulationTransformer
        self.torch, self.dev, self.coords = torch, device, coords
        self.cls, self.path, self.rand, self.seed = PopulationTransformer, path, random_init, seed
        self.bb = brainbert                             # BrainBERTEnc（预训练；PopT 的随机初始化版本也用预训练 BrainBERT 的输入）
        self.sfreq = self.bb.sfreq
        self._m: dict = {}
        self.n_reset = -1 if random_init else 0

    def _model(self, n):
        if n not in self._m:
            self._m[n], _ = load_braindecode(self.cls, self.path, self.dev, self.rand, self.seed, n_chans=n)
        return self._m[n]

    def embed(self, x, ctx):
        F = self.bb.features(x, ctx)                    # (n_ch, 768)，Laplacian 后算不出的为 NaN
        out = np.full((len(x), 512), np.nan, np.float32)
        ok = np.isfinite(F).all(axis=1)
        if ok.sum() < 2:
            return out
        t = self.torch
        kw = {}
        if self.coords == "mni" and ctx.get("xyz") is not None:
            xyz = np.nan_to_num(ctx["xyz"][ok])
            kw["coords"] = t.from_numpy(xyz[None].astype(np.float32)).to(self.dev)
        with t.no_grad():
            f = self._model(int(ok.sum()))(t.from_numpy(F[ok][None]).to(self.dev), return_features=True, **kw)
        tok = f["tokens"][0]
        if tok.shape[0] == ok.sum() + 1:                # 去掉 CLS
            tok = tok[1:]
        out[ok] = tok.reshape(int(ok.sum()), -1).float().cpu().numpy()
        return out


REGISTRY = {
    "spec_test": lambda c, path, dev, rand, seed: SpecTest(**c),
    "chronos": lambda c, path, dev, rand, seed: ChronosEnc(path, dev, rand, seed, **c),
    "moment": lambda c, path, dev, rand, seed: MomentEnc(path, dev, rand, seed, **c),
    "brainbert": lambda c, path, dev, rand, seed: BrainBERTEnc(path, dev, rand, seed, **c),
    "mvpformer_s": lambda c, path, dev, rand, seed: MVPFormerEnc(path, dev, rand, seed, **c),
    "mvpformer_m": lambda c, path, dev, rand, seed: MVPFormerEnc(path, dev, rand, seed, **c),
    "popt": None,                                       # 需要 BrainBERT 的输入，见 make()
}


def make(name: str, enc_cfg: dict, models_dir: str | Path, device: str) -> Encoder:
    """name = 配置里的模型名，可加 “-rand”。enc_cfg = p206_encoders.yaml 全文。"""
    base, rand = (name[:-5], True) if name.endswith("-rand") else (name, False)
    if base not in REGISTRY:
        raise KeyError(f"未知模型 {name}；可用：{sorted(REGISTRY)}（加 -rand 为随机初始化）")
    c = dict(enc_cfg.get(base, {}))
    path = Path(models_dir) / c.pop("path", base)
    seed = int(c.pop("rand_seed", enc_cfg.get("rand_seed", 0)))
    if base == "popt":
        bc = dict(enc_cfg.get("brainbert", {}))
        bpath = Path(models_dir) / bc.pop("path", "brainbert")
        bc.pop("rand_seed", None)
        enc = PopTEnc(path, device, rand, seed, brainbert=BrainBERTEnc(bpath, device, False, seed, **bc), **c)
    else:
        enc = REGISTRY[base](c, path, device, rand, seed)
    enc.name = name
    return enc
