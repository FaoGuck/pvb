# 来自 S2R 505b298 的 s2r/embeddings/models.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S7 三个基础模型的统一封装（依据 docs/02 第 4 节核实的官方代码；配置见 configs/models/*.yaml）。

权重和官方源码由 tools/fm_download.py 下载到 $PRJ_DATA/models/<model>/：
    reve/reve-base/、reve/reve-positions/     Hugging Face 快照（含 modeling_reve.py，trust_remote_code）
    labram/src/                               GitHub 935963004/LaBraM 固定提交的完整源码（含 checkpoints/labram-base.pth）
    cbramod/src/ + cbramod/pretrained_weights.pth   GitHub wjq-learning/CBraMod 源码 + HF 权重
用法：w = get_wrapper("reve", models_dir, device, pretrained=True)；layers = w.layers(x200, ch_names)
随机初始化对照（C1）：pretrained=False, seed=k —— 同结构、同构造参数，只是不加载权重。
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch
import yaml

from pvb.embeddings.common import FMWrapper, patch_tail


def load_model_cfg(code_root: Path, name: str) -> dict:
    return yaml.safe_load(open(Path(code_root) / "configs" / "models" / f"{name}.yaml", encoding="utf-8"))


# ---------------- REVE ----------------
class Reve(FMWrapper):
    name = "reve"

    def __init__(self, cfg, device, models_dir: Path, pretrained=True, seed=0):
        super().__init__(cfg, device)
        # modeling_reve.py 在 import 时检测 flash_attn：装了就让所有层走 FlashAttention（只支持 fp16/bf16，权重是 fp32 → 报错，
        # 11-23 s2r-fm-setup-1/2）；没装就走 F.scaled_dot_product_attention（同一公式，fp32 下自动选 EFFICIENT/MATH 实现）。
        # nvcr 镜像自带 flash_attn，这里让它对 import 不可见，固定走 fp32 的 SDPA 路径。
        sys.modules["flash_attn"] = None
        from transformers import AutoConfig, AutoModel

        d = Path(models_dir) / cfg["local_dir"]
        self.pos_bank = AutoModel.from_pretrained(str(d / "reve-positions"), trust_remote_code=True).to(device).eval()
        if pretrained:
            m = AutoModel.from_pretrained(str(d / "reve-base"), trust_remote_code=True)
        else:
            torch.manual_seed(seed)
            conf = AutoConfig.from_pretrained(str(d / "reve-base"), trust_remote_code=True)
            m = AutoModel.from_config(conf, trust_remote_code=True)
        self.model = m.to(device).eval()
        self.alias = cfg.get("channel_alias", {})
        self.size, self.step = int(cfg["patch"]["size"]), int(cfg["patch"]["step"])

    @torch.no_grad()
    def positions(self, ch_names):
        names = [self.alias.get(c, c) for c in ch_names]
        pos = self.pos_bank(names)
        if pos.shape[0] != len(names):
            raise ValueError(f"REVE 位置库只找到 {pos.shape[0]} / {len(names)} 个电极: {names}")
        return pos

    @torch.no_grad()
    def layers(self, x, ch_names):
        x = patch_tail(x, self.size, self.step)                     # 最后一个 patch 对齐窗口末端
        B, C, T = x.shape
        h = (T - self.size) // self.step + 1
        pos = self.positions(ch_names).expand(B, -1, -1)
        outs = self.model(x, pos, return_output=True)               # 23 项，每项 (B, C·h, 512)
        return [o.reshape(B, C, h, -1) for o in outs]

    @torch.no_grad()
    def final(self, x, ch_names):
        """官方默认输出 (B, C, h, 512)，冒烟测试用来核对 layers() 的 token 排列。"""
        x = patch_tail(x, self.size, self.step)
        pos = self.positions(ch_names).expand(x.shape[0], -1, -1)
        return self.model(x, pos)


# ---------------- LaBraM ----------------
def _install_timm_shim():
    """LaBraM 依赖 timm==0.4.12 的旧 API；推理只用到下面几个函数，这里提供等价实现，免装旧版 timm。
    drop_path / DropPath 只在训练时起作用（eval 时恒等）；trunc_normal_ 只用于初始化（随后被权重覆盖）。"""
    if "timm.models.layers" in sys.modules:
        return

    def drop_path(x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep = 1 - drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob or 0.0

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def trunc_normal_(t, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return torch.nn.init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)

    def register_model(fn):
        return fn

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    registry = types.ModuleType("timm.models.registry")
    for k, v in dict(drop_path=drop_path, DropPath=DropPath, to_2tuple=to_2tuple, trunc_normal_=trunc_normal_).items():
        setattr(layers, k, v)
    registry.register_model = register_model
    models.layers, models.registry, models.register_model = layers, registry, register_model
    timm.models = models
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers,
                        "timm.models.registry": registry})


def labram_channel_list(src: Path) -> list[str]:
    """从官方 utils.py 里读出 standard_1020 名表（用 ast 解析，不 import utils.py，避免它的大量依赖）。"""
    tree = ast.parse((src / "utils.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "standard_1020" for t in node.targets):
            return [str(x) for x in ast.literal_eval(node.value)]
    raise ValueError("utils.py 里没找到 standard_1020")


def labram_state_dict(path: Path) -> dict:
    """按官方 run_class_finetuning.py：取 'model' 或 'module'，只保留 'student.' 开头的键并去前缀；去掉 relative_position_index。"""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck
    for k in ("model", "module"):
        if isinstance(ck, dict) and k in ck:
            sd = ck[k]
            break
    if any(k.startswith("student.") for k in sd):
        sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
    sd = {k: v for k, v in sd.items() if "relative_position_index" not in k}
    if not sd:
        raise ValueError("LaBraM 权重为空（键名格式不符合预期）")
    return sd


class Labram(FMWrapper):
    name = "labram"

    def __init__(self, cfg, device, models_dir: Path, pretrained=True, seed=0):
        super().__init__(cfg, device)
        src = Path(models_dir) / cfg["local_dir"] / "src"
        _install_timm_shim()
        spec = importlib.util.spec_from_file_location("labram_modeling_finetune", src / "modeling_finetune.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.ch_list = labram_channel_list(src)
        sd = labram_state_dict(src / cfg["source"]["weights"])
        keys = set(sd)
        self.build_kw = dict(use_abs_pos_emb="pos_embed" in keys,
                             use_rel_pos_bias=any("relative_position_bias_table" in k for k in keys),
                             qkv_bias=any(k.endswith("q_bias") for k in keys),
                             init_values=0.1 if any("gamma_1" in k for k in keys) else 0.0)
        torch.manual_seed(seed)
        m = mod.labram_base_patch200_200(**self.build_kw)
        self.load_report = None
        if pretrained:
            r = m.load_state_dict(sd, strict=False)
            self.load_report = {"missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys),
                                "n_ckpt": len(sd), "n_model": len(m.state_dict())}
            bad = [k for k in r.missing_keys if not k.startswith(("head.", "fc_norm."))]
            if bad:
                raise ValueError(f"LaBraM 权重缺少键: {bad[:10]} …（共 {len(bad)}）")
        self.model = m.to(device).eval()

    def input_chans(self, ch_names):
        missing = [c for c in ch_names if c not in self.ch_list]
        if missing:
            raise ValueError(f"LaBraM 名表里没有: {missing}")
        return [0] + [self.ch_list.index(c) + 1 for c in ch_names]

    @torch.no_grad()
    def layers(self, x, ch_names):
        B, C, T = x.shape
        S = T // 200
        x = x[..., T - S * 200:].reshape(B, C, S, 200)
        caps = []
        hooks = [self.model.blocks[0].register_forward_pre_hook(lambda m, a: caps.append(a[0]))]
        hooks += [b.register_forward_hook(lambda m, a, o: caps.append(o)) for b in self.model.blocks]
        try:
            self.model.forward_features(x, input_chans=self.input_chans(ch_names))
        finally:
            for h in hooks:
                h.remove()
        return [c[:, 1:].reshape(B, C, S, -1) for c in caps]   # 去掉 CLS；patch token 按 (导联, 时间) 排列


# ---------------- CBraMod ----------------
class Cbramod(FMWrapper):
    name = "cbramod"

    def __init__(self, cfg, device, models_dir: Path, pretrained=True, seed=0):
        super().__init__(cfg, device)
        d = Path(models_dir) / cfg["local_dir"]
        sys.path.insert(0, str(d / "src"))
        try:
            from models.cbramod import CBraMod
        finally:
            sys.path.pop(0)
        torch.manual_seed(seed)
        m = CBraMod()
        if pretrained:
            m.load_state_dict(torch.load(str(d / cfg["source"]["weights"]), map_location="cpu"), strict=True)
        m.proj_out = torch.nn.Identity()
        self.model = m.to(device).eval()

    @torch.no_grad()
    def layers(self, x, ch_names):
        B, C, T = x.shape
        S = T // 200
        x = x[..., T - S * 200:].reshape(B, C, S, 200)
        h = self.model.patch_embedding(x, None)
        outs = [h]
        for layer in self.model.encoder.layers:
            h = layer(h)
            outs.append(h)
        return outs

    @torch.no_grad()
    def final(self, x, ch_names):
        B, C, T = x.shape
        S = T // 200
        return self.model(x[..., T - S * 200:].reshape(B, C, S, 200))


WRAPPERS = {"reve": Reve, "labram": Labram, "cbramod": Cbramod}


def get_wrapper(name: str, code_root: Path, models_dir: Path, device, pretrained=True, seed=0) -> FMWrapper:
    return WRAPPERS[name](load_model_cfg(code_root, name), device, models_dir, pretrained=pretrained, seed=seed)
