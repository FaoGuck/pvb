# 来自 S2R 505b298 的 s2r/embeddings/common.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S7 公共部分：模型输入预处理、各层 token 的池化、模型统一接口。

输入 = S3 的 master（脉冲前 4 s、1000 Hz、只去均值 + 抗混叠降采样，原参考 POz）。
每个窗口单独处理（防泄漏 6.3）：取 master 最后 W 秒 → 窗口内对称延拓 + 零相位陷波/带通 → 降到 200 Hz → 平均参考 → µV ÷ scale。
（与 S3 clean 的做法相同，只是频段和采样率按各模型官方设置，见 configs/models/*.yaml。）

统一接口：每个模型封装实现 layers(x200, ch_names) → list[Tensor]，每个元素形状 (B, C, S, D)，
C = 导联，S = 时间 patch（最后一个 patch 结束于窗口末端），D = 表征维度；列表第 0 项是进入第一个 block 之前的嵌入。
"""
from __future__ import annotations

import numpy as np
import torch

from pvb.preprocess.epoch import clean_filters, clean_window, spatial_matrix


def preprocess(master: np.ndarray, fs_in: int, window_s: float, pp: dict) -> np.ndarray:
    """master: (n, C, T) µV → (n, C, window_s·fs) float32，已除以 scale。"""
    n_in = int(round(window_s * fs_in))
    x = master[..., -n_in:].astype(np.float64)
    sos = clean_filters(fs_in, pp["notch_hz"], pp.get("notch_q", 30), pp["band_hz"], pp.get("filter_order", 4))
    spatial = spatial_matrix(x.shape[-2], None, average_ref=pp.get("reference") == "average")
    y = clean_window(x, fs_in, int(pp["fs"]), sos, spatial)
    return (y / float(pp["scale"])).astype(np.float32)


def patch_tail(x: torch.Tensor, size: int, step: int) -> torch.Tensor:
    """取窗口最后 size + step·(h−1) 个样本，使最后一个 patch 对齐窗口末端（REVE 的 unfold 从开头切、丢弃末尾余数）。"""
    T = x.shape[-1]
    h = (T - size) // step + 1
    return x[..., T - (size + step * (h - 1)):]


POOLS = ("meanpool", "last1s", "c4")
C4_NEIGHBORHOOD = ["C4", "FC2", "FC6", "CP2", "CP6"]


def pool_layers(layers: list[torch.Tensor], ch_names: list[str], pools=POOLS) -> dict[str, np.ndarray]:
    """layers: L 个 (B, C, S, D) → {pool: (B, L, D) float32}。
    meanpool = 所有 token 平均；last1s = 最后一个时间 patch（最后 1 s）上所有导联平均；c4 = C4 及其 4 个邻居、所有时间平均。"""
    idx = [ch_names.index(c) for c in C4_NEIGHBORHOOD]
    out = {p: [] for p in pools}
    for h in layers:
        h = h.float()
        if "meanpool" in out:
            out["meanpool"].append(h.mean(dim=(1, 2)))
        if "last1s" in out:
            out["last1s"].append(h[:, :, -1].mean(dim=1))
        if "c4" in out:
            out["c4"].append(h[:, idx].mean(dim=(1, 2)))
    return {p: torch.stack(v, dim=1).cpu().numpy().astype(np.float32) for p, v in out.items()}


class FMWrapper:
    """各模型封装的基类。子类设置 self.model、self.cfg，并实现 layers()。"""
    name = "base"

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def layers(self, x: torch.Tensor, ch_names: list[str]) -> list[torch.Tensor]:
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
