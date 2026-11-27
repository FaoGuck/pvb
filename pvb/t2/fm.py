"""P206：冻结基础模型的表征在 T2 上怎么变成预测特征（docs/06 第 3 节；框架 2026-11-27）。

与 P204 / P205 用同一套观测段和防泄漏规则，只是把“段内连接度”换成“段内每个通道过一遍冻结模型的表征”：
  1. 观测段 = observation.segment_table（[上一个脉冲 + T_tail, 下一个脉冲 − pre_gap)，长段切块，归属 owner = 上一个
     脉冲的刺激对）；每段的可用通道 = observation.segment_valid（屏蔽 owner / 下一个脉冲的刺激电极、饱和、NaN）
  2. 每段：可用通道降采样到模型的输入采样率 → encoder.embed → 每个通道一个 D 维向量 E_c；
     同时算段内“表征相似度”= 各通道表征减去本段的通道均值后的余弦（基础模型版的连接度）
  3. **按 owner 累加**：emb_sums[owner, ch, :]、emb_counts[owner, ch]、sim_sums[owner, i, j]、sim_counts[owner, i, j]
     → 预测留出刺激对时，只对“可用折”的 owner 求和再平均（EmbedPool.mean），留出刺激对自己的段不会进入
  4. 刺激对 p × 记录电极 r 的特征（fm_blocks）：
       Mh（高维，预测头里只在训练集上 PCA）：E_r、s = (E_e1 + E_e2) / 2、E_r ⊙ s、|E_r − s|
       Ms（标量）：cos(E_r, s)、相似度 S(e1, r)、S(e2, r) 的均值与最大值
  只在段内、通道之间做的运算（降采样、减通道均值、z 分数）不跨段、不跨刺激对，所以不会泄漏。
另有评估用的两个小函数：逐刺激对内的 Spearman（次要指标，00 第 14 节 11-26）、按距离分段的 AUROC。
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from pvb.t2 import baselines as bl, observation as obs


def resample(x: np.ndarray, sf_in: float, sf_out: float) -> np.ndarray:
    """x (n_ch, n)。vhdr 的采样率可能是 2048.0002 这种舍入值，先取整（P201）。"""
    a = round(sf_in) if abs(sf_in - round(sf_in)) < 0.01 else sf_in
    if abs(a - sf_out) < 1e-9:
        return x
    fr = Fraction(sf_out / a).limit_denominator(4096)
    return resample_poly(x, fr.numerator, fr.denominator, axis=1)


def owner_embed_accumulators(reader, chans: list[str], picks: list[int], segs: pd.DataFrame, sfreq: float,
                             cfg_obs: dict, encoders: dict, xyz: dict | None = None) -> dict:
    """reader 需有 segments(picks, starts, n_samp)。encoders：{模型名: Encoder}（pvb/t2/encoders.py）。
    每段只读一次信号，所有模型共用（同一采样率只降采样一次）。返回 {模型名: 累加量 dict}。"""
    owners = sorted(segs.owner_pair.unique())
    oi = {o: i for i, o in enumerate(owners)}
    ci = {c: i for i, c in enumerate(chans)}
    n_o, n_ch = len(owners), len(chans)
    st = {m: dict(emb=None, ec=np.zeros((n_o, n_ch), np.int32), sim=np.zeros((n_o, n_ch, n_ch)),
                  sc=np.zeros((n_o, n_ch, n_ch), np.int32), nseg=np.zeros(n_o, np.int32), sec=0.0, n_chseg=0)
          for m in encoders}
    t_read = 0.0
    for r in segs.itertuples(index=False):
        t0 = time.time()
        x = reader.segments(picks, np.array([r.start]), r.stop - r.start)[0].astype(np.float64)
        valid = obs.segment_valid(x, r.owner_pair, r.next_pair, ci, cfg_obs)
        t_read += time.time() - t0
        if valid.sum() < 2:
            continue
        idx = np.where(valid)[0]
        xv = np.nan_to_num(x[valid])
        names = [chans[i] for i in idx]
        ctx = dict(names=names, xyz=None if xyz is None else np.array([xyz.get(n, (np.nan,) * 3) for n in names], float))
        o = oi[r.owner_pair]
        cache = {}
        for m, enc in encoders.items():
            t0 = time.time()
            if enc.sfreq not in cache:
                cache[enc.sfreq] = resample(xv, sfreq, enc.sfreq).astype(np.float32)
            E = np.asarray(enc.embed(cache[enc.sfreq], ctx), np.float64)          # (n_valid, D)
            s = st[m]
            if s["emb"] is None:
                s["emb"] = np.zeros((n_o, n_ch, E.shape[1]))
            ok = np.isfinite(E).all(axis=1)
            io, Eo = idx[ok], E[ok]
            s["emb"][o, io] += Eo
            s["ec"][o, io] += 1
            s["nseg"][o] += 1
            s["n_chseg"] += int(ok.sum())
            if len(io) >= 2:
                Ec = Eo - Eo.mean(axis=0, keepdims=True)
                nrm = np.linalg.norm(Ec, axis=1)
                good = nrm > 0
                if good.sum() >= 2:
                    Ec, ig = Ec[good] / nrm[good, None], io[good]
                    s["sim"][o][np.ix_(ig, ig)] += Ec @ Ec.T
                    s["sc"][o][np.ix_(ig, ig)] += 1
            s["sec"] += time.time() - t0
    out = {}
    for m, s in st.items():
        D = 0 if s["emb"] is None else s["emb"].shape[2]
        out[m] = dict(owners=np.array(owners), channels=np.array(chans),
                      emb_sums=(s["emb"] if D else np.zeros((n_o, n_ch, 0))).astype(np.float32), emb_counts=s["ec"],
                      sim_sums=s["sim"].astype(np.float32), sim_counts=s["sc"], n_segments=s["nseg"],
                      dim=np.int64(D), seconds_model=np.float64(s["sec"]), n_chseg=np.int64(s["n_chseg"]),
                      seconds_read=np.float64(t_read))
    return out


class EmbedPool:
    """把一名患者各 run 的表征累加量合到患者级通道表上，按“可用 owner 集合”求平均（与 baselines.ObsPool 同一写法）。"""

    def __init__(self, accs: list[dict]):
        self.chans = sorted({c for a in accs for c in a["channels"]})
        self.ci = {c: i for i, c in enumerate(self.chans)}
        self.owners = sorted({o for a in accs for o in a["owners"]})
        oi = {o: i for i, o in enumerate(self.owners)}
        D = int(max(int(a["dim"]) for a in accs))
        n, k = len(self.chans), len(self.owners)
        self.dim = D
        self.emb = np.zeros((k, n, D)); self.ec = np.zeros((k, n))
        self.sim = np.zeros((k, n, n)); self.sc = np.zeros((k, n, n))
        for a in accs:
            if int(a["dim"]) != D:                     # 这个 run 一段都没有可用通道（dim = 0）
                continue
            idx = np.array([self.ci[c] for c in a["channels"]])
            for j, o in enumerate(a["owners"]):
                self.emb[oi[o]][idx] += a["emb_sums"][j]
                self.ec[oi[o]][idx] += a["emb_counts"][j]
                self.sim[oi[o]][np.ix_(idx, idx)] += a["sim_sums"][j]
                self.sc[oi[o]][np.ix_(idx, idx)] += a["sim_counts"][j]
        self._cache: dict = {}

    def owner_mask(self, allowed: set) -> np.ndarray:
        return np.array([o in allowed for o in self.owners], dtype=bool)

    def mean(self, owner_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """→ E (n_ch, D)、S (n_ch, n_ch)；没有可用段的位置为 NaN，S 的对角为 NaN。"""
        key = owner_mask.tobytes()
        if key not in self._cache:
            with np.errstate(all="ignore"):
                E = self.emb[owner_mask].sum(axis=0) / self.ec[owner_mask].sum(axis=0)[:, None]
                S = self.sim[owner_mask].sum(axis=0) / self.sc[owner_mask].sum(axis=0)
            d = np.arange(S.shape[0])
            S[d, d] = np.nan
            self._cache[key] = (E, S)
        return self._cache[key]


def fm_blocks(units: pd.DataFrame, pool: EmbedPool, allowed: set) -> dict:
    """units 需有 stim_pair、channel。→ {"Mh": (n, 4D) float32, "Ms": (n, 3)}；缺的为 NaN（预测头里用训练集中位数补）。"""
    E, S = pool.mean(pool.owner_mask(allowed))
    n, D = len(units), pool.dim
    ci = pool.ci
    r = np.array([ci.get(c, -1) for c in units.channel])
    e = np.array([[ci.get(x, -1) for x in p.split("-")] for p in units.stim_pair]).reshape(n, 2)
    Ep = np.vstack([E, np.full((1, D), np.nan)])          # 下标 −1 → 最后一行 NaN
    Sp = np.full((S.shape[0] + 1, S.shape[1] + 1), np.nan); Sp[:-1, :-1] = S
    R = Ep[r]
    with np.errstate(all="ignore"):
        s = np.nanmean(np.stack([Ep[e[:, 0]], Ep[e[:, 1]]]), axis=0)
        cos = (R * s).sum(axis=1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(s, axis=1))
        s1, s2 = Sp[e[:, 0], r], Sp[e[:, 1], r]
        smean = np.nanmean(np.stack([s1, s2]), axis=0)
        smax = np.nanmax(np.stack([s1, s2]), axis=0)
    Mh = np.hstack([R, s, R * s, np.abs(R - s)]).astype(np.float32)
    return {"Mh": Mh, "Ms": np.stack([cos, smean, smax], axis=1)}


def make_extra(pool: EmbedPool):
    """给 baselines.cv_predict 的 extra 参数。"""
    return lambda units, allowed: fm_blocks(units, pool, allowed)


# ---------------- 评估用 ----------------

def within_pair_spearman(df: pd.DataFrame, y, s, min_n: int) -> tuple[float, int]:
    """逐刺激对内的 Spearman（00 第 14 节 11-26 写死）：每个刺激对内有 N1 的记录电极 ≥ min_n 时算，取刺激对的中位数。"""
    d = pd.DataFrame({"p": df.stim_pair.to_numpy(), "y": np.asarray(y, float), "s": np.asarray(s, float)})
    v = [bl.safe_spearman(g.y, g.s) for _, g in d.groupby("p") if len(g) >= min_n]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.median(v)) if v else float("nan")), len(v)


def auc_by_distance(y, s, dist, edges: list[float]) -> list[float]:
    """按距离分段（< e0、e0–e1、…、≥ e_last）的 AUROC。"""
    y, s, dist = np.asarray(y, bool), np.asarray(s, float), np.asarray(dist, float)
    b = np.digitize(dist, edges)
    return [bl.safe_auc(y[b == k], s[b == k]) for k in range(len(edges) + 1)]
