# 来自 S2R 505b298 的 s2r/probing/cv.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S8 探测：被试内嵌套留一 block 交叉验证（总纲 6.8），岭回归 / L2 逻辑回归，置换检验（总纲 S9 的被试水平部分）。

做法要点
- 外层：留一 block（D1 为 6 折）；内层：在外层训练集上再留一 block，选正则强度；选定后用整个外层训练集重拟合，预测外层测试 block
- 标准化（和可选 PCA）只在当前训练集上估计，再作用到测试集（防泄漏 6.3 第 3 条）
- 置换：在 block 内打乱目标 P 次。真实目标与 P 个置换目标作为 Y 的 1+P 列**一起**走完整套嵌套交叉验证
  （每一列独立选正则强度），所以置换分布包含了"选超参"带来的乐观偏差
- 指标：各外层折的测试预测拼接后算（回归 Spearman，分类 AUC）；p = (1 + #{置换 ≥ 真实}) / (1 + P)
- 计算用 torch：有 GPU 用 GPU，没有就用 CPU（结果相同，只是慢）
"""
from __future__ import annotations

import numpy as np
import torch


def grid(spec) -> np.ndarray:
    if isinstance(spec, dict) and "logspace" in spec:
        a, b, n = spec["logspace"]
        return np.logspace(a, b, int(n))
    if isinstance(spec, dict) and "linspace" in spec:
        a, b, n = spec["linspace"]
        return np.linspace(a, b, int(n))
    return np.asarray(spec, dtype=float)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 目标与置换 ----------
def block_median_split(y: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """block 内中位数切分：大于该 block 中位数 → 1（并列归 0）。"""
    out = np.zeros(len(y), dtype=np.float64)
    for b in np.unique(blocks):
        m = blocks == b
        out[m] = (y[m] > np.median(y[m])).astype(float)
    return out


def permutation_index(blocks: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """返回 (n_perm, n) 的下标矩阵，只在 block 内打乱（全体打乱时传入全 0 的 blocks）。"""
    n = len(blocks)
    P = np.tile(np.arange(n), (n_perm, 1))
    for b in np.unique(blocks):
        ids = np.where(blocks == b)[0]
        P[:, ids] = rng.permuted(np.tile(ids, (n_perm, 1)), axis=1)
    return P


def with_permutations(y: np.ndarray, perm_idx: np.ndarray) -> np.ndarray:
    """n × (1 + P)：第 0 列是真实目标。"""
    return np.column_stack([y] + [y[p] for p in perm_idx])


# ---------- 指标（按列批量） ----------
def _ranks(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.argsort(torch.argsort(x, dim=dim), dim=dim).to(x.dtype)


def spearman_cols(pred: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """pred, Y: (..., n, C) → (..., C)。"""
    rp, ry = _ranks(pred, -2), _ranks(Y, -2)
    rp = rp - rp.mean(-2, keepdim=True)
    ry = ry - ry.mean(-2, keepdim=True)
    return (rp * ry).sum(-2) / torch.sqrt((rp ** 2).sum(-2) * (ry ** 2).sum(-2))


def auc_cols(score: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """score: (..., n, C)，Y: (n, C) 取 0/1 → (..., C)。Mann–Whitney 公式。"""
    r = _ranks(score, -2) + 1
    n1 = Y.sum(0)
    n0 = Y.shape[0] - n1
    return ((r * Y).sum(-2) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------- 训练集上估计的预处理 ----------
def standardize(Ztr: torch.Tensor, Zte: torch.Tensor, pca: int | None, boost_cols=None, boost: float = 100.0):
    """boost_cols：标准化后放大这些列 boost 倍 → 它们的系数受到的 L2 惩罚缩小 boost² 倍，等于不惩罚
    （E110：把"上一个 MEP"和脑电特征放在一起时，不让 125 个脑电特征的惩罚强度压住这 1 列）。不能与 PCA 同用。"""
    mu = Ztr.mean(0, keepdim=True)
    sd = Ztr.std(0, keepdim=True)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    if boost_cols:
        if pca:
            raise ValueError("boost_cols 不能与 PCA 同用")
        Ztr[:, boost_cols] *= boost
        Zte[:, boost_cols] *= boost
    if pca and pca < Ztr.shape[1]:
        _, _, Vt = torch.linalg.svd(Ztr, full_matrices=False)
        V = Vt[:pca].T
        Ztr, Zte = Ztr @ V, Zte @ V
    return Ztr, Zte


# ---------- 模型：对所有正则强度、所有列同时求解 ----------
def ridge_all(Ztr, Ytr, Zte, alphas: torch.Tensor) -> torch.Tensor:
    """返回 (A, n_te, C) 的预测。Ztr 已在训练集上标准化（列均值 0）。"""
    ym = Ytr.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Ztr, full_matrices=False)
    UtY = U.T @ (Ytr - ym)                                   # r × C
    d = S[None, :] / (S[None, :] ** 2 + alphas[:, None])     # A × r
    ZV = Zte @ Vt.T                                          # n_te × r
    return torch.einsum("tr,ar,rc->atc", ZV, d, UtY) + ym


def _spd_solve(H: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """批量解 H x = G（H 对称正定）：先 Cholesky（快约 2 倍），个别失败的退回通用求解。"""
    Lc, info = torch.linalg.cholesky_ex(H)
    x = torch.cholesky_solve(G, Lc)
    bad = info != 0
    if bool(bad.any()):
        x[bad] = torch.linalg.solve(H[bad], G[bad])
    return x


def _rotate(Ztr: torch.Tensor, Zte: torch.Tensor, eps: float = 1e-6):
    """把训练集特征转到它的行空间坐标：Ztr = U S Vᵀ → F = U S（n × r），Fte = Zte V。
    L2 惩罚是各向同性的，最优权重一定落在 Ztr 的行空间里（w = V c，‖w‖ = ‖c‖），所以在 r 维坐标里拟合
    与原坐标完全等价，预测相同；而 r ≤ n−1，特征维数大于样本数时（如 REVE 512 维）维数变小。
    Ztr 已按列中心化，所以 F 的列与全 1 列正交，XᵀX 在这组坐标下是对角阵 diag(n, S²)。"""
    U, S, Vt = torch.linalg.svd(Ztr, full_matrices=False)
    keep = S > eps * S[0]
    return U[:, keep] * S[keep], Zte @ Vt[keep].T


def logistic_all(Ztr, Ytr, Zte, lambdas: torch.Tensor, iters: int, tol: float = 1e-6,
                 cg_iters: int = 200, cg_tol: float = 1e-5) -> torch.Tensor:
    """L2 逻辑回归（截距不惩罚），截断牛顿法（11-23 起的默认解法）；λ 从大到小热启动。返回 (L, n_te, C) 的 logit。

    与 logistic_all_newton 求的是同一个凸问题的唯一最优解，区别只在怎么解牛顿方程：
    - 先转到行空间坐标（_rotate），维数 p → r ≤ n−1
    - 牛顿方程 H d = g 用预条件共轭梯度（PCG）解，只需要矩阵乘 X(W⊙(X v))，不显式构造 1001 个 (r+1)² 的
      Hessian、也不做 1001 次 Cholesky（后者是旧解法 90% 的耗时）；预条件用 Hessian 的对角线
    - 收敛判据与旧解法相同：max|牛顿步| < tol
    """
    F, Fte = _rotate(Ztr, Zte)
    n, r = F.shape
    C = Ytr.shape[1]
    dt, dv = Ztr.dtype, Ztr.device
    Xb = torch.cat([torch.ones(n, 1, dtype=dt, device=dv), F], 1)
    Xt = torch.cat([torch.ones(Fte.shape[0], 1, dtype=dt, device=dv), Fte], 1)
    pen = torch.ones(r + 1, 1, dtype=dt, device=dv)
    pen[0] = 0
    jitter = 1e-6
    Xb2 = Xb ** 2
    B = torch.zeros(r + 1, C, dtype=dt, device=dv)
    order = torch.argsort(lambdas, descending=True)
    out = [None] * len(lambdas)
    tiny = torch.finfo(dt).tiny
    for li in order.tolist():
        lam = lambdas[li]
        reg = lam * pen + jitter                                 # (r+1) × 1
        for _ in range(iters):
            mu = torch.sigmoid(Xb @ B)                            # n × C
            W = mu * (1 - mu)
            G = Xb.T @ (mu - Ytr) + lam * pen * B                 # (r+1) × C
            Minv = 1.0 / (Xb2.T @ W + reg)                        # Hessian 对角线的倒数
            # PCG：各列独立，向量化
            x = torch.zeros_like(G)
            res = G.clone()
            z = Minv * res
            d = z.clone()
            rz = (res * z).sum(0)
            g_norm = G.norm(dim=0).clamp_min(tiny)
            for _k in range(cg_iters):
                Hd = Xb.T @ (W * (Xb @ d)) + reg * d
                dHd = (d * Hd).sum(0)
                alpha = torch.where(dHd > 0, rz / dHd.clamp_min(tiny), torch.zeros_like(rz))
                x = x + alpha * d
                res = res - alpha * Hd
                if float((res.norm(dim=0) / g_norm).max()) < cg_tol:
                    break
                z = Minv * res
                rz_new = (res * z).sum(0)
                beta = torch.where(rz > 0, rz_new / rz.clamp_min(tiny), torch.zeros_like(rz))
                d = z + beta * d
                rz = rz_new
            B = B - x
            if float(x.abs().max()) < tol:
                break
        out[li] = Xt @ B
    return torch.stack(out)


def logistic_all_newton(Ztr, Ytr, Zte, lambdas: torch.Tensor, iters: int, tol: float = 1e-6) -> torch.Tensor:
    """旧解法（E101 用的）：L2 逻辑回归（截距不惩罚），完整牛顿法；λ 从大到小热启动。返回 (L, n_te, C) 的 logit。
    每步对每一列显式构造 (p+1)² 的 Hessian 并做 Cholesky；p=512、1001 列时很慢。保留作对照。"""
    n, p = Ztr.shape
    C = Ytr.shape[1]
    one_tr = torch.ones(n, 1, dtype=Ztr.dtype, device=Ztr.device)
    Xb = torch.cat([one_tr, Ztr], 1)
    Xt = torch.cat([torch.ones(Zte.shape[0], 1, dtype=Ztr.dtype, device=Ztr.device), Zte], 1)
    R = torch.eye(p + 1, dtype=Ztr.dtype, device=Ztr.device)
    R[0, 0] = 0
    jitter = 1e-6 * torch.eye(p + 1, dtype=Ztr.dtype, device=Ztr.device)
    B = torch.zeros(p + 1, C, dtype=Ztr.dtype, device=Ztr.device)
    order = torch.argsort(lambdas, descending=True)
    out = [None] * len(lambdas)
    for li in order.tolist():
        lam = lambdas[li]
        for _ in range(iters):
            mu = torch.sigmoid(Xb @ B)                        # n × C
            W = mu * (1 - mu)
            G = Xb.T @ (mu - Ytr) + lam * (R @ B)             # (p+1) × C
            H = torch.einsum("ni,nc,nj->cij", Xb, W, Xb) + lam * R + jitter
            step = _spd_solve(H, G.T.unsqueeze(-1)).squeeze(-1).T
            B = B - step
            if float(step.abs().max()) < tol:
                break
        out[li] = Xt @ B
    return torch.stack(out)


# ---------- 嵌套留一 block 交叉验证 ----------
def nested_lobo(X: np.ndarray, Y: np.ndarray, blocks: np.ndarray, task: str, cfg: dict,
                pca: int | None, dev: torch.device, boost_cols=None) -> dict:
    """X: n × p；Y: n × C（第 0 列真实目标）。返回 {'pred': n × C 外层测试预测, 'metric': C, 'chosen': 每折的超参下标}。"""
    dt = torch.float32
    Xt_ = torch.as_tensor(X, dtype=dt, device=dev)
    Yt_ = torch.as_tensor(Y, dtype=dt, device=dev)
    if task == "regression":
        hp = torch.as_tensor(grid(cfg["ridge"]["alphas"]), dtype=dt, device=dev)
        fit = lambda a, b, c: ridge_all(a, b, c, hp)  # noqa: E731
        inner_metric = cfg["ridge"].get("inner_metric", "mse")
    else:
        hp = torch.as_tensor(grid(cfg["logistic"]["lambdas"]), dtype=dt, device=dev)
        iters = int(cfg["logistic"].get("newton_iters", 25))
        tol = float(cfg["logistic"].get("tol", 1e-6))
        solver = cfg["logistic"].get("solver", "newton_cg")
        if solver == "newton_cg":
            fit = lambda a, b, c: logistic_all(a, b, c, hp, iters, tol)  # noqa: E731
        elif solver == "newton":
            fit = lambda a, b, c: logistic_all_newton(a, b, c, hp, iters, tol)  # noqa: E731
        else:
            raise ValueError(solver)
        inner_metric = cfg["logistic"].get("inner_metric", "auc")

    n, C = Y.shape
    pred = torch.empty(n, C, dtype=dt, device=dev)
    chosen = {}
    ub = np.unique(blocks)
    for b_out in ub:
        te = np.where(blocks == b_out)[0]
        tr = np.where(blocks != b_out)[0]
        # 内层：外层训练集上留一 block
        inner_pred = torch.empty(len(hp), len(tr), C, dtype=dt, device=dev)
        btr = blocks[tr]
        for b_in in np.unique(btr):
            vi = np.where(btr == b_in)[0]
            ti = np.where(btr != b_in)[0]
            Ztr, Zva = standardize(Xt_[tr[ti]], Xt_[tr[vi]], pca, boost_cols)
            inner_pred[:, vi, :] = fit(Ztr, Yt_[tr[ti]], Zva)
        Ytr = Yt_[tr]
        if inner_metric == "mse":
            score = -((inner_pred - Ytr[None]) ** 2).mean(1)      # A × C，越大越好
        elif inner_metric == "auc":
            score = auc_cols(inner_pred, Ytr)
        else:
            raise ValueError(inner_metric)
        best = torch.argmax(score, 0)                             # C
        Ztr, Zte = standardize(Xt_[tr], Xt_[te], pca, boost_cols)
        outer_all = fit(Ztr, Ytr, Zte)                            # A × n_te × C
        pred[te] = outer_all[best, :, torch.arange(C, device=dev)].T
        chosen[int(b_out)] = float(hp[best[0]])
    metric = spearman_cols(pred, Yt_) if task == "regression" else auc_cols(pred, Yt_)
    return {"pred": pred.cpu().numpy(), "metric": metric.cpu().numpy(), "chosen": chosen}


# ---------- P2-B：两层堆叠（07 第 11 节 B1 / B3） ----------
def _ridge_select_predict(Xtr, Ytr, btr: np.ndarray, Xte, hp: torch.Tensor, pca):
    """训练集内留一 block 选正则强度（每列各自，MSE），再用整个训练集重拟合预测 Xte → (n_te, C)。
    与 nested_lobo 一个外层折里做的事完全相同（回归、inner_metric = mse）。"""
    C = Ytr.shape[1]
    inner_pred = torch.empty(len(hp), len(btr), C, dtype=Xtr.dtype, device=Xtr.device)
    for b_in in np.unique(btr):
        vi = np.where(btr == b_in)[0]
        ti = np.where(btr != b_in)[0]
        Ztr, Zva = standardize(Xtr[ti], Xtr[vi], pca)
        inner_pred[:, vi, :] = ridge_all(Ztr, Ytr[ti], Zva, hp)
    best = torch.argmax(-((inner_pred - Ytr[None]) ** 2).mean(1), 0)
    Ztr, Zte = standardize(Xtr, Xte, pca)
    return ridge_all(Ztr, Ytr, Zte, hp)[best, :, torch.arange(C, device=Xtr.device)].T


def _stack_fit(H: torch.Tensor, E: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """第二层：每列 c 拟合 y_c = b0 + H·bh + be·e_c（最小二乘，be ≥ 0）。H: n × q（各列共用），E、Y: n × C。
    返回 (C, q + 2)：[截距, 历史系数…, 脑电系数]。只有一个系数有约束，二次型凸 → 无约束解 be < 0 时最优解在 be = 0，
    即只用 [1, H] 重新拟合（精确的非负最小二乘解）。用 float64。"""
    H, E, Y = H.double(), E.double(), Y.double()
    n, C = Y.shape
    one = torch.ones(n, 1, dtype=H.dtype, device=H.device)
    D0 = torch.cat([one, H], 1)                                          # n × (q+1)
    D = torch.cat([D0[None].expand(C, n, D0.shape[1]), E.T[:, :, None]], 2)   # C × n × (q+2)
    G = D.transpose(1, 2) @ D + 1e-9 * torch.eye(D.shape[2], dtype=H.dtype, device=H.device)
    beta = torch.linalg.solve(G, D.transpose(1, 2) @ Y.T[:, :, None]).squeeze(-1)   # C × (q+2)
    neg = beta[:, -1] < 0
    if bool(neg.any()):
        b0 = torch.linalg.solve(D0.T @ D0 + 1e-9 * torch.eye(D0.shape[1], dtype=H.dtype, device=H.device),
                                D0.T @ Y[:, neg])                        # (q+1) × C_neg
        beta[neg, :-1] = b0.T
        beta[neg, -1] = 0
    return beta


def nested_stack(X: np.ndarray, H: np.ndarray, Y: np.ndarray, blocks: np.ndarray, cfg: dict,
                 pca: int | None, dev: torch.device) -> dict:
    """B1 / B3 两层堆叠，只做回归。X: n × p 脑电特征；H: n × q 反应历史；Y: n × C（第 0 列真实目标）。
    外层留一 block；每个外层折内：
      第一层 = 脑电岭回归（同 E106b：内层留一 block 选正则）；在外层训练集上再做一次留一 block，得到训练试验的
               **样本外**脑电预测 e_tr（每个 e 由不含该 block 的 4 个 block 训练、正则也只在这 4 个 block 里选），
               测试 block 的 e_te 由整个外层训练集（5 个 block）训练
      第二层 = y ~ 1 + H + e（e 的系数 ≥ 0），在 (H_tr, e_tr) 上拟合，预测测试 block
    置换列与真实目标一起走完两层（每列各自选正则、各自拟合第二层）。
    返回 {'pred', 'metric', 'coef': 每折真实目标的第二层系数 (n_fold × (q+2))}。"""
    dt = torch.float32
    Xt_ = torch.as_tensor(X, dtype=dt, device=dev)
    Ht_ = torch.as_tensor(H, dtype=dt, device=dev)
    Yt_ = torch.as_tensor(Y, dtype=dt, device=dev)
    hp = torch.as_tensor(grid(cfg["ridge"]["alphas"]), dtype=dt, device=dev)
    if cfg["ridge"].get("inner_metric", "mse") != "mse":
        raise ValueError("stack 只支持 inner_metric = mse")
    n, C = Y.shape
    pred = torch.empty(n, C, dtype=dt, device=dev)
    coefs = []
    for b_out in np.unique(blocks):
        te = np.where(blocks == b_out)[0]
        tr = np.where(blocks != b_out)[0]
        btr = blocks[tr]
        Xtr, Ytr = Xt_[tr], Yt_[tr]
        e_tr = torch.empty(len(tr), C, dtype=dt, device=dev)
        for b_mid in np.unique(btr):
            vi = np.where(btr == b_mid)[0]
            ti = np.where(btr != b_mid)[0]
            e_tr[vi] = _ridge_select_predict(Xtr[ti], Ytr[ti], btr[ti], Xtr[vi], hp, pca)
        e_te = _ridge_select_predict(Xtr, Ytr, btr, Xt_[te], hp, pca)
        beta = _stack_fit(Ht_[tr], e_tr, Ytr)                              # C × (q+2)，float64
        Hte = Ht_[te].double()
        p = beta[:, 0][None] + Hte @ beta[:, 1:-1].T + beta[:, -1][None] * e_te.double()
        pred[te] = p.to(dt)
        coefs.append(beta[0].cpu().numpy())
    return {"pred": pred.cpu().numpy(), "metric": spearman_cols(pred, Yt_).cpu().numpy(), "coef": np.array(coefs)}


def perm_p(metric: np.ndarray) -> float:
    obs, null = metric[0], metric[1:]
    return float((1 + np.sum(null >= obs)) / (1 + len(null))) if len(null) else float("nan")
