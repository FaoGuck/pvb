"""P206 准备 A：候选模型的冒烟测试与接口核实（V8a–c）。离线运行（HF_HUB_OFFLINE=1），只读 --models-dir 里的快照。

对每个模型：
  - braindecode 的 BrainBERT / PopulationTransformer / MVPFormer：打印类说明（前 80 行）、__init__ 参数（采样率、输入长度、
    通道数等就在这里）、from_pretrained 后的参数量与关键属性；用假数据前向一次（输出形状、显存峰值）；
    同结构随机初始化一份（参数量应相同）；列出名字里带 generat / predict / forecast / encode / feature 的方法（P207 能否生成原始信号）
  - MOMENT（momentfm，embedding 任务）：(2, 1, 512) → embeddings 形状
  - Chronos（chronos-t5-large）：(2, 512) → embed 形状
报告写到 --report，并打印。最后一行 SUMMARY：每个模型 OK / FAIL
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
import traceback
from pathlib import Path

LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def nparam(m):
    return sum(p.numel() for p in m.parameters())


def gpu_peak():
    import torch
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def find_cls(names):
    import braindecode.models as bm
    for n in names:
        if hasattr(bm, n):
            return getattr(bm, n)
    low = {k.lower(): k for k in dir(bm)}
    for n in names:
        for k in low:
            if n.lower() in k:
                return getattr(bm, low[k])
    return None


def try_forward(model, dev, shapes):
    import torch
    for shp in shapes:
        try:
            x = torch.randn(*shp, device=dev)
            with torch.no_grad():
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                t0 = time.time()
                y = model(x)
                dt_ = time.time() - t0
            desc = {k: tuple(v.shape) for k, v in y.items()} if isinstance(y, dict) else (tuple(y.shape) if hasattr(y, "shape") else type(y))
            say(f"    forward{shp} → {desc}  {dt_ * 1000:.0f} ms  显存峰值 {gpu_peak():.2f} GiB")
            for kw in ("return_features",):
                if kw in inspect.signature(model.forward).parameters:
                    with torch.no_grad():
                        yf = model(x, **{kw: True})
                    descf = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in yf.items()} if isinstance(yf, dict) else (tuple(yf.shape) if hasattr(yf, "shape") else type(yf))
                    say(f"    forward{shp}, {kw}=True → {descf}")
            return True
        except Exception as e:  # noqa: BLE001
            say(f"    forward{shp} 失败：{e!r}"[:300])
    return False


def braindecode_model(tag, names, path, dev, shapes):
    import torch
    say(f"\n===== {tag}（{path.name}）")
    cls = find_cls(names)
    if cls is None:
        import braindecode.models as bm
        say("  braindecode.models 里找不到；现有模型：", [k for k in dir(bm) if k[0].isupper()][:200])
        return False
    say(f"  类：{cls.__module__}.{cls.__name__}")
    doc = (inspect.getdoc(cls) or "").splitlines()
    say("  --- 类说明（前 80 行）"); [say("  | " + ln) for ln in doc[:80]]
    try:
        say("  --- __init__ 参数：", str(inspect.signature(cls.__init__))[:1500])
    except Exception as e:  # noqa: BLE001
        say("  __init__ 参数读不到", e)
    meth = sorted(n for n in dir(cls) if any(k in n.lower() for k in ("generat", "predict", "forecast", "encode", "feature", "embed")))
    say("  --- 相关方法：", meth)
    for n in meth:
        f = getattr(cls, n, None)
        if callable(f):
            try:
                say(f"    {n}{inspect.signature(f)}"[:400])
            except Exception:  # noqa: BLE001
                pass
    try:
        m = cls.from_pretrained(str(path))
    except Exception as e:  # noqa: BLE001
        say(f"  from_pretrained 失败：{e!r}"[:400]); say(traceback.format_exc()[-1500:]); return False
    m = m.to(dev).eval()
    say(f"  预训练：参数 {nparam(m) / 1e6:.1f} M")
    for at in ("sfreq", "n_times", "n_chans", "n_outputs", "input_window_seconds", "chs_info"):
        if hasattr(m, at):
            v = getattr(m, at)
            say(f"    {at} = {str(v)[:200]}")
    cfgs = [p for p in path.glob("*.json")]
    for p in cfgs:
        say(f"  --- {p.name}（前 1500 字）"); say("  " + p.read_text(encoding="utf-8")[:1500].replace("\n", "\n  "))
    n_ch = getattr(m, "n_chans", None) or 4
    n_t = getattr(m, "n_times", None)
    sh = [s for s in shapes(n_ch, n_t)]
    ok = try_forward(m, dev, sh)
    try:
        kw = {k: getattr(m, k) for k in ("n_chans", "n_times", "n_outputs", "sfreq", "chs_info") if hasattr(m, k) and getattr(m, k) is not None
              and k in inspect.signature(cls.__init__).parameters}
        r = cls(**kw)
        say(f"  随机初始化（同结构，参数 {kw.keys()}）：参数 {nparam(r) / 1e6:.1f} M（应与预训练相同）")
    except Exception as e:  # noqa: BLE001
        say(f"  随机初始化失败：{e!r}"[:300])
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return ok


def check_moment(md, dev):
    import torch
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained(str(md / "moment_large"), model_kwargs={"task_name": "embedding"})
    m.init(); m = m.to(dev).eval()
    with torch.no_grad():
        o = m(x_enc=torch.randn(2, 1, 512, device=dev))
    say(f"  参数 {nparam(m) / 1e6:.1f} M；(2,1,512) → embeddings {tuple(o.embeddings.shape)}；显存峰值 {gpu_peak():.2f} GiB")
    return True


def check_chronos(md, dev):
    import torch
    from chronos import ChronosPipeline
    p = ChronosPipeline.from_pretrained(str(md / "chronos_t5_large"), device_map=dev, torch_dtype=torch.float32)
    emb, _ = p.embed(torch.randn(2, 512))
    say(f"  参数 {nparam(p.model) / 1e6:.1f} M；(2,512) → embed {tuple(emb.shape)}；显存峰值 {gpu_peak():.2f} GiB")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--only", default="", help="逗号分隔：BrainBERT,MVPFormer-S,MVPFormer-M,PopT,MOMENT,Chronos；空 = 全部（2026-11-27 加：两个环境分开测）")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"python {sys.version.split()[0]} torch {torch.__version__} device {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    md = Path(a.models_dir)
    want = [x for x in a.only.split(",") if x] or ["BrainBERT", "MVPFormer-S", "MVPFormer-M", "PopT", "MOMENT", "Chronos"]
    # 输入形状候选：模型自带 n_times 就用它；否则试 5 s @ 2048 Hz、5 s @ 512 Hz、2 s @ 512 Hz
    shapes_bd = lambda c, t: ([(1, c, t)] if t else []) + [(1, c, 10240), (1, c, 2560), (1, c, 1024)]  # noqa: E731
    jobs = {
        "BrainBERT": lambda: braindecode_model("BrainBERT", ["BrainBERT"], md / "brainbert", dev, shapes_bd),
        "MVPFormer-S": lambda: braindecode_model("MVPFormer-S", ["MVPFormer"], md / "mvpformer_s", dev, shapes_bd),
        "MVPFormer-M": lambda: braindecode_model("MVPFormer-M", ["MVPFormer"], md / "mvpformer_m", dev, shapes_bd),
        # PopT 的输入是逐电极的 BrainBERT 表征（768 维）
        "PopT": lambda: braindecode_model("PopT", ["PopulationTransformer", "PopT"], md / "popt", dev,
                                          lambda c, t: [(1, 16, 768), (1, c, t or 768)]),
        "MOMENT": lambda: (say("\n===== MOMENT（momentfm，embedding）"), check_moment(md, dev))[1],
        "Chronos": lambda: (say("\n===== Chronos（chronos-t5-large，embed）"), check_chronos(md, dev))[1],
    }
    res = {}
    for tag in want:
        try:
            res[tag] = bool(jobs[tag]())
        except Exception as e:  # noqa: BLE001
            say(f"{tag} 失败 {e!r}"[:400]); say(traceback.format_exc()[-1500:]); res[tag] = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    say("\nSUMMARY_SMOKE " + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in res.items()))
    Path(a.report).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
