"""Pull Demo 2 results from the Modal volume, compute metrics with bootstrap CIs, write web/theater_data.json.

    uv run python theater/analyze.py [tag] [layer]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
LAYER = sys.argv[2] if len(sys.argv) > 2 else "auto"  # "auto" picks the layer with the best validation accuracy
FRACS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CACHE = Path("cache") / f"theater_{TAG}"
OUT = Path("web/theater_data.json")
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
rng = np.random.default_rng(0)


def pull(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "run", "modal", "volume", "get", "xray-vol", f"{TAG}/{name}", str(CACHE / name), "--force"], check=True, capture_output=True)
    return json.load(open(CACHE / name))


def boot(vals, fn=np.mean, n=1000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return [float("nan")] * 3
    est = fn(vals)
    bs = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    meta = {m["id"]: m for m in pull("meta.json")}
    global LAYER
    if LAYER == "auto":
        cands = {L: pull(f"probe_L{L}.json") for L in (14, 20)}
        LAYER = max(cands, key=lambda L: cands[L]["val_acc"]["attn"])
        print({L: round(c["val_acc"]["attn"], 3) for L, c in cands.items()}, "-> using layer", LAYER)
    LAYER = int(LAYER)
    pf = pull(f"probe_L{LAYER}.json")
    probe = pf["per_question"]
    base = pull("baselines.json")
    ids = [i for i in base["forced"] if i in probe]
    print("questions with all methods:", len(ids), "easy:", sum(meta[i]["group"] == "easy" for i in ids), "hard:", sum(meta[i]["group"] == "hard" for i in ids))
    tgt = {i: "ABCD".index(meta[i]["pred"]) for i in ids}
    methods = {"probe": lambda i: probe[i]["attn_frac"], "linear_probe": lambda i: probe[i]["lin_frac"], "forced": lambda i: base["forced"][i], "reader": lambda i: base["reader"][i]}

    res = {"layer": LAYER, "fracs": FRACS, "n": len(ids), "acc": {}, "gap": {}, "probe_val_acc": pf.get("val_acc"), "probe_temperature": pf.get("temperature"), "probe_config": pf.get("config")}
    for grp in ("all", "easy", "hard"):
        sel = [i for i in ids if grp == "all" or meta[i]["group"] == grp]
        if not sel:
            continue
        res["acc"][grp] = {"n": len(sel)}
        hits = {}
        for name, fn in methods.items():
            H = np.array([[int(np.argmax(fn(i)[k]) == tgt[i]) for k in range(len(FRACS))] for i in sel])
            hits[name] = H
            res["acc"][grp][name] = [boot(H[:, k]) for k in range(len(FRACS))]
        for a in ("probe", "forced"):  # performativity gap: internal readout minus what an outside reader can tell
            D = hits[a] - hits["reader"]
            res["gap"].setdefault(grp, {})[a] = [boot(D[:, k]) for k in range(len(FRACS))]

    # early exit with the attention probe, on the token grid
    ee = {}
    for grp in ("all", "easy", "hard"):
        sel = [i for i in ids if grp == "all" or meta[i]["group"] == grp]
        if not sel:
            continue
        gold_hit_full = np.array([int(meta[i]["pred"] == meta[i]["gold"]) for i in sel])
        rows = []
        for tau in TAUS:
            saved, cot, hit = [], [], []
            for j, i in enumerate(sel):
                P = np.array(probe[i]["attn_grid"]); pos = probe[i]["grid_cot_pos"]; L = meta[i]["cot_len"]
                k = next((k for k in range(len(P)) if P[k].max() >= tau and pos[k] < L), None)
                if k is None:
                    saved.append(0); hit.append(gold_hit_full[j])
                else:
                    saved.append(L - pos[k]); hit.append(int("ABCD"[int(P[k].argmax())] == meta[i]["gold"]))
                cot.append(L)
            saved, cot, hit = map(np.array, (saved, cot, hit))
            idx = np.arange(len(sel))
            def tok_saved(ix): return saved[ix].sum() / cot[ix].sum()
            def ratio(ix): return hit[ix].mean() / max(1e-9, gold_hit_full[ix].mean())
            bs_s = [tok_saved(rng.integers(0, len(sel), len(sel))) for _ in range(500)]
            bs_r = [ratio(rng.integers(0, len(sel), len(sel))) for _ in range(500)]
            rows.append({"tau": tau, "tokens_saved": [float(tok_saved(idx)), float(np.percentile(bs_s, 2.5)), float(np.percentile(bs_s, 97.5))],
                         "acc_vs_baseline": [float(ratio(idx)), float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))],
                         "exit_acc": float(hit.mean()), "baseline_acc": float(gold_hit_full.mean())})
        ee[grp] = rows
    res["early_exit"] = ee

    # curated examples for the live viewer
    cand = [i for i in ids if 150 <= meta[i]["cot_len"] <= 1100]
    pick = []
    for grp in ("easy", "hard"):
        for corr in (True, False):
            pool = [i for i in cand if meta[i]["group"] == grp and meta[i]["correct"] == corr]
            pool.sort(key=lambda i: meta[i]["id"])
            pick += pool[:6 if corr else 3]
    ex = []
    for i in pick:
        m = meta[i]
        ex.append({"id": i, "group": m["group"], "subject": m["subject"], "question": m["question"], "choices": m["choices"], "gold": m["gold"], "pred": m["pred"],
                   "tokens": m["cot_tokens"], "grid_pos": probe[i]["grid_cot_pos"], "probe_grid": probe[i]["attn_grid"],
                   "fracs": FRACS, "probe_frac": probe[i]["attn_frac"], "forced_frac": base["forced"][i], "reader_frac": base["reader"][i]})
    res["examples"] = ex
    OUT.write_text(json.dumps(res))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB; examples:", len(ex))
    a = res["acc"]["all"]
    for k, f in enumerate(FRACS):
        print(f"frac {f:>4}: probe {a['probe'][k][0]:.2f}  linear {a['linear_probe'][k][0]:.2f}  forced {a['forced'][k][0]:.2f}  reader {a['reader'][k][0]:.2f}")
    for r in ee["all"]:
        print(f"tau {r['tau']}: tokens saved {r['tokens_saved'][0]:.1%} [{r['tokens_saved'][1]:.1%},{r['tokens_saved'][2]:.1%}]  acc ratio {r['acc_vs_baseline'][0]:.3f}")


if __name__ == "__main__":
    main()
