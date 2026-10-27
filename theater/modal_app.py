"""Demo 2 pipeline on Modal: generate chain-of-thought traces, cache activations, train probes, run baselines.

    modal run theater/modal_app.py::gen --n-per-subject 5      # smoke test
    modal run theater/modal_app.py::gen --n-per-subject 90     # full run
Stages write to the Modal Volume `xray-vol` (no keys needed: all models and data here are ungated).
"""
import json
import re

import modal

app = modal.App("interp-xray-theater")
vol = modal.Volume.from_name("xray-vol", create_if_missing=True)
VOL = "/vol"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

gen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "datasets", "transformers")
)

EASY = ["global_facts", "high_school_geography", "high_school_us_history", "human_aging", "miscellaneous", "public_relations",
        "clinical_knowledge", "anatomy", "astronomy", "business_ethics", "virology", "philosophy"]
HARD = ["college_mathematics", "formal_logic", "high_school_mathematics", "college_physics", "high_school_physics",
        "high_school_statistics", "college_chemistry", "electrical_engineering", "econometrics", "machine_learning"]
LETTERS = "ABCD"


def build_prompt(q, choices):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices))
    return (f"Answer the following multiple choice question. Think step by step, then finish with a line of the form "
            f"'Answer: X' where X is A, B, C or D.\n\n{q}\n\n{opts}")


def extract(final):
    m = re.findall(r"Answer:\s*\**\(?([ABCD])\b", final)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", final)
    return m[-1] if m else None


@app.function(image=gen_image, gpu=["L4", "A10G", "L40S"], volumes={VOL: vol}, timeout=3 * 3600, retries=modal.Retries(max_retries=4, initial_delay=5.0))
def gen(n_per_subject: int = 90, max_tokens: int = 3500, seed: int = 0):
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rows = []
    for group, subjects in (("easy", EASY), ("hard", HARD)):
        for s in subjects:
            ds = load_dataset("edinburgh-dawg/mmlu-redux", s, split="test")
            ok = [r for r in ds if r["error_type"] == "ok"][:n_per_subject]
            for i, r in enumerate(ok):
                rows.append(dict(id=f"{s}-{i}", subject=s, group=group, question=r["question"], choices=r["choices"], gold=LETTERS[r["answer"]]))
    print("questions:", len(rows))
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=max_tokens + 800, gpu_memory_utilization=0.90, seed=seed, enforce_eager=True)
    tok = llm.get_tokenizer()
    prompts = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": build_prompt(r["question"], r["choices"])}], tokenize=False, add_generation_prompt=True)
        if not p.rstrip().endswith("<think>"):
            p = p + "<think>\n"
        r["prompt"] = p
        prompts.append(p)
    import os
    os.makedirs(f"{VOL}/gen_parts", exist_ok=True)
    CH = 245  # checkpoint every CH prompts so a preempted container resumes instead of restarting
    n_ok = 0
    for c0 in range(0, len(rows), CH):
        part = f"{VOL}/gen_parts/{n_per_subject}_{c0}.json"
        vol.reload()
        if os.path.exists(part):
            done = json.load(open(part))
            for r, d in zip(rows[c0:c0 + CH], done):
                r.update(d)
            print("resumed chunk", c0, flush=True)
            continue
        outs = llm.generate(prompts[c0:c0 + CH], SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens, seed=seed))
        extra = []
        for r, o in zip(rows[c0:c0 + CH], outs):
            text = o.outputs[0].text
            fin = o.outputs[0].finish_reason == "stop" and "</think>" in text
            cot, _, final = text.partition("</think>")
            d = dict(finished=fin, cot=cot, final=final, n_cot_tokens=len(tok(cot, add_special_tokens=False).input_ids))
            d["pred"] = extract(final) if fin else None
            d["correct"] = d["pred"] == r["gold"]
            r.update(d)
            extra.append(d)
        json.dump(extra, open(part, "w"), ensure_ascii=False)
        vol.commit()
        print("chunk done", c0, flush=True)
    n_ok = sum(r["correct"] for r in rows)
    path = f"{VOL}/traces.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    fin = sum(r["finished"] for r in rows)
    print(f"wrote {path}: {len(rows)} rows, finished={fin}, accuracy={n_ok / len(rows):.3f}, "
          f"mean cot tokens easy={sum(r['n_cot_tokens'] for r in rows if r['group']=='easy')/max(1,sum(r['group']=='easy' for r in rows)):.0f} "
          f"hard={sum(r['n_cot_tokens'] for r in rows if r['group']=='hard')/max(1,sum(r['group']=='hard' for r in rows)):.0f}")


@app.local_entrypoint()
def main():
    print("use ::gen")


# ---------------------------------------------------------------------------------------------------------------------
# Stage 2: cache activations, train attention probes, run baselines. All on the traces from `gen`.
# ---------------------------------------------------------------------------------------------------------------------
torch_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate", "numpy", "scikit-learn")
LAYERS = [14, 20]
FRACS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STEP = 32  # token grid for probe curves / early exit


def _split(i: str) -> str:
    import hashlib
    h = int(hashlib.md5(i.encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else "val" if h < 30 else "train"


def _load(tag):
    rows = [json.loads(l) for l in open(f"{VOL}/traces.jsonl")]
    rows = [r for r in rows if r["finished"] and r["pred"] is not None and r["n_cot_tokens"] >= 20]
    for r in rows:
        r["split"] = _split(r["id"])
    return rows


@app.function(image=torch_image, gpu=["L4", "A10G", "L40S"], volumes={VOL: vol}, timeout=2 * 3600, memory=32768)
def acts(tag: str = "full"):
    """Forward each prompt+CoT once and store fp16 hidden states at LAYERS."""
    import os
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    rows = _load(tag)
    os.makedirs(f"{VOL}/{tag}", exist_ok=True)
    for L in LAYERS:
        os.makedirs(f"{VOL}/{tag}/L{L}", exist_ok=True)
    meta = []
    for n, r in enumerate(rows):
        p_ids = tok(r["prompt"], add_special_tokens=False).input_ids
        c_ids = tok(r["cot"], add_special_tokens=False).input_ids[: 4096 - len(p_ids)]
        ids = torch.tensor([p_ids + c_ids]).cuda()
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        for L in LAYERS:
            np.save(f"{VOL}/{tag}/L{L}/{r['id']}.npy", hs[L][0].to(torch.float16).cpu().numpy())
        meta.append(dict(id=r["id"], split=r["split"], group=r["group"], subject=r["subject"], pred=r["pred"], gold=r["gold"],
                         correct=r["correct"], prompt_len=len(p_ids), cot_len=len(c_ids), cot_tokens=[tok.decode([t]) for t in c_ids],
                         question=r["question"], choices=r["choices"]))
        if n % 100 == 0:
            print(f"acts {n}/{len(rows)}", flush=True)
    json.dump(meta, open(f"{VOL}/{tag}/meta.json", "w"), ensure_ascii=False)
    vol.commit()
    print("done", len(meta), "questions; split counts:", {s: sum(m["split"] == s for m in meta) for s in ("train", "val", "test")})


@app.function(image=torch_image, gpu=["L4", "A10G", "L40S"], volumes={VOL: vol}, timeout=3 * 3600, memory=32768, retries=modal.Retries(max_retries=2, initial_delay=5.0))
def probes(tag: str = "full", layer: int = 20, epochs: int = 60, k: int = 64, dropout: float = 0.3, wd: float = 0.05, lr: float = 5e-4):
    """Attention-pooling probe + single-token linear probe, with validation early stopping and temperature calibration."""
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    meta = json.load(open(f"{VOL}/{tag}/meta.json"))
    H = {m["id"]: torch.from_numpy(np.load(f"{VOL}/{tag}/L{layer}/{m['id']}.npy")) for m in meta}  # fp16 on CPU
    d = next(iter(H.values())).shape[1]
    lab = {m["id"]: "ABCD".index(m["pred"]) for m in meta}
    by = lambda s: [m for m in meta if m["split"] == s]
    dev = "cuda"
    torch.manual_seed(0)

    class AttnProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln, self.key, self.val = nn.LayerNorm(d), nn.Linear(d, k), nn.Linear(d, k)
            self.q = nn.Parameter(torch.randn(k) * 0.02)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(2 * k, k), nn.GELU(), nn.Dropout(dropout), nn.Linear(k, 4))
            self.T = 1.0

        def forward(self, x, lens):
            x = self.drop(self.ln(x))
            T = x.shape[1]
            mask = torch.arange(T, device=x.device)[None] < lens[:, None]
            a = (self.key(x) @ self.q) / k ** 0.5
            w = a.masked_fill(~mask, -1e9).softmax(1)
            v = self.val(x)
            pooled = (w.unsqueeze(-1) * v).sum(1)
            last = v[torch.arange(len(x)), lens - 1]
            return self.head(torch.cat([pooled, last], -1))

    class LinearProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln, self.fc = nn.LayerNorm(d), nn.Linear(d, 4)
            self.T = 1.0

        def forward(self, x, lens):
            return self.fc(self.ln(x[torch.arange(len(x)), lens - 1]))

    def batch(ms, ts):
        T = max(ts)
        x = torch.zeros(len(ms), T, d, dtype=torch.float16)
        for i, (m, t) in enumerate(zip(ms, ts)):
            x[i, :t] = H[m["id"]][:t]
        return x.to(dev).float(), torch.tensor(ts, device=dev)

    VAL_F = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    val_ms = by("val")[:100]

    def val_pass(model):
        """accuracy and logits on a fixed grid of val prefixes."""
        model.eval()
        logits, ys = [], []
        with torch.no_grad():
            for m in val_ms:
                ts = [min(m["prompt_len"] + m["cot_len"], m["prompt_len"] + int(f * m["cot_len"]) + 1) for f in VAL_F]
                x, lens = batch([m] * len(ts), ts)
                logits.append(model(x, lens).cpu()); ys += [lab[m["id"]]] * len(ts)
        L = torch.cat(logits); y = torch.tensor(ys)
        return (L.argmax(-1) == y).float().mean().item(), L, y

    def train(model, name):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        tr = by("train")
        rng = np.random.default_rng(0)
        best, best_state, bad = -1, None, 0
        for ep in range(epochs):
            model.train()
            rng.shuffle(tr)
            tot = 0
            for i in range(0, len(tr), 32):
                ms = tr[i:i + 32] * 2  # two random prefixes per question per step
                ts = [int(m["prompt_len"] + rng.uniform(0, 1) * m["cot_len"]) + 1 for m in ms]
                x, lens = batch(ms, ts)
                y = torch.tensor([lab[m["id"]] for m in ms], device=dev)
                loss = nn.functional.cross_entropy(model(x, lens), y, label_smoothing=0.05)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(ms)
            if ep % 3 == 2 or ep == epochs - 1:
                acc, _, _ = val_pass(model)
                print(f"{name} ep{ep} train loss {tot / (2 * len(tr)):.3f} val acc {acc:.3f}", flush=True)
                if acc > best:
                    best, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= 4:
                        break
        model.load_state_dict(best_state)
        model.eval()
        # temperature scaling on the validation prefixes
        _, L, y = val_pass(model)
        Ts = torch.linspace(0.5, 5, 46)
        nll = [nn.functional.cross_entropy(L / t, y).item() for t in Ts]
        model.T = float(Ts[int(np.argmin(nll))])
        print(f"{name}: best val acc {best:.3f}, temperature {model.T:.2f}", flush=True)
        return model, best

    @torch.no_grad()
    def predict(model, m, ts):
        out = []
        for i in range(0, len(ts), 16):
            x, lens = batch([m] * len(ts[i:i + 16]), ts[i:i + 16])
            out.append((model(x, lens) / model.T).softmax(-1).cpu().numpy())
        return np.concatenate(out)

    attn, attn_val = train(AttnProbe().to(dev), "attn")
    lin, lin_val = train(LinearProbe().to(dev), "linear")
    res = {"layer": layer, "val_acc": {"attn": attn_val, "linear": lin_val}, "temperature": {"attn": attn.T, "linear": lin.T},
           "config": dict(k=k, dropout=dropout, wd=wd, lr=lr), "per_question": {}}
    for m in by("test"):
        ts_frac = [m["prompt_len"] + int(f * m["cot_len"]) + (1 if f == 0 else 0) for f in FRACS]
        ts_grid = list(range(m["prompt_len"] + 1, m["prompt_len"] + m["cot_len"] + 1, STEP))
        if ts_grid[-1] != m["prompt_len"] + m["cot_len"]:
            ts_grid.append(m["prompt_len"] + m["cot_len"])
        res["per_question"][m["id"]] = dict(
            attn_frac=predict(attn, m, ts_frac).round(4).tolist(), lin_frac=predict(lin, m, ts_frac).round(4).tolist(),
            attn_grid=predict(attn, m, ts_grid).round(4).tolist(), grid_cot_pos=[t - m["prompt_len"] for t in ts_grid])
    json.dump(res, open(f"{VOL}/{tag}/probe_L{layer}.json", "w"))
    vol.commit()
    print("probes done for", len(res["per_question"]), "test questions; val acc attn", round(attn_val, 3))


@app.function(image=torch_image, gpu=["L4", "A10G", "L40S"], volumes={VOL: vol}, timeout=3 * 3600, memory=32768, retries=modal.Retries(max_retries=3, initial_delay=5.0))
def baselines(tag: str = "full", reader_model: str = "Qwen/Qwen2.5-7B-Instruct", max_q: int = 300):
    """Forced answering (the model itself) and an outside reader LLM, at the same fractional checkpoints."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = [m for m in json.load(open(f"{VOL}/{tag}/meta.json")) if m["split"] == "test"][:max_q]
    out = {"forced": {}, "reader": {}}

    def letter_ids(tok, prefix=" "):
        return [tok(prefix + l, add_special_tokens=False).input_ids[-1] for l in "ABCD"]

    # forced answering with the reasoning model itself
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    cand = letter_ids(tok)
    rows = {r["id"]: r for r in map(json.loads, open(f"{VOL}/traces.jsonl"))}
    import os
    fpath, rpath = f"{VOL}/{tag}/forced.json", f"{VOL}/{tag}/reader_partial.json"
    if os.path.exists(fpath):
        out["forced"] = json.load(open(fpath))
        print("loaded saved forced answers", len(out["forced"]))
    for n, m in enumerate(meta):
        if m["id"] in out["forced"]:
            continue
        r = rows[m["id"]]
        p_ids = tok(r["prompt"], add_special_tokens=False).input_ids
        c_ids = tok(r["cot"], add_special_tokens=False).input_ids[: m["cot_len"]]
        tail = tok("\n</think>\n\nAnswer:", add_special_tokens=False).input_ids
        res = []
        for f in FRACS:
            k = int(f * len(c_ids))
            ids = torch.tensor([p_ids + c_ids[:k] + tail]).cuda()
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float()
            res.append(lg[cand].softmax(0).cpu().numpy().round(4).tolist())
        out["forced"][m["id"]] = res
        if n % 50 == 0:
            print(f"forced {n}/{len(meta)}", flush=True)
    json.dump(out["forced"], open(fpath, "w")); vol.commit()
    del model
    torch.cuda.empty_cache()

    # outside reader sees only the visible text so far
    rtok = AutoTokenizer.from_pretrained(reader_model)
    rmodel = AutoModelForCausalLM.from_pretrained(reader_model, dtype=torch.bfloat16).cuda().eval()
    rcand = letter_ids(rtok, prefix="")
    if os.path.exists(rpath):
        out["reader"] = json.load(open(rpath))
        print("loaded partial reader answers", len(out["reader"]))
    for n, m in enumerate(meta):
        if m["id"] in out["reader"]:
            continue
        r = rows[m["id"]]
        c_ids = tok(r["cot"], add_special_tokens=False).input_ids[: m["cot_len"]]
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", m["choices"]))
        res = []
        for f in FRACS:
            k = int(f * len(c_ids))
            partial = tok.decode(c_ids[:k]) if k else "(the student has not written anything yet)"
            msg = (f"A student is working on this multiple choice question.\n\n{m['question']}\n\n{opts}\n\n"
                   f"Here is the student's reasoning so far (it may be unfinished):\n<<<\n{partial}\n>>>\n\n"
                   f"Based only on this, which option will the student finally choose? Reply with a single letter A, B, C or D.")
            text = rtok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
            ids = rtok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                lg = rmodel(**ids).logits[0, -1].float()
            res.append(lg[rcand].softmax(0).cpu().numpy().round(4).tolist())
        out["reader"][m["id"]] = res
        if n % 25 == 0:
            print(f"reader {n}/{len(meta)}", flush=True)
            json.dump(out["reader"], open(rpath, "w")); vol.commit()
    json.dump(out, open(f"{VOL}/{tag}/baselines.json", "w"))
    vol.commit()
    print("baselines done", len(meta))
