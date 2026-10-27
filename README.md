# Watch the AI Pretend to Think

**How early does a reasoning model know its final answer?** We trained small probes on the hidden states of DeepSeek-R1-Distill-Qwen-1.5B while it wrote its chain of thought on MMLU-Redux questions. A probe can name the model's eventual answer while the reasoning is still going, which lets us stop the model early and save tokens.

**▶ Live demo:** https://samkoduru.github.io/watch-ai-pretend-to-think/theater.html

Pick a question, press play, and watch the model's internal answer belief next to what an outside reader can tell from the visible text.

![status](https://img.shields.io/badge/model-DeepSeek--R1--Distill--1.5B-purple) ![compute](https://img.shields.io/badge/compute-Modal%20L4%20GPUs-blue) ![license](https://img.shields.io/badge/code-MIT-blue)

## Headline results

300 held-out questions (196 recall-type, 104 reasoning-type). 95% CIs are bootstrap intervals over questions.

**Early exit.** Stop the model once the probe is at least 70% sure of its answer:
- saves **37%** of reasoning tokens (95% CI 32–42%)
- keeps **99%** of full-reasoning accuracy (CI 95–104%)

At a 60% threshold: 49% saved, 92% accuracy kept.

**Inside versus outside.** How often each method names the model's *final* answer, given the first X% of its reasoning:

| Reasoning shown | Attention probe (ours) | Single-token linear probe | Forced answer (model asked to answer now) | Outside reader (Qwen2.5-7B) |
|---|---|---|---|---|
| 0% | 0.36 | 0.21 | 0.47 | 0.52 |
| 20% | 0.38 | 0.30 | 0.50 | 0.54 |
| 50% | 0.50 | 0.30 | 0.68 | 0.66 |
| 80% | 0.78 | 0.31 | 0.90 | 0.85 |
| 100% | 0.98 | 0.42 | 1.00 | 0.98 |

- **On recall-type questions, the model's internals beat the outside reader** by about 11 points at 60% of the reasoning (95% CI +4 to +18). That is the gap between what the model "knows" and what the visible text reveals.
- **Our probe is not the best very-early predictor.** Asking the model to answer right away (forced answer) does better at the start. The probe's value here is as a cheap, calibrated stopping signal, not as the sharpest knowledge detector.
- **Single-token linear probes are near chance** at every point, consistent with answer information being spread across the context. The attention-pooling probe does the work.

## Try it

You need a Modal account (free tier works; GPU runs cost a few dollars in total) and the Modal CLI logged in on your machine. No API keys are stored in the repo.

```bash
git clone https://github.com/Samkoduru/watch-ai-pretend-to-think
cd watch-ai-pretend-to-think
uv sync
uv run modal setup                          # one-time: links the Modal CLI to your account
```

**Just look at the results** (no GPU needed). The committed `web/theater_data.json` drives the page, and the same page is live at the link above:

```bash
cd web && python3 -m http.server 8000      # open http://localhost:8000/theater.html
```

**Re-run the pipeline** (a few dollars of GPU time; generation is the slow step, and preemptions can extend it):

```bash
uv run modal run --detach theater/modal_app.py::gen --n-per-subject 95        # 1,960 CoT traces (checkpointed, resumes after preemption)
uv run modal run theater/modal_app.py::acts --tag full                       # cache hidden states at layers 14 and 20
uv run modal run theater/modal_app.py::probes --tag full --layer 20          # train probes (run for 14 too; analysis picks the better one)
uv run modal run theater/modal_app.py::probes --tag full --layer 14
uv run modal run theater/modal_app.py::baselines --tag full --max-q 300       # forced answering + outside reader
uv run python theater/analyze.py full auto                                   # CIs, early-exit table -> web/theater_data.json
```

## How it works

1. **Generate.** Sample chain-of-thought traces from the reasoning model with vLLM, 1,960 MMLU-Redux questions across 22 subjects: 12 recall-type (for example anatomy, virology, clinical knowledge) and 10 reasoning-type (for example college mathematics, formal logic).
2. **Cache.** Run each prompt plus its reasoning once and store the hidden states at layers 14 and 20.
3. **Probe.** Train two predictors of the model's *final answer* from any prefix of its reasoning: an attention-pooling probe and a single-token linear probe. Splits are by question (train 1,087, validation 157, test 327). We use validation early stopping and temperature calibration, so the probe's confidence means something for the early-exit threshold.
4. **Baselines.** At the same checkpoints (0% to 100% of the reasoning), ask the model itself to answer now (forced answering), and ask an outside 7B model to guess from the visible text only.
5. **Analyse.** Accuracy by checkpoint, performativity gap (internal readout minus outside reader), and an early-exit curve over stopping thresholds, with bootstrap CIs.

## What this does and does not show

- A 1.5B model. Larger reasoning models know their answers earlier on easy questions, so the gaps here are probably smaller than in frontier models. We are not comparing to any published number.
- "Knows the answer" means a trained probe can predict the final answer from the model's internals. It does not claim the model is deceptive.
- The outside reader is a 7B model that can solve many questions itself, so its early accuracy is not pure text reading. A stronger reader would narrow the gap.
- One layer pair, one dataset (multiple-choice), one seed for the probes. We did not sweep hyperparameters across seeds.
- **Development note:** a first version of the probe overfit badly (no early stopping). Its early-exit accuracy collapsed to 0.56–0.70 of baseline at the thresholds that save the most tokens. The reported probe is the retrained one.

## Layout

```
theater/modal_app.py   generation, activation caching, probe training, baselines (Modal)
theater/analyze.py     metrics with bootstrap CIs, early-exit curves, examples for the page
web/theater.html       the interactive page
web/theater.js         playback, probe/reader/forced-answer charts, early-exit table
web/theater_data.json  committed results used by the page
```

## Credits and licences

- The probing idea follows public work on performative chain-of-thought (Goodfire's "Reasoning Theater"). This is an independent small-scale re-implementation on an open model, not their code or results.
- DeepSeek-R1-Distill-Qwen-1.5B: MIT. Qwen2.5-7B-Instruct: Apache-2.0. MMLU-Redux: CC-BY-4.0.
- Code in this repository is MIT licensed (see `LICENSE`).
