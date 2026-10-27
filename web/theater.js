const $ = s => document.querySelector(s);
let D; try { D = await (await fetch("theater_data.json")).json() } catch (e) { D = null }
if (!D) { document.body.insertAdjacentHTML("beforeend", "<p style='padding:120px 6vw'>theater_data.json missing: run theater/analyze.py first.</p>"); throw new Error("no data") }
const L4 = ["A","B","C","D"], pct = x => (100*x).toFixed(0) + "%";
const GREEN = "#7CFFB2", ORANGE = "#ffb86b", PURPLE = "#9b8cff";
let ex = D.examples, cur = null, idx = 0, timer = null, playing = false;

/* ---------- picker ---------- */
const pick = $("#pick");
ex.forEach((e, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${e.group === "easy" ? "Recall" : "Reasoning"} · ${e.subject.replace(/_/g, " ")} · ${e.tokens.length} tokens · model ${e.pred === e.gold ? "right" : "wrong"}`; pick.appendChild(o) });
pick.onchange = () => load(+pick.value);

const mon = (el) => { el.innerHTML = L4.map(l => `<span class="l">${l}</span><div class="bar"><i></i></div><span class="pv">0%</span>`).join("") };
mon($("#xr")); mon($("#rd"));
const setMon = (el, p, mark) => { el.querySelectorAll(".bar i").forEach((b, i) => b.style.width = pct(p[i])); el.querySelectorAll(".pv").forEach((v, i) => v.textContent = pct(p[i])); el.querySelectorAll(".l").forEach((l, i) => l.style.color = i === mark ? "#fff" : "") };

/* ---------- derived series for the current example ---------- */
const stepAt = (arr, pos, i) => { let k = 0; while (k + 1 < pos.length && pos[k + 1] <= i) k++; return arr[k] };
function fracIdx(e, i) { const f = i / e.tokens.length; let k = 0; while (k + 1 < e.fracs.length && e.fracs[k + 1] <= f + 1e-9) k++; return k }
function exitIndex(e, tau) { for (let k = 0; k < e.grid_pos.length; k++) if (Math.max(...e.probe_grid[k]) >= tau && e.grid_pos[k] < e.tokens.length) return k; return -1 }

function load(i) {
  cur = ex[i]; idx = 0; stop(); const e = cur;
  $("#qbox").innerHTML = `<p class="q"><b>${e.question}</b></p>` + e.choices.map((c, j) => `<div class="opt">${L4[j]}. ${c}</div>`).join("") + `<div class="chips"><span class="chip">model's final answer: <b>${e.pred}</b></span><span class="chip">correct answer: <b>${e.gold}</b></span></div>`;
  render();
}
function render() {
  const e = cur, tau = +$("#tau").value, k = exitIndex(e, tau), exitPos = k < 0 ? null : e.grid_pos[k];
  $("#tauV").textContent = pct(tau);
  const shown = e.tokens.slice(0, idx).join("");
  let html;
  if (exitPos != null && idx > exitPos) html = esc(e.tokens.slice(0, exitPos).join("")) + `<span class="cut">✋ X-ray is ${pct(Math.max(...e.probe_grid[k]))} sure → stop here and answer ${L4[e.probe_grid[k].indexOf(Math.max(...e.probe_grid[k]))]}</span><span class="skip">${esc(e.tokens.slice(exitPos, idx).join(""))}</span>`;
  else html = esc(shown);
  $("#cot").innerHTML = html; $("#cot").scrollTop = 1e9;
  const pr = idx === 0 ? e.probe_frac[0] : stepAt(e.probe_grid, e.grid_pos, idx), fi = fracIdx(e, idx), rd = e.reader_frac[fi], fo = e.forced_frac[fi];
  const finalI = L4.indexOf(e.pred); setMon($("#xr"), pr, finalI); setMon($("#rd"), rd, finalI);
  const chips = [`token ${idx} / ${e.tokens.length}`];
  if (exitPos != null) { const ans = L4[e.probe_grid[k].indexOf(Math.max(...e.probe_grid[k]))]; chips.push(`stop point: token ${exitPos} · saves ${pct(1 - exitPos / e.tokens.length)}`, `answer at stop: ${ans} ${ans === e.gold ? "✓ correct" : "✗ wrong"}`) } else chips.push("X-ray never reaches the threshold: full reasoning used");
  $("#status").innerHTML = chips.map(c => `<span class="chip">${c}</span>`).join("");
  drawLine(k, exitPos);
}
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function drawLine(k, exitPos) {
  const e = cur, c = $("#line"), g = c.getContext("2d"), W = 1000, H = 400, N = e.tokens.length, fi = L4.indexOf(e.pred);
  g.clearRect(0, 0, W, H); g.font = "22px system-ui"; g.fillStyle = "#98a0b8"; g.strokeStyle = "rgba(255,255,255,.1)";
  for (const y of [0, .5, 1]) { const Y = 350 - y * 320; g.beginPath(); g.moveTo(60, Y); g.lineTo(980, Y); g.stroke(); g.fillText(pct(y), 6, Y + 7) }
  const X = t => 60 + t / N * 920, Y = p => 350 - p * 320;
  g.fillText("0", 60, 385); g.fillText(N + " tokens", 880, 385);
  const upto = idx;
  g.strokeStyle = GREEN; g.lineWidth = 4; g.beginPath(); let first = true;
  e.grid_pos.forEach((t, j) => { if (t > upto) return; const p = e.probe_grid[j][fi]; first ? g.moveTo(X(t), Y(p)) : g.lineTo(X(t), Y(p)); first = false }); g.stroke();
  const dots = (arr, col) => { g.fillStyle = col; e.fracs.forEach((f, j) => { const t = f * N; if (t > upto) return; g.beginPath(); g.arc(X(t), Y(arr[j][fi]), 7, 0, 7); g.fill() }) };
  g.strokeStyle = ORANGE; g.lineWidth = 3; g.beginPath(); e.fracs.forEach((f, j) => { const t = f * N; if (t > upto) return; const p = e.reader_frac[j][fi]; j ? g.lineTo(X(t), Y(p)) : g.moveTo(X(t), Y(p)) }); g.stroke();
  dots(e.reader_frac, ORANGE); dots(e.forced_frac, PURPLE);
  g.strokeStyle = "rgba(255,255,255,.25)"; g.lineWidth = 2; g.setLineDash([6, 6]); g.beginPath(); g.moveTo(X(upto), 20); g.lineTo(X(upto), 350); g.stroke(); g.setLineDash([]);
  if (exitPos != null) { g.strokeStyle = "#fff"; g.lineWidth = 3; g.beginPath(); g.moveTo(X(exitPos), 20); g.lineTo(X(exitPos), 350); g.stroke(); g.fillStyle = "#fff"; g.fillText("stop", X(exitPos) + 8, 40) }
}

/* ---------- playback ---------- */
function stop() { playing = false; clearInterval(timer); $("#play").textContent = "▶ Play" }
function play() { if (idx >= cur.tokens.length) idx = 0; playing = true; $("#play").textContent = "⏸ Pause"; clearInterval(timer);
  timer = setInterval(() => { idx = Math.min(cur.tokens.length, idx + Math.max(1, Math.round(+$("#speed").value / 30))); render(); if (idx >= cur.tokens.length) stop() }, 33) }
$("#play").onclick = () => playing ? stop() : play(); $("#restart").onclick = () => { idx = 0; render(); play() };
$("#tau").oninput = () => { render(); eeUpdate() };

/* ---------- aggregate chart ---------- */
let grp = "all";
document.querySelectorAll("[data-g]").forEach(b => b.onclick = () => { grp = b.dataset.g; document.querySelectorAll("[data-g]").forEach(c => c.classList.toggle("on", c === b)); drawAgg(); eeUpdate() });
function drawAgg() {
  const A = D.acc[grp], c = $("#agg"), g = c.getContext("2d"), W = 1200, H = 380, F = D.fracs; g.clearRect(0, 0, W, H); g.font = "22px system-ui"; g.fillStyle = "#98a0b8"; g.strokeStyle = "rgba(255,255,255,.1)";
  for (const y of [0, .25, .5, .75, 1]) { const Y = 330 - y * 300; g.beginPath(); g.moveTo(70, Y); g.lineTo(1180, Y); g.stroke(); g.fillText(pct(y), 8, Y + 7) }
  const X = f => 70 + f * 1090, Y = p => 330 - p * 300;
  [0, .25, .5, .75, 1].forEach(f => g.fillText(pct(f), X(f) - 20, 366)); g.fillText("share of the reasoning shown →", 830, 24);
  const S = [["probe", GREEN], ["linear_probe", "#6c7aa8"], ["forced", PURPLE], ["reader", ORANGE]];
  S.forEach(([m, col]) => { const v = A[m]; g.fillStyle = col + "26"; g.beginPath(); F.forEach((f, i) => g[i ? "lineTo" : "moveTo"](X(f), Y(v[i][2]))); [...F].reverse().forEach((f, i) => g.lineTo(X(f), Y(v[F.length - 1 - i][1]))); g.fill();
    g.strokeStyle = col; g.lineWidth = 4; g.beginPath(); F.forEach((f, i) => g[i ? "lineTo" : "moveTo"](X(f), Y(v[i][0]))); g.stroke() });
  const k2 = F.indexOf(0.2), k0 = 0, p = A.probe, r = A.reader;
  $("#resNote").innerHTML = `${A.n} held-out questions. With <b>0%</b> of the reasoning written, the X-ray names the model's final answer <b style="color:var(--ok)">${pct(p[k0][0])}</b> of the time vs <b style="color:#ffb86b">${pct(r[k0][0])}</b> for the outside reader; at <b>20%</b> it is <b style="color:var(--ok)">${pct(p[k2][0])}</b> vs <b style="color:#ffb86b">${pct(r[k2][0])}</b>.`;
}
function eeUpdate() {
  const rows = D.early_exit[grp], tau = +$("#tau").value; const r = rows.reduce((a, b) => Math.abs(b.tau - tau) < Math.abs(a.tau - tau) ? b : a);
  const f = x => `${(100 * x[0]).toFixed(0)}% <small>[${(100 * x[1]).toFixed(0)}–${(100 * x[2]).toFixed(0)}]</small>`;
  $("#ee").innerHTML = `<table><tr><th>Stop when X-ray ≥</th><th>Reasoning tokens saved</th><th>Accuracy kept (vs full reasoning)</th></tr>` + rows.map(x => `<tr style="${x === r ? "background:rgba(124,255,178,.08)" : ""}"><td>${pct(x.tau)}</td><td>${f(x.tokens_saved)}</td><td>${f(x.acc_vs_baseline)}</td></tr>`).join("") + "</table>";
  $("#eeNote").textContent = `Highlighted row matches the slider at the top. Brackets: 95% bootstrap CI over questions. Baseline: model's own accuracy with full reasoning = ${(100 * r.baseline_acc).toFixed(0)}% on these questions.`;
}
$("#cav").innerHTML = ["Small model (1.5B). Larger reasoning models know their answers earlier on easy questions, so these gaps are probably smaller than in frontier models.",
  "'Knows the answer' here means a trained probe can predict the model's eventual answer from its internals. It does not mean the model is being deceptive.",
  "One layer, one dataset (multiple-choice), probes trained on 70% of questions and tested on held-out ones.",
  "The outside reader is an open 7B model, not a frontier model, so a stronger reader would narrow the gap. It can also solve many questions itself, so its early guesses are not pure text-reading.",
  "Early-exit accuracy is measured against the true answer; baseline is the model's own accuracy with full reasoning."].map(t => `<li>${t}</li>`).join("");
function stats() {
  const rows = D.early_exit.all, tau = +$("#tau").value, r = rows.reduce((a, b) => Math.abs(b.tau - tau) < Math.abs(a.tau - tau) ? b : a), ci = x => `${(100*x[1]).toFixed(0)}–${(100*x[2]).toFixed(0)}%`;
  // recall-subject lead of forced answering over the outside reader, at the fraction where it is most clearly positive
  const g = D.gap.easy.forced; let bi = -1; g.forEach((v, i) => { if (v[1] > 0 && (bi < 0 || v[0] > g[bi][0])) bi = i });
  const lead = bi < 0 ? null : {f: D.fracs[bi], v: g[bi]};
  $("#tiles").innerHTML = `<div class="kv" style="gap:40px;margin:0">
   <div><small>Reasoning tokens saved (stop at ${pct(r.tau)})</small><span class="stat" style="color:var(--ok)">${pct(r.tokens_saved[0])}</span><small>95% CI ${ci(r.tokens_saved)}</small></div>
   <div><small>Accuracy kept vs full reasoning</small><span class="stat">${pct(r.acc_vs_baseline[0])}</span><small>95% CI ${ci(r.acc_vs_baseline)}</small></div>
   <div><small>Recall subjects: inside beats the text at ${lead ? pct(lead.f) + " of the reasoning" : "any point"}</small><span class="stat" style="color:#ffb86b">${lead ? "+" + (100*lead.v[0]).toFixed(0) + " pts" : "n/a"}</span><small>${lead ? "95% CI +" + (100*lead.v[1]).toFixed(0) + " to +" + (100*lead.v[2]).toFixed(0) : "no clear lead"}</small></div></div>`;
  const a = D.acc.all, k0 = 0;
  const best = rows.filter(x => x.acc_vs_baseline[0] >= .98).sort((x, y) => y.tokens_saved[0] - x.tokens_saved[0])[0];
  $("#take").innerHTML = [
    best ? `<b>Early exit works.</b> Stopping when the X-ray is ≥${pct(best.tau)} sure saves ${pct(best.tokens_saved[0])} of the reasoning tokens (95% CI ${ci(best.tokens_saved)}) while keeping ${pct(best.acc_vs_baseline[0])} of the accuracy. Lower thresholds save more but cost accuracy (see table).` : "",
    lead ? `<b>The model sometimes knows before the text shows it.</b> On recall subjects, asking the model to answer right now (forced answer, an internal readout) beats an outside reader of the visible text by ${(100*lead.v[0]).toFixed(0)} points at ${pct(lead.f)} of the reasoning (95% CI +${(100*lead.v[1]).toFixed(0)} to +${(100*lead.v[2]).toFixed(0)}). That gap is the "performative" part of the reasoning.` : "",
    `<b>But our small probe is not the best early predictor.</b> At 0% of the reasoning it names the final answer ${pct(a.probe[k0][0])} of the time, versus ${pct(a.forced[k0][0])} for forced answering and ${pct(a.reader[k0][0])} for the outside reader. Its value is as a cheap, calibrated stopping signal, not as the sharpest knowledge detector.`,
    `<b>Single-token linear probes are weak here.</b> They sit near chance (${pct(a.linear_probe[k0][0])} at the start, ${pct(a.linear_probe.at(-1)[0])} at the end), in line with earlier reports that answer information is spread across the context and needs a pooling probe.`,
    `<b>On reasoning subjects the outside reader is ahead early.</b> It is a larger model and can solve many math questions itself, so this comparison is not purely about reading text.`].filter(Boolean).map(t => `<li>${t}</li>`).join("");
}
$("#tau").addEventListener("input", stats);
load(0); drawAgg(); eeUpdate(); stats();
