"""Render the weekly summary (Markdown + HTML page) from reports/weekly_summary_data.json.

    python training/inf_retriever/tools/weekly_collect.py > reports/weekly_summary_data.json
    python training/inf_retriever/tools/weekly_render.py

Writes reports/weekly_summary_2026-10-12.md and reports/weekly_summary_2026-10-12.html. Tables, the
figures and run status come from the data file; the prose is below and is edited by hand as results land.
"""
import datetime, html, json, re

ROOT = "/scratch/hc3337/projects/autoregressive"
DATA = f"{ROOT}/reports/weekly_summary_data.json"
MD = f"{ROOT}/reports/weekly_summary_2026-10-12.md"
HTML = f"{ROOT}/reports/weekly_summary_2026-10-12.html"

runs = json.load(open(DATA))
R = {r["label"]: r for r in runs}


def ev(label, step=None):
    r = R[label]
    step = r["selected"] if step is None else step
    return next((e for e in r["evals"] if e["step"] == step), None)


def f1(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def gv(e, key, split="test"):
    return ((e or {}).get(split + "_geom") or {}).get(key)


def status_text(r):
    s = r["status"]
    if s == "not launched" and "wave 2" in r["group"]:
        return "starts after hps 0.1"
    if s == "done":
        return "done"
    if s == "full evals" and r["evals"]:
        return f"full evals ({len(r['evals'])} of {len(r['planned'])})"
    return s


# ---------------------------------------------------------------- tables
def main_table():
    head = ["group", "run", "step", "dev", "test", "test, k = 1", "q–q cos", "g–g cos", "q–q − g–g", "status"]
    rows, group = [], None
    for r in runs:
        e = ev(r["label"])
        g = r["group"] if r["group"] != group else ""
        group = r["group"]
        if e is None:
            rows.append([g, r["label"], "", "", "", "", "", "", "", ("pending", status_text(r))])
            continue
        step = str(r["selected"]) + (" *" if r["ties"] else "")
        k1 = f1(e.get("test_k1")) if r["mode"] == "multi" else "–"
        st = status_text(r)
        rows.append([g, r["label"], step, f1(e["dev"], 1 if r["mode"] == "multi" else 2), f1(e["test"]), k1,
                     f1(gv(e, "query_pair_cos")), f1(gv(e, "gold_pair_cos")), signed(gv(e, "gap")),
                     ("ok", st) if st == "done" else ("pending", st)])
    return head, rows


def signed(x):
    return "–" if x is None else f"{x:+.2f}"


def sim_table():
    head = ["run", "step", "q–q cos", "g–g cos", "q–q − g–g", "g–g above random", "query→gold cos", "margin",
            "sibling rank", "dev q–q", "dev g–g"]
    rows = []
    for r in runs:
        e = ev(r["label"])
        if e is None:
            continue
        rows.append([r["label"], str(r["selected"]), f1(gv(e, "query_pair_cos")), f1(gv(e, "gold_pair_cos")),
                     signed(gv(e, "gap")), f1(gv(e, "gold_cohesion")), f1(gv(e, "align_cos")),
                     signed(gv(e, "align_margin")), f1(gv(e, "sibling_rank_median"), 0),
                     f1(gv(e, "query_pair_cos", "dev")), f1(gv(e, "gold_pair_cos", "dev"))])
    return head, rows


def drift_table():
    head = ["run", "selected step", "dev", "test", "test, k = 1", "g–g cos", "final (2500): dev", "test",
            "test, k = 1", "g–g cos"]
    rows = []
    for r in runs:
        e, f = ev(r["label"]), ev(r["label"], 2500)
        if e is None or f is None or r["selected"] == 2500:
            continue
        multi = r["mode"] == "multi"
        rows.append([r["label"], str(r["selected"]), f1(e["dev"], 1), f1(e["test"]),
                     f1(e.get("test_k1")) if multi else "–", f1(gv(e, "gold_pair_cos")),
                     f1(f["dev"], 1), f1(f["test"]), f1(f.get("test_k1")) if multi else "–",
                     f1(gv(f, "gold_pair_cos"))])
    return head, rows


def perstep_ev(label):
    """Selected checkpoint if its per-embedding lists exist, else the evaluated checkpoint nearest it on full dev."""
    r = R[label]
    sel = ev(label)
    if sel and sel.get("test_perstep"):
        return sel
    cands = [e for e in r["evals"] if e.get("test_perstep") and e["step"] != 2500]
    return min(cands, key=lambda e: abs(e["dev"] - sel["dev"])) if (cands and sel) else None


def perstep_table(key):
    head = ["run", "step", "emb 1", "emb 2", "emb 3", "emb 4", "emb 5", "all 5, RRF"]
    rows = []
    for r in runs:
        if r["mode"] != "multi":
            continue
        e = perstep_ev(r["label"])
        if e is None:
            continue
        step = str(e["step"]) + ("" if e["step"] == r["selected"] else f" (selected {r['selected']})")
        allk = f1(e["test"]) if key == "test_perstep" else "–"
        rows.append([r["label"], step] + [f1(v) for v in e[key]] + [allk])
    s = ev("single")
    if s:
        rows.append(["single (reference)", str(R["single"]["selected"]), f1(s["test"]) if key == "test_perstep" else "–",
                     "", "", "", "", ""])
    return head, rows


# ---------------------------------------------------------------- figures (SVG + toggle legends)
# Every run has one colour (class r<i>, defined in run_css()) in every figure. Each figure carries a legend of
# run chips; the page script hides/shows a run's <g class="run"> and re-spaces the end labels.
RCLS = {r["label"]: f"r{i}" for i, r in enumerate(runs)}


def run_css():
    light, dark = [], []
    for i in range(len(runs)):
        h = (i * 137.508 + 200) % 360
        light.append(f".r{i}{{--c:hsl({h:.0f} 62% {38 if i % 2 else 46}%)}}")
        dark.append((i, f"--c:hsl({h:.0f} 70% {66 if i % 2 else 74}%)"))
    auto = "".join(f':root:not([data-theme="light"]) .r{i}{{{c}}}' for i, c in dark)
    forced = "".join(f':root[data-theme="dark"] .r{i}{{{c}}}' for i, c in dark)
    return "".join(light) + "@media (prefers-color-scheme: dark){" + auto + "}" + forced


def legend(fig, labels, refs=()):
    """Run chips grouped like the main table; refs = single-query runs drawn as dashed reference lines."""
    o = [f'<div class="legend" data-fig="{fig}"><div class="lg-ctl"><span>click a run to hide or show it; hover to highlight</span>'
         '<button data-act="all">show all</button><button data-act="none">hide all</button></div>']
    for g in dict.fromkeys(r["group"] for r in runs):
        rs = [r["label"] for r in runs if r["group"] == g and r["label"] in labels]
        if not rs:
            continue
        o.append(f'<div class="lg-row"><button class="lg-grp" data-grp="{html.escape(g)}">{html.escape(g)}</button>')
        for lab in rs:
            o.append(f'<button class="chip {RCLS[lab]}{" ref" if lab in refs else ""}" data-run="{html.escape(lab)}" '
                     f'data-grp="{html.escape(g)}"><span class="sw"></span>{html.escape(lab)}</button>')
        o.append("</div>")
    o.append("</div>")
    return "".join(o)


def frame(W, H, L, Rm, T, B, ylo, yhi, ystep, aria, xlabel, ylabel):
    y = lambda v: T + (yhi - v) * (H - T - B) / (yhi - ylo)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria)}">']
    for v in range(ylo, yhi + 1, ystep):
        s.append(f'<line x1="{L}" x2="{W - Rm}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
    s.append(f'<text x="{(L + W - Rm) / 2:.1f}" y="{H - 6}" class="axis" text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append(f'<text x="14" y="{(T + H - B) / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(T + H - B) / 2:.1f})">{html.escape(ylabel)}</text>')
    return s, y


def g_run(label, inner):
    return f'<g class="run {RCLS[label]}" data-run="{html.escape(label)}">{inner}</g>'


def line_fig(fig, aria, xlabel, series):
    """series: label -> (values for x = 1..5, end-label text). Single-query runs become dashed reference lines."""
    W, H, L, Rm, T, B = 640, 360, 52, 190, 16, 44
    x = lambda k: L + (k - 1) * (W - L - Rm) / 4
    s, y = frame(W, H, L, Rm, T, B, 0, 60, 10, aria, xlabel, "test MRecall@100")
    for k in range(1, 6):
        s.append(f'<text x="{x(k):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{k}</text>')
    refs = [r["label"] for r in runs if r["mode"] == "single" and ev(r["label"])]
    for lab in refs:
        v = ev(lab)["test"]
        s.append(g_run(lab, f'<line x1="{L}" x2="{W - Rm}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="ref"><title>{html.escape(lab)} (single query): {v:.2f}</title></line>'
                            f'<text x="{W - Rm + 8}" y="{y(v) + 4:.1f}" data-y="{y(v) + 4:.1f}" class="lab">{html.escape(lab)} {v:.2f}</text>'))
    for lab, (vals, end) in series.items():
        pts = " ".join(f"{x(k + 1):.1f},{y(v):.1f}" for k, v in enumerate(vals))
        inner = f'<polyline points="{pts}" class="ln"/>'
        inner += "".join(f'<circle cx="{x(k + 1):.1f}" cy="{y(v):.1f}" r="3.2" class="pt"><title>{html.escape(lab)}, k = {k + 1}: {v:.2f}</title></circle>'
                         for k, v in enumerate(vals))
        inner += f'<text x="{W - Rm + 8}" y="{y(vals[-1]) + 4:.1f}" data-y="{y(vals[-1]) + 4:.1f}" class="lab">{html.escape(end)}</text>'
        s.append(g_run(lab, inner))
    s.append("</svg>")
    return legend(fig, set(series) | set(refs), refs) + "".join(s)


def svg_kcurve():
    series = {r["label"]: (ev(r["label"])["test_curve"], r["label"]) for r in runs
              if r["mode"] == "multi" and ev(r["label"]) and ev(r["label"]).get("test_curve")}
    return line_fig("kcurve", "Test MRecall@100 against the number of queries k (RRF)", "queries used (k), RRF", series)


def svg_perstep():
    series = {}
    for r in runs:
        e = perstep_ev(r["label"]) if r["mode"] == "multi" else None
        if e:
            series[r["label"]] = (e["test_perstep"], r["label"] + ("" if e["step"] == r["selected"] else f" (step {e['step']})"))
    return line_fig("perstep", "Test MRecall@100 of each query embedding used alone", "query embedding used alone (generation order)", series)


def svg_scatter():
    W, H, L, Rm, T, B = 640, 360, 52, 24, 16, 44
    x0, x1 = 0.40, 0.65
    x = lambda v: L + (v - x0) * (W - L - Rm) / (x1 - x0)
    s, y = frame(W, H, L, Rm, T, B, 25, 55, 5, "Test score against gold–gold cosine, selected and final checkpoints",
                 "gold–gold cosine (test, the model's own document space)", "test MRecall@100")
    for i in range(0, 6):
        v = x0 + i * 0.05
        s.append(f'<text x="{x(v):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{v:.2f}</text>')
    shown = set()
    for r in runs:
        inner = ""
        for which, e in (("sel", ev(r["label"])), ("fin", ev(r["label"], 2500) if r["selected"] != 2500 else None)):
            g = gv(e, "gold_pair_cos")
            if e is None or g is None or not (x0 <= g <= x1) or not (25 <= e["test"] <= 56):
                continue
            inner += (f'<circle cx="{x(g):.1f}" cy="{y(e["test"]):.1f}" r="{5.5 if which == "sel" else 3.5}" '
                      f'class="sc {"single" if r["mode"] == "single" else "multi"} {which}"><title>{html.escape(r["label"])}, step {e["step"]}'
                      f'{" (selected)" if which == "sel" else " (final)"}: g–g {g:.3f}, test {e["test"]:.2f}</title></circle>')
            if which == "sel":
                inner += f'<text x="{x(g) + 8:.1f}" y="{y(e["test"]) + 4:.1f}" data-y="{y(e["test"]) + 4:.1f}" class="lab sl">{html.escape(r["label"])}</text>'
        if inner:
            s.append(g_run(r["label"], inner))
            shown.add(r["label"])
    s.append("</svg>")
    return legend("scatter", shown) + "".join(s)


def scatter_offchart():
    out = []
    for r in runs:
        e = ev(r["label"]); g = gv(e, "gold_pair_cos")
        if e is not None and g is not None and not (0.40 <= g <= 0.65 and 25 <= e["test"] <= 56):
            out.append(f"{r['label']} (g–g {g:.2f}, test {e['test']:.2f})")
    return out


FIG_JS = r"""
(function(){
  function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
  document.querySelectorAll('figure.ifig').forEach(function(fig){
    var lg=fig.querySelector('.legend'), key='wk-hidden-'+lg.dataset.fig, hidden;
    try{hidden=new Set(JSON.parse(store(key)||'[]'))}catch(e){hidden=new Set()}
    var chips=[].slice.call(lg.querySelectorAll('.chip')), gs=[].slice.call(fig.querySelectorAll('svg g.run'));
    function layout(){
      var labs=[];
      gs.forEach(function(g){if(!g.classList.contains('off'))g.querySelectorAll('text.lab').forEach(function(t){labs.push(t)})});
      var line=labs.filter(function(t){return !t.classList.contains('sl')}), sc=labs.filter(function(t){return t.classList.contains('sl')});
      line.sort(function(a,b){return a.dataset.y-b.dataset.y});
      var prev=-1e9, ys=line.map(function(t){prev=Math.max(+t.dataset.y,prev+13); return prev});
      var bottom=+fig.querySelector('svg').getAttribute('viewBox').split(' ')[3]-4;
      for(var i=ys.length-1;i>=0;i--){ys[i]=Math.min(ys[i],i==ys.length-1?bottom:ys[i+1]-13)}
      line.forEach(function(t,i){t.setAttribute('y',ys[i])});
      sc.sort(function(a,b){return a.dataset.y-b.dataset.y}); var placed=[];
      sc.forEach(function(t){var x=+t.getAttribute('x'), w=t.textContent.length*6.2, yy=+t.dataset.y, moved=true;
        while(moved){moved=false; for(var i=0;i<placed.length;i++){var p=placed[i];
          if(x<p.x+p.w&&p.x<x+w&&Math.abs(yy-p.y)<12){yy=p.y+12;moved=true}}}
        t.setAttribute('y',yy); placed.push({x:x,w:w,y:yy})});
    }
    function apply(){
      gs.forEach(function(g){g.classList.toggle('off',hidden.has(g.dataset.run))});
      chips.forEach(function(c){c.classList.toggle('off',hidden.has(c.dataset.run)); c.setAttribute('aria-pressed',!hidden.has(c.dataset.run))});
      store(key,JSON.stringify([].slice.call(hidden))); layout();
    }
    chips.forEach(function(c){
      c.addEventListener('click',function(){hidden.has(c.dataset.run)?hidden.delete(c.dataset.run):hidden.add(c.dataset.run); apply()});
      c.addEventListener('mouseenter',function(){if(hidden.has(c.dataset.run))return; fig.classList.add('hl');
        gs.forEach(function(g){g.classList.toggle('hot',g.dataset.run===c.dataset.run)})});
      c.addEventListener('mouseleave',function(){fig.classList.remove('hl')});
    });
    lg.querySelectorAll('.lg-grp').forEach(function(b){b.addEventListener('click',function(){
      var mine=chips.filter(function(c){return c.dataset.grp===b.dataset.grp}).map(function(c){return c.dataset.run});
      var anyOn=mine.some(function(r){return !hidden.has(r)});
      mine.forEach(function(r){anyOn?hidden.add(r):hidden.delete(r)}); apply()})});
    lg.querySelector('[data-act=all]').addEventListener('click',function(){hidden.clear(); apply()});
    lg.querySelector('[data-act=none]').addEventListener('click',function(){chips.forEach(function(c){hidden.add(c.dataset.run)}); apply()});
    apply();
  });
})();
"""


# ---------------------------------------------------------------- content
def nums():
    g = lambda lab, k="test": (ev(lab) or {}).get(k)
    return dict(single_dev=g("single", "dev"), single_test=g("single"), masked_dev=g("masked", "dev"),
                masked_test=g("masked"), hps_k1=g("hps 0.5", "test_k1"))


PERSTEP_NOTES = [
    "**Runs with diverse queries (`multi`, sampling, LR 3e-5, `B multi`): every embedding is weak alone, and they are complementary.** "
    "Each finds 27–48% of a question's golds and scores 1.3–16.6 MRecall on its own; fused, they reach 38–43. "
    "For comparison, `single`'s one query finds 77.62% of the golds (53.30 MRecall).",
    "**`masked`: five copies of one good query.** Every embedding alone scores 46–49 and finds 74–76% of the golds; fusing them gives 48.59, no gain.",
    "**`hps 0.5` and `hps 0.25`: one strong query plus weak extras.** Embedding 1 finds 75–77% of the golds (46.5–52.0 MRecall). Embeddings 2–5 find 40–56% (8–23 MRecall), "
    "and fusing them in lowers the score (`hps 0.5` step 1000: 51.98 alone, 47.27 fused). "
    "`hps + full sampling` is the same shape with even weaker extras: embedding 1 finds 75–76% of the golds (48.4–50.1 MRecall), embeddings 2–5 find 20–36% (0.2–3.2), "
    "and fusion again lowers the score (step 750: 50.09 alone, 46.33 fused). "
    "`hps + saturate 50%` is the same: embedding 1 finds 76% of the golds (49.34 MRecall), embeddings 2–5 find 28–36% (2.5–6.0), "
    "and fusion lowers the score (step 1000: 49.34 alone, 46.14 fused).",
    "**`hps 0.1`: the single-query term is too weak to make the first query strong.** Embedding 1 alone scores 36.35 (66% of the golds); "
    "the others are diverse-run-like (3–6 MRecall, 33–38%), and here fusion helps: 44.44 for all five.",
    "**Diversity penalty: the later embeddings get weaker, not complementary.** Embedding 1 is best (45–49), the rest decline (34–43), "
    "and fusion does not beat embedding 1 alone.",
    "No run has embeddings that are both strong alone and complementary. The diverse runs get the complementarity; the others get the strength.",
]


def content():
    n = nums()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pend = [r["label"] for r in runs if r["status"] != "done"]
    B = []
    B.append(("lede", f"QAMPARI only (AmbigQA is on hold). Week of 2026-10-05 to 2026-10-11. Numbers as of {now}; "
                      + (f"runs still in progress are marked pending: {', '.join(pend)}." if pend else
                         "every run in this report is fully evaluated; the report is closed. Follow-up work (frozen document "
                         "encoder, and the objective audit of 2026-10-13) is tracked in reports/objective_audit_2026-10-13/PLAN.md.")))

    B.append(("h2", "Summary"))
    B.append(("ul", [
        f"**Single-query is still the best QAMPARI retriever on test.** `single` scores {n['single_test']:.2f} test "
        f"({n['single_dev']:.2f} dev). The best multi-query run is round 1's `masked`: {n['masked_test']:.2f} test, "
        f"{n['masked_dev']:.1f} dev (k = 5, RRF). It is higher on dev but 5.65 points lower on test. None of the " + str(sum(r["status"] == "done" and r["label"] != "single" for r in runs)) + " other finished runs closes that gap.",
        "**Every objective that helped made the model behave more like a single-query retriever.** `masked`'s five queries are almost "
        f"the same vector (cosine 0.94). `hps 0.5` is best with its first query alone: {n['hps_k1']:.2f} test at k = 1, "
        "the only multi-query number above `single`. Adding its other four queries lowers the score.",
        "**Scored one at a time, the embeddings of the diverse runs are weak but complementary.** Each finds 27–43% of a "
        "question's golds (`single`'s one query: 77.6%); fused, they reach 38–43 test. `masked`'s five embeddings are each as good "
        "as the fused result, and `hps` has one strong embedding plus weak ones. No run has embeddings that are both.",
        "**Forcing the queries apart does not help recall.** The diversity penalty lowers the query–query cosine from 0.94 to 0.63, "
        "as intended, but test falls from 47.65 to 44.63 (λ = 2).",
        "**Hungarian multi-query training leaves a question's golds further apart than single-query training.** Gold–gold cosine is 0.50–0.53 "
        "against 0.59 for `single`, and the late checkpoints that collapse drop to 0.42–0.47. The masked loss keeps the golds "
        "together (0.57–0.61) and gives the best multi-query run, but that is not enough on its own: the diversity-penalty runs "
        "keep them together too and score 3–4.5 points lower.",
        "**Wave 2: adding sampling to `hps 0.5` gains nothing at the best checkpoint and brings back the late collapse.** "
        "`hps + full sampling` (62.8 dev / 46.33 test) and `hps + saturate 50%` (63.8 / 46.14) tie with `hps 0.5` (63.0 / 46.14). "
        "By step 2500 both fall to about 35 test (35.03, 35.40), like the plain sampling runs, and in both the first query alone "
        "(50.09, 49.34 test) beats the five fused.",
        "**Earlier full sampling, earlier saturation and a higher learning rate all hurt.** The sampling runs peak mid-training, then "
        "their first query collapses (test at k = 1 falls below 1 by step 2500). LR 3e-5 is worse than 1e-5 for both modes.",
        "**Measurement fixes.** The earlier multi-query 44.26 was inflated by 44 QAMPARI test questions that leaked into the v1 "
        "training data. Cheap dev cannot pick the best checkpoint within a run, so every run is now selected by full dev over "
        "the top 2 cheap-dev checkpoints plus the final one.",
    ]))

    B.append(("h2", "What was done this week"))
    B.append(("table", ["date", "work"], [
        ["Oct 5–7", "Round 1 (`clean_v3_abl`) on QAMPARI: `single`, `multi`, `masked`, `full sampling`, `hps 0.5`, each fully evaluated at three checkpoints, with k = 1–5 and round-robin/RRF from one retrieval."],
        ["Oct 6", "Headline metric fixed as MRecall@100 at k = 5 with RRF; round-robin and other k are secondary."],
        ["Oct 7", "Found the v1 test leak (44 of 531 QAMPARI test questions), which explains the drop from the grid's 44.26."],
        ["Oct 7", "Cheap-eval validity study: good for ranking runs, close to a coin flip for picking steps within a run. Selection protocol changed."],
        ["Oct 7", "Round 2 code: sampling saturation, masked + diversity penalty, RRF-based cheap selection, keep the final checkpoint."],
        ["Oct 7–11", "Round 2, wave 1: eight runs (hps weights, diversity penalty, sampling saturation, LR 3e-5). All evaluated except `hps 0.1`."],
        ["Oct 9–10", "Wave 3 code: frozen document encoder under FSDP (fixed by making the frozen copy an FSDP-ignored module, no DDP fallback), causal query attention for every model class, one shared corpus embedding for all frozen-encoder checkpoints."],
        ["Oct 10–11", "Wave 3 launched (cells C, A, B). Wave 2 launches automatically once the hps weight is decided."],
        ["all week", "Operations: /scratch quota incident (Oct 8), B200 nodes back with automatic requeue, disk-aware launch gates, an eval-queue bug that flagged good results as failed."],
    ], {}))

    B.append(("h3", "Common setup (all QAMPARI runs on this page)"))
    B.append(("p", "Every run shares the settings below; the run tables list only what each one changes."))
    B.append(("ul", [
        "**Model**: inf-retriever-v1-1.5b, full fine-tuning (no LoRA), 2 GPUs with FSDP, bf16 mixed precision, seed 0.",
        "**Optimisation**: 2,500 steps, AdamW (weight decay 0.01), LR 1e-5 (3e-5 in the two LR runs), 100 warmup steps then linear decay; "
        "a checkpoint every 250 steps.",
        "**Batch and loss**: 50 questions per GPU (100 per step); InfoNCE with temperature 0.05 on normalised embeddings; in-batch negatives.",
        "**Hard negatives (every run)**: training data `clean_v3_hn` with mined hard negatives (30 for 95–97% of questions). "
        "Each question gets one extra negative per gold (one in single-query mode), which is a mined hard negative with probability 0.5 and a random passage otherwise. "
        "The frozen-encoder cells (A, C) see the same negatives; only their document encoder does not update.",
        "**Passages**: 512 tokens. **Multi-query**: as many queries as golds in training, k = 5 at inference.",
    ]))

    B.append(("h2", "Main results"))
    B.append(("p", "Each run at the checkpoint with the best **full dev** score; test is never used for selection. Multi-query runs are "
                   "scored at k = 5 with RRF; single-query runs use their single query. MRecall@100 on the 500 dev and 531 test questions."))
    h, rows = main_table()
    B.append(("table", h, rows, {"num": [2, 3, 4, 5, 6, 7, 8], "status": 9}))
    B.append(("caption", [
        "**q–q cos**: mean cosine between a question's 5 query embeddings (test). Blank for single-query runs.",
        "**g–g cos**: mean cosine between the gold passages of the same question, in that model's own document space (test). Training changes the document encoder, so this differs between runs.",
        "**q–q − g–g**: positive means the queries are more alike than the golds they should cover (collapse); near 0 means they are spread like the golds.",
        "\\* the selected step beats another evaluated step by less than 2 dev points (about two paired standard errors): treat as a tie.",
    ]))
    B.append(("h3", "Test score against the number of queries"))
    B.append(("figure", svg_kcurve(), "Test MRecall@100 at k = 1–5 (RRF over the first k queries), selected checkpoints. "
              "Runs whose queries differ (`multi`, the sampling runs) need all five queries; runs that act like a single "
              "query (`masked`, `hps`) are flat or fall with k. None reaches `single` at k = 5. Dashed lines: single-query runs (one query, so no curve)."))

    B.append(("h3", "Each query embedding on its own"))
    B.append(("p", "The same retrieval, scored with one embedding's top 100 alone (embedding 1 = the first generated). "
                   "Round 1 deleted the per-embedding lists of most selected checkpoints, so `multi`, `masked` and `hps 0.5` use "
                   "their other evaluated checkpoint (marked with its step); it is within 2 dev points of the selected one."))
    B.append(("figure", svg_perstep(), "Test MRecall@100 of each query embedding used alone. MRecall needs every gold of a question "
              "in the top 100, so an embedding that targets one gold scores low by design; Recall@100 (second table) shows how much each one covers."))
    h, rows = perstep_table("test_perstep")
    B.append(("table", h, rows, {"num": [2, 3, 4, 5, 6, 7]}))
    B.append(("p", "Recall@100 (share of a question's golds found) of each embedding alone, test:"))
    h, rows = perstep_table("test_perstep_recall")
    B.append(("table", h[:-1], [r[:-1] for r in rows if not r[0].startswith("single")], {"num": [2, 3, 4, 5, 6]}))
    B.append(("ul", PERSTEP_NOTES))

    B.append(("h2", "Findings by question"))
    B.append(("h3", "Training objective"))
    B.append(("ul", [
        "`masked` (other golds are not negatives) gives the best multi-query dev and test, by collapsing its queries: cosine 0.94, flat k curve.",
        "The diversity penalty (relu of query–query cosine minus the matched golds' cosine) works as designed: q–q − g–g falls from +0.38 to +0.25 (λ = 0.5) and +0.02 (λ = 2). Test falls by 3–4.5 points. Spreading the queries does not make each one point at a different gold: the margin stays negative.",
        "`hps` weights 0.5 and 0.25 tie on dev (63.0 and 62.8). The single-query term is what makes `hps` work: its first query alone beats `single` on test (0.5: 54.05).",
    ]))
    B.append(("h3", "Scheduled sampling"))
    B.append(("ul", [
        "Saturating the sampling rate early (25% or 50% of training) peaks around steps 500–1000 at 57–60 dev, below `multi`, then decays. By step 2500 the first query is useless (test k = 1: 0.75 and 0.38).",
        "Full sampling from the start (round 1) is worse: the first three queries are close to useless (test k = 1: 9.6).",
    ]))
    B.append(("h3", "Learning rate"))
    B.append(("ul", ["LR 3e-5 is worse than 1e-5 for both modes: single 61.4 / 48.21 against 64.80 / 53.30; multi 55.8 / 39.92 against 61.2 / 38.42 (dev / test)."]))
    B.append(("h3", "Late training"))
    B.append(("p", "Round-2 multi-query runs lose 6–23 dev points between their best checkpoint and step 2500; single-query loses 1.4. "
                   "Keeping the final checkpoint (new this round) is what shows this."))
    h, rows = drift_table()
    B.append(("table", h, rows, {"num": list(range(1, 10))}))

    B.append(("h2", "Query and gold similarity"))
    B.append(("p", "Computed on the full corpus for every evaluated checkpoint (`tools/geometry_eval.py`), test unless marked dev."))
    B.append(("ul", [
        "**q–q cos**: mean pairwise cosine between a question's 5 query embeddings.",
        "**g–g cos**: mean pairwise cosine between a question's gold passages; **above random** subtracts the cosine of random passage pairs, so it compares across document spaces.",
        "**query→gold cos**: each query's cosine with the gold it is matched to (Hungarian); **margin**: that minus its cosine with the closest other gold of the question. A negative margin means a query is not specific to its gold.",
        "**sibling rank**: median number of corpus passages closer to a gold than its nearest sibling gold. Low means the golds sit together.",
    ]))
    h, rows = sim_table()
    B.append(("table", h, rows, {"num": list(range(1, 11))}))
    base = ev("C single")
    B.append(("p", "**Untrained document space (reference).** Cell C keeps the base model's document encoder, so its g–g is the base retriever's: " +
              (f"{gv(base, 'gold_pair_cos'):.2f} raw, but random passage pairs already sit at {gv(base, 'gold_rand_cos'):.2f}, so only "
               f"{gv(base, 'gold_cohesion'):.2f} above random (test). The base space is very anisotropic; training spreads the corpus out "
               "(random pairs 0.02–0.05 in every trained run), so raw g–g is not comparable with it, only **above random** is. On that scale every trained "
               "document encoder holds a question's golds far closer together than the base one (0.47–0.59 at the selected checkpoints, against "
               f"{gv(base, 'gold_cohesion'):.2f}); the differences between runs are differences in how much closer. "
               if base and gv(base, 'gold_pair_cos') is not None else "pending (C single full eval running). ")))
    B.append(("figure", svg_scatter(), "Test MRecall@100 against gold–gold cosine. Large dots: selected checkpoints; small dots: final checkpoints (step 2500). "
              "Filled: single-query; open: multi-query; hover a dot for its numbers. Off the chart: " + "; ".join(scatter_offchart()) +
              " (the frozen-encoder runs sit in the untrained, anisotropic document space; see above)."))
    B.append(("ul", [
        "Single-query training keeps a question's golds together (g–g 0.59, sibling rank 4). Hungarian multi-query training separates them (0.50–0.53; sibling rank up to 16–20), presumably because each gold is a negative for the queries matched to its siblings.",
        "The masked loss removes that push (other golds are not negatives) and keeps g–g at single-query level (0.57–0.61, sibling rank 3–4). The diversity-penalty runs do too, yet score 3–4.5 points below `masked`: keeping the golds together is not enough on its own.",
        "The collapsing late checkpoints have the lowest g–g (0.42–0.47).",
        "The q–q cosine is about 0.6–0.67 for every multi-query run except `masked` and the penalised runs. Query similarity alone does not separate good runs from bad ones.",
    ]))

    B.append(("h2", "Measurement"))
    B.append(("h3", "The v1 test leak"))
    B.append(("p", "44 of the 531 QAMPARI test questions had a near-duplicate in the v1 training data (gold-set Jaccard ≥ 0.8); v2/v3 remove them. "
                   "The grid's best multi-query RRF score drops from 44.26 on all 531 to 41.48 on the 487 clean questions. On those, the current "
                   "runs show no regression against the grid. Grid test numbers are not comparable with anything since."))
    B.append(("table", ["test MRecall@100, round-robin k = 5", "all 531", "44 leaked", "487 clean"], [
        ["grid single (v1 data)", "52.73", "84.09", "49.90"], ["v3 single", "53.30", "68.18", "51.95"],
        ["grid multi (v1 data)", "38.23", "63.64", "35.93"], ["v3 multi, step 750", "37.10", "36.36", "37.17"],
        ["v3 masked", "49.34", "61.36", "48.25"]], {"num": [1, 2, 3]}))
    B.append(("h3", "Cheap dev and checkpoint selection"))
    B.append(("ul", [
        "Cheap dev scores each checkpoint on a reduced corpus (about 500k passages). Across runs it agrees with full dev (Spearman 0.91–0.96).",
        "Within a run it picks the full-dev-best checkpoint about half the time. Neighbouring checkpoints differ by about 1 point with a paired standard error of about 1 point, on full dev too.",
        "New protocol: full-evaluate the top 2 by cheap dev plus the final checkpoint; select by full dev; treat differences under 2 points as ties.",
    ]))

    B.append(("h2", "In progress" if pend else "Status of the week's work (all done)"))
    B.append(("table", ["work", "runs", "status"], [
        ["Wave 2 (done): best hps weight combined with sampling", "`hps + full sampling`, `hps + saturate 50%`",
         "launched 2026-10-12 10:14 with weight 0.5, the best full dev of 0.5 / 0.25 / 0.1 (63.0 / 62.8 / 62.6: a tie under the 2-point rule); " + "; ".join(f"{r['label']}: {status_text(r)}" for r in runs if "wave 2" in r["group"])],
        ["Wave 3: document encoder frozen or joint × query attention causal or bidirectional",
         "C (frozen, bidirectional), A (frozen, causal), B (joint, causal); single and multi each. D = round 1.",
         "; ".join(f"{r['label']}: {status_text(r)}" for r in runs if r["group"].startswith("doc encoder"))],
    ], {}))
    cs, cm, as_ = ev("C single"), ev("C multi"), ev("A single")
    w3 = []
    if cs:
        w3.append(f"`C single` (frozen document encoder) scores {cs['dev']:.1f} full dev / {cs['test']:.2f} test, against "
                  f"{n['single_dev']:.1f} / {n['single_test']:.2f} for `single` with a trained document encoder: freezing the document encoder roughly halves the score.")
    if cs and cm:
        w3.append(f"With the encoder frozen, multi-query matches single-query: `C multi` scores {cm['dev']:.1f} / {cm['test']:.2f} (RRF, k = 5), a tie with `C single`. "
                  "With a trained encoder the gap is 5.7–15.3 test points. So the gap appears only once the document encoder is trained, which fits the gold–gold finding "
                  "(Hungarian training leaves golds further apart than single-query training). A frozen encoder may also simply cap both near 31, leaving no room for a gap.")
        if cm.get("test_perstep"):
            w3.append("Its five embeddings are similar in strength on their own (test " + ", ".join(f"{x:.1f}" for x in cm["test_perstep"]) +
                      f"); fusing them adds {cm['test'] - max(cm['test_perstep']):.1f} over the best one.")
    am = ev("A multi")
    if am and cm:
        w3.append(f"`A multi` (frozen, causal) scores {am['dev']:.1f} / {am['test']:.2f}, a tie with `C multi`: causal attention does not change the multi-query result either.")
    if as_:
        w3.append(f"`A single` (frozen, causal attention) scores {as_['dev']:.1f} / {as_['test']:.2f}, also a tie with `C single`: with a frozen encoder, causal vs bidirectional query attention makes no difference for single-query.")
    bs = ev("B single")
    if bs:
        w3.append(f"`B single` (trained encoder, causal attention) scores {bs['dev']:.1f} / {bs['test']:.2f}"
                  + (" so far" if R["B single"]["status"] != "done" else "") +
                  f", against {n['single_dev']:.1f} / {n['single_test']:.2f} for `single` (trained encoder, bidirectional): "
                  + ("a tie on dev." if abs(bs['dev'] - n['single_dev']) < 2 else "not a tie on dev."))
    bm, mu = ev("B multi"), ev("multi")
    if bm and mu:
        tie = lambda a, b: "a tie" if abs(a - b) < 2 else ("above" if a > b else "below")
        s = (f"`B multi` (trained encoder, causal attention) scores {bm['dev']:.1f} / {bm['test']:.2f}"
             + (" so far" if R["B multi"]["status"] != "done" else "") +
             f", against {mu['dev']:.1f} / {mu['test']:.2f} for `multi` (bidirectional) — {tie(bm['dev'], mu['dev'])} on dev, "
             f"{bm['test'] - mu['test']:+.1f} on test")
        if bs:
            s += f" — and {bs['dev'] - bm['dev']:.1f} dev / {bs['test'] - bm['test']:.1f} test below `B single`"
        s += "."
        if bm.get("test_perstep"):
            s += (" As in `multi`, the first embedding alone is weak (test " + ", ".join(f"{x:.1f}" for x in bm["test_perstep"]) +
                  "); the score comes from fusing the five.")
        fin = next((e for e in R["B multi"]["evals"] if e["step"] == 2500), None)
        if fin and fin.get("test_perstep_recall") and bm["dev"] - fin["dev"] >= 2:
            s += (f" It degrades late in training: step 2500 scores {fin['dev']:.1f} / {fin['test']:.2f}, each embedding alone finds only "
                  f"{min(fin['test_perstep_recall']):.0f}–{max(fin['test_perstep_recall']):.0f}% of the golds "
                  f"({min(fin['test_perstep']):.1f}–{max(fin['test_perstep']):.1f} MRecall), so nearly all of its score comes from fusion.")
        w3.append(s)
    B.append(("p", " ".join(w3) if w3 else "Early sign from wave 3: `C single` scores 37–39 on cheap dev at every step, against 71–74 for `single`."))

    B.append(("h2", "Next steps and open questions"))
    B.append(("ul", [
        "**Multi-query as an add-on to a strong single query.** `hps`'s first query already beats `single`. Make the later queries add only what the first misses, for example by weighting the first query's list in the fusion or by stopping early when later queries add nothing.",
        "**Keep the golds together.** Gold–gold similarity follows quality. Test a multi-query loss that does not push sibling golds apart: masked loss without collapse, or a document-side term.",
        "**Second seed.** Many differences here are 1–3 points, about the size of seed and checkpoint noise. A second seed of the main runs would show which are real (on hold).",
        "**AmbigQA** is on hold until the QAMPARI picture settles.",
    ]))

    B.append(("h2", "Appendix: setup and runs"))
    B.append(("ul", [
        "Model: inf-retriever-v1-1.5b, fine-tuned in full, 2 GPUs (FSDP), 2,500 steps, LR 1e-5 unless stated, one seed.",
        "Data: clean v3 QAMPARI with hard negatives (ratio 0.5; see Common setup); dev = clean v2 dev (500); test = the unchanged AMER test set (531).",
        "Corpus: Wikipedia chunks embedded at 512 tokens. Multi-query: k = number of golds in training, k = 5 at inference.",
        "Scheduled sampling: the rate of feeding back the model's own query embedding grows linearly to 1 at the end of training unless stated.",
    ]))
    B.append(("table", ["run", "group", "change"], [[r["label"], r["group"], r["desc"]] for r in runs], {}))
    B.append(("p", "Detailed reports: `reports/ablation_clean_v3_2026-10-07.md` (round 1, leak), `reports/cheap_eval_validity_2026-10-07.md`, "
                   "`reports/wave3_prep_2026-10-09.md`. Data behind this page: `reports/weekly_summary_data.json`, built by "
                   "`training/inf_retriever/tools/weekly_collect.py`."))
    return B


# ---------------------------------------------------------------- renderers
def md_inline(t):
    return t


def to_md(B):
    out = ["# QAMPARI multi-query: week of Oct 5–11", ""]
    for b in B:
        k = b[0]
        if k == "lede":
            out += [b[1], ""]
        elif k in ("h2", "h3"):
            out += [("## " if k == "h2" else "### ") + b[1], ""]
        elif k == "p":
            out += [b[1], ""]
        elif k in ("ul", "caption"):
            out += [f"- {x}" for x in b[1]] + [""]
        elif k == "table":
            h, rows = b[1], b[2]
            out.append("| " + " | ".join(h) + " |")
            out.append("| " + " | ".join("---" for _ in h) + " |")
            for r in rows:
                out.append("| " + " | ".join(c[1] if isinstance(c, tuple) else (c or "") for c in r) + " |")
            out.append("")
        elif k == "figure":
            out += [f"*Figure (see the HTML page): {b[2]}*", ""]
    return "\n".join(out)


def h_inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\\*", "*")


CSS = """
:root{
  /* layout: one reading column; wide tables scroll inside their own frame */
  --bg:#f6f7f9; --panel:#ffffff; --fg:#17202b; --muted:#5b6675; --rule:#d9dee5;
  --accent:#1f6f8b; --accent-soft:#e3eff4; --warn:#9a5b00; --warn-soft:#fbf0dc; --ok:#2f7a4b;
  --c1:#1f6f8b; --c2:#b5482f; --c3:#6b4fa3; --c4:#7a7f87; --c5:#c08a12; --c6:#2f8a6a; --c7:#c2407e;
  --display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#11161c; --panel:#171e26; --fg:#e4e9ef; --muted:#9aa6b4; --rule:#2c3642;
  --accent:#6fb6d0; --accent-soft:#1b2d36; --warn:#e5b25a; --warn-soft:#2e2615; --ok:#79c495;
  --c1:#6fb6d0; --c2:#e58a72; --c3:#b39ce6; --c4:#a5abb3; --c5:#e5b84a; --c6:#6fcfa9; --c7:#ee8ab8; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#11161c; --panel:#171e26; --fg:#e4e9ef; --muted:#9aa6b4; --rule:#2c3642;
  --accent:#6fb6d0; --accent-soft:#1b2d36; --warn:#e5b25a; --warn-soft:#2e2615; --ok:#79c495;
  --c1:#6fb6d0; --c2:#e58a72; --c3:#b39ce6; --c4:#a5abb3; --c5:#e5b84a; --c6:#6fcfa9; --c7:#ee8ab8; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body);margin:0}
.wrap{max-width:980px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
.prose{max-width:68ch}
header{border-bottom:2px solid var(--fg);padding-bottom:14px;margin-bottom:22px}
.eyebrow{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(28px,4.5vw,40px)/1.1 var(--display);margin:.35em 0 .3em;text-wrap:balance}
h2{font:600 24px/1.2 var(--display);margin:2.2em 0 .6em;padding-top:.6em;border-top:1px solid var(--rule);text-wrap:balance}
h3{font:600 17px/1.3 var(--display);margin:1.6em 0 .4em;color:var(--fg)}
.lede{color:var(--muted);max-width:75ch;margin:0}
p,ul{max-width:75ch}
ul{padding-left:1.2em} li{margin:.35em 0}
code{font:13px var(--mono);background:var(--accent-soft);padding:.05em .3em;border-radius:3px}
strong{font-weight:600}
.tbl{overflow-x:auto;margin:.8em 0 1em;background:var(--panel);border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule)}
th{font:600 11.5px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:var(--panel);white-space:nowrap}
tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
th.num{text-align:right}
td.grp{color:var(--accent);font-weight:600;white-space:nowrap}
tr.grpstart td{border-top:1.5px solid var(--rule)}
.pill{display:inline-block;font:500 11.5px/1.5 var(--mono);padding:0 .55em;border-radius:99px;white-space:nowrap}
.pill.ok{color:var(--ok);border:1px solid var(--ok)}
.pill.pending{color:var(--warn);background:var(--warn-soft);border:1px solid var(--warn)}
tr.pending td{color:var(--muted)}
.cap{font-size:13px;color:var(--muted);margin-top:-.3em}
.cap li{margin:.2em 0}
figure{margin:1em 0 1.4em;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:6px;max-width:80ch}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .tick{fill:var(--muted);font:11px var(--mono)}
svg .axis{fill:var(--muted);font:12px var(--body)}
svg .lab{fill:var(--fg);font:12px var(--body)}
svg .ref{stroke:var(--fg);stroke-width:1.5;stroke-dasharray:5 4}
svg .ref-t{fill:var(--fg);font-weight:600}
svg .ln{fill:none;stroke-width:2.2;stroke:var(--c)}
svg .pt{fill:var(--c);stroke:var(--panel);stroke-width:1.5}
svg .run .lab{fill:var(--c);font-weight:600}
svg .run .ref{stroke:var(--c)}
svg .run.off{display:none}
svg .run{transition:opacity .15s}
figure.hl svg .run:not(.hot){opacity:.12}
figure.hl svg .run.hot .ln{stroke-width:3.4}
svg .sc{stroke-width:2;stroke:var(--c)}
svg .sc.single{fill:var(--c)}
svg .sc.multi{fill:var(--panel)}
svg .sc.fin{opacity:.7}
svg .lab.sl{font-size:11px;font-weight:500}
.legend{margin:0 0 10px;font-size:12.5px}
.lg-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;color:var(--muted);margin-bottom:6px}
.lg-ctl span{margin-right:auto;font-size:12px}
.lg-row{display:flex;flex-wrap:wrap;gap:4px 6px;align-items:center;margin:3px 0}
.legend button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--rule);border-radius:99px;padding:1px 9px;cursor:pointer;line-height:1.6}
.legend button:hover{border-color:var(--fg)}
.legend button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.legend .lg-grp{font:600 11px/1.9 var(--mono);text-transform:uppercase;letter-spacing:.03em;color:var(--accent);border-color:transparent;background:var(--accent-soft)}
.chip .sw{display:inline-block;width:14px;height:3px;background:var(--c);vertical-align:middle;margin-right:6px;border-radius:2px}
.chip.ref .sw{background:repeating-linear-gradient(90deg,var(--c) 0 4px,transparent 4px 7px)}
.chip.off{opacity:.45;text-decoration:line-through}
.chip.off .sw{background:var(--muted)}
@media (max-width:520px){body{font-size:14px} .wrap{padding-inline:16px}}
"""


def to_html(B):
    o = ['<title>QAMPARI Multi-Query Week</title>',
         '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">',
         f"<style>{CSS}{run_css()}</style>", '<div class="wrap">']
    for b in B:
        k = b[0]
        if k == "lede":
            o.append('<header><div class="eyebrow">AMER · multi-query retrieval · weekly update</div>'
                     '<h1>QAMPARI multi-query: week of Oct 5–11</h1>'
                     f'<p class="lede">{h_inline(b[1])}</p></header>')
        elif k == "h2":
            o.append(f"<h2>{h_inline(b[1])}</h2>")
        elif k == "h3":
            o.append(f"<h3>{h_inline(b[1])}</h3>")
        elif k == "p":
            o.append(f"<p>{h_inline(b[1])}</p>")
        elif k == "ul":
            o.append("<ul>" + "".join(f"<li>{h_inline(x)}</li>" for x in b[1]) + "</ul>")
        elif k == "caption":
            o.append('<ul class="cap">' + "".join(f"<li>{h_inline(x)}</li>" for x in b[1]) + "</ul>")
        elif k == "table":
            h, rows, opt = b[1], b[2], b[3]
            num = set(opt.get("num", []))
            t = ['<div class="tbl"><table><thead><tr>']
            t += [f'<th class="{"num" if i in num else ""}">{h_inline(c)}</th>' for i, c in enumerate(h)]
            t.append("</tr></thead><tbody>")
            for r in rows:
                pending = any(isinstance(c, tuple) and c[0] == "pending" for c in r) and not r[3]
                cls = ["pending" if pending else "", "grpstart" if r and r[0] and h[0] == "group" else ""]
                t.append(f'<tr class="{" ".join(c for c in cls if c)}">')
                for i, c in enumerate(r):
                    if isinstance(c, tuple):
                        t.append(f'<td><span class="pill {c[0]}">{html.escape(c[1])}</span></td>')
                    elif h[0] == "group" and i == 0:
                        t.append(f'<td class="grp">{h_inline(c)}</td>')
                    else:
                        t.append(f'<td class="{"num" if i in num else ""}">{h_inline(c or "")}</td>')
                t.append("</tr>")
            t.append("</tbody></table></div>")
            o.append("".join(t))
        elif k == "figure":
            o.append(f'<figure class="ifig">{b[1]}<figcaption>{h_inline(b[2])}</figcaption></figure>')
    o.append("</div>")
    o.append(f"<script>{FIG_JS}</script>")
    return "\n".join(o)


if __name__ == "__main__":
    B = content()
    open(MD, "w").write(to_md(B) + "\n")
    open(HTML, "w").write(to_html(B) + "\n")
    print("wrote", MD, HTML)
