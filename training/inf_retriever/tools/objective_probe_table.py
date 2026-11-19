"""Markdown tables from tools/objective_probe.py output.  python objective_probe_table.py <probe.json>"""
import json
import re
import sys

r = json.load(open(sys.argv[1]))


def name(spec, v):
    if spec.startswith("base:"):
        return "base (HF weights)"
    m = re.search(r"qampari_(.+)/checkpoint/step-(\d+)", spec)
    return f"{m.group(1)} @{m.group(2)}" if m else spec


def fmt(x):
    return "" if x is None else f"{x:.3f}"


def table(title, cols, get):
    print(f"\n### {title}\n")
    print("| run | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for spec, v in r.items():
        print(f"| {name(spec, v)} | " + " | ".join(fmt(get(v, key)) for _, key in cols) + " |")


def g(v, path):
    sec, rate, key = path
    return v.get(sec, {}).get(rate, {}).get(key)


table("D2, teacher-forced (rate 0): does output t+1 copy the gold fed at step t?",
      [("top-1 = fed gold", ("d2", "rate_0", "tf_top1_fed")),
       ("top-1 = unfed gold", ("d2", "rate_0", "tf_top1_unfed")),
       ("matched to fed gold", ("d2", "rate_0", "tf_copy_match")),
       ("cos fed gold", ("d2", "rate_0", "tf_copy_cos")),
       ("best cos unfed gold", ("d2", "rate_0", "tf_best_unfed_cos"))],
      lambda v, p: g(v, p))
table("D2, free-running (rate 1): repeats",
      [("cos(q_t, q_t-1)", ("d2", "rate_1", "free_repeat_cos")),
       ("same top-1 gold", ("d2", "rate_1", "free_repeat_top1")),
       ("top-1 is a gold", ("d2", "rate_1", "free_top1_gold")),
       ("cos(q1, q0)", ("d2", "rate_1", "free_cos_q1_q0")),
       ("cos(q4, q0)", ("d2", "rate_1", "free_cos_q4_q0"))],
      lambda v, p: g(v, p))
table("D2, norms (fed gold vs own output vs question tokens)",
      [("gold", ("d2", "rate_0", "norm_gold")),
       ("output (free)", ("d2", "rate_1", "norm_output")),
       ("token", ("d2", "rate_0", "norm_token"))],
      lambda v, p: g(v, p))
for rate in ("rate_0.5", "rate_1"):
    table(f"D3, gradient into output 0 at sampling {rate[5:]}: own term vs later steps through the fed-back input",
          [("|g_feed|/|g_own|", ("d3", rate, "feed_over_own")),
           ("cos(g_own, g_feed)", ("d3", rate, "cos_own_feed")),
           ("share opposing", ("d3", rate, "share_feed_opposes")),
           ("batch |g_feed|/|g_own|", ("d3", rate, "batch_feed_over_own")),
           ("batch cos", ("d3", rate, "batch_cos_own_feed"))],
          lambda v, p: g(v, p))
