"""Score E9 draft-bits arms against per-prompt break-even acceptance floors.

  python3 research/e9_arms.py LABEL [LABEL...]

Reads .mlxfast-private/draft-bits/e9-<label>-b<bits>/amdahl.json.

The floor is derived per prompt, because the break-even acceptance drop depends
on the prompt's own drafts-per-round D and round time, not just on the kernel.
The only thing low-bit readout changes mechanically is readout cost, so:

    saving_per_round = kernel_speedup * R4 * D
    tokens_per_round_breakeven = tpr4 * (1 - saving_per_round / ms_per_round4)
    acceptance_floor = (tokens_per_round_breakeven - 1) / D

R4 is the per-readout 4-bit cost, a host property, solved once from the English
arms in research/e9_breakeven.py. This floor prices a rejection as a lost token
only; it ignores rollback and re-forward, so it is an OPTIMISTIC upper bound and
an arm landing at the floor should be read as a loss.
"""

import json
import pathlib
import sys

TOKENS = 512
KERNEL_SPEEDUP = {4: 0.0, 3: 0.242, 2: 0.424}
R4_MS = 1.0480  # per-readout 4-bit cost, solved from the English arms
ROOT = pathlib.Path(__file__).resolve().parent.parent / ".mlxfast-private" / "draft-bits"


def load(label, bits):
    p = ROOT / f"e9-{label}-b{bits}" / "amdahl.json"
    if not p.exists():
        return None
    leg = json.loads(p.read_text())["mtp_leg"]
    ident = (p.parent / "identity.txt").read_text()
    sha = next(
        (l.split("=", 1)[1].strip() for l in ident.splitlines() if l.startswith("worker_sha256=")),
        "?",
    )
    leg["spt_ms"] = leg["parent_measured_seconds_per_token"] * 1000
    leg["ms_per_round"] = leg["spt_ms"] * TOKENS / leg["round_count"]
    leg["D"] = leg["effective_mean_draft_len"]
    leg["worker_sha256"] = sha
    return leg


def report(label):
    arms = {b: load(label, b) for b in (4, 3, 2)}
    arms = {b: a for b, a in arms.items() if a}
    if 4 not in arms:
        print(f"\n### {label}: no control arm, skipping")
        return
    ctl = arms[4]
    print(f"\n### prompt: {label}   (control acceptance {ctl['accepted_draft_rate']:.6f})")
    print(f"  worker_sha256 (all arms): {' '.join(sorted({a['worker_sha256'][:12] for a in arms.values()}))}")
    share = ctl["D"] * R4_MS / ctl["ms_per_round"]
    print(f"  ms/round {ctl['ms_per_round']:.3f}   D {ctl['D']:.4f}   readout share {share * 100:.2f}%")
    print(f"  ceiling if draft head were FREE: {-share * 100:.2f}% ms/token\n")
    hdr = f"  {'arm':>4} {'ms/token':>10} {'vs ctl':>8} {'accept':>9} {'floor':>9} {'margin':>9} {'rounds':>7} {'tok/rnd':>8} {'match':>6}"
    print(hdr)
    for b in (4, 3, 2):
        a = arms.get(b)
        if not a:
            continue
        # Control D throughout: the counterfactual is "acceptance falls while the
        # schedule holds". Using the arm's own D is circular, because a degraded
        # arm proposes fewer drafts and would inflate its own floor above control.
        saving = KERNEL_SPEEDUP[b] * R4_MS * ctl["D"]
        tpr_be = ctl["tokens_per_round"] * (1 - saving / ctl["ms_per_round"])
        floor = (tpr_be - 1) / ctl["D"] if b != 4 else ctl["accepted_draft_rate"]
        margin = a["accepted_draft_rate"] - floor
        dt = (a["spt_ms"] / ctl["spt_ms"] - 1) * 100
        print(
            f"  {b}b   {a['spt_ms']:10.4f} {dt:+7.3f}% {a['accepted_draft_rate']:9.6f} "
            f"{floor:9.6f} {margin:+9.6f} {a['round_count']:7d} {a['tokens_per_round']:8.4f} "
            f"{str(a['all_tokens_matched']):>6}"
        )
    for b in (3, 2):
        a = arms.get(b)
        if not a:
            continue
        dr = (a["round_count"] / ctl["round_count"] - 1) * 100
        dm = (a["ms_per_round"] / ctl["ms_per_round"] - 1) * 100
        print(f"  {b}b split: per-round time {dm:+.3f}%  |  round count {dr:+.3f}% "
              f"({ctl['round_count']}->{a['round_count']})")
    if 2 in arms:
        eq = arms[2]["accepted_draft_rate"] == ctl["accepted_draft_rate"]
        print(f"  prediction-2 check: 2b acceptance == 4b acceptance exactly? {eq}")


for label in sys.argv[1:]:
    report(label)
