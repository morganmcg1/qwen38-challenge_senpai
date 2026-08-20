#!/usr/bin/env python3
"""Split every E56 round cost by width AND by the previous round's rejection.

  python3 research/e56_repair_census.py [--session s4] [--out PATH]

WHY. The advisor reports that a rejection charges one extra pass over the 48
Gated DeltaNet layers to the FOLLOWING round, that the charge is binary in
whether anything was rejected, and that width and repair are confounded by
construction because a round can only be wide when the previous round rejected
nothing. A cost table indexed by width alone therefore charges narrow rounds
for a repair the previous round caused. The quantity the cost model needs is
`T(M, repaired)`, never `T(M)`.

WHAT THIS READS. Every E56 timed leg already emits one `mtp-trace:` line per
round carrying `round_us` and its components, so the split needs no GPU and no
new instrument. The dispatch census that motivated this is a count; this is the
same effect in seconds, which is the unit the scheduler actually spends.

WHAT `repaired` MEANS HERE. `repaired[i]` is true when round `i-1` had at least
one draft rejected, so round `i` is the round that pays. The first round of a
leg has no predecessor and is dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics as st
import subprocess
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUANTIZED_H = (ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
               "kernels/quantized.h")

TRACE_RE = re.compile(
    r"mtp-trace: round=(?P<round>\d+) d=(?P<d>\d+) acc=(?P<acc>\d+) "
    r"draft_build_us=(?P<draft_build>\d+) verify_build_us=(?P<verify_build>\d+) "
    r"eval_wall_us=(?P<eval_wall>\d+) readout_us=(?P<readout>\d+) "
    r"commit_us=(?P<commit>\d+) upkeep_us=(?P<upkeep>\d+) "
    r"round_us=(?P<round_us>\d+)")
COMPONENTS = ("draft_build", "verify_build", "eval_wall", "readout", "commit",
              "upkeep")

LAYOUTS = {
    "s3": (("base1", "base"), ("s45a", "s45"), ("s89a", "s89"),
           ("sfulla", "sfull"), ("sfullb", "sfull"), ("s89b", "s89"),
           ("s45b", "s45"), ("base2", "base")),
    "s4": (("base1", "base"), ("s45a", "s45"), ("s89a", "s89"),
           ("h224a", "h224"), ("mixa", "s45h224"), ("mixb", "s45h224"),
           ("h224b", "h224"), ("s89b", "s89"), ("s45b", "s45"),
           ("base2", "base")),
}

# The commit each session's binaries were built at. A session must be scored
# against the dispatch table it actually ran on: E55 changed `case 9` between
# these two, so scoring s3 with the live table misclassifies its 8 -> 9 step.
SESSION_BASE = {"s3": "aded0f5", "s4": "7040406"}


def ipg_table(ref: str | None = None) -> dict[int, int]:
    """The `out_vec_size >= 4096` inputs-per-group switch, from source.

    Read the TEMPLATE ARGUMENT, never the prose. The comment above `case 8` in
    this file describes a 3+3+2 split at inputs-per-group 3; the shipped
    template argument is 4. A campaign table built from the comment would be
    wrong at exactly the width where the walk spends most of its rounds.
    """
    if ref is None:
        source = QUANTIZED_H.read_text(errors="replace")
    else:
        rel = QUANTIZED_H.relative_to(ROOT)
        source = subprocess.run(
            ["git", "show", f"{ref}:{rel}"], cwd=ROOT, check=True,
            capture_output=True, text=True, errors="replace").stdout
    window = source[source.index("out_vec_size >= 4096"):]
    table = {}
    for case_body in re.finditer(r"case (\d+):(.*?)return;", window, re.S):
        template = re.search(
            r"qmv_fast_crossrow_affine4_g64_m<\s*T\s*,\s*(\d+)\s*,\s*(\d+)",
            case_body.group(2))
        if not template:
            # The narrow-output branch of the same switch calls the kernel
            # without an inputs-per-group argument; stop at that boundary.
            if table:
                break
            continue
        width, ipg = int(case_body.group(1)), int(template.group(2))
        if int(template.group(1)) != width:
            raise SystemExit(f"e56_repair_census: case {width} dispatches a "
                             f"kernel templated on {template.group(1)}")
        table[width] = ipg
    return table


def ipg_identity(ref: str | None = None) -> dict:
    """Digest the dispatch table so a leg records which world it ran in."""
    table = ipg_table(ref)
    if not table:
        return {"inputs_per_group": {}, "digest": None,
                "note": "could not parse the dispatch table"}
    widths = sorted(table)
    canonical = ",".join(f"{w}:{table[w]}" for w in widths)
    streams = {w: -(-w // table[w]) for w in widths}
    return {
        "source_ref": ref or "worktree",
        "inputs_per_group": {str(w): table[w] for w in widths},
        "weight_streams": {str(w): streams[w] for w in widths},
        "canonical": canonical,
        "digest": hashlib.sha256(canonical.encode()).hexdigest(),
    }


# The end-state dispatch table, if thorfinn's `t55` (`<T,5,5>`) and askeladd's
# `t6` (`<T,6,6>`) both land. Recorded here so the prediction below is fixed
# before either arm lands, not fitted after.
FUTURE_IPG = {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}

# askeladd's measured cross-row read bandwidth, GB/s, by group width.
BANDWIDTH = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946, 6: 117.8,
             7: 97.9}
WEIGHT_GB = 14.412


def stream_inverse_bandwidth(width: int, ipg: dict[int, int]) -> float:
    """`sum_g 1 / bw(NA_g)` for one verify width under one dispatch table.

    A width-M dispatch splits its rows into `ceil(M / p)` weight streams of
    `p` rows, with the remainder in the last stream. Each stream re-reads the
    backbone at the bandwidth its own row count sustains.
    """
    p = ipg[width]
    groups = [p] * (width // p) + ([width % p] if width % p else [])
    return sum(1.0 / BANDWIDTH[g] for g in groups)


def fit_row_cost_model(marginals: dict[int, float],
                       ipg: dict[int, int]) -> dict:
    """Fit `m(M-1 -> M) = c + s * d(sum_g 1/bw)` on clean marginals.

    The advisor's form is purely bandwidth-driven. Adding one row also costs a
    fixed amount that no re-read explains, so the marginal is fitted with an
    intercept. `c` is that fixed per-row cost and `s` absorbs `b * W`.
    """
    points = []
    for width, value in sorted(marginals.items()):
        if width in ipg and width - 1 in ipg:
            delta = (stream_inverse_bandwidth(width, ipg)
                     - stream_inverse_bandwidth(width - 1, ipg))
            points.append((width, delta, value / 1e6))
    if len(points) < 3:
        return {}
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    return {"fixed_row_cost_s": intercept, "bandwidth_slope": slope,
            "implied_b": slope / WEIGHT_GB,
            "points": [{"width": w, "delta_inv_bw": d, "measured_s": y}
                       for (w, d, y) in points]}


def score_cost_model(model: dict, marginals: dict[int, float],
                     ipg: dict[int, int], label: str) -> dict:
    """Apply a fitted model to one session's clean marginals and report error."""
    rows = []
    print(f"\n  {label}")
    print(f"  {'step':>8}{'measured ms':>14}{'predicted ms':>15}"
          f"{'residual ms':>14}{'error %':>10}")
    for width, value in sorted(marginals.items()):
        if width not in ipg or width - 1 not in ipg:
            continue
        delta = (stream_inverse_bandwidth(width, ipg)
                 - stream_inverse_bandwidth(width - 1, ipg))
        predicted = model["fixed_row_cost_s"] + model["bandwidth_slope"] * delta
        measured = value / 1e6
        residual = predicted - measured
        rows.append({"step": f"{width-1}->{width}", "measured_s": measured,
                     "predicted_s": predicted, "residual_s": residual,
                     "error_pct": 100.0 * residual / measured})
        print(f"  {width-1:>3} ->{width:>2}{measured*1000:>14.3f}"
              f"{predicted*1000:>15.3f}{residual*1000:>14.3f}"
              f"{100.0*residual/measured:>10.2f}")
    worst = max(abs(r["error_pct"]) for r in rows) if rows else None
    print(f"  worst absolute error: {worst:.2f} %")
    return {"rows": rows, "worst_abs_error_pct": worst}


def stop_width(ipg: dict[int, int], accept: float, ratio: float,
               head: float = 0.18, cap: int = 8) -> int:
    """Verify width the greedy walk selects under one dispatch table.

    This is `costModelDepth`'s rule with a stream-aware price: the same greedy
    marginal-value walk, with the mean price pinned to `head` so the table can
    only move the walk by moving cost between steps.

    This models the MARGINAL WALK ONLY. The shipped session also has the
    `fullAcceptStreak >= 2 ? 8 : 5` escape, which jumps past the walk's stop
    without pricing the steps it skips. That escape is why E56 R5 saw rounds at
    width 4 and width 9 but none between: the walk stops at 4 and the escape
    lands on the cap, so no schedule ever prices the widths in between.
    """
    # Widths below the table dispatch the narrow-output kernel, which reads the
    # backbone once. Defaulting them to zero would invent a boundary at the
    # table's first width and charge the walk a premium that no dispatch makes.
    streams = {w: -(-w // ipg[w]) for w in ipg}
    crosses = [streams.get(d + 2, 1) > streams.get(d + 1, 1)
               for d in range(cap)]
    count = sum(crosses)
    within = cap * head / (cap - count + count * ratio)
    marginal = [within * ratio if cross else within for cross in crosses]
    reach, expected, depth = 1.0, 0.0, 0
    cumulative = 1.0
    while depth < cap:
        reach *= accept
        if reach <= marginal[depth] * (1.0 + expected) / cumulative:
            break
        expected += reach
        cumulative += marginal[depth]
        depth += 1
    return depth + 1


def preregister_width_caps() -> dict:
    """Predicted verify-width cap under today's table and the end-state table.

    The advisor's falsifier: on the end-state table the only weight-stream
    boundary left is 6 -> 7, so a stream-aware walk should discover a cap at
    width 6 on its own. The shipped `widthCap = fullAcceptStreak >= 2 ? 8 : 5`
    rule cannot express a cap of 6, so if this prediction holds the shipped
    rule is structurally unable to reach the optimum.
    """
    today = ipg_table()
    ratio = 2.6596  # clean-round boundary ratio measured in this session
    block = {"crossing_ratio_used": ratio, "today": {}, "end_state": {},
             "future_inputs_per_group": {str(k): v
                                         for k, v in FUTURE_IPG.items()}}
    print("\nPREREGISTERED WIDTH CAP, recorded before `t55` or `t6` lands.")
    print("The stream-aware walk is run offline against both dispatch tables")
    print(f"at the clean-round boundary ratio {ratio:.4f}.")
    for label, table in (("today", today), ("end state", FUTURE_IPG)):
        streams = {w: -(-w // table[w]) for w in sorted(table)}
        boundaries = [f"{w-1}->{w}" for w in sorted(table)
                      if w - 1 in streams and streams[w] > streams[w - 1]]
        block[f"{label.replace(' ', '_')}_boundaries"] = boundaries
        print(f"  {label:<10} streams "
              + ",".join(f"{w}:{s}" for w, s in streams.items())
              + f"   boundaries {boundaries}")
    print(f"{'accept p':>10}{'cap today':>12}{'cap end state':>16}")
    for accept in (0.75, 0.80, 0.8351, 0.8750, 0.90, 0.9625, 0.98, 0.99):
        now = stop_width(today, accept, ratio)
        future = stop_width(FUTURE_IPG, accept, ratio)
        block["today"][f"{accept}"] = now
        block["end_state"][f"{accept}"] = future
        print(f"{accept:>10.4f}{now:>12}{future:>16}")
    return block


def leg_rounds(path: pathlib.Path) -> list[dict]:
    """Rounds of the longest drafting leg in one trace file."""
    if not path.exists():
        return []
    legs, current, last = [], [], -1
    for line in path.read_text(errors="replace").splitlines():
        match = TRACE_RE.search(line)
        if not match:
            continue
        record = {k: int(v) for k, v in match.groupdict().items()}
        if record["round"] <= last and current:
            legs.append(current)
            current = []
        last = record["round"]
        current.append(record)
    if current:
        legs.append(current)
    drafting = [leg for leg in legs if any(r["d"] > 0 for r in leg)]
    if not drafting:
        return []
    rounds = max(drafting, key=len)
    # `repaired` is a property of the PREVIOUS round, so round 1 cannot carry
    # it and is dropped rather than assumed clean.
    out = []
    for previous, this in zip(rounds, rounds[1:]):
        this = dict(this)
        this["repaired"] = previous["acc"] < previous["d"]
        this["width"] = this["d"] + 1
        out.append(this)
    return out


def summarize(rounds: list[dict], key: str = "round_us") -> dict:
    values = [r[key] for r in rounds]
    return {
        "n": len(values),
        "median_us": st.median(values),
        "mean_us": st.mean(values),
        "sd_us": st.pstdev(values) if len(values) > 1 else 0.0,
    }


def pool_session(session: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Every scored round of one session, tagged with its leg and arm."""
    pooled: list[dict] = []
    per_arm: dict[str, list[dict]] = defaultdict(list)
    for suffix, arm in LAYOUTS[session]:
        tag = f"{session}{suffix}"
        rounds = leg_rounds(ROOT / "research" / "out" / tag / "trace.txt")
        if not rounds:
            rounds = leg_rounds(ROOT / "research" / "out" / f"{tag}r" /
                                "trace.txt")
        for record in rounds:
            record["tag"] = tag
            record["arm"] = arm
        pooled.extend(rounds)
        per_arm[arm].extend(rounds)
    return pooled, per_arm


def clean_marginals(pooled: list[dict]) -> dict[int, float]:
    """Median clean round cost differences between adjacent verify widths."""
    median = {}
    for width in sorted({r["width"] for r in pooled}):
        clean = [r["round_us"] for r in pooled
                 if r["width"] == width and not r["repaired"]]
        if clean:
            median[width] = st.median(clean)
    return {w: median[w] - median[w - 1] for w in sorted(median)
            if w - 1 in median}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="s4", choices=sorted(LAYOUTS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pooled, per_arm = pool_session(args.session)

    base = SESSION_BASE[args.session]
    report = {"session": args.session, "session_base": base,
              "ipg_identity": ipg_identity(base),
              "ipg_identity_worktree": ipg_identity(),
              "rounds_pooled": len(pooled)}

    print(f"E56 repair census, session {args.session}: {len(pooled)} rounds "
          f"pooled over {len(per_arm)} arms")
    ident = report["ipg_identity"]
    print(f"\nIPG identity of the binary under the clock (mandatory), read at "
          f"the session base {base}:")
    print(f"  inputs per group  {ident['canonical']}")
    print(f"  weight streams    " +
          ",".join(f"{w}:{s}" for w, s in ident["weight_streams"].items()))
    print(f"  digest            {ident['digest']}")
    live = report["ipg_identity_worktree"]
    if live["digest"] != ident["digest"]:
        print(f"  NOTE this session did NOT run the worktree table "
              f"({live['canonical']}, digest {live['digest'][:16]}). Crossing "
              f"steps below are classified with the session's own table.")

    print("\nSTEP 0b. Round wall time by (width, repaired). `repaired` means")
    print("the PREVIOUS round rejected at least one draft, so this round pays.")
    print(f"{'M':>3}{'clean n':>9}{'clean med ms':>14}"
          f"{'repaired n':>12}{'repaired med ms':>17}{'repair cost ms':>16}")
    table = {}
    for width in sorted({r["width"] for r in pooled}):
        clean = [r for r in pooled if r["width"] == width and not r["repaired"]]
        dirty = [r for r in pooled if r["width"] == width and r["repaired"]]
        cell = {"clean": summarize(clean) if clean else None,
                "repaired": summarize(dirty) if dirty else None}
        cost = None
        if clean and dirty:
            cost = cell["repaired"]["median_us"] - cell["clean"]["median_us"]
        cell["repair_cost_us"] = cost
        table[width] = cell
        print(f"{width:>3}{len(clean):>9}"
              f"{(cell['clean']['median_us'] / 1000 if clean else float('nan')):>14.3f}"
              f"{len(dirty):>12}"
              f"{(cell['repaired']['median_us'] / 1000 if dirty else float('nan')):>17.3f}"
              f"{(cost / 1000 if cost is not None else float('nan')):>16.3f}")
    report["width_by_repaired"] = {str(k): v for k, v in table.items()}

    widths_with_both = [w for w, c in table.items()
                        if c["repair_cost_us"] is not None]
    if widths_with_both:
        costs = [table[w]["repair_cost_us"] for w in widths_with_both]
        report["repair_cost_us_median_over_widths"] = st.median(costs)
        print(f"\nSTEP 1. Repair cost in seconds, matched on width:")
        print(f"  widths with both cells: {widths_with_both}")
        print(f"  median over those widths: "
              f"{st.median(costs) / 1000:.3f} ms per repaired round")
        clean_all = [r["round_us"] for r in pooled if not r["repaired"]]
        if clean_all:
            print(f"  as a share of a clean round: "
                  f"{100.0 * st.median(costs) / st.median(clean_all):.2f} %")
            report["repair_cost_share_of_clean_round_pct"] = (
                100.0 * st.median(costs) / st.median(clean_all))
    else:
        print("\nSTEP 1. No width has both a clean and a repaired cell, which "
              "is itself the finding: width and repair are fully confounded "
              "in this session.")
        report["repair_cost_us_median_over_widths"] = None

    print("\nConfounding check. Share of rounds at each width that were "
          "repaired:")
    conf = {}
    for width in sorted(table):
        clean_n = table[width]["clean"]["n"] if table[width]["clean"] else 0
        dirty_n = (table[width]["repaired"]["n"]
                   if table[width]["repaired"] else 0)
        total = clean_n + dirty_n
        conf[width] = dirty_n / total if total else None
        print(f"  M={width}  repaired {dirty_n:>4} of {total:>4}"
              f"  = {100.0 * dirty_n / total if total else float('nan'):6.2f} %")
    report["repaired_share_by_width"] = {str(k): v for k, v in conf.items()}

    print("\nSTEP 0b (continued). Clean-only marginals and the boundary ratio.")
    clean_med = {w: table[w]["clean"]["median_us"]
                 for w in sorted(table) if table[w]["clean"]}
    marginals = {}
    for width in sorted(clean_med):
        if width - 1 in clean_med:
            marginals[width] = clean_med[width] - clean_med[width - 1]
    for width, value in marginals.items():
        print(f"  m({width-1} -> {width}) = {value / 1000:8.3f} ms"
              f"   (n={table[width]['clean']['n']} clean at M={width})")
    report["clean_marginals_us"] = {str(k): v for k, v in marginals.items()}

    streams = ident["weight_streams"]
    crossing, within = [], []
    for width, value in marginals.items():
        if str(width) in streams and str(width - 1) in streams:
            if streams[str(width)] > streams[str(width - 1)]:
                crossing.append(value)
            else:
                within.append(value)
    if crossing and within:
        ratio = st.mean(crossing) / st.mean(within)
        report["clean_boundary_ratio"] = ratio
        print(f"  crossing steps {[round(c/1000, 3) for c in crossing]} ms")
        print(f"  within steps   {[round(w/1000, 3) for w in within]} ms")
        print(f"  BOUNDARY RATIO FROM CLEAN ROUNDS ONLY: {ratio:.4f}")
    else:
        report["clean_boundary_ratio"] = None
        print("  Not enough clean widths on both sides of a live crossing to "
              "re-derive the ratio.")

    print("\nSTEP 0c. P(at least one draft rejected | drafts proposed d).")
    print(f"{'d':>3}{'rounds':>9}{'rejected':>10}{'P(reject)':>12}"
          f"{'marginal dP':>13}")
    reject = {}
    previous = None
    for d in sorted({r["d"] for r in pooled if r["d"] > 0}):
        at_d = [r for r in pooled if r["d"] == d]
        rejected = [r for r in at_d if r["acc"] < r["d"]]
        p = len(rejected) / len(at_d)
        delta = p - previous if previous is not None else None
        reject[d] = {"rounds": len(at_d), "rejected": len(rejected), "p": p,
                     "marginal_dp": delta}
        print(f"{d:>3}{len(at_d):>9}{len(rejected):>10}{p:>12.4f}"
              f"{(delta if delta is not None else float('nan')):>13.4f}")
        previous = p
    report["p_reject_by_depth"] = {str(k): v for k, v in reject.items()}

    print("\nPer-arm repaired share, to show the arms do not all sit in the")
    print("same repair regime (a schedule change moves this on purpose):")
    arm_block = {}
    for arm, rounds in per_arm.items():
        if not rounds:
            continue
        dirty = [r for r in rounds if r["repaired"]]
        clean = [r for r in rounds if not r["repaired"]]
        arm_block[arm] = {
            "rounds": len(rounds),
            "repaired_share": len(dirty) / len(rounds),
            "median_round_us_clean": st.median([r["round_us"] for r in clean])
            if clean else None,
            "median_round_us_repaired": st.median([r["round_us"] for r in dirty])
            if dirty else None,
        }
        print(f"  {arm:<9} rounds {len(rounds):>4}"
              f"  repaired {100.0 * len(dirty) / len(rounds):6.2f} %"
              f"  clean med {st.median([r['round_us'] for r in clean]) / 1000 if clean else float('nan'):8.3f} ms"
              f"  repaired med {st.median([r['round_us'] for r in dirty]) / 1000 if dirty else float('nan'):8.3f} ms")
    report["per_arm"] = arm_block

    print("\nSTEP 2 INPUT. Cost-model functional form, fitted on the clean")
    print("marginals of this session and applied to the other session, which")
    print("ran a different dispatch table on a different base.")
    own_ipg = {int(k): v for k, v in ident["inputs_per_group"].items()}
    model = fit_row_cost_model(marginals, own_ipg)
    if model:
        print(f"  fixed cost of one extra verify row  "
              f"{model['fixed_row_cost_s'] * 1000:.3f} ms")
        print(f"  bandwidth slope (absorbs b * W)      "
              f"{model['bandwidth_slope']:.4f}")
        print(f"  implied b at W = {WEIGHT_GB} GB           "
              f"{model['implied_b']:.4f}")
        fits = {args.session: score_cost_model(
            model, marginals, own_ipg, f"in sample, session {args.session}")}
        for other in sorted(LAYOUTS):
            if other == args.session:
                continue
            other_pooled, _ = pool_session(other)
            if not other_pooled:
                continue
            other_ipg = ipg_table(SESSION_BASE[other])
            fits[other] = score_cost_model(
                model, clean_marginals(other_pooled), other_ipg,
                f"OUT OF SAMPLE, session {other} on base "
                f"{SESSION_BASE[other]}")
        report["cost_model"] = {"fit": model, "scored": fits}

    report["width_cap_preregistration"] = preregister_width_caps()

    out = args.out or f"research/out/e56-{args.session}-repair-census.json"
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
