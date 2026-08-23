#!/usr/bin/env python3
"""Price the chunk-sum fill dispatch, and convert it to the Idea 3 ceiling.

F41 asks how much of a decode round the `xsums_v1` fill dispatch costs, so the
xsums fill fusion can be priced before a slot is committed to it.

`Qwen35CustomQMV.Arm` already contains the instrument. `fill_noconsume` runs
the fill dispatch and binds its output, but compiles the wide kernel with
`USE_TABLE=false` so the kernel recomputes the sums itself and never reads the
table. `replica` is the same kernel with no fill at all. The two arms emit
identical tokens and identical schedules, so the contrast is a pure cost
measurement and Rule 79 does not bind on it.

  check   read one leg's pipeline log and prove which arm ran
  report  the ABBA contrast over the timed legs, in per cent, us/round and
          us/fill, then the Idea 3 coverage ceiling

Coverage. `qwen35FusedResidualRMSNorm` produces the activation for three of the
seven wide-QMV shapes: `gdn.in_proj` on 48 layers, `fa.qkv` on 16 and
`mlp.gate_up` on 64. That is 128 of the 257 wide QMV calls in one target
forward pass. A fusion that emits the table from that kernel can therefore
remove at most 128 fills, never all 257.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

QMV_PER_FORWARD = 257
FUSABLE_PER_FORWARD = 128
ARM_FIELD = {"repl": "replica", "fill": "fill_noconsume"}


def check(args: argparse.Namespace) -> int:
    log = json.load(open(args.log))
    want = ARM_FIELD[args.want]
    bad = []
    if log.get("arm") != want:
        bad.append(f"arm {log.get('arm')!r} wanted {want!r}")

    fills = log.get("by_key", {}).get("xsums_v1", 0)
    if args.want == "fill" and fills == 0:
        bad.append("no xsums_v1 dispatch, so the fill never ran")
    if args.want == "repl" and fills != 0:
        bad.append(f"{fills} xsums_v1 dispatches on an arm that fills nothing")

    consuming = [k for k in log.get("by_key", {}) if "USE_TABLE=true" in k]
    if consuming:
        bad.append(f"a table-consuming entry point ran: {consuming}")

    print(f"arm            {log.get('arm')!r}")
    print(f"xsums_v1 fills {fills}")
    print(f"by_key         {json.dumps(log.get('by_key', {}))}")
    for line in bad:
        print(f"  FAIL {line}")
    print("OK" if not bad else "FAILED")
    return 1 if bad else 0


def leg(path: pathlib.Path) -> dict:
    score = json.load(open(path / "score.json"))["metrics"]
    meta = dict(
        line.split("=", 1)
        for line in (path / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    return {
        "tag": path.name,
        "arm": meta.get("e135_arm", "?"),
        "rep": int(meta.get("e135_rep", 1)),
        "position": int(meta.get("e135_position", 0)),
        "mtp": score["mtp_seconds_per_token"],
        "serial": score["serial_seconds_per_token"],
        "edl": score["effective_mean_draft_len"],
        "tokens": score["decode_tokens"],
        "entry_c": meta.get("gpu_temp_entry_c", "?"),
        "exit_c": meta.get("gpu_temp_exit_c", "?"),
        "gated": meta.get("cool_gate_passed_real_gate", "?"),
    }


def load_legs(label: str) -> list[dict]:
    return sorted(
        (leg(p) for p in pathlib.Path("research/out").glob(f"e135{label}k*")
         if (p / "score.json").exists()),
        key=lambda d: (d["rep"], d["position"]),
    )


def fit(legs: list[dict]) -> dict:
    """Least squares for mean + arm + within-session drift + session offset.

    Every session is its own palindrome, so the arm indicator is orthogonal to
    the drift regressor and to the session offsets. Each coefficient is
    therefore read off directly. Drift is centred inside its own session: two
    sessions separated by hours do not lie on one line, and a session offset
    absorbs the difference in base clock or base temperature between them.

    The reduced fit drops only the drift term. It keeps the session offsets,
    which are real, so with one session it is exactly the two-parameter fit.
    """
    y = [d["mtp"] for d in legs]
    a = [1.0 if d["arm"] == "fill" else -1.0 for d in legs]
    n = len(legs)

    reps = sorted({d["rep"] for d in legs})
    centre = {r: statistics.fmean(d["position"] for d in legs if d["rep"] == r)
              for r in reps}
    x = [d["position"] - centre[d["rep"]] for d in legs]
    offsets = [[1.0 if d["rep"] == r else 0.0 for d in legs] for r in reps[1:]]
    offsets = [[v - statistics.fmean(col) for v in col] for col in offsets]

    def project(c: list[float]) -> float:
        return sum(ci * yi for ci, yi in zip(c, y)) / sum(ci * ci for ci in c)

    grand = statistics.fmean(y)
    arm_c = project(a)
    drift_c = project(x) if any(x) else 0.0
    off_c = [project(col) for col in offsets]

    def residuals(with_drift: bool) -> list[float]:
        out = []
        for i, yi in enumerate(y):
            pred = grand + arm_c * a[i] + sum(
                c * col[i] for c, col in zip(off_c, offsets))
            if with_drift:
                pred += drift_c * x[i]
            out.append(yi - pred)
        return out

    kf = 2 + len(offsets) + (1 if any(x) else 0)
    varf = sum(r * r for r in residuals(True)) / (n - kf)
    sef = 2.0 * (varf / sum(ai * ai for ai in a)) ** 0.5

    kr = 2 + len(offsets)
    varr = sum(r * r for r in residuals(False)) / (n - kr)
    ser = 2.0 * (varr / sum(ai * ai for ai in a)) ** 0.5

    base = grand - arm_c
    return {
        "effect": 2.0 * arm_c,
        "effect_pct": 2.0 * arm_c / base * 100.0,
        "se3_pct": sef / base * 100.0,
        "se2_pct": ser / base * 100.0,
        "dof3": n - kf,
        "dof2": n - kr,
        "drift_pct_per_leg": drift_c / base * 100.0,
        "session_offsets_pct": [c / base * 100.0 for c in off_c],
        "resid_sd3_pct": varf ** 0.5 / base * 100.0,
        "resid_sd2_pct": varr ** 0.5 / base * 100.0,
        "replica": base,
        "fill": grand + arm_c,
    }


def report(args: argparse.Namespace) -> int:
    legs = load_legs(args.label)
    if not legs:
        print("no timed legs found")
        return 1

    print("## Legs")
    for d in legs:
        print(f"  k{d['rep']}p{d['position']} {d['arm']:<5} mtp {d['mtp']:.6f}  "
              f"serial {d['serial']:.6f}  edl {d['edl']:.6f}  "
              f"entry {d['entry_c']} exit {d['exit_c']} gated {d['gated']}")

    edls = {round(d["edl"], 9) for d in legs}
    print(f"\nschedule identity: {len(edls)} distinct effective_mean_draft_len "
          f"{'OK, the arms share one schedule' if len(edls) == 1 else 'DIFFERENT, the contrast is not a pure cost'}")

    by_arm = {}
    for d in legs:
        by_arm.setdefault(d["arm"], []).append(d)
    if set(by_arm) != {"repl", "fill"}:
        print(f"expected arms repl and fill, found {sorted(by_arm)}")
        return 1

    def mean(arm: str, key: str) -> float:
        return statistics.fmean(d[key] for d in by_arm[arm])

    repl, fill = mean("repl", "mtp"), mean("fill", "mtp")
    delta_pct = (fill - repl) / repl * 100.0
    stats = fit(legs)
    print(f"\n## Fill cost, candidate MTP leg")
    print(f"  replica        {repl:.6f} s/tok  n={len(by_arm['repl'])}")
    print(f"  fill_noconsume {fill:.6f} s/tok  n={len(by_arm['fill'])}")
    print(f"  fill costs     {delta_pct:+.4f} % of candidate MTP time")
    print(f"  se, drift fitted, {stats['dof3']} dof   {stats['se3_pct']:.4f} pp"
          f"  t {abs(delta_pct) / stats['se3_pct']:.2f}")
    print(f"  se, no drift,     {stats['dof2']} dof   {stats['se2_pct']:.4f} pp"
          f"  t {abs(delta_pct) / stats['se2_pct']:.2f}")
    print(f"  position drift            {stats['drift_pct_per_leg']:+.4f} %/leg")
    if stats["session_offsets_pct"]:
        off = " ".join(f"{v:+.4f}" for v in stats["session_offsets_pct"])
        print(f"  session offsets           {off} %")
    for r in sorted({d['rep'] for d in legs}):
        rl = [d for d in legs if d["rep"] == r]
        rr = statistics.fmean(d["mtp"] for d in rl if d["arm"] == "repl")
        rf = statistics.fmean(d["mtp"] for d in rl if d["arm"] == "fill")
        print(f"  session k{r} alone        {(rf - rr) / rr * 100.0:+.4f} %")

    srepl, sfill = mean("repl", "serial"), mean("fill", "serial")
    print(f"  serial null    {(sfill - srepl) / srepl * 100.0:+.4f} %  "
          f"(the serial leg routes no wide QMV at m>=4, so this is drift)")

    tokens = legs[0]["tokens"]
    edl = legs[0]["edl"]
    rounds = args.rounds
    per_round_us = (fill - repl) * tokens / rounds * 1e6
    per_fill_us = per_round_us / args.fills_per_round
    print(f"\n## Per round, {rounds} rounds over {tokens} tokens, edl {edl:.6f}")
    print(f"  whole fill set {per_round_us:+.1f} us/round over "
          f"{args.fills_per_round} fills")
    print(f"  one fill       {per_fill_us:+.3f} us")

    shipped = per_fill_us * args.shipped_fills_per_round
    ceiling = per_fill_us * FUSABLE_PER_FORWARD
    round_us = repl * tokens / rounds * 1e6
    print(f"\n## Idea 3 ceiling, harness=local")
    print(f"  shipped arm pays {shipped:.1f} us/round for "
          f"{args.shipped_fills_per_round} fills")
    print(f"  fusable          {FUSABLE_PER_FORWARD} of {QMV_PER_FORWARD} "
          f"shapes = {FUSABLE_PER_FORWARD / QMV_PER_FORWARD * 100:.1f} %")
    print(f"  ceiling          {ceiling:.1f} us/round")
    print(f"  local round      {round_us:.1f} us  -> ceiling is "
          f"{ceiling / round_us * 100:.4f} % of the candidate MTP leg")
    print("\n  This is a local M4 Pro cost. It transfers to the ranked host "
          "only as\n  a dispatch count, not as microseconds. Rule 83.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("log")
    c.add_argument("--want", choices=sorted(ARM_FIELD), required=True)
    c.set_defaults(fn=check)

    r = sub.add_parser("report")
    r.add_argument("--label", default="fill")
    r.add_argument("--rounds", type=int, default=78)
    r.add_argument("--fills-per-round", type=int, default=QMV_PER_FORWARD)
    r.add_argument("--shipped-fills-per-round", type=int, default=QMV_PER_FORWARD)
    r.set_defaults(fn=report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
