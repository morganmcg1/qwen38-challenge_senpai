#!/usr/bin/env python3
"""E165: witness, exactness gate and report for the head-chain prefetch.

usage:
  research/e165_prefetch.py witness TRACE --want on|off
  research/e165_prefetch.py rows OFF_TRACE ON_TRACE [--positive-control]
  research/e165_prefetch.py report --label LABEL

WITNESS. The arm is not taken on trust. Every traced round writes `pf=`, the
arm the process compiled in, `pf_hit=`, whether this round consumed a stashed
head step, and the cumulative session counters `pf_made=`, `pf_hits=` and
`pf_undo=`. The `on` arm must satisfy `made == hits + undo + pending`, where
at most one step may still be stashed at the end; the `off` arm must leave
every counter at zero. Each check is run against the arm it expects AND
against the other arm, and the second run has to fail, so a witness that
cannot fail is caught before it certifies anything.

EXACTNESS. `mtp-row` dumps every declared row's top-2 ids and its logit values
as `%a` hexfloats, so the two arms are compared BIT FOR BIT by token position
rather than by argmax. `--positive-control` perturbs one value by one unit in
the last place and requires the same comparison to fail.

REPORT. Reads each timed leg's `score.json` and `meta.txt`, refuses to price
the contrast when the accept ledger moves between arms (RULE 179), and prints
the arm means, the paired palindrome contrast, the pooled 2 sigma and the
predeclared stop-rule verdict. harness=local throughout: no number here is an
official or ranked score.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=(-?[\d.]+)")
ROW_RE = re.compile(r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=(\S+)$")

MINIMUM_USEFUL_PCT = 0.35


def rounds(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        fields = {k: float(v) for k, v in KV_RE.findall(m.group(4))}
        fields["round"] = int(m.group(1))
        fields["d"] = int(m.group(2))
        fields["acc"] = int(m.group(3))
        out.append(fields)
    return out


def witness(args) -> int:
    """Prove which arm ran, from the trace rather than from the launcher.

    `pf` and `pf_hit` are per-round flags. `pf_made`, `pf_hits` and `pf_undo`
    are cumulative session counters, so the last traced round carries the
    session totals. Skip rounds return before the trace emit, so they can
    raise `pf_undo` without adding a record here; the accounting identity
    below is written to survive that.
    """
    recs = rounds(Path(args.trace))
    if not recs:
        print("witness: no traced rounds")
        return 1
    required = ("pf", "pf_hit", "pf_made", "pf_hits", "pf_undo")
    missing = [r for r in recs if any(k not in r for k in required)]
    if missing:
        print(f"witness: {len(missing)} rounds lack a pf field; this build "
              "predates the instrument")
        return 1

    for name in ("pf_made", "pf_hits", "pf_undo"):
        series = [r[name] for r in recs]
        if any(b < a for a, b in zip(series, series[1:])):
            print(f"witness: {name} is not monotone; it is not a counter")
            return 1

    last = recs[-1]
    made, hits, undo = int(last["pf_made"]), int(last["pf_hits"]), int(last["pf_undo"])
    arm_on = sum(1 for r in recs if r["pf"] == 1)
    hit_rounds = sum(1 for r in recs if r["pf_hit"] == 1)
    hit_rate = hit_rounds / len(recs)
    print(f"rounds={len(recs)} pf_on={arm_on} pf_made={made} pf_hits={hits} "
          f"pf_undo={undo} hit_rounds={hit_rounds} hit_rate={hit_rate:.4f}")

    checks: list[tuple[str, bool]] = []
    if args.want == "on":
        # Every step made is consumed, undone, or still stashed at the end.
        checks = [
            ("arm compiled on in every round", arm_on == len(recs)),
            ("counter agrees with per-round flag", hits == hit_rounds),
            ("made == hits + undo + pending<=1", 0 <= made - hits - undo <= 1),
            ("steps were actually made", made >= 1),
            (f"hit_rate >= {args.min_hit_rate}", hit_rate >= args.min_hit_rate),
        ]
    else:
        checks = [
            ("arm compiled off in every round", arm_on == 0),
            ("no step made", made == 0),
            ("no step consumed", hits == 0 and hit_rounds == 0),
            ("no step undone", undo == 0),
        ]
    for label, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    verdict = all(ok for _, ok in checks)
    print(f"want={args.want} verdict={'ok' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def read_rows(path: Path) -> list[tuple[int, int, int, tuple[float, ...]]]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        values = tuple(float.fromhex(v) for v in m.group(4).split(","))
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), values))
    return out


def next_after(value: float) -> float:
    """One unit in the last place above `value`, for the positive control."""
    bits = struct.unpack("<q", struct.pack("<d", value))[0]
    return struct.unpack("<d", struct.pack("<q", bits + 1))[0]


def compare_rows(args) -> int:
    left = read_rows(Path(args.left))
    right = read_rows(Path(args.right))
    if not left or not right:
        print("rows: one of the traces carries no row dump")
        return 1
    if args.positive_control:
        pos, a, b, values = right[len(right) // 2]
        right = list(right)
        right[len(right) // 2] = (pos, a, b,
                                  (next_after(values[0]),) + values[1:])
        print(f"positive control: perturbed pos={pos} by one ulp")
    lmap = {r[0]: r for r in left}
    rmap = {r[0]: r for r in right}
    shared = sorted(set(lmap) & set(rmap))
    if not shared:
        print("rows: the two traces share no token position")
        return 1
    bad = [p for p in shared if lmap[p][1:] != rmap[p][1:]]
    print(f"positions_left={len(lmap)} positions_right={len(rmap)} "
          f"compared={len(shared)} mismatched={len(bad)}")
    if bad:
        p = bad[0]
        print(f"  first mismatch pos={p} left={lmap[p][1:]} right={rmap[p][1:]}")
        return 1
    return 0


def meta(path: Path) -> dict:
    return dict(
        line.strip().split("=", 1) for line in path.open() if "=" in line)


def ledger(edl: float, acc: float, tokens: int) -> tuple[int, int, int]:
    """Rounds, proposed drafts and accepted drafts, from the exact rationals."""
    for count in range(1, tokens + 1):
        proposed = edl * count
        if abs(proposed - round(proposed)) > 1e-6:
            continue
        proposed = round(proposed)
        accepted = acc * proposed
        if abs(accepted - round(accepted)) > 1e-6:
            continue
        return count, proposed, round(accepted)
    raise SystemExit(f"no small-denominator ledger fits edl={edl} acc={acc}")


def load_legs(label: str) -> list[dict]:
    legs = []
    for score_path in sorted(glob.glob(f"research/out/e165{label}k*/score.json")):
        m = json.load(open(score_path))["metrics"]
        d = meta(Path(score_path).with_name("meta.txt"))
        count, proposed, accepted = ledger(
            m["effective_mean_draft_len"], m["accepted_draft_rate"],
            m["decode_tokens"])
        legs.append({
            "tag": d["tag"], "arm": d["e165_arm"], "rep": int(d["e165_rep"]),
            "position": int(d["e165_position"]),
            "mtp": m["mtp_seconds_per_token"],
            "serial": m["serial_seconds_per_token"],
            "edl": m["effective_mean_draft_len"],
            "acc": m["accepted_draft_rate"],
            "matched": m["all_tokens_matched"],
            "divergences": m["residual_divergence_count"],
            "tokens": m["decode_tokens"],
            "rounds": count, "proposed": proposed, "accepted": accepted,
            "entry_c": float(d["gpu_temp_entry_c"]),
            "exit_c": float(d["gpu_temp_exit_c"]),
            "real_gate": d["cool_gate_passed_real_gate"],
            "qualified": d["gate_qualified_for_timing"],
            "worker_sha256": d["worker_sha256"],
            "post_run_worker_sha256": d["post_run_worker_sha256"],
            "commit": d.get("e165_session_commit", d["base_sha"]),
        })
    if not legs:
        raise SystemExit(f"no timed legs under research/out/e165{label}k*")
    return legs


def report(args) -> int:
    legs = load_legs(args.label)
    arms = sorted({leg["arm"] for leg in legs})
    print(f"legs={len(legs)} arms={arms}")
    print(f"{'tag':28s} {'arm':4s} {'pos':>3s} {'s/tok':>10s} {'edl':>8s} "
          f"{'acc':>8s} {'in C':>6s} {'out C':>6s} {'gate':>5s} {'match':>5s}")
    for leg in legs:
        print(f"{leg['tag']:28s} {leg['arm']:4s} {leg['position']:3d} "
              f"{leg['mtp']:10.6f} {leg['edl']:8.4f} {leg['acc']:8.4f} "
              f"{leg['entry_c']:6.1f} {leg['exit_c']:6.1f} "
              f"{str(leg['real_gate']):>5s} {str(leg['matched']):>5s}")

    bad = [leg for leg in legs if not leg["matched"] or leg["divergences"]]
    if bad:
        print(f"FAIL exactness: {len(bad)} leg(s) did not match every token")
        return 2

    identity = {(leg["rounds"], leg["proposed"], leg["accepted"])
                for leg in legs}
    print(f"accept ledger across every leg: {sorted(identity)}")
    if len(identity) != 1:
        print("FAIL RULE 179: the accept ledger moved between legs; the "
              "contrast is not a pure cost measurement")
        return 3

    means = {a: st.fmean(leg["mtp"] for leg in legs if leg["arm"] == a)
             for a in arms}
    for arm in arms:
        values = [leg["mtp"] for leg in legs if leg["arm"] == arm]
        print(f"arm {arm:3s} n={len(values)} mean={means[arm]:.6f} "
              f"min={min(values):.6f} max={max(values):.6f}")

    if set(arms) != {"off", "on"}:
        print("report: expected exactly the off and on arms")
        return 4

    effect = (means["on"] - means["off"]) / means["off"] * 100.0
    residuals, dof = [], 0
    for arm in arms:
        values = [leg["mtp"] for leg in legs if leg["arm"] == arm]
        mu = st.fmean(values)
        residuals += [v - mu for v in values]
        dof += len(values) - 1
    var = sum(r * r for r in residuals) / dof
    n = len(legs) / 2
    two_se = 2 * (2 * var / n) ** 0.5 / means["off"] * 100.0

    rounds_per_leg = legs[0]["rounds"]
    leg_seconds = means["off"] * legs[0]["tokens"]
    delta_us_round = (
        (means["on"] - means["off"]) * legs[0]["tokens"] / rounds_per_leg * 1e6)

    print(f"\ncontrast on - off      {effect:+.4f} %  (2 sigma {two_se:.4f} %)")
    print(f"per decode round       {delta_us_round:+.1f} us "
          f"over {rounds_per_leg} rounds of a {leg_seconds:.4f} s leg")
    print(f"minimum useful effect  -{MINIMUM_USEFUL_PCT:.2f} %")

    if effect + two_se < 0 and effect <= -MINIMUM_USEFUL_PCT:
        verdict = "PROMOTE: the effect clears the minimum and 2 sigma clears 0"
    elif effect + two_se < 0:
        verdict = ("REAL BUT SMALL: 2 sigma clears 0 and the effect is under "
                   "the predeclared minimum")
    else:
        verdict = "STOP: the 2 sigma interval contains zero"
    print(f"verdict                {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("witness")
    w.add_argument("trace")
    w.add_argument("--want", choices=["on", "off"], required=True)
    w.add_argument("--min-hit-rate", type=float, default=0.75)

    r = sub.add_parser("rows")
    r.add_argument("left")
    r.add_argument("right")
    r.add_argument("--positive-control", action="store_true")

    p = sub.add_parser("report")
    p.add_argument("--label", default="pf")

    args = ap.parse_args()
    if args.cmd == "witness":
        return witness(args)
    if args.cmd == "rows":
        return compare_rows(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
