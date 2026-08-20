#!/usr/bin/env python3
"""E56 palindrome analysis.

  python3 research/e56_analyze.py [--session s4] [--out research/e56-abba.json]

A session runs every arm twice in palindrome order, so every arm gets one early
and one late slot and monotone thermal drift cancels to first order. The two
`base` legs are two byte-identical builds measured in one session. Their spread
-- not a nominal noise figure -- is what this instrument calls a difference
when there is none.

Session 3 ran on the pre-E55 base, where the dispatch table still crossed a
weight stream at width 9:

  base   no crossing priced; one scalar price per extra draft step
  s45    the 4 -> 5 crossing priced (the SDPA width wall)
  s89    the 8 -> 9 crossing priced (the segmented-verify wall)
  sfull  both crossings priced

Session 4 runs on the post-E55 base, where `case 9:` dispatches two streams and
the 8 -> 9 crossing no longer exists. It keeps `s89` unchanged to price the
erosion directly, and replaces `sfull` with the head-price correction arms:

  h224     flat price at the directly measured head cost ratio 0.224
  s45h224  the 4 -> 5 crossing priced at that corrected head cost

An arm that only redistributes cost between steps keeps the same mean price, so
it can move the walk only by moving cost, never by making drafting globally
cheaper or dearer. The `h224` arms deliberately break that: they change the
mean, which is the quantity the advisor says was mismeasured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Suffix -> arm, in run order, matching research/e56_session.sh.
LAYOUTS = {
    "s3": (("base1", "base"), ("s45a", "s45"), ("s89a", "s89"),
           ("sfulla", "sfull"), ("sfullb", "sfull"), ("s89b", "s89"),
           ("s45b", "s45"), ("base2", "base")),
    "s4": (("base1", "base"), ("s45a", "s45"), ("s89a", "s89"),
           ("h224a", "h224"), ("mixa", "s45h224"), ("mixb", "s45h224"),
           ("h224b", "h224"), ("s89b", "s89"), ("s45b", "s45"),
           ("base2", "base")),
}
NULL_FLOOR_PCT = 0.0629
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")

CORRECTNESS = ("all_tokens_matched", "residual_divergence_count",
               "public_drift_tripwire_passed", "mtp_depth", "decode_tokens",
               "head_provenance_sha256", "uses_pinned_mtp_head")


def session_order(session: str) -> list[tuple[str, str]]:
    """Tags in run order, matching research/e56_session.sh."""
    return [(f"{session}{suffix}", arm) for suffix, arm in LAYOUTS[session]]


def session_arms(session: str) -> list[str]:
    """Arms in first-appearance order, so `base` leads the report."""
    seen = []
    for _, arm in LAYOUTS[session]:
        if arm not in seen:
            seen.append(arm)
    return seen


def read_meta(path: pathlib.Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                out[key.strip()] = value.strip()
    return out


def rounds_from_trace(path: pathlib.Path) -> list[tuple[int, int]]:
    """(drafted, accepted) for the longest drafting leg in one trace."""
    if not path.exists():
        return []
    legs, current, last = [], [], -1
    for line in path.read_text(errors="replace").splitlines():
        match = ROUND_RE.search(line)
        if not match:
            continue
        index, depth, accepted = (int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)))
        if index <= last and current:
            legs.append(current)
            current = []
        last = index
        current.append((depth, accepted))
    if current:
        legs.append(current)
    drafting = [leg for leg in legs if any(d > 0 for d, _ in leg)]
    return max(drafting, key=len) if drafting else []


def width_histogram(rounds: list[tuple[int, int]]) -> dict:
    if not rounds:
        return {}
    widths = Counter(d + 1 for d, _ in rounds)
    # A round whose drafts were all accepted stopped because the walk refused
    # to buy another step, not because the target rejected a draft. At width 5
    # that is the 4 -> 5 crossing binding (R5).
    full = Counter(d + 1 for d, a in rounds if d > 0 and a == d)
    total = len(rounds)
    # `accepted_draft_rate` divides accepted drafts by PROPOSED drafts, which
    # counts every draft behind a rejection as a failure. The depth walk models
    # acceptance as a sequential Bernoulli chain, so it needs the conditional
    # per-draft rate: successes over trials that were actually reached, where a
    # round that lost a draft contributes exactly one observed failure and a
    # round that kept every draft contributes none.
    successes = sum(a for _, a in rounds)
    trials = sum(a + (1 if a < d else 0) for d, a in rounds)
    return {
        "rounds": total,
        "mean_verify_width": sum(d + 1 for d, _ in rounds) / total,
        "sequential_accept_mle": successes / trials if trials else None,
        "accept_trials": trials,
        "share": {w: round(widths.get(w, 0) / total, 5) for w in range(1, 10)},
        "count": {w: widths.get(w, 0) for w in range(1, 10)},
        "full_accept_count": {w: full.get(w, 0) for w in range(1, 10)},
        "width5_rounds": widths.get(5, 0),
        "width5_share": round(widths.get(5, 0) / total, 5),
        "width5_full_accept": full.get(5, 0),
    }


def load_legs(session: str) -> list[dict]:
    legs = []
    for tag, arm in session_order(session):
        out_dir = ROOT / "research" / "out" / tag
        score_path = out_dir / "score.json"
        if not score_path.exists():
            # The session retries a failed leg once under a suffixed tag.
            retry = ROOT / "research" / "out" / f"{tag}r"
            if (retry / "score.json").exists():
                out_dir, score_path, tag = retry, retry / "score.json", f"{tag}r"
            else:
                legs.append({"tag": tag, "arm": arm, "status": "missing"})
                continue
        score = json.loads(score_path.read_text())
        rounds = rounds_from_trace(out_dir / "trace.txt")
        legs.append({
            "tag": tag,
            "arm": arm,
            "status": "ok",
            "score": score.get("score"),
            "passed": score.get("passed"),
            "metrics": score.get("metrics", {}),
            "meta": read_meta(out_dir / "meta.txt"),
            "widths": width_histogram(rounds),
        })
    return legs


def arm_values(legs: list[dict], arm: str, key: str) -> list[float]:
    return [leg["metrics"][key] for leg in legs
            if leg["status"] == "ok" and leg["arm"] == arm
            and isinstance(leg["metrics"].get(key), (int, float))]


def contrast(legs: list[dict], key: str, arms: list[str]) -> dict | None:
    base = arm_values(legs, "base", key)
    if not base:
        return None
    mb = st.mean(base)
    row = {
        "base": base,
        "base_mean": mb,
        "base_null_spread_pct": (100.0 * (max(base) - min(base)) / mb
                                 if len(base) > 1 and mb else None),
        "arms": {},
    }
    for arm in arms:
        if arm == "base":
            continue
        values = arm_values(legs, arm, key)
        if not values:
            continue
        mean = st.mean(values)
        row["arms"][arm] = {
            "values": values,
            "mean": mean,
            "delta_pct": 100.0 * (mean / mb - 1.0) if mb else None,
            "spread_pct": (100.0 * (max(values) - min(values)) / mean
                           if len(values) > 1 and mean else None),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="s4", choices=sorted(LAYOUTS))
    parser.add_argument("--out", default="research/e56-abba.json")
    args = parser.parse_args()

    legs = load_legs(args.session)
    arms = session_arms(args.session)
    report = {"session": args.session, "arms": arms, "legs": legs,
              "null_floor_pct": NULL_FLOOR_PCT, "contrasts": {}}

    print(f"{'tag':<10}{'arm':<8}{'serial s/tok':>14}{'mtp s/tok':>14}"
          f"{'speedup':>10}{'draft':>8}{'acc':>8}{'tokens':>8}")
    for leg in legs:
        if leg["status"] != "ok":
            print(f"{leg['tag']:<10}{leg['arm']:<8}  MISSING")
            continue
        m = leg["metrics"]
        print(f"{leg['tag']:<10}{leg['arm']:<8}"
              f"{m.get('serial_seconds_per_token', float('nan')):>14.8f}"
              f"{m.get('mtp_seconds_per_token', float('nan')):>14.8f}"
              f"{m.get('mtp_decode_speedup', float('nan')):>10.5f}"
              f"{m.get('effective_mean_draft_len', float('nan')):>8.3f}"
              f"{m.get('accepted_draft_rate', float('nan')):>8.3f}"
              f"{str(m.get('decode_tokens')):>8}")

    print()
    print("Arm contrasts against the base pair. The base null spread is the")
    print(f"bar; the pre-registered floor is {NULL_FLOOR_PCT:.4f} %.")
    for key, label in (("mtp_seconds_per_token", "candidate s/token (PRIMARY)"),
                       ("mtp_decode_speedup", "local ratio"),
                       ("serial_seconds_per_token", "serial leg (falsifier)"),
                       ("effective_mean_draft_len", "mean draft length"),
                       ("accepted_draft_rate", "accepted draft rate")):
        row = contrast(legs, key, arms)
        if row is None:
            continue
        report["contrasts"][key] = row
        null = row["base_null_spread_pct"]
        print(f"\n  {label}")
        print(f"    base  {row['base_mean']:.8f}"
              + (f"   null arm spread {null:.4f} %" if null is not None else ""))
        for arm, cell in row["arms"].items():
            print(f"    {arm:<8} {cell['mean']:.8f}   {cell['delta_pct']:+.4f} %"
                  + (f"   own spread {cell['spread_pct']:.4f} %"
                     if cell["spread_pct"] is not None else ""))

    print()
    print("Correctness and provenance (must be identical across all arms):")
    for key in CORRECTNESS:
        values = {leg["tag"]: leg["metrics"].get(key)
                  for leg in legs if leg["status"] == "ok"}
        unique = set(map(str, values.values()))
        verdict = "IDENTICAL" if len(unique) == 1 else "DIFFERS"
        shown = next(iter(unique)) if len(unique) == 1 else values
        print(f"  {key:<32}{verdict}  {shown}")
    report["correctness"] = {
        key: {leg["tag"]: leg["metrics"].get(key)
              for leg in legs if leg["status"] == "ok"}
        for key in CORRECTNESS}

    print()
    print("Verify-width share by leg (the mechanism readout):")
    print(f"{'tag':<10}{'arm':<8}{'rounds':>7}{'mean W':>8}{'accept p':>10}" +
          "".join(f"{w:>8}" for w in range(1, 10)))
    for leg in legs:
        widths = leg.get("widths") or {}
        if not widths:
            continue
        print(f"{leg['tag']:<10}{leg['arm']:<8}{widths['rounds']:>7}"
              f"{widths['mean_verify_width']:>8.3f}"
              f"{widths['sequential_accept_mle']:>10.4f}"
              + "".join(f"{widths['share'][w]:>8.3f}" for w in range(1, 10)))
    print("`accept p` is the sequential per-draft acceptance the depth walk")
    print("models, not accepted/proposed. It is the p the walk should be fed.")

    print()
    print("Rounds that terminate at exactly verify width 5 (R5). `full accept`")
    print("counts the width-5 rounds whose four drafts were all accepted, so")
    print("the walk stopped there by price, not by target rejection:")
    print(f"{'tag':<10}{'arm':<8}{'rounds':>8}{'W=5':>8}{'share':>9}"
          f"{'full accept':>13}")
    width5 = {}
    for leg in legs:
        widths = leg.get("widths") or {}
        if not widths:
            continue
        width5.setdefault(leg["arm"], []).append(widths["width5_rounds"])
        print(f"{leg['tag']:<10}{leg['arm']:<8}{widths['rounds']:>8}"
              f"{widths['width5_rounds']:>8}{widths['width5_share']:>9.4f}"
              f"{widths['width5_full_accept']:>13}")
    report["width5_rounds_by_arm"] = width5

    print()
    print("Thermal record, taken from each leg's own benchmark.sh output:")
    for leg in legs:
        meta = leg.get("meta") or {}
        print(f"  {leg['tag']:<10} entry={meta.get('entry_gpu_temp_c')}"
              f" exit={meta.get('exit_gpu_temp_c')}"
              f" gate_passes={meta.get('cool_gate_passes')}"
              f" gate_skips={meta.get('cool_gate_skips')}"
              f" cool_gate_passed_real_gate={meta.get('cool_gate_passed_real_gate')}")

    print()
    print("Arm provenance. The checkout blob is HEAD on every leg; the arm is")
    print("selected by which prebuilt worker binary the leg ran.")
    for leg in legs:
        meta = leg.get("meta") or {}
        print(f"  {leg['tag']:<10} arm={meta.get('e56_arm')}"
              f" arm_schedule_blob={str(meta.get('arm_schedule_blob'))[:12]}"
              f" worker_sha256={str(meta.get('worker_sha256'))[:12]}"
              f" metallib_sha256={str(meta.get('metallib_sha256'))[:12]}"
              f" checkout_schedule_blob={str(meta.get('checkout_schedule_blob'))[:12]}")

    out_path = ROOT / args.out
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
