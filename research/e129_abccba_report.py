#!/usr/bin/env python3
"""Report one E129 ABCCBA session.

The headline is absolute `mtp_seconds_per_token` per arm, paired inside each
replicate. The local serial-to-MTP ratio is reported next to it so a reader can
watch the cancellation the ratio suffers on this change: the arm is in the QMV
verify kernel, which both local legs run.

The session is ungated by construction. Entry and exit temperature per leg and
the entry spread per arm are printed, and the gate labels are reproduced
verbatim, because an ungated reading is directional evidence inside its own
counterbalanced session and nothing more.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e129_entry_point_census as census  # noqa: E402
import e129_local_to_ranked as transfer  # noqa: E402

ARMS = ("shipped", "onepass67", "onepass678")
OUT = pathlib.Path("research/out")

# The realised local width histogram, and the exact mean draft length it
# implies. A leg whose `effective_mean_draft_len` equals this to the last bit
# ran the same schedule, so the arms differ in kernel only. The equality is
# exact, not statistical: the metric is a ratio of two integers the parent
# counted.
LOCAL_HISTOGRAM = census.LOCAL_HISTOGRAM
LOCAL_ROUNDS = sum(LOCAL_HISTOGRAM.values())
LOCAL_MEAN_WIDTH = sum(w * n for w, n in LOCAL_HISTOGRAM.items()) / LOCAL_ROUNDS
LOCAL_MEAN_DRAFT_LEN = LOCAL_MEAN_WIDTH - 1.0


def read_meta(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def legs(label: str) -> list[dict]:
    found = []
    for d in sorted(OUT.glob(f"e129{label}k*")):
        meta = read_meta(d / "meta.txt")
        arm = meta.get("e129_arm")
        if arm not in ARMS:
            continue
        score = d / "score.json"
        metrics = {}
        if score.exists():
            try:
                metrics = json.loads(score.read_text()).get("metrics", {})
            except json.JSONDecodeError:
                metrics = {}
        found.append({"tag": d.name, "arm": arm, "meta": meta,
                      "metrics": metrics,
                      "rep": int(meta.get("e129_replicate", 0)),
                      "pos": int(meta.get("e129_position", 0))})
    return found


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s1")
    args = ap.parse_args()

    rows = legs(args.label)
    if not rows:
        print(f"e129_abccba_report: no legs for label {args.label!r}")
        return 1

    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]
    print(f"E129 ABCCBA session {args.label}: {len(rows)} legs, "
          f"{len(complete)} with a score")
    gate = {r["meta"].get("gate_qualified_for_timing") for r in rows}
    real = {r["meta"].get("cool_gate_passed_real_gate") for r in rows}
    print(f"gate_qualified_for_timing={sorted(x for x in gate if x)} "
          f"cool_gate_passed_real_gate={sorted(x for x in real if x)}")
    workers = {r["meta"].get("worker_sha256") for r in rows}
    print(f"worker_sha256 across the session: "
          f"{'one build' if len(workers) == 1 else sorted(workers)} "
          f"{list(workers)[0][:16] if len(workers) == 1 else ''}")
    print()

    print("realised local width histogram, `harness=local`")
    print("  " + "  ".join(f"M={w}:{n}" for w, n in
                           sorted(LOCAL_HISTOGRAM.items()))
          + f"   rounds {LOCAL_ROUNDS}")
    print(f"  mean width {LOCAL_MEAN_WIDTH:.9f}, "
          f"implied effective_mean_draft_len {LOCAL_MEAN_DRAFT_LEN:.15f}")
    seen = {fnum(r["metrics"].get("effective_mean_draft_len"))
            for r in complete}
    seen.discard(None)
    identical = len(seen) == 1
    matches = seen == {LOCAL_MEAN_DRAFT_LEN}
    print(f"  schedule identical across every scored leg: "
          f"{'yes' if identical else 'NO, THE ARMS ARE CONFOUNDED: ' + str(sorted(seen))}")
    print(f"  and equal to the recorded histogram: "
          f"{'yes' if matches else 'no, the histogram is stale: ' + str(sorted(seen))}")
    print()

    print("per leg")
    print(f"{'tag':34s} {'arm':11s} {'rep':>3s} {'pos':>3s} "
          f"{'mtp s/tok':>10s} {'serial':>9s} {'ratio':>7s} "
          f"{'in C':>6s} {'out C':>6s} {'draft':>6s}")
    for r in sorted(rows, key=lambda r: (r["rep"], r["pos"])):
        m = r["metrics"]
        mtp = fnum(m.get("mtp_seconds_per_token"))
        ser = fnum(m.get("serial_seconds_per_token"))
        print(f"{r['tag']:34s} {r['arm']:11s} {r['rep']:3d} {r['pos']:3d} "
              f"{mtp if mtp else float('nan'):10.6f} "
              f"{ser if ser else float('nan'):9.6f} "
              f"{fnum(m.get('mtp_decode_speedup')) or float('nan'):7.4f} "
              f"{fnum(r['meta'].get('gpu_temp_entry_c')) or float('nan'):6.1f} "
              f"{fnum(r['meta'].get('gpu_temp_exit_c')) or float('nan'):6.1f} "
              f"{fnum(m.get('effective_mean_draft_len')) or float('nan'):6.3f}")
    print()

    by_arm: dict[str, list[dict]] = {a: [] for a in ARMS}
    for r in complete:
        by_arm[r["arm"]].append(r)

    print("per arm")
    print(f"{'arm':11s} {'n':>2s} {'mtp mean':>10s} {'sd':>9s} "
          f"{'serial mean':>11s} {'ratio':>7s} {'entry C mean':>12s} "
          f"{'entry spread':>12s} {'draft':>6s}")
    means = {}
    for arm in ARMS:
        rs = by_arm[arm]
        if not rs:
            continue
        mtp = [fnum(r["metrics"]["mtp_seconds_per_token"]) for r in rs]
        ser = [fnum(r["metrics"].get("serial_seconds_per_token")) or 0 for r in rs]
        rat = [fnum(r["metrics"].get("mtp_decode_speedup")) or 0 for r in rs]
        ent = [fnum(r["meta"].get("gpu_temp_entry_c")) for r in rs]
        ent = [e for e in ent if e is not None]
        drf = [fnum(r["metrics"].get("effective_mean_draft_len")) or 0 for r in rs]
        means[arm] = statistics.fmean(mtp)
        print(f"{arm:11s} {len(rs):2d} {statistics.fmean(mtp):10.6f} "
              f"{(statistics.stdev(mtp) if len(mtp) > 1 else 0):9.6f} "
              f"{statistics.fmean(ser):11.6f} {statistics.fmean(rat):7.4f} "
              f"{(statistics.fmean(ent) if ent else float('nan')):12.1f} "
              f"{((max(ent) - min(ent)) if len(ent) > 1 else 0):12.1f} "
              f"{statistics.fmean(drf):6.3f}")
    print()

    if "shipped" in means:
        base = means["shipped"]
        print("against shipped, unpaired session means")
        for arm in ("onepass67", "onepass678"):
            if arm in means:
                d = means[arm] - base
                print(f"  {arm:11s} {d:+.6f} s/token  {100 * d / base:+.3f} %")
        print()

    reps = sorted({r["rep"] for r in complete})
    if len(reps) >= 1:
        print("paired inside each replicate, position-balanced")
        pairs: dict[str, list[float]] = {"onepass67": [], "onepass678": []}
        for rep in reps:
            rs = [r for r in complete if r["rep"] == rep]
            per = {}
            for arm in ARMS:
                v = [fnum(r["metrics"]["mtp_seconds_per_token"])
                     for r in rs if r["arm"] == arm]
                if v:
                    per[arm] = statistics.fmean(v)
            if "shipped" not in per:
                continue
            line = f"  rep {rep}: shipped {per['shipped']:.6f}"
            for arm in ("onepass67", "onepass678"):
                if arm in per:
                    pct = 100 * (per[arm] - per["shipped"]) / per["shipped"]
                    pairs[arm].append(pct)
                    line += f"   {arm} {per[arm]:.6f} ({pct:+.3f} %)"
            print(line)
        print()
        paired_pct = {}
        for arm, vals in pairs.items():
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            paired_pct[arm] = statistics.fmean(vals)
            print(f"  {arm:11s} mean {statistics.fmean(vals):+.3f} % "
                  f"over {len(vals)} replicate(s), sd {sd:.3f}")
        print()
        ranked_frame(paired_pct)
    print()
    print("This session is ungated and counterbalanced. It is directional "
          "causal evidence inside itself, not a gate-qualified reading and "
          "not any kind of official score.")
    return 0


def ranked_frame(paired_pct: dict[str, float]) -> None:
    """Re-weight each local effect into the ranked width mix.

    A local effect is priced against a histogram whose mass sits at M=8. The
    ranked mix is much flatter, so a pair whose differing widths are rare
    locally and common on the runner is scaled UP, and the reverse is scaled
    down. This is a first-order transfer: it assumes the per-width saving is
    the same fraction of per-width QMV time on both generations, which the
    measured spill difference between g16s and g17s specifically contradicts
    for the one-pass bodies. It is a re-frame, never a ranked prediction.
    """
    local = transfer.local_histogram()
    ranked = transfer.ranked_histogram()
    print("re-weighted into the ranked width mix (first order, NOT a "
          "ranked prediction)")
    print(f"  {'pair':28s} {'widths':12s} {'local %':>9s} "
          f"{'x rounds':>9s} {'ranked %':>9s}")
    named = {"onepass67": ("shipped", "onepass67"),
             "onepass678": ("shipped", "onepass678")}
    for arm, pct in paired_pct.items():
        if arm not in named:
            continue
        a, b = named[arm]
        widths = transfer.differing_widths(a, b)
        factor = (transfer.mass(ranked, widths, None)
                  / transfer.mass(local, widths, None))
        print(f"  {a + ' -> ' + b:28s} {str(list(widths)):12s} "
              f"{pct:+9.3f} {factor:9.3f} {pct * factor:+9.3f}")
    print("  The pre-registered table is stated in the ranked frame on the "
          "eight-prompt candidate-leg mean. Neither the local column nor the "
          "re-weighted column may be read against it as though it were a "
          "receipt.")


if __name__ == "__main__":
    raise SystemExit(main())
