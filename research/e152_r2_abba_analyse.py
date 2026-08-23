#!/usr/bin/env python3
"""E152 R2: read the leaf-width ABBA session and report the effect with its noise.

Sign convention, stated in words because Rule 144 requires it: a POSITIVE
`leaf16 gain` means the leaf-16 arm decoded FASTER than the leaf-8 arm, that is
its candidate MTP seconds per token was lower. `harness=local` throughout; this
is a same-build local pair and it is not a ranked score.

Two frames are reported and they are not interchangeable:

  * absolute candidate MTP seconds per token, which is the frame that transfers
    to the ranked candidate leg; and
  * the local serial-to-MTP ratio, which is only valid because both arms differ
    on the proposal-side index alone, so the change cannot move the serial leg.

The arm is read from each leg's own trace (`leaf=`, `leaves=`, `probes=`) and
not from the launch environment, per Rule 114. A leg whose trace disagrees with
its intended arm is dropped and reported.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDER = ["leaf8", "leaf16", "leaf16", "leaf8"]
EXPECTED_LEAF = {"leaf8": 8, "leaf16": 16}


def read_meta(path: pathlib.Path) -> dict[str, str]:
    return dict(re.findall(r"^([\w.]+)=(.*)$", path.read_text(), re.M))


def trace_geometry(path: pathlib.Path) -> dict[str, int]:
    """Last observed geometry and census counters in a leg's trace."""
    if not path.exists():
        return {}
    fields: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        for key in ("leaf", "leaves", "probes", "xs_hit", "xs_fill", "round"):
            m = re.search(rf"\b{key}=(\d+)", line)
            if m:
                fields[key] = int(m.group(1))
    return fields


def collect(label: str, first: int, replicates: int) -> list[dict]:
    legs = []
    for rep in range(first, first + replicates):
        for pos, arm in enumerate(ORDER, start=1):
            out = ROOT / "research" / "out" / f"e152{label}k{rep}p{pos}{arm}"
            meta_path, score_path = out / "meta.txt", out / "score.json"
            if not (meta_path.exists() and score_path.exists()):
                continue
            meta = read_meta(meta_path)
            score = json.loads(score_path.read_text())["metrics"]
            geom = trace_geometry(out / "trace.txt")
            legs.append(
                {
                    "tag": out.name,
                    "arm": arm,
                    "rep": rep,
                    "pos": pos,
                    "mtp_spt": score["mtp_seconds_per_token"],
                    "serial_spt": score["serial_seconds_per_token"],
                    "speedup": score["mtp_decode_speedup"],
                    "edl": score["effective_mean_draft_len"],
                    "accept": score["accepted_draft_rate"],
                    "matched": score["all_tokens_matched"],
                    "divergence": score["residual_divergence_count"],
                    "tripwire": score["public_drift_tripwire_passed"],
                    "entry_c": float(meta.get("gpu_temp_entry_c", "nan")),
                    "exit_c": float(meta.get("gpu_temp_exit_c", "nan")),
                    "gated": meta.get("cool_gate_passed_real_gate") == "true",
                    "worker": meta.get("worker_sha256", "")[:12],
                    "base_sha": meta.get("base_sha", "")[:8],
                    "geom": geom,
                }
            )
    return legs


def pct(a: float, b: float) -> float:
    """Percent by which `b` is faster than `a`. Positive means b is faster."""
    return 100.0 * (a - b) / a


def main() -> None:
    label = os.environ.get("E152_LABEL", "r2")
    first = int(os.environ.get("E152_FIRST", "1"))
    replicates = int(os.environ.get("E152_REPLICATES", "3"))

    legs = collect(label, first, replicates)
    if not legs:
        raise SystemExit("e152_r2_abba_analyse: no legs found")

    print("=== legs ===")
    header = (
        f"{'tag':<24}{'arm':<8}{'leaf':<6}{'leaves':<8}{'probes':<8}"
        f"{'mtp s/tok':<12}{'serial':<10}{'edl':<8}{'accept':<9}"
        f"{'entry_C':<9}{'exit_C':<9}{'xs_hit':<8}{'xs_fill':<8}{'ok'}"
    )
    print(header)
    dropped = []
    for leg in legs:
        g = leg["geom"]
        ok = (
            leg["matched"]
            and leg["divergence"] == 0
            and leg["tripwire"]
            and leg["gated"]
            and g.get("leaf") == EXPECTED_LEAF[leg["arm"]]
        )
        if not ok:
            dropped.append(leg)
        print(
            f"{leg['tag']:<24}{leg['arm']:<8}{g.get('leaf', '?'):<6}"
            f"{g.get('leaves', '?'):<8}{g.get('probes', '?'):<8}"
            f"{leg['mtp_spt']:<12.8f}{leg['serial_spt']:<10.6f}"
            f"{leg['edl']:<8.4f}{leg['accept']:<9.6f}"
            f"{leg['entry_c']:<9.2f}{leg['exit_c']:<9.2f}"
            f"{g.get('xs_hit', '?'):<8}{g.get('xs_fill', '?'):<8}{ok}"
        )

    good = [leg for leg in legs if leg not in dropped]
    if dropped:
        print(f"\nDROPPED {len(dropped)} leg(s) that failed a gate or arm witness")
    if len(good) < 2:
        raise SystemExit("e152_r2_abba_analyse: not enough valid legs")

    print("\n=== arm means (valid legs only) ===")
    stats = {}
    for arm in ("leaf8", "leaf16"):
        v = [leg for leg in good if leg["arm"] == arm]
        if not v:
            continue
        mtp = [x["mtp_spt"] for x in v]
        stats[arm] = {
            "n": len(v),
            "mtp": statistics.fmean(mtp),
            "mtp_sd": statistics.stdev(mtp) if len(mtp) > 1 else float("nan"),
            "serial": statistics.fmean([x["serial_spt"] for x in v]),
            "ratio": statistics.fmean([x["speedup"] for x in v]),
            "edl": statistics.fmean([x["edl"] for x in v]),
            "accept": statistics.fmean([x["accept"] for x in v]),
            "entry": statistics.fmean([x["entry_c"] for x in v]),
        }
        s = stats[arm]
        print(
            f"{arm:<8} n={s['n']}  mtp={s['mtp']:.8f} sd={s['mtp_sd']:.8f}  "
            f"serial={s['serial']:.6f}  ratio={s['ratio']:.6f}  "
            f"edl={s['edl']:.6f}  accept={s['accept']:.6f}  "
            f"entry_C={s['entry']:.2f}"
        )

    if "leaf8" in stats and "leaf16" in stats:
        a, b = stats["leaf8"], stats["leaf16"]
        # Paired by replicate: each replicate contributes one leaf8 mean and one
        # leaf16 mean, so the pair difference removes between-replicate drift.
        pairs = []
        for rep in range(first, first + replicates):
            r8 = [x["mtp_spt"] for x in good if x["arm"] == "leaf8" and x["rep"] == rep]
            r16 = [
                x["mtp_spt"] for x in good if x["arm"] == "leaf16" and x["rep"] == rep
            ]
            if r8 and r16:
                pairs.append(pct(statistics.fmean(r8), statistics.fmean(r16)))
        print("\n=== effect, harness=local ===")
        print("sign convention: POSITIVE means leaf16 decoded FASTER than leaf8")
        print(f"absolute candidate MTP s/token  leaf8 {a['mtp']:.8f} -> "
              f"leaf16 {b['mtp']:.8f}")
        print(f"e152_r2_leaf16_local_pct (pooled means)  {pct(a['mtp'], b['mtp']):+.4f} %")
        if pairs:
            mean = statistics.fmean(pairs)
            sd = statistics.stdev(pairs) if len(pairs) > 1 else float("nan")
            se = sd / len(pairs) ** 0.5 if len(pairs) > 1 else float("nan")
            print(f"per-replicate paired pcts  {['%+.4f' % p for p in pairs]}")
            print(f"paired mean {mean:+.4f} %  sd {sd:.4f}  se {se:.4f}  "
                  f"2sigma {2 * se:.4f}")
            print(f"gated-leg noise floor 0.052 %  clears_2sigma="
                  f"{abs(mean) > 2 * se and abs(mean) > 0.052}")
        print(f"local serial-to-MTP ratio  {a['ratio']:.6f} -> {b['ratio']:.6f}  "
              f"({pct(1 / a['ratio'], 1 / b['ratio']):+.4f} %)")
        print(f"serial leg  {a['serial']:.6f} -> {b['serial']:.6f}  "
              f"({pct(a['serial'], b['serial']):+.4f} %, must be noise)")
        print(f"acceptance  edl {a['edl']:.6f} -> {b['edl']:.6f}   "
              f"rate {a['accept']:.6f} -> {b['accept']:.6f}")
        print(f"entry temperature spread  leaf8 {a['entry']:.2f} C  "
              f"leaf16 {b['entry']:.2f} C  delta {b['entry'] - a['entry']:+.2f} C")


if __name__ == "__main__":
    main()
