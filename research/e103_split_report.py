#!/usr/bin/env python3
"""Price the E103 rung-2b split removal and the stacked pack + merge ceiling.

Inputs, all produced by research-only harnesses:

  research/out/e103/rung2.json            arms a..j at M = 1..5
  research/out/e103/rung2c_widths678.json arms a..j at M = 6, 7, 8
  research/out/e103/rung2c_width9.json    arms a..j at M = 9
  research/out/e103/rung2b_split.json     split vs single at M = 6, 7, 8
  research/out/e103/rung2b_split_m9.json  split vs single at M = 9

Every quantity used below is a *within-session* contrast:

  merge saving(M, N)  = t_split(M, N) - t_single(M, N)     [split harness]
  pack  saving(M, N)  = t_a(M, N) - min(t_pack2, t_pack3)  [arm harness]
  stacked(M, N)       = merge saving + pack saving

because what ships today at M >= 6 is the split, and the best a Swift-side
custom kernel could do is one merged, packed dispatch.
"""

from __future__ import annotations

import json
import pathlib
import statistics as st

OUT = pathlib.Path("research/out/e103")
FA_LAYERS = 16          # full-attention layers per round, census leg
DRAFT_DISPATCHES = 4    # draft-head SDPA dispatches per round, all at M = 1
MAX_QL = 5              # trusted supports_sdpa_vector cap at gqa 6

# Ledger 207 verify-width shares.
WIDTH_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122, 8: 0.0735,
               9: 0.0575}
PACK_ARMS = ("d_pack2_c", "d_pack3_c")
MIN_USEFUL_US_PER_ROUND = 383.0
LOCAL_ROUND_US = 127176.0
LATENCY_CLASS_FACTOR = 2.40
INSITU_DISCOUNT = (1.65, 2.59)


def arm_medians(paths: list[str]) -> dict[tuple[int, int], dict[str, float]]:
    med: dict[tuple[int, int], dict[str, float]] = {}
    for p in paths:
        payload = json.loads((OUT / p).read_text())
        cells: dict[tuple[int, int], list[dict]] = {}
        for r in payload["measurements"]:
            if r["kind"] != "timing":
                continue
            cells.setdefault((r["n"], r["m"]), []).append(r)
        for key, rows in cells.items():
            med[key] = {a: st.median(x["seconds"][a] for x in rows) * 1e6
                        for a in payload["arms"]}
    return med


def split_medians(paths: list[str]) -> dict[tuple[int, int], dict[str, float]]:
    med: dict[tuple[int, int], dict[str, float]] = {}
    for p in paths:
        payload = json.loads((OUT / p).read_text())
        cells: dict[tuple[int, int], list[dict]] = {}
        for r in payload["measurements"]:
            if r["kind"] != "timing":
                continue
            cells.setdefault((r["n"], r["m"]), []).append(r)
        for key, rows in cells.items():
            med[key] = {v: st.median(x["seconds"][v] for x in rows) * 1e6
                        for v in payload["variants"]}
    return med


def window_mean(values: dict[int, float]) -> float:
    ns = sorted(values)
    total = 0.0
    for lo, hi in zip(ns, ns[1:]):
        total += 0.5 * (values[lo] + values[hi]) * (hi - lo)
    return total / (ns[-1] - ns[0])


def main() -> None:
    arms = arm_medians(["rung2.json", "rung2c_widths678.json",
                        "rung2c_width9.json"])
    splits = split_medians(["rung2b_split.json", "rung2b_split_m9.json"])
    lens = sorted({n for n, _ in arms})
    widths = sorted({m for _, m in arms})

    print("=== per-dispatch microseconds, arm a (shipped transcription) ===")
    print(f"{'N':>6}" + "".join(f"{m:>9}" for m in widths))
    for n in lens:
        print(f"{n:>6}" + "".join(
            f"{arms[(n, m)]['a_shipped_c']:9.2f}" if (n, m) in arms else " " * 9
            for m in widths))

    print()
    print("=== merge saving, us per dispatch: split(5 + r) minus single(M) ===")
    print(f"{'N':>6}" + "".join(f"{m:>9}" for m in widths if m > MAX_QL))
    merge: dict[tuple[int, int], float] = {}
    for n in lens:
        row = ""
        for m in widths:
            if m <= MAX_QL:
                continue
            if (n, m) not in splits:
                row += " " * 9
                continue
            s = splits[(n, m)]
            merge[(n, m)] = s["split"] - s["single"]
            row += f"{merge[(n, m)]:9.2f}"
        print(f"{n:>6}{row}")

    print()
    print("=== pack saving, us per dispatch: arm a minus best of pack2/pack3 ===")
    print(f"{'N':>6}" + "".join(f"{m:>9}" for m in widths))
    pack: dict[tuple[int, int], float] = {}
    pack_which: dict[tuple[int, int], str] = {}
    for n in lens:
        row = ""
        for m in widths:
            if (n, m) not in arms:
                row += " " * 9
                continue
            cell = arms[(n, m)]
            best = min(PACK_ARMS, key=lambda a: cell[a])
            pack[(n, m)] = cell["a_shipped_c"] - cell[best]
            pack_which[(n, m)] = best
            row += f"{pack[(n, m)]:9.2f}"
        print(f"{n:>6}{row}")

    print()
    print("=== us saved per round, by verify width ===")
    print("  merge  = FA_LAYERS x merge saving")
    print("  pack   = FA_LAYERS x pack saving + draft-head pack saving at M=1")
    print("  stack  = merge + pack")
    print()
    header = (f"{'M':>3} {'share':>7} {'merge/round':>12} {'pack/round':>11} "
              f"{'stack/round':>12} {'x bar':>7}  arm")
    print(header)
    per_round: dict[int, dict[str, float]] = {}
    for m in widths:
        merge_w = window_mean({n: merge[(n, m)] for n in lens
                               if (n, m) in merge}) if m > MAX_QL else 0.0
        pack_w = window_mean({n: pack[(n, m)] for n in lens if (n, m) in pack})
        draft = window_mean({n: pack[(n, 1)] for n in lens if (n, 1) in pack})
        merge_r = FA_LAYERS * merge_w
        pack_r = FA_LAYERS * pack_w + DRAFT_DISPATCHES * draft
        stack_r = merge_r + pack_r
        per_round[m] = {"merge": merge_r, "pack": pack_r, "stack": stack_r}
        share = WIDTH_SHARE.get(m, 0.0)
        which = {pack_which[(n, m)] for n in lens if (n, m) in pack_which}
        print(f"{m:>3} {share:7.3f} {merge_r:12.0f} {pack_r:11.0f} "
              f"{stack_r:12.0f} {stack_r / MIN_USEFUL_US_PER_ROUND:7.2f}  "
              f"{','.join(sorted(w.replace('d_', '').replace('_c', '') for w in which))}")

    print()
    covered = sum(WIDTH_SHARE.get(m, 0.0) for m in widths)
    for label, key in (("merge only", "merge"), ("pack only", "pack"),
                       ("stacked", "stack")):
        weighted = sum(WIDTH_SHARE.get(m, 0.0) * per_round[m][key]
                       for m in widths)
        norm = weighted / covered if covered else 0.0
        print(f"session average, {label:>10}: {weighted:7.0f} us/round over "
              f"{100 * covered:.1f} % of width mass, {norm:7.0f} us/round "
              f"renormalised, {100 * norm / LOCAL_ROUND_US:6.3f} % of the local "
              f"round, {norm / MIN_USEFUL_US_PER_ROUND:5.2f} x bar")

    print()
    m_ge_6 = sum(WIDTH_SHARE[m] for m in WIDTH_SHARE if m >= 6)
    merge_only_rounds = window_mean(
        {n: st.mean([merge[(n, m)] for m in widths if (n, m) in merge])
         for n in lens})
    print(f"M >= 6 share of width mass: {100 * m_ge_6:.2f} %")
    print(f"mean merge saving per dispatch over M >= 6: "
          f"{merge_only_rounds:.2f} us")
    print(f"advisor section 2 predicted a merge saving of 20.9 us per "
          f"dispatch, which is the fitted fixed term")

    print()
    print("=== fidelity ===")
    for p in ("rung2b_split.json", "rung2b_split_m9.json"):
        payload = json.loads((OUT / p).read_text())
        fid = [r for r in payload["measurements"] if r["kind"] == "fidelity"]
        pos = [r for r in payload["measurements"]
               if r["kind"] == "positive_control"]
        bad = [r for r in fid if not r["bit_identical"]]
        print(f"{p}: {len(fid)} cells, bit-exact violations {len(bad)}, "
              f"positive controls {len(pos)} all detected "
              f"{all(r['detected'] for r in pos)}")

    print()
    print("=== ranked pricing of the stacked ceiling ===")
    weighted = sum(WIDTH_SHARE.get(m, 0.0) * per_round[m]["stack"]
                   for m in widths) / covered
    local_pct = 100.0 * weighted / LOCAL_ROUND_US
    ranked = LATENCY_CLASS_FACTOR * local_pct
    print(f"local  {weighted:7.0f} us/round  {local_pct:6.3f} %")
    print(f"ranked undiscounted            {ranked:6.3f} %")
    print(f"ranked after in-situ discount  {ranked / INSITU_DISCOUNT[1]:6.3f} "
          f"% to {ranked / INSITU_DISCOUNT[0]:6.3f} %")
    print("published detection floor       0.277 %")


if __name__ == "__main__":
    main()
