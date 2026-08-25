#!/usr/bin/env python3
"""E217 step 3 (RULE 403): deterministic write-coverage walk for all mappings.

The staged mapping changes three things at once: the threadgroup shape (2 to 4
simdgroups), the grid (`(32, (n/8)*4, 1)`), and where a consumer's packed
weight word comes from. Any one of those can silently drop or duplicate output
rows, or feed a consumer the wrong weight row, and a timing run would still
produce a number. This walk enumerates every thread coordinate the launcher
produces and checks two separate covers:

  1. OUTPUT COVER. The multiset of `(input_row, output_row)` pairs written must
     be the full `m x n` product, exactly once, for every mapping and width.
  2. TILE COVER. For the staged mapping, every tile word must be produced
     exactly once per k-block, and every word a consumer reads must be the
     SAME device word the shipped mapping would have read for that output row
     and column. This is the check that a row-half or lane indexing error
     cannot survive.

Both checks carry deliberate-defect controls. If a control reports a clean
cover, the walk has no power and its passes mean nothing.

Widths 2..5 have `G(m) == 1` and never leave the shipped mapping, so the walk
runs them too and requires them to be identical under all three environment
settings: that is the control band the timing arms depend on.

No GPU, no build:

    python3 research/e217_coverage.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

# The shipped width plan, `Qwen35QMVKernelVariant.staged.pairs`.
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}

# n is any multiple of 8. 64 is a small exact case, 4104 is the odd-looking
# down-projection width, 5120 is the hidden size.
NS = (64, 4104, 5120)

VALUES_PER_THREAD = 16
BLOCK_SIZE = VALUES_PER_THREAD * 32
WORDS_PER_ROW = BLOCK_SIZE // 4
BYTES_PER_LANE = 8


def groups(m: int, ipg: int) -> int:
    return (m + ipg - 1) // ipg


def split_na(m: int, ipg: int, first_m: int) -> int:
    """`qwen_e120_qmv_m`: the tail group compiles `max(M % IPG, 2)` rows."""
    tail = m % ipg
    if tail == 0 or m - first_m >= ipg:
        return ipg
    return max(tail, 2)


def staged_na(m: int, ipg: int, col_group: int) -> int:
    """`qwen_e217_qmv_staged`: `TAIL` is `IPG` when the width divides."""
    tail = ipg if m % ipg == 0 else m % ipg
    return ipg if col_group == 0 else tail


def cover_split(m: int, n: int, ipg: int, rows_per_tg: int = 8) -> Counter:
    seen: Counter = Counter()
    for tx in range(groups(m, ipg)):
        first_m = tx * ipg
        if first_m >= m:
            continue
        na = split_na(m, ipg, first_m)
        for ty in range(n // rows_per_tg):
            for sgid in range(2):
                out_row = ty * rows_per_tg + sgid * 4
                for r in range(4):
                    for j in range(na):
                        seen[(first_m + j, out_row + r)] += 1
    return seen


def cover_coop(m: int, n: int, ipg: int, rows_per_tg: int = 4) -> Counter:
    seen: Counter = Counter()
    for ty in range(n // rows_per_tg):
        for sgid in range(2):
            first_m = sgid * ipg
            if first_m >= m:
                continue
            na = split_na(m, ipg, first_m)
            out_row = ty * rows_per_tg
            for r in range(4):
                for j in range(na):
                    seen[(first_m + j, out_row + r)] += 1
    return seen


def cover_staged(m: int, n: int, ipg: int, rows_per_tg: int = 8,
                 simdgroups: int = 4, row_half_stride: int = 4) -> Counter:
    seen: Counter = Counter()
    for ty in range(n // rows_per_tg):
        out_row_base = ty * rows_per_tg
        for sgid in range(simdgroups):
            row_half = sgid // 2
            col_group = sgid % 2
            out_row = out_row_base + row_half * row_half_stride
            first_m = col_group * ipg
            na = staged_na(m, ipg, col_group)
            for r in range(4):
                for j in range(na):
                    seen[(first_m + j, out_row + r)] += 1
    return seen


def effective(env: str, m: int, ipg: int) -> str:
    """`Qwen35CustomQMV.mappingForLaunch`: only `G == 2` leaves the shipped
    mapping, whatever the environment asks for."""
    return env if groups(m, ipg) == 2 else "split"


def output_cover(env: str, m: int, n: int, ipg: int) -> Counter:
    return {"split": cover_split, "coop": cover_coop,
            "staged": cover_staged}[effective(env, m, ipg)](m, n, ipg)


def verdict(seen: Counter, m: int, n: int) -> dict:
    want = {(j, r) for j in range(m) for r in range(n)}
    got = set(seen)
    return {
        "pairs_expected": len(want),
        "pairs_written": len(got),
        "never_written_count": len(want - got),
        "never_written_sample": sorted(want - got)[:6],
        "written_twice_count": sum(1 for v in seen.values() if v > 1),
        "written_twice_sample": sorted(k for k, v in seen.items() if v > 1)[:6],
        "out_of_range_count": len(got - want),
        "out_of_range_sample": sorted(got - want)[:6],
        "exact_cover": got == want and all(v == 1 for v in seen.values()),
    }


def tile_walk(produce_rows: int = 2, consumer_row_half: bool = True,
              lane_stride_words: int = 4) -> dict:
    """Produce and consume one 8-row staged tile for one k-block.

    A tile word is identified by the DEVICE word it must hold:
    `(row_in_block, byte_offset_in_block)`. The producer writes what it read,
    and the consumer's read is checked against the word the shipped mapping
    would have used for the same output row and the same column, so a wrong
    row half or a wrong lane stride is a mismatch and not merely a different
    number.
    """
    tile: dict[int, tuple[int, int]] = {}
    writes: Counter = Counter()
    for sgid in range(4):
        for rr in range(produce_rows):
            row = sgid * 2 + rr
            for lane in range(32):
                for i in range(4):
                    slot = row * WORDS_PER_ROW + lane * lane_stride_words + i
                    device_word = (row, lane * BYTES_PER_LANE + i * 2)
                    tile[slot] = device_word
                    writes[slot] += 1

    mismatches = []
    unwritten = 0
    reads = 0
    for sgid in range(4):
        # The output row always follows the true row half, because the kernel
        # derives it from `sgid`. Only the TILE POINTER can be wrong, so the
        # two halves are tracked separately or the control has no power.
        row_half = sgid // 2
        tile_row_half = row_half if consumer_row_half else 0
        for r in range(4):
            out_row = row_half * 4 + r
            for lane in range(32):
                for i in range(4):
                    slot = ((tile_row_half * 4 + r) * WORDS_PER_ROW
                            + lane * lane_stride_words + i)
                    reads += 1
                    if slot not in tile:
                        unwritten += 1
                        continue
                    want = (out_row, lane * BYTES_PER_LANE + i * 2)
                    if tile[slot] != want:
                        mismatches.append(
                            {"sgid": sgid, "r": r, "lane": lane, "i": i,
                             "got": list(tile[slot]), "want": list(want)})
    return {
        "tile_words": 8 * WORDS_PER_ROW,
        "words_written": len(tile),
        "words_written_twice": sum(1 for v in writes.values() if v > 1),
        "reads": reads,
        "reads_of_unwritten_words": unwritten,
        "wrong_device_word_count": len(mismatches),
        "wrong_device_word_sample": mismatches[:6],
        "exact_cover": (
            len(tile) == 8 * WORDS_PER_ROW
            and all(v == 1 for v in writes.values())
            and unwritten == 0
            and not mismatches),
    }


# Every control is a real indexing mistake the staged rewrite could have made.
OUTPUT_CONTROLS = [
    ("CONTROL staged keeps coop 4-row tiles",
     lambda m, n, ipg: cover_staged(m, n, ipg, rows_per_tg=4)),
    ("CONTROL staged uses E216 first_m = sgid * IPG",
     lambda m, n, ipg: Counter(
         {(sgid * ipg + j, ty * 8 + r): 1
          for ty in range(n // 8) for sgid in range(4)
          for r in range(4) for j in range(ipg)})),
    ("CONTROL staged drops the row-half offset",
     lambda m, n, ipg: cover_staged(m, n, ipg, row_half_stride=0)),
    ("CONTROL staged runs 2 simdgroups",
     lambda m, n, ipg: cover_staged(m, n, ipg, simdgroups=2)),
    ("CONTROL coop keeps the 8-row split grid",
     lambda m, n, ipg: cover_coop(m, n, ipg, rows_per_tg=8)),
]

TILE_CONTROLS = [
    ("CONTROL producer writes 1 row per simdgroup",
     lambda: tile_walk(produce_rows=1)),
    ("CONTROL consumer ignores the row half",
     lambda: tile_walk(consumer_row_half=False)),
    ("CONTROL lane stride 8 words instead of 4",
     lambda: tile_walk(lane_stride_words=8)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="research/e217-artifacts/e217_coverage.json")
    args = ap.parse_args()

    result = {
        "experiment": "e217-staged-dequant",
        "step": "3-rule-403-write-coverage",
        "harness": "static_enumeration",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "n_values": list(NS),
        "output_cover": [],
        "output_controls": [],
        "tile_cover": tile_walk(),
        "tile_controls": [],
    }

    failures = []
    for env in ("split", "coop", "staged"):
        for m in range(2, 10):
            ipg = SHIPPED_IPG[m]
            for n in NS:
                row = {
                    "env": env, "m": m, "ipg": ipg, "n": n,
                    "groups": groups(m, ipg),
                    "effective_mapping": effective(env, m, ipg),
                    **verdict(output_cover(env, m, n, ipg), m, n),
                }
                result["output_cover"].append(row)
                if not row["exact_cover"]:
                    failures.append(f"{env} m={m} n={n} is not an exact cover")

    for label, fn in OUTPUT_CONTROLS:
        for m in (6, 7, 8, 9):
            ipg = SHIPPED_IPG[m]
            n = 64
            row = {"control": label, "m": m, "ipg": ipg, "n": n,
                   **verdict(fn(m, n, ipg), m, n)}
            result["output_controls"].append(row)
            if row["exact_cover"]:
                failures.append(f"{label} m={m} reports a clean cover")

    if not result["tile_cover"]["exact_cover"]:
        failures.append("the staged tile walk is not an exact cover")
    for label, fn in TILE_CONTROLS:
        row = {"control": label, **fn()}
        result["tile_controls"].append(row)
        if row["exact_cover"]:
            failures.append(f"{label} reports a clean tile cover")

    result["failures"] = failures
    result["passed"] = not failures

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")

    covers = result["output_cover"]
    clean = sum(1 for r in covers if r["exact_cover"])
    print(f"output cover: {clean}/{len(covers)} exact "
          f"(3 environments x widths 2..9 x n in {NS})")
    for env in ("split", "coop", "staged"):
        moved = sorted({r["m"] for r in covers
                        if r["env"] == env and r["effective_mapping"] != "split"})
        print(f"  {env:7s} leaves the shipped mapping at widths {moved}")
    print(f"output controls: "
          f"{sum(1 for r in result['output_controls'] if not r['exact_cover'])}"
          f"/{len(result['output_controls'])} correctly rejected")
    tc = result["tile_cover"]
    print(f"tile cover: {tc['words_written']}/{tc['tile_words']} words written "
          f"once, {tc['reads']} reads, "
          f"{tc['wrong_device_word_count']} wrong device words, "
          f"exact={tc['exact_cover']}")
    print(f"tile controls: "
          f"{sum(1 for r in result['tile_controls'] if not r['exact_cover'])}"
          f"/{len(result['tile_controls'])} correctly rejected")
    for line in failures:
        print("FAIL:", line)
    print("\nwrote", out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
