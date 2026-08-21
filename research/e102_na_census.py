#!/usr/bin/env python3
"""E102 rung 0b: census every submission tree that instantiates a wide QMV NA >= 5.

Two sources have to agree before a tree counts. The readable
`kernels/quantized.h` is what the AOT metallib build compiles; the checked-in
`mlx-generated/quantized.cpp` twin is the string MLX hands to `newLibrary` on
the JIT path. A tree that widened only one of the two did not ship the wider
kernel to the scored worker, so the pair is reported separately and never
collapsed.

    python3 research/e102_na_census.py            # kernel census only
    python3 research/e102_na_census.py --board    # join with /tmp/yukon-board
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess

HDR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
PATHS = (HDR, TWIN)

NA_RE = re.compile(r"static_assert\(NA >= 2 && NA <= (\d+)")
TBL_RE = re.compile(r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+)")
RPS_RE = re.compile(r"constexpr int rows_per_simd = (\d+)")


def git(*args: str, text: bool = True) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=text,
                          check=True).stdout


def submission_blobs() -> list[dict]:
    refs = [line.split() for line in git(
        "for-each-ref", "--format=%(refname:short) %(objectname)",
        "refs/remotes/upstream/submissions/").splitlines() if line.strip()]
    query = "".join(f"{sha}:{p}\n" for _, sha in refs for p in PATHS)
    lines = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        input=query, capture_output=True, text=True).stdout.splitlines()
    lines = [l for l in lines if l.strip()]
    assert len(lines) == len(refs) * len(PATHS), (len(lines), len(refs))
    out, i = [], 0
    for name, sha in refs:
        entry = {}
        for path in PATHS:
            tok = lines[i].split()
            i += 1
            entry[path] = tok[0] if len(tok) > 1 and tok[1] == "blob" else None
        out.append({"sub": name.split("/")[-1], "commit": sha,
                    "hdr": entry[HDR], "twin": entry[TWIN]})
    return out


def scan(blob: str | None) -> dict | None:
    if blob is None:
        return None
    text = git("cat-file", "blob", blob)
    table = sorted({(int(m), int(ipg)) for m, ipg in TBL_RE.findall(text)})
    return {
        "na_bounds": sorted({int(m) for m in NA_RE.findall(text)}),
        "table": table,
        "max_ipg": max((ipg for _, ipg in table), default=0),
        "rows_per_simd": sorted({int(m) for m in RPS_RE.findall(text)}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true")
    ap.add_argument("--out", default="/tmp/e102-na-census.json")
    args = ap.parse_args()

    subs = submission_blobs()
    cache: dict[str, dict | None] = {}
    for s in subs:
        for key in ("hdr", "twin"):
            blob = s[key]
            if blob not in cache:
                cache[blob] = scan(blob)
            s[key + "_info"] = cache[blob]

    groups = collections.defaultdict(list)
    for s in subs:
        key = json.dumps([s["hdr_info"], s["twin_info"]], sort_keys=True)
        groups[key].append(s["sub"])

    print(f"branches {len(subs)}  distinct kernel configurations {len(groups)}")
    print(f"{'n':>4}  {'hdr NA':>7} {'hdr IPG':>7}  {'twin NA':>7} "
          f"{'twin IPG':>8}  rps  example")
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        h, t = json.loads(key)
        print(f"{len(members):>4}  "
              f"{str(h and h['na_bounds']):>7} {str(h and h['max_ipg']):>7}  "
              f"{str(t and t['na_bounds']):>7} {str(t and t['max_ipg']):>8}  "
              f"{str(h and h['rows_per_simd'])}  {members[0][:8]}")

    wide = [s for s in subs if _wide(s)]
    print(f"\ntrees with NA >= 5 reachable in the wide kernel: {len(wide)}")

    board = {}
    if args.board:
        board = load_board()
    for s in sorted(wide, key=lambda s: s["sub"]):
        h, t = s["hdr_info"], s["twin_info"]
        row = board.get(s["sub"][:8], {})
        print(f"  {s['sub']}  hdr_NA<={h['na_bounds']} tbl={h['table']}  "
              f"twin_NA<={t['na_bounds'] if t else None} "
              f"tbl={t['table'] if t else None}")
        print(f"      board: {row or 'NOT ON BOARD'}")

    json.dump({"subs": [{k: v for k, v in s.items()} for s in subs]},
              open(args.out, "w"))
    print(f"\nwrote {args.out}")


def _wide(s: dict) -> bool:
    h, t = s["hdr_info"], s["twin_info"]
    for info in (h, t):
        if info and info["max_ipg"] >= 5:
            return True
        if info and info["na_bounds"] and max(info["na_bounds"]) >= 5:
            # A widened bound with no >= 5 table entry does not reach the
            # hardware, so it is reported but flagged by the table column.
            return True
    return False


def load_board() -> dict:
    full = json.load(open("/tmp/yukon-board/full.json"))
    rows = full["submissions"] if isinstance(full, dict) else full
    out = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        sid = str(r.get("id") or r.get("submissionId") or "")
        metrics = r.get("officialMetrics") or {}
        out[sid[:8]] = {
            "status": r.get("status"),
            "score": r.get("score") or metrics.get("score"),
            "created": r.get("createdAt"),
            "n_per_prompt": len(metrics.get("per_prompt") or []),
        }
    return out


if __name__ == "__main__":
    main()
