"""E128 F7 item 3: stratify the board submissions by their QMV width-dispatch
table and fit the ranked round-cost curve inside each stratum.

Model P (advisor):  T(M) = a + b*M + f*G(M),   G(M) = passes at width M.
                    b is a shared per-row slope, f is the ranked price of one
                    extra QMV pass. The break in the observed cost curve moves
                    with the stratum's own dispatch table.

Model R (student):  T(M) = a + b_lo*M below the regime change and
                    a' + b_hi*M above it, with no pass term. The break sits at
                    the same M in every stratum.

The two models are separated by whether the fitted break follows the stratum's
G(M) vector or stays put.

Inputs
  /tmp/tree_ids.json        submission id -> git tree sha (456 trees, local)
  /tmp/yukon-board/full.json board rows with officialMetrics.per_prompt
"""

import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "research")
from rankedcurve import PROMPTS, ROUNDS, T  # noqa: E402

REPO = "/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-edward/workspace/target"
QH = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"

CROSSROW_M_RE = re.compile(r"qmv_fast_crossrow_affine4_g64_m<\s*T\s*,\s*(\d+)\s*,\s*(\d+)\s*>")
CROSSROW_PLAIN_RE = re.compile(r"qmv_fast_crossrow_affine4_g64<\s*T\s*,\s*(\d+)\s*>")
E120_RE = re.compile(r"qwen_e120_qmv_m<\s*(\d+)\s*,\s*(\d+)\s*")
MAX_M = 9


def cat(tree, path):
    p = subprocess.run(
        ["git", "cat-file", "-p", f"{tree}:{path}"], cwd=REPO, capture_output=True
    )
    return None if p.returncode != 0 else p.stdout.decode("utf-8", "replace")


def split_blocks(pairs):
    """Split an ordered (M, IPG) match list into ascending-M switch blocks."""
    blocks, cur = [], []
    for m, ipg in pairs:
        if cur and m <= cur[-1][0]:
            blocks.append(cur)
            cur = []
        cur.append((m, ipg))
    if cur:
        blocks.append(cur)
    return blocks


def ipg_table(tree):
    """Return the wide-output (out_vec_size >= 4096) inputs-per-group table."""
    text = cat(tree, QH)
    if text is None:
        return None, "no-quantized.h"
    pairs = [(int(a), int(b)) for a, b in CROSSROW_M_RE.findall(text)]
    if not pairs:
        if CROSSROW_PLAIN_RE.search(text):
            return None, "crossrow-no-m-table"
        return None, "no-crossrow"
    blocks = split_blocks(pairs)
    tbl = dict(blocks[0])
    for m, _ in CROSSROW_PLAIN_RE.findall(text)[:0]:
        pass
    plain = {int(a) for a in CROSSROW_PLAIN_RE.findall(text)}
    for m in plain:
        tbl.setdefault(m, m)
    return tbl, f"blocks={len(blocks)}"


def passes(tbl):
    """G(M) for M = 1..9. Widths outside the table run one pass."""
    out = {}
    for m in range(1, MAX_M + 1):
        ipg = tbl.get(m, m)
        out[m] = math.ceil(m / max(ipg, 1))
    return out


def load_board():
    rows = json.load(open("/tmp/yukon-board/full.json"))
    if isinstance(rows, dict):
        rows = rows.get("submissions") or rows.get("rows") or list(rows.values())[0]
    return rows


def round_points(row):
    """Per-prompt (mean width, round microseconds) for one board row."""
    om = row.get("officialMetrics") or {}
    pp = om.get("per_prompt") or []
    out = {}
    for p in pp:
        name = PROMPTS.get((p.get("prompt_sha256") or "")[:8])
        if name is None:
            return None
        spt = p.get("mtp_seconds_per_token_mean")
        dl = p.get("effective_mean_draft_len")
        if spt is None or dl is None:
            return None
        r = ROUNDS[name]
        out[name] = {
            "mbar": dl + 1.0,
            "round_us": T * spt / r * 1e6,
            "spt": spt,
            "draft_len": dl,
        }
    return out if len(out) == 8 else None


def ols(a, y):
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    return beta, float(np.sqrt(np.mean(resid**2))), resid


def main():
    ids = json.load(open("/tmp/tree_ids.json"))["full"]
    board = load_board()
    by_id = {r.get("id"): r for r in board if r.get("id")}

    tables, reasons = {}, Counter()
    for i, (sid, tree) in enumerate(sorted(ids.items())):
        tbl, why = ipg_table(tree)
        reasons[why] += 1
        if tbl:
            tables[sid] = tbl
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(ids)}", file=sys.stderr)

    print("dispatch-table extraction:")
    for why, n in reasons.most_common():
        print(f"  {n:4d}  {why}")

    strata = defaultdict(list)
    for sid, tbl in tables.items():
        g = passes(tbl)
        key = ",".join(str(g[m]) for m in range(1, MAX_M + 1))
        strata[key].append(sid)

    print(f"\nstrata by G(1..9) over {len(tables)} submissions with a table:")
    for key, sids in sorted(strata.items(), key=lambda kv: -len(kv[1])):
        ipgs = tables[sids[0]]
        ipgstr = " ".join(f"{m}:{ipgs.get(m,m)}" for m in range(3, MAX_M + 1))
        print(f"  n={len(sids):4d}  G={key}   IPG[{ipgstr}]")

    usable = defaultdict(list)
    for key, sids in strata.items():
        for sid in sids:
            row = by_id.get(sid)
            if row is None:
                continue
            pts = round_points(row)
            if pts is None:
                continue
            usable[key].append((sid, pts))
    print("\nstrata with complete per-prompt board evidence:")
    for key, v in sorted(usable.items(), key=lambda kv: -len(kv[1])):
        print(f"  n={len(v):4d}  G={key}")

    json.dump(
        {
            "tables": {k: {str(m): v for m, v in t.items()} for k, t in tables.items()},
            "strata": {k: v for k, v in strata.items()},
            "usable_counts": {k: len(v) for k, v in usable.items()},
        },
        open("/tmp/e128_strata.json", "w"),
    )
    print("\nwrote /tmp/e128_strata.json")


if __name__ == "__main__":
    main()
