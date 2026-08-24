#!/usr/bin/env python3
"""E175: price Q (software-pipelined qmm_t) from ranked receipt telemetry.

Q is the four campaign quantized kernel files. The campaign priced it at
+1.00 % from the published-score contrast B/C. This script prices it instead
from the candidate-side per-prompt fields of the same receipts, and measures
the null contrast A/C, whose two trees are functionally identical on the
ranked path.

harness=ranked throughout. Input is a board snapshot with officialMetrics.

Run: python3 research/e175_board_q_decomposition.py [board.json] [--wandb]
"""

from __future__ import annotations

import collections
import glob
import json
import statistics as st
import sys

RECEIPTS = {
    "5a9f130a": ("A", "organizer-pure"),
    "180db842": ("B", "organizer + Q + inert instrumentation"),
    "fda590bb": ("C", "organizer + inert instrumentation"),
    "2c885d64": ("D", "organizer + Q + E165 + strip"),
}

# Receipts whose submitted tree carries the software-pipelined qmm_t.
Q_IN = {"7226dc9a", "180db842", "2c885d64"}

FIELDS = [
    "mtp_seconds_per_token_mean",
    "prefill_seconds_per_token",
    "serial_seconds_per_token_mean",
    "raw_ratio_of_means",
]


def load(path: str | None) -> list[dict]:
    if path is None:
        path = max(glob.glob("/tmp/board_t*.json"), key=lambda p: len(p) + ord(p[-6]))
    with open(path) as fh:
        rows = json.load(fh)["submissions"]
    print(f"board: {path}, {len(rows)} rows")
    return rows


def ours(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if (r.get("note") or "").lower().startswith("model: senpai")
        and (r.get("officialMetrics") or {}).get("per_prompt")
    ]


def by_prompt(rec: dict) -> dict[str, dict]:
    return {
        p["prompt_sha256"][:8]: p for p in rec["officialMetrics"]["per_prompt"]
    }


def paired(a: dict, b: dict, field: str) -> tuple[int, float, float, int]:
    """Per-prompt percent change from a to b, paired on prompt digest."""
    pa, pb = by_prompt(a), by_prompt(b)
    keys = sorted(set(pa) & set(pb))
    deltas = [100.0 * (pb[k][field] / pa[k][field] - 1.0) for k in keys]
    return (
        len(keys),
        st.mean(deltas),
        st.stdev(deltas),
        sum(1 for d in deltas if d > 0),
    )


def main(board_path: str | None = None) -> dict:
    rows = ours(load(board_path))
    index = {r["id"][:8]: r for r in rows}
    out: dict = {}

    print("\n== our receipts, candidate-side telemetry ==")
    header = f"{'id':11}{'when':21}{'score':>12}{'prefill_us':>12}{'cand_ms':>10}{'serial_ms':>11}  Q"
    print(header)
    for r in sorted(rows, key=lambda r: r["createdAt"]):
        rid = r["id"][:8]
        m = r["officialMetrics"]
        tag = RECEIPTS.get(rid, ("", ""))[0]
        print(
            f"{rid + ('/' + tag if tag else ''):11}{r['createdAt'][:19]:21}"
            f"{r['officialScore']:12.6f}"
            f"{m['prefill_seconds_per_token'] * 1e6:12.1f}"
            f"{m['candidate_mtp_seconds_per_token_mean'] * 1e3:10.4f}"
            f"{m['baseline_serial_seconds_per_token_mean'] * 1e3:11.4f}"
            f"  {'IN' if rid in Q_IN else '-'}"
        )

    print("\n== prefill population split by Q ==")
    ins = [
        r["officialMetrics"]["prefill_seconds_per_token"] * 1e6
        for r in rows
        if r["id"][:8] in Q_IN
    ]
    outs = [
        r["officialMetrics"]["prefill_seconds_per_token"] * 1e6
        for r in rows
        if r["id"][:8] not in Q_IN
    ]
    print(
        f"  Q out: n={len(outs)} mean {st.mean(outs):.1f} us sd {st.stdev(outs):.1f} "
        f"range [{min(outs):.1f}, {max(outs):.1f}]"
    )
    print(f"  Q in : n={len(ins)} values " + " ".join(f"{v:.1f}" for v in ins))
    out["prefill_q_out_mean_us"] = st.mean(outs)
    out["prefill_q_out_sd_us"] = st.stdev(outs)
    out["prefill_q_in_min_us"] = min(ins)
    out["prefill_gap_pct"] = 100.0 * (min(ins) / max(outs) - 1.0)
    print(
        f"  lowest Q-in prefill is {out['prefill_gap_pct']:+.3f} % above the "
        f"highest Q-out prefill"
    )

    contrasts = [
        ("5a9f130a", "180db842", "A->B  add Q, else inert"),
        ("5a9f130a", "fda590bb", "A->C  inert only (NULL CONTROL)"),
        ("fda590bb", "180db842", "C->B  add Q given instrumentation"),
    ]
    print("\n== paired per-prompt contrasts, percent change ==")
    for field in FIELDS:
        print(f"  {field}")
        for a, b, label in contrasts:
            if a not in index or b not in index:
                print(f"    {label:34} MISSING")
                continue
            n, mean, sd, pos = paired(index[a], index[b], field)
            print(
                f"    {label:34} n={n} mean {mean:+.4f} %  sd {sd:.4f}  "
                f"same sign {max(pos, n - pos)}/{n}"
            )
            out[f"{field}|{a}->{b}"] = mean

    print("\n== duplicate submitted commits (byte-identical trees) ==")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r.get("submissionCommitSha")].append(r)
    dups = {k: v for k, v in groups.items() if k and len(v) > 1}
    if not dups:
        print("  none")
    for sha, group in dups.items():
        ids = ", ".join(r["id"][:8] for r in group)
        for field in ("prefill_seconds_per_token", "candidate_mtp_seconds_per_token_mean"):
            vals = [r["officialMetrics"][field] for r in group]
            print(
                f"  {sha[:8]} [{ids}] {field}: "
                f"spread {100 * (max(vals) / min(vals) - 1):+.3f} %"
            )
        scores = [r["officialScore"] for r in group]
        print(
            f"      officialScore spread {100 * (max(scores) / min(scores) - 1):+.3f} %"
        )

    print("\n== Q priced on the leg ==")
    a, b = index["5a9f130a"], index["180db842"]
    pre = a["officialMetrics"]["prefill_seconds_per_token"]
    dec = a["officialMetrics"]["candidate_mtp_seconds_per_token_mean"]
    share = pre / (pre + dec)
    _, d_pre, _, _ = paired(a, b, "prefill_seconds_per_token")
    _, d_dec, _, _ = paired(a, b, "mtp_seconds_per_token_mean")
    _, d_pub, _, _ = paired(a, b, "raw_ratio_of_means")
    modelled = -(share * d_pre + (1 - share) * d_dec)
    out.update(
        prefill_share_of_leg=share,
        q_prefill_pct=d_pre,
        q_decode_pct=d_dec,
        q_published_measured_pct=d_pub,
        q_published_modelled_pct=modelled,
    )
    print(f"  prefill share of the candidate leg  {100 * share:.2f} %")
    print(f"  Q prefill                           {d_pre:+.3f} %")
    print(f"  Q decode                            {d_dec:+.3f} %")
    print(f"  modelled published effect           {modelled:+.3f} %")
    print(f"  measured published effect (A->B)    {d_pub:+.3f} %")
    print(
        f"  predicted organizer-pure+Q receipt  "
        f"{a['officialScore'] * (1 + d_pub / 100):.4f}"
    )
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
