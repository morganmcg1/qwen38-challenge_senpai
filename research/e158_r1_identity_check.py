#!/usr/bin/env python3
"""E158 R1 -- test the advisor's corrected round identity on the measured legs.

Advisor ERROR 202 corrected `effective_mean_draft_len` from ACCEPTED to
PROPOSED draft length and supplied a replacement identity:

    rounds = N - acceptedDraftTotal
    R      = decodeSeconds / rounds
    a      = acceptedDraftTotal / rounds
    q      = effective_mean_draft_len
    mtp    = R / (1 + a)

Both identities are checked here against the recorded integers rather than
assumed. `rounds` is reported independently by the driver, so `rounds + acc == N`
is a falsifiable check and not a restatement.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in sorted((ROOT / ".mlxfast-private/e128").glob(
        "runs-e158r1-*/*/report.json"
    )):
        report = json.loads(path.read_text())
        tag = path.parent.parent.name.replace("runs-e158r1-", "")
        n = report["decode_token_count"]
        rounds = report["round_count"]
        acc = report["accepted_draft_total"]
        q = report["effective_mean_draft_len"]
        a = acc / rounds
        r_per_round = report["decode_seconds"] / rounds
        mtp = report["parent_measured_seconds_per_token"]
        rows.append(
            {
                "tag": tag,
                "n": n,
                "rounds": rounds,
                "acc": acc,
                "q": q,
                "a": a,
                "alpha": report["accepted_draft_rate"],
                "R": r_per_round,
                "mtp": mtp,
                "rej": report["rejected_draft_total"],
                "nd": report["non_drafting_round_count"],
                "matched": report["all_tokens_matched"],
                "id_rounds": rounds + acc == n,
                "id_mtp": abs(mtp - r_per_round / (1.0 + a)) / mtp,
                "id_alpha": abs(report["accepted_draft_rate"] - acc / (q * rounds)),
            }
        )

    header = (
        f"{'leg':34s} {'N':>4s} {'rnds':>5s} {'acc':>4s} {'rej':>4s} "
        f"{'q':>7s} {'a':>7s} {'alpha':>7s} {'R':>10s} {'mtp':>10s} "
        f"{'nd':>3s} {'r+a=N':>6s} {'mtp-R/(1+a)':>11s} {'alpha-id':>9s}"
    )
    print(header)
    for x in rows:
        print(
            f"{x['tag']:34s} {x['n']:4d} {x['rounds']:5d} {x['acc']:4d} "
            f"{x['rej']:4d} {x['q']:7.4f} {x['a']:7.4f} {x['alpha']:7.5f} "
            f"{x['R']:10.7f} {x['mtp']:10.7f} {x['nd']:3d} "
            f"{str(x['id_rounds']):>6s} {x['id_mtp']:11.2e} {x['id_alpha']:9.2e}"
        )

    print()
    print("legs                :", len(rows))
    print("all_tokens_matched  :", all(x["matched"] for x in rows))
    print("rounds + acc == N   :", all(x["id_rounds"] for x in rows))
    print("max |mtp - R/(1+a)| :", max(x["id_mtp"] for x in rows))
    print("max alpha residual  :", max(x["id_alpha"] for x in rows))
    print("non-drafting rounds :", sorted({x["nd"] for x in rows}))


if __name__ == "__main__":
    main()
