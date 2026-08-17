#!/usr/bin/env python3
"""Research-only (qwen38-r1-e11-depth-lever-showdown): render the arm table
from the collector's --json-out, scored against the C replicates.

C1/C2 are byte-identical builds, so their spread is the pure timing noise
floor; every arm is judged against it rather than against a single run.

usage: research/e11_table.py /tmp/e11-t1.json [--ref C1 C2]
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_in", type=Path)
    ap.add_argument("--ref", nargs="+", default=["C1", "C2"])
    args = ap.parse_args()

    rows = {r["label"]: r for r in json.loads(args.json_in.read_text())}
    ref = [rows[a] for a in args.ref if a in rows]
    if not ref:
        sys.exit("no reference arms present")
    ref_mtp = sum(r["mtp_seconds_per_token"] for r in ref) / len(ref)
    ref_ser = sum(r["serial_seconds_per_token"] for r in ref) / len(ref)
    if len(ref) > 1:
        lo = min(r["mtp_seconds_per_token"] for r in ref)
        hi = max(r["mtp_seconds_per_token"] for r in ref)
        floor = 100 * (hi - lo) / lo
    else:
        floor = float("nan")

    hdr = (f'{"arm":<4} {"MTP s/tok":>11} {"vs ref%":>8} {"ratio":>7} '
           f'{"drift%":>7} {"rnds":>5} {"rows":>5} {"mean":>6} {"accR":>7} '
           f'{"rej":>4} {"maxd":>4} {"match":>6} {"rowchk":>7}')
    print(hdr)
    print("-" * len(hdr))
    for arm, r in rows.items():
        d = 100 * (r["mtp_seconds_per_token"] - ref_mtp) / ref_mtp
        drift = 100 * (r["serial_seconds_per_token"] - ref_ser) / ref_ser
        chk = "OK" if r["declared_rows_total"] == r["reference_checked_row_total"] else "MISMATCH"
        print(f'{arm:<4} {r["mtp_seconds_per_token"]:>11.6f} {d:>+8.3f} '
              f'{r["mtp_decode_speedup"]:>7.4f} {drift:>+7.3f} '
              f'{r["round_count"]:>5} {r["declared_rows_total"]:>5} '
              f'{r["effective_mean_draft_len"]:>6.3f} {r["accepted_draft_rate"]:>7.4f} '
              f'{r["rejected_draft_total"]:>4} {r["effective_max_draft_len"]:>4} '
              f'{str(r["all_tokens_matched"]):>6} {chk:>7}')

    print(f"\nnoise floor (ref spread on MTP s/tok): {floor:.3f}%")
    print(f"reference mean MTP s/tok: {ref_mtp:.9f}   serial: {ref_ser:.9f}\n")

    for arm, r in rows.items():
        print(f'{arm:<4} hist={r["parent_depth_hist"]} replay={r["verify_block_replayed_round_count"]} '
              f'resid={r["residual_divergence_count"]} head={r["head_provenance_sha256"][:8]} '
              f'pass={r["pass"]} env="{r["mlx_qwen_env"]}"')

    # Two-parameter cost fit: decode_seconds = a*rounds + b*rows.
    # Exactly determined at n=2, so only meaningful once a third arm exists.
    print()
    names = list(rows)
    if len(names) >= 2:
        import itertools
        fits = []
        for x, y in itertools.combinations(names, 2):
            rx, ry = rows[x], rows[y]
            n1, w1 = rx["round_count"], rx["declared_rows_total"]
            n2, w2 = ry["round_count"], ry["declared_rows_total"]
            t1 = rx["mtp_seconds_per_token"] * rx["decode_tokens"]
            t2 = ry["mtp_seconds_per_token"] * ry["decode_tokens"]
            det = n1 * w2 - n2 * w1
            if abs(det) < 1e-9:
                continue
            a = (t1 * w2 - t2 * w1) / det
            b = (n1 * t2 - n2 * t1) / det
            fits.append((f"{x}/{y}", a, b))
        print(f'{"pair":<10} {"s/round":>10} {"s/row":>10}')
        for nm, a, b in fits:
            print(f"{nm:<10} {a:>10.6f} {b:>10.6f}")


if __name__ == "__main__":
    main()
