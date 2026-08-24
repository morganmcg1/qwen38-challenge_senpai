#!/usr/bin/env python3
"""E1: price a uniform candidate-leg time saving through the real median rule.

The published ranked score is NOT the mean of the eight raw ratios.  It is the
median of eight order statistics, and with n=8 that is the mean of the 4th and
5th values after sorting.  A lever that is uniform in milliseconds is therefore
*not* uniform in published score: it moves each prompt's raw by a different
amount, it can reorder the prompts, and only the two central order statistics
are paid.

Usage:
    python3 research/e164_e1_median_lever.py [--board PATH] [--receipt SHA]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_BOARD = "/tmp/yukon-board/full.json"
DELTAS_MS = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def load_board(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("rows", "submissions", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise SystemExit("cannot find row list in board json")
    return data


def short(sha):
    return (sha or "")[:8]


def find_receipt(rows, prefix):
    hits = []
    for row in rows:
        for key in ("submissionId", "id", "submissionCommitSha", "sourceRef"):
            val = row.get(key)
            if isinstance(val, str) and val.startswith(prefix):
                hits.append(row)
                break
    if not hits:
        raise SystemExit(f"receipt {prefix} not found")
    return hits[0]


def per_prompt_cells(row):
    metrics = row.get("officialMetrics") or {}
    cells = metrics.get("per_prompt") or []
    out = []
    for cell in cells:
        out.append(
            {
                "prompt_sha256": cell.get("prompt_sha256"),
                "serial_spt": cell.get("serial_seconds_per_token_mean"),
                "mtp_spt": cell.get("mtp_seconds_per_token_mean"),
                "raw": cell.get("raw_ratio_of_means"),
                "edl": cell.get("effective_mean_draft_len"),
                "non_drafting": cell.get("non_drafting_round_count"),
            }
        )
    return out


PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def label(cell, idx):
    sha = cell.get("prompt_sha256") or ""
    return PROMPT_NAMES.get(sha[:8], f"p{idx}:{sha[:6]}")


def published_from_raws(raws):
    """Median of eight values == mean of the 4th and 5th order statistics."""
    order = sorted(range(len(raws)), key=lambda i: raws[i])
    n = len(raws)
    if n % 2 == 1:
        mid = order[n // 2]
        return raws[mid], (mid,), order
    lo = order[n // 2 - 1]
    hi = order[n // 2]
    return 0.5 * (raws[lo] + raws[hi]), (lo, hi), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=DEFAULT_BOARD)
    ap.add_argument("--receipt", default="5a9f130a")
    ap.add_argument("--reference", default="ec24d591")
    ap.add_argument("--out", default="research/e164-artifacts/e164-e1-median-lever.json")
    args = ap.parse_args()

    rows = load_board(args.board)
    row = find_receipt(rows, args.receipt)
    metrics = row.get("officialMetrics") or {}
    decode_tokens = metrics.get("decode_tokens")
    if not decode_tokens:
        raise SystemExit("decode_tokens missing; cannot convert ms to seconds/token")

    cells = per_prompt_cells(row)
    if len(cells) != 8:
        raise SystemExit(f"expected 8 prompts, got {len(cells)}")

    names = [label(c, i) for i, c in enumerate(cells)]
    serial = [c["serial_spt"] for c in cells]
    mtp0 = [c["mtp_spt"] for c in cells]
    raw0 = [s / m for s, m in zip(serial, mtp0)]

    # Consistency: board raw_ratio_of_means must equal serial/mtp.
    max_raw_err = max(abs(r - c["raw"]) for r, c in zip(raw0, cells))

    base_pub, base_central, base_order = published_from_raws(raw0)
    board_score = row.get("officialScore")

    report = []
    report.append("=" * 78)
    report.append("E1  uniform candidate-leg lever priced through the real median rule")
    report.append("=" * 78)
    report.append(f"receipt          : {short(row.get('id') or row.get('submissionId'))}")
    report.append(f"solver           : {row.get('solverUsername')}")
    report.append(f"commit           : {short(row.get('submissionCommitSha'))}")
    report.append(f"status           : {row.get('status')}  createdAt {row.get('createdAt')}")
    report.append(f"decode_tokens    : {decode_tokens}")
    report.append(f"pairs_per_prompt : {metrics.get('pairs_per_prompt')}")
    report.append(f"board officialScore          : {board_score!r}")
    report.append(f"recomputed published (delta=0): {base_pub:.9f}")
    report.append(f"max |serial/mtp - raw_ratio_of_means| = {max_raw_err:.3e}")
    report.append("")
    report.append("baseline per-prompt state (sorted by raw)")
    report.append(
        f"  {'rank':>4}  {'prompt':<16} {'serial_spt':>12} {'mtp_spt':>12} "
        f"{'cand_leg_ms':>12} {'raw':>10}  central"
    )
    for rank, idx in enumerate(base_order, start=1):
        leg_ms = mtp0[idx] * decode_tokens * 1000.0
        mark = "<-- paid" if idx in base_central else ""
        report.append(
            f"  {rank:>4}  {names[idx]:<16} {serial[idx]:>12.7f} {mtp0[idx]:>12.7f} "
            f"{leg_ms:>12.3f} {raw0[idx]:>10.6f}  {mark}"
        )
    report.append("")

    results = {
        "receipt": row.get("submissionId") or row.get("id"),
        "solver": row.get("solverName") or row.get("solver"),
        "decode_tokens": decode_tokens,
        "board_official_score": board_score,
        "baseline_published": base_pub,
        "baseline_central_prompts": [names[i] for i in base_central],
        "baseline_order": [names[i] for i in base_order],
        "max_raw_identity_error": max_raw_err,
        "prompts": [
            {
                "name": names[i],
                "serial_spt": serial[i],
                "mtp_spt": mtp0[i],
                "cand_leg_ms": mtp0[i] * decode_tokens * 1000.0,
                "raw": raw0[i],
                "edl": cells[i]["edl"],
                "non_drafting": cells[i]["non_drafting"],
            }
            for i in range(8)
        ],
        "sweep": [],
    }

    report.append("sweep: uniform -delta ms removed from the candidate leg of ALL 8 prompts")
    report.append(
        f"  {'delta_ms':>8} {'published':>11} {'d_pub':>10} {'d_pub_%':>9} "
        f"{'per_ms':>10}  {'central prompts':<34} {'reorder?'}"
    )
    prev_central = set(base_central)
    crossings = []
    for delta_ms in DELTAS_MS:
        dspt = (delta_ms / 1000.0) / decode_tokens
        mtp_new = [m - dspt for m in mtp0]
        if min(mtp_new) <= 0:
            raise SystemExit("delta exceeds a candidate leg time")
        raw_new = [s / m for s, m in zip(serial, mtp_new)]
        pub, central, order = published_from_raws(raw_new)
        d_pub = pub - base_pub
        central_names = ", ".join(sorted(names[i] for i in central))
        changed = set(central) != set(base_central)
        order_changed = [names[i] for i in order] != [names[i] for i in base_order]
        tag = ""
        if changed:
            tag = "CENTRAL PAIR CHANGED"
        elif order_changed:
            tag = "order moved (central pair same)"
        if changed:
            crossings.append(
                {
                    "delta_ms": delta_ms,
                    "from": [names[i] for i in base_central],
                    "to": [names[i] for i in central],
                }
            )
        report.append(
            f"  {delta_ms:>8.1f} {pub:>11.6f} {d_pub:>+10.6f} "
            f"{100.0 * d_pub / base_pub:>+8.4f}% {d_pub / delta_ms:>10.6f}  "
            f"{central_names:<34} {tag}"
        )
        results["sweep"].append(
            {
                "delta_ms": delta_ms,
                "published": pub,
                "delta_published": d_pub,
                "delta_published_pct": 100.0 * d_pub / base_pub,
                "published_per_ms": d_pub / delta_ms,
                "central_prompts": sorted(names[i] for i in central),
                "central_pair_changed": changed,
                "sort_order_changed": order_changed,
                "order": [names[i] for i in order],
            }
        )
        prev_central = set(central)

    report.append("")

    # Fine scan to locate the exact crossing deltas.
    report.append("fine scan for order-statistic crossings (0.1 ms grid, 0..40 ms)")
    fine = []
    last_pair = tuple(sorted(names[i] for i in base_central))
    last_order = tuple(names[i] for i in base_order)
    cross_events = []
    d = 0.0
    while d <= 40.0 + 1e-9:
        dspt = (d / 1000.0) / decode_tokens
        mtp_new = [m - dspt for m in mtp0]
        if min(mtp_new) <= 0:
            break
        raw_new = [s / m for s, m in zip(serial, mtp_new)]
        pub, central, order = published_from_raws(raw_new)
        pair = tuple(sorted(names[i] for i in central))
        ordn = tuple(names[i] for i in order)
        if pair != last_pair:
            cross_events.append(
                {"delta_ms": round(d, 3), "kind": "central_pair", "from": list(last_pair), "to": list(pair)}
            )
            last_pair = pair
        if ordn != last_order:
            moved = [
                (r + 1, last_order[r], ordn[r])
                for r in range(8)
                if last_order[r] != ordn[r]
            ]
            cross_events.append(
                {"delta_ms": round(d, 3), "kind": "sort_order", "swaps": moved}
            )
            last_order = ordn
        fine.append((round(d, 3), pub))
        d += 0.1
    if cross_events:
        for ev in cross_events:
            if ev["kind"] == "central_pair":
                report.append(
                    f"  delta={ev['delta_ms']:>6.1f} ms  CENTRAL PAIR "
                    f"{ev['from']} -> {ev['to']}"
                )
            else:
                sw = "; ".join(f"rank{r}: {a}->{b}" for r, a, b in ev["swaps"])
                report.append(f"  delta={ev['delta_ms']:>6.1f} ms  sort order changed  {sw}")
    else:
        report.append("  none: the 4th/5th order statistics keep the same identity over 0..40 ms")
    results["crossing_events"] = cross_events
    report.append("")

    # Exact crossing delta for every adjacent pair in the baseline order.
    # raw_i(d) > raw_j(d)  <=>  s_i*m_j - s_j*m_i > (d/T)*(s_i - s_j)
    # so the crossing delta is  d* = T*(s_i*m_j - s_j*m_i)/(s_i - s_j).
    report.append("exact crossing delta for each adjacent pair (closed form, ms)")
    report.append(
        "  raw_i(d) = s_i/(m_i - d/T) so raw_i > raw_j  <=>  "
        "s_i*m_j - s_j*m_i > (d/T)*(s_i - s_j)"
    )
    adjacent = []
    for rank in range(7):
        i = base_order[rank]
        j = base_order[rank + 1]
        num = serial[i] * mtp0[j] - serial[j] * mtp0[i]
        den = serial[i] - serial[j]
        if abs(den) < 1e-15:
            dstar = None
            txt = "never (identical serial legs)"
        else:
            dstar = decode_tokens * num / den * 1000.0
            txt = f"{dstar:+.1f} ms"
            if dstar < 0:
                txt += "  (wrong sign: a saving never causes this swap)"
        report.append(f"  rank{rank + 1}/{rank + 2}  {names[i]:<10} vs {names[j]:<10}  d* = {txt}")
        adjacent.append(
            {"lower": names[i], "upper": names[j], "crossing_delta_ms": dstar}
        )
    results["adjacent_crossing_deltas_ms"] = adjacent
    max_evaluable = min(mtp0) * decode_tokens * 1000.0
    positive = [a["crossing_delta_ms"] for a in adjacent if a["crossing_delta_ms"] and a["crossing_delta_ms"] > 0]
    smallest_positive = min(positive) if positive else None
    report.append(
        f"  max evaluable delta = shortest candidate leg = {max_evaluable:.1f} ms "
        f"(a larger saving would make a leg non-positive)"
    )
    if smallest_positive is None:
        report.append("  no adjacent swap has a positive crossing delta: the order is fixed.")
    elif smallest_positive > max_evaluable:
        report.append(
            f"  smallest positive crossing delta = {smallest_positive:.1f} ms > "
            f"{max_evaluable:.1f} ms, so NO reordering is reachable by any uniform saving."
        )
    else:
        report.append(f"  smallest reachable crossing delta = {smallest_positive:.1f} ms")
    results["max_evaluable_delta_ms"] = max_evaluable
    results["smallest_positive_crossing_delta_ms"] = smallest_positive
    results["any_reachable_crossing"] = bool(
        smallest_positive is not None and smallest_positive <= max_evaluable
    )
    report.append("")

    # Extended sweep: how far does the lever have to go before it matters?
    prefill_probe = metrics.get("prefill_seconds_per_token")
    prefill_leg_ms = None
    if prefill_probe:
        prefill_leg_ms = prefill_probe * 512 * 1000.0
    extended = [1.1396, 50.0, 100.0, 250.0]
    if prefill_leg_ms:
        extended.append(round(prefill_leg_ms, 3))
    report.append("extended sweep: larger levers, including full prefill removal")
    if prefill_leg_ms:
        report.append(
            f"  published prefill probe = {prefill_probe:.9f} s/seed-token "
            f"-> {prefill_leg_ms:.2f} ms over 512 seed tokens"
        )
    report.append(
        f"  {'delta_ms':>10} {'published':>11} {'d_pub':>10} {'d_pub_%':>9} "
        f"{'per_ms':>10}  {'central prompts':<26} {'note'}"
    )
    ext_rows = []
    for delta_ms in extended:
        dspt = (delta_ms / 1000.0) / decode_tokens
        mtp_new = [m - dspt for m in mtp0]
        if min(mtp_new) <= 0:
            report.append(f"  {delta_ms:>10.3f}  delta exceeds a candidate leg; not evaluable")
            continue
        raw_new = [s / m for s, m in zip(serial, mtp_new)]
        pub, central, order = published_from_raws(raw_new)
        d_pub = pub - base_pub
        note = ""
        if set(central) != set(base_central):
            note = "CENTRAL PAIR CHANGED"
        elif [names[i] for i in order] != [names[i] for i in base_order]:
            note = "order moved"
        if prefill_leg_ms and abs(delta_ms - prefill_leg_ms) < 1e-6:
            note = ("full prefill removal (upper bound) " + note).strip()
        if abs(delta_ms - 1.1396) < 1e-9:
            note = ("measured ours-vs-ec24d591 prefill gap " + note).strip()
        report.append(
            f"  {delta_ms:>10.3f} {pub:>11.6f} {d_pub:>+10.6f} "
            f"{100.0 * d_pub / base_pub:>+8.4f}% {d_pub / delta_ms:>10.6f}  "
            f"{', '.join(sorted(names[i] for i in central)):<26} {note}"
        )
        ext_rows.append(
            {
                "delta_ms": delta_ms,
                "published": pub,
                "delta_published": d_pub,
                "delta_published_pct": 100.0 * d_pub / base_pub,
                "published_per_ms": d_pub / delta_ms,
                "central_prompts": sorted(names[i] for i in central),
                "central_pair_changed": set(central) != set(base_central),
                "note": note,
            }
        )
    results["extended_sweep"] = ext_rows
    results["prefill_probe_seconds_per_seed_token"] = prefill_probe
    results["prefill_leg_ms_if_512_seed_tokens"] = prefill_leg_ms
    report.append("")

    # Naive alternatives that the median rule is often confused with.
    report.append("what the median rule costs you versus the two common shortcuts")
    report.append(
        f"  {'delta_ms':>8} {'median rule':>12} {'mean-of-raw':>12} {'central-only':>13} "
        f"{'median/mean':>12}"
    )
    shortcut = []
    base_mean = sum(raw0) / 8.0
    for delta_ms in DELTAS_MS:
        dspt = (delta_ms / 1000.0) / decode_tokens
        mtp_new = [m - dspt for m in mtp0]
        raw_new = [s / m for s, m in zip(serial, mtp_new)]
        pub, _, _ = published_from_raws(raw_new)
        d_true = pub - base_pub
        d_mean = sum(raw_new) / 8.0 - base_mean
        # "central-only": pretend the baseline central pair keeps paying.
        lo, hi = base_central
        d_frozen = 0.5 * (raw_new[lo] + raw_new[hi]) - base_pub
        report.append(
            f"  {delta_ms:>8.1f} {d_true:>+12.6f} {d_mean:>+12.6f} {d_frozen:>+13.6f} "
            f"{d_true / d_mean if d_mean else float('nan'):>12.4f}"
        )
        shortcut.append(
            {
                "delta_ms": delta_ms,
                "median_rule_delta": d_true,
                "mean_of_raw_delta": d_mean,
                "frozen_central_delta": d_frozen,
                "median_over_mean": d_true / d_mean if d_mean else None,
            }
        )
    results["shortcut_comparison"] = shortcut
    report.append("")

    # Local sensitivity: analytic derivative at delta=0.
    # raw_i = s_i / (m_i - d/T). d raw_i / d d_ms = s_i / (T*1000*m_i^2)
    deriv = [serial[i] / (decode_tokens * 1000.0 * mtp0[i] ** 2) for i in range(8)]
    lo, hi = base_central
    analytic = 0.5 * (deriv[lo] + deriv[hi])
    report.append("analytic marginal published score per ms at delta=0")
    report.append(f"  d(published)/d(delta_ms) = {analytic:.6f}   (mean of the two central prompts)")
    report.append(f"  per-prompt d(raw)/d(delta_ms):")
    for rank, idx in enumerate(base_order, start=1):
        mark = "  <-- paid" if idx in base_central else ""
        report.append(f"    rank{rank} {names[idx]:<16} {deriv[idx]:.6f}{mark}")
    spread = max(deriv) / min(deriv)
    report.append(f"  spread max/min across prompts = {spread:.4f}x")
    report.append(
        "  a uniform ms lever is NOT uniform in raw: the fastest-candidate prompt "
        "gains the most raw per ms."
    )
    results["marginal"] = {
        "analytic_published_per_ms_at_zero": analytic,
        "per_prompt_raw_per_ms": {names[i]: deriv[i] for i in range(8)},
        "spread_max_over_min": spread,
    }
    report.append("")

    # How much of the published gain is "paid" versus "wasted" on non-central prompts.
    report.append("who pays: share of the uniform lever that reaches the published number")
    for delta_ms in DELTAS_MS:
        dspt = (delta_ms / 1000.0) / decode_tokens
        mtp_new = [m - dspt for m in mtp0]
        raw_new = [s / m for s, m in zip(serial, mtp_new)]
        pub, central, _ = published_from_raws(raw_new)
        total_raw_gain = sum(raw_new[i] - raw0[i] for i in range(8))
        paid = sum(raw_new[i] - raw0[i] for i in central) / 2.0
        report.append(
            f"  delta={delta_ms:>5.1f} ms: sum of all 8 raw gains = {total_raw_gain:+.6f}, "
            f"published gain = {pub - base_pub:+.6f} "
            f"({100.0 * (pub - base_pub) / total_raw_gain:.1f}% of the total raw movement)"
        )
    report.append("")

    # Gap accounting against the promoted frontier receipt.
    ref = None
    for cand in rows:
        if str(cand.get("id", "")).startswith(args.reference):
            ref = cand
            break
    if ref is not None:
        ref_score = ref.get("officialScore")
        ref_pref = (ref.get("officialMetrics") or {}).get("prefill_seconds_per_token")
        gap = ref_score - base_pub

        def pub_at(d):
            dspt = (d / 1000.0) / decode_tokens
            r = sorted(s / (m - dspt) for s, m in zip(serial, mtp0))
            return 0.5 * (r[3] + r[4])

        lo_d, hi_d = 0.0, max_evaluable * 0.999
        for _ in range(200):
            mid = 0.5 * (lo_d + hi_d)
            if pub_at(mid) < ref_score:
                lo_d = mid
            else:
                hi_d = mid
        need_ms = lo_d
        report.append("gap accounting versus the promoted frontier receipt")
        report.append(f"  reference          : {short(ref.get('id'))} score {ref_score:.9f}")
        report.append(f"  ours               : {short(row.get('id'))} score {base_pub:.9f}")
        report.append(f"  published gap      : {gap:+.9f}")
        report.append(
            f"  uniform candidate-leg saving needed to close it: {need_ms:.3f} ms"
        )
        pref_gap_ms = None
        if prefill_probe and ref_pref:
            pref_gap_ms = (prefill_probe - ref_pref) * 512 * 1000.0
            share = 100.0 * pref_gap_ms / need_ms
            report.append(
                f"  measured prefill-probe gap (ours - reference): {pref_gap_ms:+.3f} ms "
                f"= {share:.2f}% of the {need_ms:.1f} ms needed"
            )
            report.append(
                f"  that prefill gap priced through the median rule: "
                f"{pub_at(pref_gap_ms) - base_pub:+.6f} published "
                f"({100.0 * (pub_at(pref_gap_ms) - base_pub) / gap:.2f}% of the gap)"
            )
        results["gap_accounting"] = {
            "reference_receipt": ref.get("id"),
            "reference_score": ref_score,
            "our_score": base_pub,
            "published_gap": gap,
            "uniform_ms_needed_to_close": need_ms,
            "prefill_probe_gap_ms": pref_gap_ms,
            "prefill_share_of_needed_ms_pct": (
                100.0 * pref_gap_ms / need_ms if pref_gap_ms is not None else None
            ),
            "prefill_share_of_published_gap_pct": (
                100.0 * (pub_at(pref_gap_ms) - base_pub) / gap
                if pref_gap_ms is not None
                else None
            ),
        }
        report.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    text = "\n".join(report)
    print(text)
    with open(args.out.replace(".json", ".txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
