#!/usr/bin/env python3
"""qwen38-r1-e20: turn per-forward attribution records into the four-way split.

Reads one or more arm directories produced by research/e20-run.sh and reports,
per decode width M:

  * the four assignment families (gdn / full_attention / mlp / head+top_two),
  * their seconds and share of verify-side decode work,
  * the unattributed residual, which is the closure check,

plus the width histogram, the prefill-exclusive decode wall clock, and the
correctness fields the assignment requires per timed arm.

Prefill is subtracted explicitly: the trusted report's `decode_seconds` charges
the 512-token seed to the same window, so any share-of-decode taken against it
without subtracting `seed_prefill_seconds` is wrong by that fraction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

BUCKETS = ["embed", "gdn", "full_attention", "mlp", "head", "top_two"]
# The assignment's four families. `embed` is reported separately rather than
# folded into one of them, because it is neither a layer family nor readout.
FAMILIES = {
    "gdn": ["gdn"],
    "full_attention": ["full_attention"],
    "mlp": ["mlp"],
    "head_and_top_two": ["head", "top_two"],
}

FWD = re.compile(r"^qwen-attrib: (.*)$")
SPAN = re.compile(r"^qwen-attrib-span: (.*)$")


def parse_kv(body: str) -> dict[str, int]:
    out = {}
    for tok in body.split():
        k, _, v = tok.partition("=")
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out


def load_attrib(path: str) -> list[dict[str, int]]:
    forwards: list[dict[str, int]] = []
    spans: dict[int, dict[str, int]] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = FWD.match(line.strip())
            if m:
                forwards.append(parse_kv(m.group(1)))
                continue
            m = SPAN.match(line.strip())
            if m:
                rec = parse_kv(m.group(1))
                spans.setdefault(rec.get("f", -1), {}).update(
                    {k: v for k, v in rec.items() if k.endswith("_ns")}
                )
    for rec in forwards:
        rec.update(spans.get(rec.get("f", -1), {}))
    return forwards


def find_arm(arm_dir: str) -> dict:
    """Locate the width-varied MTP decode leg and its trusted report."""
    reports = os.path.join(arm_dir, "reports")
    out: dict = {"arm_dir": arm_dir, "meta": {}}
    meta_path = os.path.join(arm_dir, "meta.txt")
    if os.path.exists(meta_path):
        for line in open(meta_path):
            k, _, v = line.strip().partition("=")
            out["meta"][k] = v
    if not os.path.isdir(reports):
        return out
    names = sorted(os.listdir(reports))
    depth = out["meta"].get("offered_depth", "")
    # e20-cli.sh stamps the offered depth into the stem, so the speculative leg
    # is named rather than guessed. Depth 0 is the serial control.
    for n in names:
        if n.endswith(".json") and f"-mtp-timed-d{depth}" in n and depth != "0":
            out["report_path"] = os.path.join(reports, n)
            stem = n[: -len(".json")]
            out["attrib_paths"] = [
                os.path.join(reports, m)
                for m in names
                if m.startswith(stem + "-attrib.")
            ]
        if n.endswith(".json") and "-mtp-timed-d0" in n:
            out["serial_report_path"] = os.path.join(reports, n)
    return out


def summarize(forwards: list[dict[str, int]], min_count: int) -> dict:
    by_m: dict[int, list[dict[str, int]]] = defaultdict(list)
    for rec in forwards:
        by_m[rec.get("rows", -1)].append(rec)

    rows = {}
    for m, recs in sorted(by_m.items()):
        total = [r["total_ns"] for r in recs if "total_ns" in r]
        entry: dict = {
            "m": m,
            "forwards": len(recs),
            "total_ns_mean": statistics.mean(total) if total else None,
            "total_ns_median": statistics.median(total) if total else None,
        }
        attributed = 0.0
        for b in BUCKETS:
            vals = [r.get(f"{b}_ns", 0) for r in recs]
            calls = [r.get(f"{b}_n", 0) for r in recs]
            entry[f"{b}_ns_mean"] = statistics.mean(vals) if vals else 0.0
            entry[f"{b}_calls_mean"] = statistics.mean(calls) if calls else 0.0
            attributed += entry[f"{b}_ns_mean"]
        entry["attributed_ns_mean"] = attributed
        if entry["total_ns_mean"]:
            # top_two runs after endForward, so it is outside total_ns.
            in_forward = attributed - entry["top_two_ns_mean"]
            entry["residual_ns_mean"] = entry["total_ns_mean"] - in_forward
            entry["residual_frac"] = entry["residual_ns_mean"] / entry["total_ns_mean"]
        denom = attributed
        for fam, parts in FAMILIES.items():
            fam_ns = sum(entry[f"{p}_ns_mean"] for p in parts)
            entry[f"{fam}_ns_mean"] = fam_ns
            entry[f"{fam}_share"] = fam_ns / denom if denom else None
        entry["embed_share"] = entry["embed_ns_mean"] / denom if denom else None
        rows[m] = entry

    kept = {m: e for m, e in rows.items() if e["forwards"] >= min_count}
    pooled = None
    if kept:
        # Weight by observed forward count: this is the split "as run", and the
        # width provenance of that weighting is the run's own histogram.
        w = {m: e["forwards"] for m, e in kept.items()}
        wt = sum(w.values())
        pooled = {"forwards": wt, "widths": sorted(kept)}
        for fam in list(FAMILIES) + ["embed"]:
            num = sum(kept[m][f"{fam}_ns_mean"] * w[m] for m in kept)
            den = sum(kept[m]["attributed_ns_mean"] * w[m] for m in kept)
            pooled[f"{fam}_share"] = num / den if den else None
    return {"by_m": rows, "pooled": pooled}


def report_fields(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path))
    except Exception:
        return {}
    keep = [
        "all_tokens_matched",
        "parity_all_ok",
        "residual_divergence_count",
        "max_rejected_tail_logit_delta",
        "accepted_draft_rate",
        "accepted_draft_total",
        "rejected_draft_total",
        "round_count",
        "seed_token_count",
        "decode_token_count",
        "decode_seconds",
        "seed_prefill_seconds",
        "parent_measured_seconds_per_token",
        "declared_rows_total",
        "reference_checked_row_total",
        "emitted_token_total",
        "target_tail_total",
        "uses_native_mtp_head",
    ]
    out = {k: d[k] for k in keep if k in d}
    if "decode_seconds" in out and "seed_prefill_seconds" in out:
        net = out["decode_seconds"] - out["seed_prefill_seconds"]
        out["decode_seconds_ex_prefill"] = net
        if out.get("decode_token_count"):
            out["sec_per_token_ex_prefill"] = net / out["decode_token_count"]
        if out["decode_seconds"]:
            out["prefill_share_of_charged_window"] = (
                out["seed_prefill_seconds"] / out["decode_seconds"]
            )
    # The trusted row ledger carries the width histogram independently of the
    # research instrumentation; disagreement between them is a real defect.
    ledger = d.get("row_ledger") or []
    if ledger:
        per_round: dict[int, int] = defaultdict(int)
        for row in ledger:
            per_round[row["round"]] += 1
        hist: dict[int, int] = defaultdict(int)
        for _, n in per_round.items():
            hist[n] += 1
        out["ledger_width_histogram"] = dict(sorted(hist.items()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    result = {}
    for arm_dir in args.arms:
        label = os.path.basename(arm_dir.rstrip("/"))
        info = find_arm(arm_dir)
        forwards: list[dict[str, int]] = []
        for p in info.get("attrib_paths", []):
            forwards.extend(load_attrib(p))
        entry = {
            "meta": info["meta"],
            "attrib_files": [os.path.basename(p) for p in info.get("attrib_paths", [])],
            "mtp": report_fields(info.get("report_path")),
            "serial": report_fields(info.get("serial_report_path")),
            "attrib": summarize(forwards, args.min_count) if forwards else None,
        }
        result[label] = entry

        print(f"\n===== {label} =====")
        m = entry["meta"]
        print(
            f"build={m.get('build')} mode={m.get('attrib_mode')} "
            f"depth={m.get('offered_depth')} tokens={m.get('tokens')} "
            f"dirty={m.get('dirty')} exit={m.get('exit')}"
        )
        print(f"  thermal_before: {m.get('thermal_before')}")
        print(f"  thermal_after : {m.get('thermal_after')}")
        r = entry["mtp"]
        if r:
            print(
                "  MTP leg: decode_s=%.4f prefill_s=%.4f net_s=%.4f "
                "s/tok_net=%.6f rounds=%s"
                % (
                    r.get("decode_seconds", float("nan")),
                    r.get("seed_prefill_seconds", float("nan")),
                    r.get("decode_seconds_ex_prefill", float("nan")),
                    r.get("sec_per_token_ex_prefill", float("nan")),
                    r.get("round_count"),
                )
            )
            print(
                "  correctness: matched=%s parity=%s residual_div=%s "
                "max_rej_tail_delta=%s accept_rate=%s"
                % (
                    r.get("all_tokens_matched"),
                    r.get("parity_all_ok"),
                    r.get("residual_divergence_count"),
                    r.get("max_rejected_tail_logit_delta"),
                    r.get("accepted_draft_rate"),
                )
            )
            print(f"  ledger width histogram: {r.get('ledger_width_histogram')}")
        s = entry["serial"]
        if s:
            print(
                "  serial leg: decode_s=%.4f prefill_s=%.4f net_s=%.4f "
                "s/tok_net=%.6f"
                % (
                    s.get("decode_seconds", float("nan")),
                    s.get("seed_prefill_seconds", float("nan")),
                    s.get("decode_seconds_ex_prefill", float("nan")),
                    s.get("sec_per_token_ex_prefill", float("nan")),
                )
            )
        a = entry["attrib"]
        if a:
            print(
                "  %-4s %-7s %-11s %8s %8s %8s %8s %8s %8s"
                % (
                    "M",
                    "n_fwd",
                    "total_us",
                    "gdn%",
                    "fullat%",
                    "mlp%",
                    "head2%",
                    "embed%",
                    "resid%",
                )
            )
            for mm, e in a["by_m"].items():
                if e["forwards"] < args.min_count:
                    continue

                def pct(x):
                    return "%8.2f" % (100 * x) if x is not None else "       -"

                print(
                    "  %-4d %-7d %-11.1f %s %s %s %s %s %s"
                    % (
                        mm,
                        e["forwards"],
                        (e["total_ns_mean"] or 0) / 1e3,
                        pct(e["gdn_share"]),
                        pct(e["full_attention_share"]),
                        pct(e["mlp_share"]),
                        pct(e["head_and_top_two_share"]),
                        pct(e["embed_share"]),
                        pct(e.get("residual_frac")),
                    )
                )
            p = a["pooled"]
            if p:
                print(
                    "  pooled(widths=%s, n=%d): gdn=%.2f%% fullat=%.2f%% "
                    "mlp=%.2f%% head+top2=%.2f%% embed=%.2f%%"
                    % (
                        p["widths"],
                        p["forwards"],
                        100 * (p["gdn_share"] or 0),
                        100 * (p["full_attention_share"] or 0),
                        100 * (p["mlp_share"] or 0),
                        100 * (p["head_and_top_two_share"] or 0),
                        100 * (p["embed_share"] or 0),
                    )
                )
            inst_hist = {mm: e["forwards"] for mm, e in a["by_m"].items()}
            print(f"  instrumented width histogram: {inst_hist}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
