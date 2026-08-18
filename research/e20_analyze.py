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

BUCKETS = [
    "drain", "embed", "gdn", "full_attention",
    "mlp_gdn", "mlp_full_attention", "head", "top_two",
]
# The assignment's four families. `embed` and `drain` are reported separately
# rather than folded in: `embed` is neither a layer family nor readout, and
# `drain` is draft-head work that finished landing inside the verify forward,
# so charging it to a target family would overstate that family.
FAMILIES = {
    "gdn": ["gdn"],
    "full_attention": ["full_attention"],
    "mlp": ["mlp_gdn", "mlp_full_attention"],
    "head_and_top_two": ["head", "top_two"],
}
# Whole-decoder-layer totals. Mode 1 splits every layer in two and mode 3 keeps
# it whole, so these are the only quantities both attributing modes measure,
# and their agreement is the instrument's own validity check.
LAYER_GROUPS = {
    "gdn_layer": ["gdn", "mlp_gdn"],
    "full_attention_layer": ["full_attention", "mlp_full_attention"],
}

FWD = re.compile(r"^qwen-attrib: (.*)$")
SPAN = re.compile(r"^qwen-attrib-span: (.*)$")
VERIFY = re.compile(r"^qwen-attrib-verify: (.*)$")


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
    scored: set[int] = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = FWD.match(line)
            if m:
                forwards.append(parse_kv(m.group(1)))
                continue
            m = SPAN.match(line)
            if m:
                rec = parse_kv(m.group(1))
                spans.setdefault(rec.get("f", -1), {}).update(
                    {k: v for k, v in rec.items() if k.endswith("_ns")}
                )
                continue
            m = VERIFY.match(line)
            if m:
                scored.add(parse_kv(m.group(1)).get("f", -1))
    for rec in forwards:
        f = rec.get("f", -1)
        rec.update(spans.get(f, {}))
        # Only forwards the session marked are shipped verify work. The harness
        # warms every legal shape 1..depth+1 before timing, and those warmup
        # forwards land at exactly the widths of interest, so pooling them in
        # would silently mix warmup into the answer.
        rec["scored"] = 1 if f in scored else 0
        rec["evals"] = sum(rec.get(f"{b}_n", 0) for b in BUCKETS)
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
            "evals_mean": statistics.mean([r.get("evals", 0) for r in recs]),
            "evals_median": statistics.median([r.get("evals", 0) for r in recs]),
        }
        attributed = 0.0
        for b in BUCKETS:
            vals = [r.get(f"{b}_ns", 0) for r in recs]
            calls = [r.get(f"{b}_n", 0) for r in recs]
            entry[f"{b}_ns_mean"] = statistics.mean(vals) if vals else 0.0
            entry[f"{b}_ns_median"] = statistics.median(vals) if vals else 0.0
            entry[f"{b}_calls_mean"] = statistics.mean(calls) if calls else 0.0
            entry[f"{b}_calls_median"] = statistics.median(calls) if calls else 0.0
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
        for grp, parts in LAYER_GROUPS.items():
            grp_ns = sum(entry[f"{p}_ns_mean"] for p in parts)
            entry[f"{grp}_ns_mean"] = grp_ns
            entry[f"{grp}_share"] = grp_ns / denom if denom else None
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


def boundary_overhead(arms: dict, key: str) -> dict:
    """Fit forward wall time against blocking-boundary count, per width.

    Mode 2 (0 boundaries), mode 3 (layers+2) and mode 1 (2*layers+2) run the
    same forward, so `total_ns ~= gpu_ns + c * evals` identifies `c`, the cost
    a blocking `eval` adds by flushing the command buffer and stopping the CPU
    from building the next subgraph while the GPU runs. That cost lands on a
    bucket in proportion to how many boundaries it contains, so subtracting
    `c * calls` is what turns a raw attributing-mode split into an estimate of
    the unperturbed one.

    `c` is fitted separately per width and never pooled: measured per-boundary
    cost rises from roughly 0.5 ms at M=2 to roughly 10 ms on the 512-row seed
    forward, so a single slope across widths is not a description of this
    machine. Two boundary counts identify a width's line and the third makes
    `max_abs_resid_frac` a real falsification test of the linear model.
    """
    points: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for label, entry in arms.items():
        a = entry.get(key)
        if not a:
            continue
        for m, e in a["by_m"].items():
            if e["total_ns_median"] is None:
                continue
            points[m].append((e["evals_median"], e["total_ns_median"], label))

    out: dict[int, dict] = {}
    for m, pts in sorted(points.items()):
        if len({round(p[0]) for p in pts}) < 2:
            continue
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if not den:
            continue
        c = (n * sxy - sx * sy) / den
        gpu = (sy - c * sx) / n
        resid = [p[1] - (gpu + c * p[0]) for p in pts]
        out[m] = {
            "per_eval_ns": c,
            "gpu_ns_intercept": gpu,
            "n_boundary_counts": len({round(p[0]) for p in pts}),
            "points": [{"evals": p[0], "total_ns": p[1], "arm": p[2]} for p in pts],
            "max_abs_resid_frac": max(abs(r) for r in resid) / gpu if gpu else None,
        }
    return {"source": key, "by_m": out}


def corrected_split(entry: dict, key: str, fit: dict) -> dict:
    """Remove the fitted `c * calls` from every bucket of an attributing arm."""
    a = entry.get(key)
    by_m = (fit or {}).get("by_m") or {}
    if not a or not by_m:
        return {}
    out: dict[int, dict] = {}
    for m, e in a["by_m"].items():
        f = by_m.get(m)
        if not f or not e["evals_median"]:
            continue
        c = f["per_eval_ns"]
        adj = {}
        for b in BUCKETS:
            adj[b] = max(0.0, e[f"{b}_ns_median"] - c * e[f"{b}_calls_median"])
        denom = sum(adj.values())
        row = {
            "m": m,
            "forwards": e["forwards"],
            "per_eval_ns": c,
            "corrected_total_ns": denom,
            "raw_total_ns": e["total_ns_median"],
            "unperturbed_total_ns": f["gpu_ns_intercept"],
            "max_abs_resid_frac": f["max_abs_resid_frac"],
        }
        for fam, parts in FAMILIES.items():
            fam_ns = sum(adj[p] for p in parts)
            row[f"{fam}_ns"] = fam_ns
            row[f"{fam}_share"] = fam_ns / denom if denom else None
        for b in ("embed", "drain"):
            row[f"{b}_ns"] = adj[b]
            row[f"{b}_share"] = adj[b] / denom if denom else None
        out[m] = row
    return out


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


def print_split_table(by_m: dict, min_count: int, title: str) -> None:
    print(f"  {title}")
    print(
        "  %-4s %-7s %-11s %-7s %8s %8s %8s %8s %8s %8s"
        % ("M", "n_fwd", "total_us", "evals", "gdn%", "fullat%", "mlp%",
           "head2%", "embed%", "resid%")
    )
    for mm, e in sorted(by_m.items()):
        if e["forwards"] < min_count:
            continue

        def pct(x):
            return "%8.2f" % (100 * x) if x is not None else "       -"

        print(
            "  %-4d %-7d %-11.1f %-7.1f %s %s %s %s %s %s"
            % (
                mm,
                e["forwards"],
                (e["total_ns_mean"] or 0) / 1e3,
                e["evals_mean"],
                pct(e["gdn_share"]),
                pct(e["full_attention_share"]),
                pct(e["mlp_share"]),
                pct(e["head_and_top_two_share"]),
                pct(e["embed_share"]),
                pct(e.get("residual_frac")),
            )
        )


def layer_group_agreement(arms: dict, key: str) -> dict:
    """Compare whole-decoder-layer shares between the two attributing modes.

    Mode 1 charges 2*layers+2 boundaries and mode 3 charges layers+2, so if the
    boundaries themselves were driving the answer the two modes would disagree
    about how much of the forward each layer family owns. They measure the same
    quantity, so their spread is an upper bound on the distortion the instrument
    adds to a layer-family claim -- a share gap wider than the gap between two
    families is a failure of the instrument, not a finding about the model.
    """
    per_mode: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels: dict[int, list[str]] = defaultdict(list)
    for label, entry in arms.items():
        if not isinstance(entry, dict):
            continue
        mode = entry.get("meta", {}).get("attrib_mode")
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            continue
        if mode not in (1, 3):
            continue
        a = entry.get(key)
        if not a:
            continue
        labels[mode].append(label)
        for mm, e in a["by_m"].items():
            for gi, grp in enumerate(LAYER_GROUPS):
                s = e.get(f"{grp}_share")
                if s is not None:
                    per_mode[mode][(mm, gi)].append(s)

    if 1 not in per_mode or 3 not in per_mode:
        return {}
    out: dict = {"source": key, "arms": {str(k): v for k, v in labels.items()},
                 "by_m": {}}
    names = list(LAYER_GROUPS)
    worst = 0.0
    for (mm, gi), fine in sorted(per_mode[1].items()):
        coarse = per_mode[3].get((mm, gi))
        if not coarse:
            continue
        f = statistics.mean(fine)
        c = statistics.mean(coarse)
        row = out["by_m"].setdefault(mm, {})
        row[names[gi]] = {
            "mode1_share": f, "mode3_share": c, "abs_delta": abs(f - c)
        }
        worst = max(worst, abs(f - c))
    out["max_abs_share_delta"] = worst
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--min-count", type=int, default=2)
    # Warmup covers every legal width with more samples per width than the few
    # scored rounds do, and the per-eval cost being fitted is a property of the
    # boundary, not of the cache state, so it is the better-conditioned source.
    ap.add_argument(
        "--fit-source", choices=["attrib_warmup", "attrib_scored"],
        default="attrib_warmup",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args()

    result = {}
    for arm_dir in args.arms:
        label = os.path.basename(arm_dir.rstrip("/"))
        info = find_arm(arm_dir)
        forwards: list[dict[str, int]] = []
        for p in info.get("attrib_paths", []):
            forwards.extend(load_attrib(p))
        scored = [r for r in forwards if r.get("scored")]
        warmup = [r for r in forwards if not r.get("scored")]
        entry = {
            "meta": info["meta"],
            "attrib_files": [os.path.basename(p) for p in info.get("attrib_paths", [])],
            "mtp": report_fields(info.get("report_path")),
            "serial": report_fields(info.get("serial_report_path")),
            "attrib_scored": summarize(scored, args.min_count) if scored else None,
            "attrib_warmup": summarize(warmup, args.min_count) if warmup else None,
        }
        result[label] = entry

    fits = {
        k: boundary_overhead(result, k) for k in ("attrib_scored", "attrib_warmup")
    }
    fit = fits.get(args.fit_source) or {}
    for entry in result.values():
        entry["corrected"] = corrected_split(entry, "attrib_scored", fit)

    for label, entry in result.items():
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
        for key, title in (
            ("attrib_scored", "RAW split, SCORED verify forwards only"),
            ("attrib_warmup", "RAW split, WARMUP forwards (context, not the answer)"),
        ):
            a = entry.get(key)
            if not a:
                continue
            print_split_table(a["by_m"], args.min_count, title)
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
        a = entry.get("attrib_scored")
        if a:
            print(
                "  instrumented scored width histogram: "
                f"{ {mm: e['forwards'] for mm, e in sorted(a['by_m'].items())} }"
            )
        elif entry.get("attrib_warmup"):
            print(
                "  WARNING: no qwen-attrib-verify markers in this arm, so every "
                "forward was classified as warmup and no scored split exists."
            )
        corr = entry.get("corrected")
        if corr:
            print("  BOUNDARY-CORRECTED split (c*calls removed per bucket)")
            print(
                "  %-4s %-7s %-10s %-10s %8s %8s %8s %8s %8s %8s"
                % ("M", "n_fwd", "raw_us", "corr_us", "gdn%", "fullat%",
                   "mlp%", "head2%", "embed%", "drain%")
            )
            for mm, row in sorted(corr.items()):
                if row["forwards"] < args.min_count:
                    continue

                def pct(x):
                    return "%8.2f" % (100 * x) if x is not None else "       -"

                print(
                    "  %-4d %-7d %-10.1f %-10.1f %s %s %s %s %s %s"
                    % (
                        mm,
                        row["forwards"],
                        (row["raw_total_ns"] or 0) / 1e3,
                        row["corrected_total_ns"] / 1e3,
                        pct(row["gdn_share"]),
                        pct(row["full_attention_share"]),
                        pct(row["mlp_share"]),
                        pct(row["head_and_top_two_share"]),
                        pct(row["embed_share"]),
                        pct(row["drain_share"]),
                    )
                )

    for k, f in fits.items():
        if not f.get("by_m"):
            continue
        used = " (used for correction)" if k == args.fit_source else ""
        print(f"\n===== boundary-overhead fit from {k}{used} =====")
        print(
            "  %-4s %-7s %-6s %-14s %-16s %-10s"
            % ("M", "n_bnd", "pts", "per_eval_us", "unperturbed_us", "resid%")
        )
        for mm, e in sorted(f["by_m"].items()):
            print(
                "  %-4d %-7d %-6d %-14.4f %-16.1f %-10.2f"
                % (mm, e["n_boundary_counts"], len(e["points"]),
                   e["per_eval_ns"] / 1e3, e["gpu_ns_intercept"] / 1e3,
                   100 * (e["max_abs_resid_frac"] or 0))
            )
    result["_boundary_fits"] = fits

    agree = {}
    for key in ("attrib_scored", "attrib_warmup"):
        a = layer_group_agreement(result, key)
        if not a:
            continue
        agree[key] = a
        print(f"\n===== mode1-vs-mode3 layer-group agreement ({key}) =====")
        print(f"  mode1 arms={a['arms'].get('1')} mode3 arms={a['arms'].get('3')}")
        print(
            "  %-4s %-22s %-12s %-12s %-10s"
            % ("M", "layer_group", "mode1%", "mode3%", "delta_pp")
        )
        for mm, row in sorted(a["by_m"].items()):
            for grp, v in row.items():
                print(
                    "  %-4d %-22s %-12.2f %-12.2f %-10.2f"
                    % (mm, grp, 100 * v["mode1_share"], 100 * v["mode3_share"],
                       100 * v["abs_delta"])
                )
        print(
            "  worst disagreement: %.2f pp -- any family gap narrower than this "
            "is not resolved by this instrument."
            % (100 * a["max_abs_share_delta"])
        )
    result["_layer_group_agreement"] = agree

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
