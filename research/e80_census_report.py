#!/usr/bin/env python3
"""E80 rung 2 -- the per-kernel GPU-time census.

Two configurations are needed, and neither alone answers the question.

  DEFAULT   MLX packs many ops into one command buffer, so a buffer's
            GPU interval covers a whole group of kernels running with real
            overlap. This is the true cost, and it cannot be split per kernel:
            the width-6 verify phase produces about six distinct buffer
            signatures over fifteen kernels, so the linear system is
            underdetermined by a wide margin.

  ISOLATED  MLX_MAX_OPS_PER_BUFFER=1 makes every buffer exactly one dispatch,
            so a buffer's GPU interval IS that kernel's GPU time. Per-kernel
            cost is then exact, but all intra-buffer concurrency is removed and
            per-buffer overhead is paid once per dispatch, so the total is an
            upper bound on the real cost.

The concurrency discount for a width is
    discount = isolated kernel-time total / default in-situ total
and it is reported beside every share so no isolated number is ever read as
though it were the shipped cost.

Dispatch counts always come from the DEFAULT run's `round` records, which are
the unperturbed schedule.

usage:
  research/e80_census_report.py --default research/out/TAG/census.jsonl \\
      [--isolated research/out/TAG-iso/census.jsonl] \\
      [--width M] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e58_census_report import family  # noqa: E402

# Ranked width histogram supplied with the E80 assignment. The local fixture
# over-weights M = 9 by 7.6x, so an unweighted local census answers the wrong
# question. Mean ranked width is 5.82 against a local fixture mean of 7.27.
RANKED_WIDTH_WEIGHTS = {
    3: 0.0325, 4: 0.1420, 5: 0.2410, 6: 0.3340,
    7: 0.1220, 8: 0.0735, 9: 0.0575,
}

# Falsification riders from the assignment. Each is a share of verify GPU time.
# `copy` carries a headline rule: above 1 % it reopens ledger entry 218.
RIDERS = [
    ("copy", ["copy"], 0.01, "ledger 218 reopens; must be in the headline"),
    ("elementwise+fusions", ["elementwise", "compiled_fusion", "reduce_scan"], 0.03,
     "unary/binary/ternary ops well under 3 %"),
    ("rms_norm", ["norm"], 0.03, "rms_norm under 3 %"),
    ("sdpa_vector", ["sdpa_fused"], 0.03, "sdpa_vector under 3 %"),
    ("gemv", ["dense_gemv"], 0.02, "gemv under 2 %"),
]

HASH_RE = re.compile(r"_\d{12,}_")
SHAPE_RE = re.compile(r"^(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
                      r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")

# The `affine_qmv_fast` dispatches all share one Metal function name and are
# told apart only by their grid. grid.y * 8 recovers the projection's output
# width, which identifies the module exactly. Verified against the config:
# hidden 5120, 64 layers = 48 GDN + 16 full attention, head_dim 256, 24 query
# heads, 4 KV heads, MLP intermediate 17408, vocabulary 248320.
#
# The count column is the decisive one. `gdn_in_proj_fused` and `fa_qkv_gate_
# fused` are the two projections that `Qwen35GatedDeltaNet` and `Qwen35Attention`
# fuse into raw `quantizedMM` calls, so they never dispatch through a child
# `Linear` and NO E71 arm can intercept them. Together they are 48 + 16 = 64 of
# the 257 qmv dispatches in a verify round.
QMV_GROUPS = {
    4352: ("mlp_gate_up", 64, True, "5120 -> 34816, gate and up fused"),
    640: ("out_projections", 128, True,
          "-> 5120: mlp_down x64, gdn_out_proj x48, fa_o_proj x16"),
    2060: ("gdn_in_proj_fused", 48, False, "5120 -> 16480, qkvzba fused"),
    1792: ("fa_qkv_gate_fused", 16, False, "5120 -> 14336, q/k/v and output gate"),
    31040: ("lm_head", 1, True, "5120 -> 248320"),
}


def parse_shape(shape: str):
    m = SHAPE_RE.match(shape)
    if not m:
        return None
    return {
        "kernel": m.group("kernel"),
        "grid": (int(m.group("gx")), int(m.group("gy")), int(m.group("gz"))),
        "threadgroup": (int(m.group("tx")), int(m.group("ty")), int(m.group("tz"))),
    }


def qmv_group(shape: str):
    parsed = parse_shape(shape)
    if not parsed or "affine_qmv" not in parsed["kernel"]:
        return None
    return QMV_GROUPS.get(parsed["grid"][1])


def short(kernel: str, width: int = 84) -> str:
    """Collapse the MLX fusion hash so a table stays readable."""
    name = HASH_RE.sub("_H_", kernel)
    return name if len(name) <= width else name[:width - 3] + "..."


def read_records(path: pathlib.Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def default_dispatches(path: pathlib.Path):
    """{width: {kernel: dispatches per round}} and {width: rounds} from `round`."""
    counts = collections.defaultdict(collections.Counter)
    rounds = collections.Counter()
    for rec in read_records(path):
        if rec.get("event") != "round":
            continue
        verify = (rec.get("phases") or {}).get("verify_block")
        if not verify:
            continue
        width = rec["width"]
        rounds[width] += 1
        for kernel, n in (verify.get("kernels") or {}).items():
            counts[width][kernel] += n
    per_round = {
        w: {k: v / rounds[w] for k, v in c.items()} for w, c in counts.items()
    }
    return per_round, dict(rounds)


def default_phase_gpu(path: pathlib.Path):
    """{width: total verify GPU ns} summed over every snapshot."""
    total = collections.Counter()
    for rec in read_records(path):
        if rec.get("event") != "gputime":
            continue
        for key, bucket in rec.get("by_width_phase", {}).items():
            w, _, phase = key.partition("|")
            if phase != "verify_block":
                continue
            total[int(w[1:])] += bucket["gpu_ns"]
    return dict(total)


def isolated_kernel_gpu(path: pathlib.Path):
    """{width: {kernel: {gpu_ns, buffers}}} from single-dispatch buffers."""
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"gpu_ns": 0, "buffers": 0}))
    for rec in read_records(path):
        if rec.get("event") != "gputime":
            continue
        for key, bucket in rec.get("exclusive_kernels", {}).items():
            w, _, rest = key.partition("|")
            phase, _, kernel = rest.partition("|")
            if phase != "verify_block":
                continue
            slot = acc[int(w[1:])][kernel]
            slot["gpu_ns"] += bucket["gpu_ns"]
            slot["buffers"] += bucket["buffers"]
    return {w: dict(k) for w, k in acc.items()}


def build_width_table(width, dispatches, iso_kernels, insitu_ns, rounds):
    """Per-kernel rows for one width, with the isolated-to-in-situ reconciliation."""
    rows = []
    iso = iso_kernels.get(width, {})
    for kernel, per_round in sorted(dispatches.get(width, {}).items()):
        entry = iso.get(kernel)
        if entry and entry["buffers"]:
            mean_ns = entry["gpu_ns"] / entry["buffers"]
            iso_ms = mean_ns * per_round / 1e6
        else:
            mean_ns, iso_ms = None, None
        rows.append({
            "kernel": kernel,
            "family": family(kernel),
            "dispatches_per_round": per_round,
            "isolated_mean_ns": mean_ns,
            "isolated_ms_per_round": iso_ms,
            "isolated_samples": entry["buffers"] if entry else 0,
        })
    iso_total = sum(r["isolated_ms_per_round"] or 0.0 for r in rows)
    insitu_ms = (insitu_ns.get(width, 0) / 1e6 / rounds[width]) if rounds.get(width) else None
    discount = (insitu_ms / iso_total) if (iso_total and insitu_ms) else None
    for r in rows:
        r["share_of_isolated"] = (
            (r["isolated_ms_per_round"] / iso_total) if (iso_total and r["isolated_ms_per_round"])
            else None)
        # Scaling the isolated per-kernel time by the measured overall discount
        # is the best in-situ estimate this instrument can produce. It assumes
        # the discount is uniform across kernels, which is stated, not proved.
        r["insitu_estimate_ms"] = (
            r["isolated_ms_per_round"] * discount
            if (discount and r["isolated_ms_per_round"]) else None)
    return {
        "width": width,
        "rounds": rounds.get(width, 0),
        "isolated_total_ms_per_round": iso_total,
        "insitu_total_ms_per_round": insitu_ms,
        "concurrency_discount": discount,
        "rows": rows,
    }


def family_totals(table):
    fam = collections.defaultdict(lambda: {"ms": 0.0, "dispatches": 0.0, "kernels": 0})
    for r in table["rows"]:
        slot = fam[r["family"]]
        slot["ms"] += r["isolated_ms_per_round"] or 0.0
        slot["dispatches"] += r["dispatches_per_round"]
        slot["kernels"] += 1
    total = table["isolated_total_ms_per_round"]
    for slot in fam.values():
        slot["share"] = slot["ms"] / total if total else None
    return dict(fam)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", required=True, type=pathlib.Path)
    ap.add_argument("--isolated", type=pathlib.Path)
    ap.add_argument("--width", type=int, action="append")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    dispatches, rounds = default_dispatches(args.default)
    insitu_ns = default_phase_gpu(args.default)
    iso_kernels = isolated_kernel_gpu(args.isolated) if args.isolated else {}

    widths = args.width or sorted(dispatches)
    print(f"widths seen = {sorted(dispatches)}  rounds per width = "
          f"{ {w: rounds[w] for w in sorted(rounds)} }")
    if args.isolated:
        print(f"isolated widths = {sorted(iso_kernels)}")
    else:
        print("no isolated run supplied: dispatch counts only, no per-kernel GPU time")

    tables = {}
    for width in widths:
        table = build_width_table(width, dispatches, iso_kernels, insitu_ns, rounds)
        tables[width] = table
        print(f"\n## width M = {width}   rounds = {table['rounds']}")
        if table["insitu_total_ms_per_round"]:
            print(f"in-situ verify GPU = {table['insitu_total_ms_per_round']:.3f} ms/round")
        if table["isolated_total_ms_per_round"]:
            print(f"isolated kernel sum = {table['isolated_total_ms_per_round']:.3f} ms/round")
        if table["concurrency_discount"]:
            print(f"concurrency discount = {table['concurrency_discount']:.3f} "
                  f"(in-situ / isolated)")

        rows = sorted(table["rows"],
                      key=lambda r: -(r["isolated_ms_per_round"] or 0.0))
        print(f"\n| kernel | family | dispatches/round | isolated ms/round "
              f"| share | in-situ est ms |")
        print("|---|---|---:|---:|---:|---:|")
        for r in rows[:args.top]:
            ms = "--" if r["isolated_ms_per_round"] is None else f"{r['isolated_ms_per_round']:.4f}"
            sh = "--" if r["share_of_isolated"] is None else f"{r['share_of_isolated'] * 100:.2f} %"
            est = "--" if r["insitu_estimate_ms"] is None else f"{r['insitu_estimate_ms']:.4f}"
            print(f"| `{short(r['kernel'])}` | {r['family']} "
                  f"| {r['dispatches_per_round']:.1f} | {ms} | {sh} | {est} |")

        fam = family_totals(table)
        print(f"\n| family | kernels | dispatches/round | isolated ms/round | share |")
        print("|---|---:|---:|---:|---:|")
        for name, slot in sorted(fam.items(), key=lambda kv: -kv[1]["ms"]):
            sh = "--" if slot["share"] is None else f"{slot['share'] * 100:.2f} %"
            print(f"| {name} | {slot['kernels']} | {slot['dispatches']:.1f} "
                  f"| {slot['ms']:.4f} | {sh} |")

        unclassified = [r for r in table["rows"] if r["family"] == "unclassified"]
        print(f"\nunclassified kernels: {len(unclassified)}")
        for r in sorted(unclassified, key=lambda r: -(r["isolated_ms_per_round"] or 0.0)):
            ms = "--" if r["isolated_ms_per_round"] is None else f"{r['isolated_ms_per_round']:.4f} ms"
            print(f"  {ms}  x{r['dispatches_per_round']:.1f}  {r['kernel']}")

    # --- riders ----------------------------------------------------------
    print("\n## falsification riders, at the ranked-dominant width M = 6\n")
    verdicts = {}
    if 6 in tables and tables[6]["isolated_total_ms_per_round"]:
        fam6 = family_totals(tables[6])
        print("| rider | families | share | limit | verdict |")
        print("|---|---|---:|---:|---|")
        for label, fams, limit, note in RIDERS:
            share = sum(fam6.get(f, {}).get("ms", 0.0) for f in fams)
            share = share / tables[6]["isolated_total_ms_per_round"]
            ok = share <= limit
            verdicts[label] = {"share": share, "limit": limit, "holds": ok, "note": note}
            print(f"| {label} | {', '.join(fams)} | {share * 100:.3f} % "
                  f"| {limit * 100:.0f} % | {'holds' if ok else '**BREAKS** -- ' + note} |")
    else:
        print("width 6 has no isolated data; riders cannot be decided")

    # --- ranked weighting -------------------------------------------------
    usable = {w: t for w, t in tables.items()
              if w in RANKED_WIDTH_WEIGHTS and t["isolated_total_ms_per_round"]}
    if usable:
        covered = sum(RANKED_WIDTH_WEIGHTS[w] for w in usable)
        print(f"\n## ranked-weighted family shares\n")
        print(f"covered ranked mass = {covered * 100:.1f} % over widths {sorted(usable)}; "
              f"weights renormalised over the covered widths")
        weighted = collections.defaultdict(float)
        for w, t in usable.items():
            weight = RANKED_WIDTH_WEIGHTS[w] / covered
            for name, slot in family_totals(t).items():
                weighted[name] += weight * (slot["share"] or 0.0)
        print("\n| family | ranked-weighted share |")
        print("|---|---:|")
        for name, share in sorted(weighted.items(), key=lambda kv: -kv[1]):
            print(f"| {name} | {share * 100:.2f} % |")
    else:
        weighted = {}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "default_census": str(args.default),
            "isolated_census": str(args.isolated) if args.isolated else None,
            "ranked_width_weights": RANKED_WIDTH_WEIGHTS,
            "rounds_per_width": rounds,
            "widths": {str(w): t for w, t in tables.items()},
            "families_per_width": {str(w): family_totals(t) for w, t in tables.items()},
            "riders": verdicts,
            "ranked_weighted_family_share": dict(weighted),
        }, indent=2, default=float) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
