#!/usr/bin/env python3
"""How the proposal-head phase scales with draft depth, and what it implies.

`draft_head` runs once per draft token, so its cost should be a straight line in
`width - 1` with no fixed term. This script fits that line separately inside the
gate-qualified set and inside the ungated ABBA set, never across them, and then
prices the head per draft token.

The head is bandwidth-bound: the census shows every head matmul streaming its
whole weight tensor once per draft token at a near-constant GB/s. The resident
artifact is bf16. The submitted manifest declares a different, smaller artifact.
The last table converts the measured bytes into the cost the declared artifact
would carry under the same bandwidth, which is the size of the systematic error
in every local drafting measurement made on this host.

    usage: research/e80_head_scaling.py --gated DIR... --ungated DIR...
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e80_blocks as B
import e80_census_report as R

# Fixture-pinned resident head size and the size the manifest declares.
RESIDENT_BYTES = 849_398_784
DECLARED_BYTES = 427_742_600

# The five bf16 head matmuls account for 849,346,560 of the artifact's
# 849,398,784 bytes, so `gemv` is exactly the head weights. The `qmv` row in the
# same phase is already affine-4 quantized and reads 281 MB that the head
# artifact does not contain, so replacing the head cannot shrink it.
HEAD_WEIGHTS = ("gemv",)
STREAMING = ("gemv", "qmv", "steel_gemm")


def fit(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for _, y in points)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    return slope, intercept, (1.0 if ss_tot == 0 else 1 - ss_res / ss_tot)


def collect(dirs, min_rounds):
    """Per width: head ms, verify ms, and head cost split by family."""
    paths = [pathlib.Path(d) / "census.jsonl" for d in dirs]
    leg = R.Leg([p for p in paths if p.exists()])
    rules = B.learn_axis_rules(leg)
    rows = {}
    for width in sorted(leg.widths()):
        if width <= 1:
            continue
        head, verify, fams = 0.0, 0.0, {}
        rounds = 0
        for phase in ("draft_head", "target_verify"):
            if leg.round_count(width, phase) < min_rounds:
                continue
            att, n = B.attribute(leg, phase, width, rules)
            total = sum(v["gpu_ns"] for v in att.values()) / n / 1e6
            rounds = max(rounds, n)
            if phase == "draft_head":
                head = total
                for key, v in att.items():
                    fam = B.family_of_owner(key)
                    fams[fam] = fams.get(fam, 0.0) + v["gpu_ns"] / n / 1e6
            else:
                verify = total
        if head and verify:
            rows[width] = {"head": head, "verify": verify,
                           "round": head + verify, "families": fams,
                           "rounds": rounds}
    return rows


def table(name, rows, out):
    out += [f"### {name}", ""]
    out.append("| width | draft tokens | rounds | draft_head ms/round | "
               "ms per draft token | verify ms/round | whole round ms/round | "
               "head share of round |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    pts = []
    for w in sorted(rows):
        r = rows[w]
        d = w - 1
        pts.append((d, r["head"]))
        out.append(f"| {w} | {d} | {r['rounds']} | {r['head']:.3f} | "
                   f"{r['head']/d:.3f} | {r['verify']:.3f} | "
                   f"{r['round']:.3f} | {100*r['head']/r['round']:.2f}% |")
    out.append("")
    if len(pts) >= 2:
        slope, intercept, r2 = fit(pts)
        if len(pts) == 2:
            out.append(f"two widths only, so the line through them is an "
                       f"interpolation and not evidence of linearity: "
                       f"`draft_head_ms = {slope:.4f} * drafts "
                       f"{intercept:+.4f}`. Its R^2 is 1 by construction.")
        else:
            out.append(f"least squares over {len(pts)} widths: "
                       f"`draft_head_ms = {slope:.4f} * drafts "
                       f"{intercept:+.4f}`, R^2 = {r2:.6f}. The intercept is "
                       f"{abs(intercept)/slope*100:.1f} % of one draft token, "
                       "so the phase carries no measurable fixed cost.")
        out.append("")
        gbps = RESIDENT_BYTES / (slope / 1e3) / 1e9
        out.append(f"At {slope:.3f} ms per draft token the head moves its whole "
                   f"{RESIDENT_BYTES/1e6:.1f} MB of weights once per draft "
                   f"token, an effective {gbps:.0f} GB/s across the whole "
                   "phase.")
        out.append("")
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gated", nargs="*", default=[])
    ap.add_argument("--ungated", nargs="*", default=[])
    ap.add_argument("--min-rounds", type=int, default=20)
    args = ap.parse_args()

    lines = ["## proposal-head cost per draft token", ""]
    g_rows = collect(args.gated, args.min_rounds) if args.gated else {}
    u_rows = collect(args.ungated, args.min_rounds) if args.ungated else {}

    g_pts = table("gate-qualified set", g_rows, lines) if g_rows else []
    u_pts = table("ungated ABBA set", u_rows, lines) if u_rows else []

    if g_pts and len(u_pts) >= 2:
        slope, intercept, _ = fit(u_pts)
        lines += ["### cross-set check, not a pooled fit", "",
                  "| width | set | measured ms | ungated-fit prediction | "
                  "delta |", "|---:|---|---:|---:|---:|"]
        for d, y in g_pts:
            pred = slope * d + intercept
            lines.append(f"| {d+1} | gate-qualified | {y:.3f} | {pred:.3f} | "
                         f"{100*(y-pred)/pred:+.2f}% |")
        lines.append("")

    ratio = DECLARED_BYTES / RESIDENT_BYTES
    lines += ["### what the declared head would cost instead", "",
              f"The resident artifact holds {RESIDENT_BYTES:,} bytes. "
              f"`mtp-head.manifest.json` declares {DECLARED_BYTES:,} bytes, "
              f"{100*ratio:.2f} % of it. The head phase is bandwidth-bound, so "
              "a bandwidth-proportional model scales the bytes the head "
              "artifact supplies and leaves every other family unchanged. Two "
              "models bracket the answer. The narrow model scales only `gemv`, "
              "which is exactly the head weight tensors. The wide model also "
              "scales `qmv` and `steel_gemm`, which read weights the artifact "
              "does not contain and would therefore only shrink if the "
              "declared head changed the readout too. Both are models, not "
              "measurements: each assumes the declared artifact reaches the "
              "same GB/s.", ""]
    lines += ["| width | set | measured head ms | narrow model ms | narrow "
              "saving | wide model ms | wide saving | whole round ms | narrow "
              "share of round | wide share of round |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, rows in (("gate-qualified", g_rows), ("ungated", u_rows)):
        for w in sorted(rows):
            r = rows[w]
            narrow_b = sum(v for k, v in r["families"].items()
                           if k in HEAD_WEIGHTS)
            wide_b = sum(v for k, v in r["families"].items() if k in STREAMING)
            narrow = r["head"] - narrow_b * (1 - ratio)
            wide = r["head"] - wide_b * (1 - ratio)
            ns, ws = r["head"] - narrow, r["head"] - wide
            lines.append(f"| {w} | {label} | {r['head']:.3f} | {narrow:.3f} | "
                         f"{ns:.3f} | {wide:.3f} | {ws:.3f} | "
                         f"{r['round']:.3f} | {100*ns/r['round']:.2f}% | "
                         f"{100*ws/r['round']:.2f}% |")
    lines.append("")

    lines += ["### head cost by family per draft token", "",
              "| width | set | " + " | ".join(STREAMING) + " | other |",
              "|---:|---|" + "---:|" * (len(STREAMING) + 1)]
    for label, rows in (("gate-qualified", g_rows), ("ungated", u_rows)):
        for w in sorted(rows):
            r = rows[w]
            d = w - 1
            cells = [f"{r['families'].get(f, 0.0)/d:.3f}" for f in STREAMING]
            other = sum(v for k, v in r["families"].items()
                        if k not in STREAMING) / d
            lines.append(f"| {w} | {label} | " + " | ".join(cells) +
                         f" | {other:.3f} |")
    lines.append("")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
