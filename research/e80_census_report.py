#!/usr/bin/env python3
"""E80 rung 2 and 3: the per-kernel GPU-time census of the verify block.

Mechanism
---------
Rung 0c picked `MLX_MAX_OPS_PER_BUFFER=1` on the premise that it puts one
dispatch in one command buffer, so the buffer's `GPUEndTime - GPUStartTime`
would BE that dispatch's GPU time. The rung-2 debug leg refuted the premise:
MLX honours the limit, but one MLX op is not one Metal dispatch, so the
isolated leg still averages 2.04 dispatches per buffer and only 0.4 % of
verify dispatches ever run alone.

The recorded data still determines every kernel's time. Each single-phase
command buffer contributes one exact linear equation

    gpu_time(buffer) = sum over shapes s of count(s, buffer) * t_s

and the same shape appears in many different pairings, which identifies the
system. `e80_nnls` solves it under a non-negativity constraint. Fitting the
default-mode and the isolated-mode legs separately and dividing gives the
concurrency discount per family.

Shapes, not kernel names, are the unknowns. Every projection in the model
dispatches the same `affine_qmv_fast` Metal function and differs only by grid.

Identifiability is reported, never assumed. Where the design matrix is rank
deficient the split inside a degenerate group is arbitrary, so the group is
reported as a group and flagged.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e80_nnls import solve_kernel_times  # noqa: E402

# Ranked verify-width histogram supplied with the assignment. It sums to
# 100.25 %, and rung 2 measures only a subset, so shares are renormalised over
# the covered widths and the covered mass is always printed beside them.
RANKED_WIDTH_WEIGHTS = {
    3: 0.0325, 4: 0.1420, 5: 0.2410, 6: 0.3340,
    7: 0.1220, 8: 0.0735, 9: 0.0575,
}

# Falsification riders from the assignment, as a share of verify GPU time.
RIDERS = [
    ("copy", ["copy"], 0.01, "ledger 218 reopens; must be in the headline"),
    ("unary/binary/ternary ops", ["elementwise", "compiled_fusion", "reduce_scan"],
     0.03, "elementwise and fused elementwise under 3 %"),
    ("rms_norm", ["norm"], 0.03, "rms_norm under 3 %"),
    ("sdpa_vector", ["sdpa"], 0.03, "sdpa_vector under 3 %"),
    ("gemv", ["dense_gemv"], 0.02, "non-quantised gemv under 2 %"),
]

SHAPE_RE = re.compile(
    r"^(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r"(?: tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+))?$")
SIG_ENTRY = re.compile(r"^(?P<name>.+)\*(?P<count>\d+)$")

# `affine_qmv_fast` grid.y * 8 is the projection's output width, which names the
# module exactly. Verified against the config: hidden 5120, 64 layers = 48 GDN +
# 16 full attention, head_dim 256, 24 query heads, 4 KV heads, MLP intermediate
# 17408, vocabulary 248320. grid.x is the verify width M.
#
# `gdn_in_proj_fused` and `fa_qkv_gate_fused` are the two projections that
# `Qwen35GatedDeltaNet` and `Qwen35Attention` fuse into raw `quantizedMM`, so
# they never dispatch through a child `Linear` and NO E71 arm can intercept
# them. Together they are 48 + 16 = 64 of the 257 qmv dispatches in a round.
QMV_UNITS = {
    4352: ("mlp_gate_up", 64, True),
    640: ("mlp_down + gdn_out_proj + fa_o_proj", 128, True),
    2060: ("gdn_in_proj_fused (qkvzba)", 48, False),
    1792: ("fa_qkv_gate_fused", 16, False),
    31040: ("lm_head", 1, True),
}

FAMILY_RULES = [
    ("qmv", ("affine_qmv",)),
    ("quant_dequant", ("affine_dequantize", "affine_quantize")),
    ("dense_gemv", ("gemv", "gemm", "steel")),
    ("norm", ("rms", "layer_norm", "layernorm")),
    ("sdpa", ("sdpa",)),
    ("gdn_recurrence", ("gated_delta", "gdn_prework", "packed_gdn")),
    ("qk_rms_rope", ("qk_rms_rope",)),
    ("top2_readout", ("top2",)),
    ("gather_scatter", ("gather", "scatter")),
    ("copy", ("copy",)),
    ("compiled_fusion", ("_VV_", "_strided_", "astype", "multiply", "sigmoid")),
    ("elementwise", ("vv_", "ss_", "unary", "binary", "ternary", "add")),
    ("reduce_scan", ("reduce", "scan", "sum", "arg")),
]


def family_of(kernel: str) -> str:
    low = kernel.lower()
    for name, needles in FAMILY_RULES:
        for needle in needles:
            if needle.lower() in low:
                return name
    return "unclassified"


def parse_shape(shape: str):
    m = SHAPE_RE.match(shape)
    if not m:
        return None
    return {"kernel": m["kernel"],
            "grid": (int(m["gx"]), int(m["gy"]), int(m["gz"]))}


def qmv_unit(shape: str):
    parsed = parse_shape(shape)
    if not parsed or "affine_qmv" not in parsed["kernel"]:
        return None
    return QMV_UNITS.get(parsed["grid"][1])


def parse_signature(sig: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in sig.split(","):
        m = SIG_ENTRY.match(part)
        if not m:
            raise ValueError(f"unparsable signature element: {part!r}")
        counts[m["name"]] = counts.get(m["name"], 0) + int(m["count"])
    return counts


def read_records(path: pathlib.Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


class Leg:
    """One census file, summed over its delta snapshots.

    Every `gputime` record is a DELTA since the previous snapshot and the
    `snapshot` field is its index, not a boolean, so totals are sums.

    A leg runs two model-holding processes: the serial reference pass, whose
    rounds are width 1 in phase `target_forward`, and the MTP candidate pass,
    whose rounds carry the forced verify width in `draft_head` and
    `target_verify`. Their records interleave in one file but never share a
    width and phase, so summing is safe.
    """

    def __init__(self, path: pathlib.Path, skip_rounds: int = 0):
        self.path = path
        self.skip_rounds = skip_rounds
        self.by_width_phase = collections.defaultdict(
            lambda: {"gpu_ns": 0, "buffers": 0, "dispatches": 0})
        self.signatures = collections.defaultdict(
            lambda: {"gpu_ns": 0, "buffers": 0, "dispatches": 0})
        self.rounds = collections.Counter()
        self.dispatches = collections.Counter()
        self.shape_dispatches = collections.defaultdict(collections.Counter)
        self.health = collections.Counter()
        self._load()

    def _load(self):
        for rec in read_records(self.path):
            event = rec.get("event")
            if event == "round":
                if rec.get("round", 0) <= self.skip_rounds:
                    continue
                width = rec["width"]
                for phase, entry in (rec.get("phases") or {}).items():
                    self.rounds[(width, phase)] += 1
                    self.dispatches[(width, phase)] += entry.get("dispatches", 0)
                    for shape, count in (entry.get("shapes") or {}).items():
                        self.shape_dispatches[(width, phase)][shape] += count
            elif event == "gputime":
                # Drop the same warmup rounds here as in the round records, or
                # the per-round means would divide warmup GPU time by a
                # post-warmup round count. With one snapshot per round the
                # filter is exact; completion handlers can still smear a
                # buffer into the next snapshot, which is immaterial over the
                # round counts this census uses.
                if (self.skip_rounds
                        and rec.get("round_last", -1) >= 0
                        and rec["round_last"] <= self.skip_rounds):
                    continue
                for key, b in (rec.get("by_width_phase") or {}).items():
                    slot = self.by_width_phase[key]
                    slot["gpu_ns"] += b.get("gpu_ns", 0)
                    slot["buffers"] += b.get("buffers", 0)
                    slot["dispatches"] += b.get("dispatches", 0)
                for key, b in (rec.get("signatures") or {}).items():
                    slot = self.signatures[key]
                    slot["gpu_ns"] += b.get("gpu_ns", 0)
                    slot["buffers"] += b.get("buffers", 0)
                    slot["dispatches"] += b.get("dispatches", 0)
                for field in ("unmapped_encoder_dispatches", "zero_time_buffers",
                              "untracked_buffers", "signature_buffers",
                              "mixed_phase_buffers", "committed_total",
                              "completed_total"):
                    self.health[field] += rec.get(field, 0)

    def widths(self, phase="target_verify"):
        return sorted({w for (w, p) in self.rounds if p == phase})

    def round_count(self, width, phase="target_verify"):
        return self.rounds[(width, phase)]

    def signature_rows(self, phase, widths=None):
        """NNLS equations for one phase, optionally pooled over widths.

        Pooling is legitimate because the unknown is keyed by SHAPE, and a
        shape already carries the verify width in grid.x wherever the kernel is
        width dependent. Pooling adds distinct pairings, which is what breaks
        degeneracies that a single width leaves behind.
        """
        rows = []
        for key, v in self.signatures.items():
            w, _, rest = key.partition("|")
            ph, _, sig = rest.partition("|")
            if ph != phase or v["buffers"] <= 0:
                continue
            if widths is not None and int(w[1:]) not in widths:
                continue
            rows.append((parse_signature(sig), v["gpu_ns"], v["buffers"]))
        return rows


class Identifiability:
    """Which reported quantities the data actually determines.

    The fit solves `A t = b` for per-dispatch times `t`. If `A` has a null
    space `N`, then `t + N a` fits equally well for any `a`, so an individual
    time is only meaningful when its own direction is orthogonal to `N`.

    The quantities this census reports are never single times. They are
    weighted sums `c . t`, where `c` holds dispatches per round for the shapes
    in a family, a qmv unit or a rider. Such a sum is determined exactly when
    `c` is orthogonal to the null space, EVEN IF several of its individual
    terms are not. Checking the reported functional directly is therefore both
    stricter and more useful than checking each shape, and it is what
    `identified` does.
    """

    def __init__(self, rows, tol=1e-8):
        self.keys = sorted({k for counts, _, _ in rows for k in counts})
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.tol = tol
        n_keys = len(self.keys)
        A = np.zeros((len(rows), n_keys))
        for r, (counts, _, n) in enumerate(rows):
            for k, c in counts.items():
                A[r, self.index[k]] = c * np.sqrt(n)
        self.A = A
        if n_keys == 0 or not len(rows):
            self.null = np.zeros((0, 0))
            self.rank = 0
            return
        _, s, vt = np.linalg.svd(A, full_matrices=True)
        cutoff = max(A.shape) * (s[0] if s.size else 0.0) * 1e-12
        self.rank = int((s > cutoff).sum())
        self.null = vt[self.rank:]                      # (n_keys - rank, n_keys)

    def identified(self, weights: dict[str, float]) -> bool:
        """Is `sum_i weights[i] * t_i` determined by the data?"""
        if self.null.size == 0:
            return True
        c = np.zeros(len(self.keys))
        for k, w in weights.items():
            if k in self.index:
                c[self.index[k]] = w
        norm = np.linalg.norm(c)
        if norm == 0:
            return True
        return bool(np.linalg.norm(self.null @ c) / norm < self.tol)

    def unidentified_shapes(self) -> list[str]:
        """Shapes whose individual time the data leaves free."""
        if self.null.size == 0:
            return []
        loading = np.linalg.norm(self.null, axis=0)
        return [self.keys[i] for i in np.where(loading > self.tol)[0]]


def fit_phase(leg, phase, widths=None):
    rows = leg.signature_rows(phase, widths)
    if not rows:
        empty = {"signatures": 0, "kernels": 0, "rank": 0,
                 "rank_deficient": False, "closure": None,
                 "unidentified_shapes": [], "buffers": 0, "measured_ms": 0.0}
        return {}, empty, None
    times, diag = solve_kernel_times(rows)
    ident = Identifiability(rows)
    diag["unidentified_shapes"] = ident.unidentified_shapes()
    diag["buffers"] = sum(n for _, _, n in rows)
    diag["measured_ms"] = sum(g for _, g, _ in rows) / 1e6
    return times, diag, ident


def build_rows(leg, width, phase, times):
    """Per-shape rows: dispatches per round, fitted GPU ms per round, share."""
    per_round_counts = leg.shape_dispatches[(width, phase)]
    rounds = leg.round_count(width, phase) or 1
    total_ns = leg.by_width_phase[f"w{width}|{phase}"]["gpu_ns"]
    out = []
    for shape, count in per_round_counts.items():
        per_round = count / rounds
        t = times.get(shape)
        ms = (t * per_round / 1e6) if t is not None else None
        parsed = parse_shape(shape)
        unit = qmv_unit(shape)
        out.append({
            "shape": shape,
            "kernel": parsed["kernel"] if parsed else shape,
            "grid": list(parsed["grid"]) if parsed else None,
            "family": family_of(parsed["kernel"] if parsed else shape),
            "unit": unit[0] if unit else None,
            "e71_interceptable": unit[2] if unit else None,
            "dispatches_per_round": per_round,
            "fitted_ns_per_dispatch": t,
            "ms_per_round": ms,
            "share": (ms * 1e6 * rounds / total_ns) if (ms and total_ns) else None,
        })
    out.sort(key=lambda r: -(r["ms_per_round"] or 0))
    return out, rounds, (total_ns / 1e6 / rounds if rounds else 0.0)


def _group(rows, key_fn, ident=None):
    acc = collections.defaultdict(
        lambda: {"kernels": 0, "dispatches_per_round": 0.0,
                 "ms_per_round": 0.0, "share": 0.0,
                 "weights": {}, "e71_interceptable": None})
    for r in rows:
        name = key_fn(r)
        if name is None:
            continue
        slot = acc[name]
        slot["kernels"] += 1
        slot["dispatches_per_round"] += r["dispatches_per_round"]
        slot["ms_per_round"] += r["ms_per_round"] or 0.0
        slot["share"] += r["share"] or 0.0
        slot["weights"][r["shape"]] = (slot["weights"].get(r["shape"], 0.0)
                                       + r["dispatches_per_round"])
        if r["e71_interceptable"] is not None:
            slot["e71_interceptable"] = r["e71_interceptable"]
    for slot in acc.values():
        slot["identified"] = (ident.identified(slot["weights"])
                              if ident is not None else None)
        slot.pop("weights")
    return dict(acc)


def family_totals(rows, ident=None):
    return _group(rows, lambda r: r["family"], ident)


def qmv_unit_totals(rows, ident=None):
    """Rung-3 table: qmv cost split by projection, with E71 reachability."""
    return _group(
        rows,
        lambda r: (None if r["family"] != "qmv" else
                   (r["unit"] or
                    f"unmapped grid.y={r['grid'][1] if r['grid'] else '?'}")),
        ident)


def fmt_ms(v):
    return "     n/a" if v is None else f"{v:8.3f}"


def fmt_pct(v):
    return "    n/a" if v is None else f"{100 * v:6.2f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--default", type=pathlib.Path, required=True)
    ap.add_argument("--isolated", type=pathlib.Path)
    ap.add_argument("--width", type=int, action="append", dest="widths")
    ap.add_argument("--skip-rounds", type=int, default=0)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args(argv)

    default = Leg(args.default, args.skip_rounds)
    isolated = Leg(args.isolated, args.skip_rounds) if args.isolated else None
    phase = args.phase
    widths = args.widths or default.widths(phase)

    print("# E80 per-kernel GPU-time census\n")
    print(f"default leg  = {args.default}")
    print(f"isolated leg = {args.isolated}")
    print(f"phase = {phase}   widths = {widths}   "
          f"warmup rounds dropped = {args.skip_rounds}")

    print("\n## phase coverage, including `begin()`\n")
    print("| leg | width and phase | GPU ms | buffers | dispatches |")
    print("|---|---|---:|---:|---:|")
    for label, leg in (("default", default), ("isolated", isolated)):
        if leg is None:
            continue
        for key in sorted(leg.by_width_phase,
                          key=lambda k: -leg.by_width_phase[k]["gpu_ns"]):
            b = leg.by_width_phase[key]
            print(f"| {label} | {key} | {b['gpu_ns']/1e6:.1f} | "
                  f"{b['buffers']} | {b['dispatches']} |")

    fits = {}
    for label, leg in (("default", default), ("isolated", isolated)):
        if leg is None:
            continue
        times, diag, ident = fit_phase(leg, phase, set(widths))
        fits[label] = (times, diag, ident)
        print(f"\n## NNLS fit, {label} leg, phase {phase}, widths pooled {widths}\n")
        print(f"- equations (distinct signatures): {diag['signatures']}")
        print(f"- buffers behind them: {diag['buffers']}")
        print(f"- unknown shapes: {diag['kernels']}")
        print(f"- design-matrix rank: {diag['rank']}, "
              f"rank deficient: {diag['rank_deficient']}")
        closure = diag.get("closure")
        print("- closure (fitted / measured): "
              + (f"{closure:.4f}" if closure else "n/a"))
        free = diag.get("unidentified_shapes") or []
        if free:
            print(f"- {len(free)} of {diag['kernels']} shapes are not "
                  f"individually identified. Every reported family, qmv unit "
                  f"and rider below carries its own identifiability verdict, "
                  f"because a weighted sum can be determined even when its "
                  f"terms are not.")
            for s in free:
                print(f"    - {s}")
        else:
            print("- every fitted shape is individually identified")

    if not fits:
        print("\nno signature data; nothing to fit")
        return 1
    primary = "isolated" if "isolated" in fits else "default"
    times, _, primary_ident = fits[primary]
    leg = isolated if primary == "isolated" else default
    print(f"\nper-kernel times below come from the **{primary}** leg")

    tables = {}
    for width in widths:
        rows, rounds, ms_round = build_rows(leg, width, phase, times)
        if not rows:
            continue
        tables[width] = {"rows": rows, "rounds": rounds,
                         "measured_ms_per_round": ms_round}
        print(f"\n## width M = {width}   rounds = {rounds}   "
              f"measured {phase} = {ms_round:.3f} ms/round\n")
        print("| kernel | grid | unit | family | disp/round | ns/disp | "
              "ms/round | share |")
        print("|---|---|---|---|---:|---:|---:|---:|")
        for r in rows:
            grid = "x".join(str(g) for g in r["grid"]) if r["grid"] else "?"
            ns = ("" if r["fitted_ns_per_dispatch"] is None
                  else str(int(r["fitted_ns_per_dispatch"])))
            print(f"| {r['kernel'][:52]} | {grid} | {r['unit'] or ''} | "
                  f"{r['family']} | {r['dispatches_per_round']:.1f} | {ns} | "
                  f"{fmt_ms(r['ms_per_round'])} | {fmt_pct(r['share'])} |")

        fam = family_totals(rows, primary_ident)
        print("\n| family | kernels | disp/round | ms/round | share | identified |")
        print("|---|---:|---:|---:|---:|---|")
        for name, slot in sorted(fam.items(),
                                 key=lambda kv: -kv[1]["ms_per_round"]):
            print(f"| {name} | {slot['kernels']} | "
                  f"{slot['dispatches_per_round']:.1f} | "
                  f"{slot['ms_per_round']:8.3f} | {100*slot['share']:6.2f}% | "
                  f"{slot['identified']} |")

        units = qmv_unit_totals(rows, primary_ident)
        if units:
            print("\n| qmv unit | reachable by an E71 arm | disp/round | "
                  "ms/round | share | identified |")
            print("|---|---|---:|---:|---:|---|")
            for name, slot in sorted(units.items(),
                                     key=lambda kv: -kv[1]["ms_per_round"]):
                print(f"| {name} | {slot['e71_interceptable']} | "
                      f"{slot['dispatches_per_round']:.1f} | "
                      f"{slot['ms_per_round']:8.3f} | "
                      f"{100*slot['share']:6.2f}% | {slot['identified']} |")

        unclassified = [r for r in rows if r["family"] == "unclassified"]
        if unclassified:
            print(f"\nunclassified kernels ({len(unclassified)}), each named "
                  f"with its GPU time:")
            for r in unclassified:
                print(f"  - {r['kernel']}  grid="
                      f"{'x'.join(str(g) for g in r['grid']) if r['grid'] else '?'}"
                      f"  {fmt_ms(r['ms_per_round'])} ms/round")
        else:
            print("\nunclassified kernels: 0")

    # -- concurrency discount, per family ---------------------------------
    discounts = {"phase_level": {}, "per_family": {}}
    if isolated is not None:
        print("\n## concurrency discount\n")
        print("The discount is the in-situ default configuration divided by the "
              "one-op-per-buffer isolated configuration. At phase level it needs "
              "no fit at all: both legs run the same dispatches, so the ratio of "
              "measured GPU time is exact.\n")
        print("| width and phase | isolated GPU ms | default GPU ms | "
              "isolated buffers | default buffers | discount |")
        print("|---|---:|---:|---:|---:|---:|")
        for width in widths:
            key = f"w{width}|{phase}"
            bi = isolated.by_width_phase.get(key)
            bd = default.by_width_phase.get(key)
            if not bi or not bd or not bi["gpu_ns"]:
                continue
            ratio = bd["gpu_ns"] / bi["gpu_ns"]
            discounts["phase_level"][key] = {
                "isolated_ms": bi["gpu_ns"] / 1e6,
                "default_ms": bd["gpu_ns"] / 1e6,
                "isolated_buffers": bi["buffers"],
                "default_buffers": bd["buffers"],
                "discount": ratio,
            }
            print(f"| {key} | {bi['gpu_ns']/1e6:.1f} | {bd['gpu_ns']/1e6:.1f} | "
                  f"{bi['buffers']} | {bd['buffers']} | {ratio:.4f} |")

    if "default" in fits and "isolated" in fits and tables:
        d_times, _, d_ident = fits["default"]
        i_times, _, i_ident = fits["isolated"]
        width = max(tables, key=lambda w: tables[w]["rounds"])
        rows_i, _, _ = build_rows(isolated, width, phase, i_times)
        rows_d, _, _ = build_rows(default, width, phase, d_times)
        fam_i = family_totals(rows_i, i_ident)
        fam_d = family_totals(rows_d, d_ident)
        print(f"\nPer family, at width M = {width}. A family is only "
              f"comparable when BOTH legs identify it.\n")
        print("| family | isolated ms/round | default ms/round | discount | "
              "both identified |")
        print("|---|---:|---:|---:|---|")
        for name in sorted(set(fam_i) | set(fam_d)):
            si, sd = fam_i.get(name, {}), fam_d.get(name, {})
            mi, md = si.get("ms_per_round"), sd.get("ms_per_round")
            both = bool(si.get("identified")) and bool(sd.get("identified"))
            ratio = (md / mi) if (mi and md) else None
            discounts["per_family"][name] = {
                "width": width, "isolated_ms": mi, "default_ms": md,
                "discount": ratio, "both_identified": both}
            print(f"| {name} | {fmt_ms(mi)} | {fmt_ms(md)} | "
                  f"{'n/a' if ratio is None else f'{ratio:.3f}'} | {both} |")

    # -- falsification riders ---------------------------------------------
    verdicts = {}
    rider_width = 6 if 6 in tables else (max(tables) if tables else None)
    if rider_width is not None:
        fam = family_totals(tables[rider_width]["rows"], primary_ident)
        print(f"\n## falsification riders, at width M = {rider_width}\n")
        print("| rider | families | measured share | limit | verdict | identified |")
        print("|---|---|---:|---:|---|---|")
        for name, families, limit, note in RIDERS:
            share = sum(fam.get(f, {}).get("share", 0.0) for f in families)
            ident_ok = all(fam[f].get("identified") for f in families if f in fam)
            ok = share < limit
            verdicts[name] = {"share": share, "limit": limit, "pass": ok,
                              "families": families, "note": note,
                              "identified": ident_ok}
            print(f"| {name} | {', '.join(families)} | {100*share:.3f}% | "
                  f"{100*limit:.1f}% | {'PASS' if ok else 'FAIL'} | {ident_ok} |")
        copy_share = verdicts["copy"]["share"]
        if copy_share > 0.01:
            print(f"\n**HEADLINE: `copy` is {100*copy_share:.2f} % of verify GPU "
                  f"time, above the 1 % rider. Ledger entry 218 reopens.**")

    # -- ranked-weighted family shares ------------------------------------
    weighted = {}
    usable = {w: t for w, t in tables.items() if w in RANKED_WIDTH_WEIGHTS}
    if usable:
        covered = sum(RANKED_WIDTH_WEIGHTS[w] for w in usable)
        print("\n## ranked-weighted family shares\n")
        print(f"covered ranked mass = {100*covered:.2f} % over widths "
              f"{sorted(usable)}; weights renormalised over the covered widths. "
              f"Widths {sorted(set(RANKED_WIDTH_WEIGHTS) - set(usable))} were "
              f"not measured and are NOT interpolated.")
        acc = collections.defaultdict(float)
        for w, t in usable.items():
            weight = RANKED_WIDTH_WEIGHTS[w] / covered
            for name, slot in family_totals(t["rows"], primary_ident).items():
                acc[name] += weight * slot["share"]
        weighted = dict(acc)
        print("\n| family | ranked-weighted share |")
        print("|---|---:|")
        for name, share in sorted(weighted.items(), key=lambda kv: -kv[1]):
            print(f"| {name} | {100*share:.2f}% |")

    print("\n## instrument health\n")
    print("| leg | " + " | ".join(sorted(default.health)) + " |")
    print("|---" * (1 + len(default.health)) + "|")
    for label, lg in (("default", default), ("isolated", isolated)):
        if lg is None:
            continue
        print(f"| {label} | " +
              " | ".join(str(lg.health[k]) for k in sorted(default.health)) + " |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "default_census": str(args.default),
            "isolated_census": str(args.isolated) if args.isolated else None,
            "phase": phase,
            "skip_rounds": args.skip_rounds,
            "ranked_width_weights": RANKED_WIDTH_WEIGHTS,
            "fits": {k: {"diag": v[1], "times_ns": v[0]}
                     for k, v in fits.items()},
            "widths": {str(w): t for w, t in tables.items()},
            "families_per_width": {str(w): family_totals(t["rows"], primary_ident)
                                   for w, t in tables.items()},
            "qmv_units_per_width": {str(w): qmv_unit_totals(t["rows"], primary_ident)
                                    for w, t in tables.items()},
            "concurrency_discount": discounts,
            "riders": verdicts,
            "ranked_weighted_family_share": weighted,
            "phase_gpu_ms": {
                "default": {k: v["gpu_ns"] / 1e6
                            for k, v in default.by_width_phase.items()},
                "isolated": ({k: v["gpu_ns"] / 1e6
                              for k, v in isolated.by_width_phase.items()}
                             if isolated else None),
            },
            "health": {"default": dict(default.health),
                       "isolated": dict(isolated.health) if isolated else None},
        }, indent=2, default=float) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
