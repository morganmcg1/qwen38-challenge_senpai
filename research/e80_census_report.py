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

    def __init__(self, paths, skip_rounds: int = 0):
        # Several legs may be merged. Every bucket key carries its verify
        # width, and each leg forces a different width, so the only bucket the
        # legs share is the serial reference pass at `w1|target_forward`, where
        # pooling simply adds rounds of the same computation.
        self.paths = [paths] if isinstance(paths, pathlib.Path) else list(paths)
        self.path = self.paths[0]
        self.skip_rounds = skip_rounds
        self.by_width_phase = collections.defaultdict(
            lambda: {"gpu_ns": 0, "buffers": 0, "dispatches": 0})
        self.signatures = collections.defaultdict(
            lambda: {"gpu_ns": 0, "buffers": 0, "dispatches": 0})
        self.rounds = collections.Counter()
        self.dispatches = collections.Counter()
        self.shape_dispatches = collections.defaultdict(collections.Counter)
        self.health = collections.Counter()
        self.exclusive = collections.defaultdict(
            lambda: {"gpu_ns": 0, "buffers": 0})
        # Round-level, not phase-level: one entry per `round` record, so a
        # round's wall clock is never multiplied by its phase count.
        self.round_total = collections.Counter()
        self.round_wall_ns = collections.Counter()
        self.round_commits = collections.Counter()
        self.round_dispatches = collections.Counter()
        # Host synchronisation points: `waitUntilCompleted` calls, and the host
        # nanoseconds blocked inside them. Present only when the census build
        # carries the H-221 wait hook.
        self.round_waits = collections.Counter()
        self.round_wait_ns = collections.Counter()
        self._load()

    def _load(self):
        for path in self.paths:
            self._load_one(path)

    def _load_one(self, path):
        for rec in read_records(path):
            event = rec.get("event")
            if event == "round":
                if rec.get("round", 0) <= self.skip_rounds:
                    continue
                width = rec["width"]
                self.round_total[width] += 1
                self.round_wall_ns[width] += rec.get("wall_ns", 0)
                for phase, entry in (rec.get("phases") or {}).items():
                    self.rounds[(width, phase)] += 1
                    self.dispatches[(width, phase)] += entry.get("dispatches", 0)
                    self.round_commits[width] += entry.get("commits", 0)
                    self.round_dispatches[width] += entry.get("dispatches", 0)
                    self.round_waits[width] += entry.get("waits", 0)
                    self.round_wait_ns[width] += entry.get("wait_ns", 0)
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
                for key, b in (rec.get("exclusive_kernels") or {}).items():
                    slot = self.exclusive[key]
                    slot["gpu_ns"] += b.get("gpu_ns", 0)
                    slot["buffers"] += b.get("buffers", 0)
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

    def exclusive_ns(self, width, phase, shape):
        """Directly measured ns for a dispatch that owned its command buffer.

        This is the only per-kernel number in the census that needs no fit, so
        it is the ground truth the NNLS solution is checked against. Coverage
        is small: on this runtime one MLX op emits about two dispatches, so
        only a few shapes ever get a buffer to themselves.
        """
        slot = self.exclusive.get(f"w{width}|{phase}|{shape}")
        if not slot or slot["buffers"] <= 0:
            return None, 0
        return slot["gpu_ns"] / slot["buffers"], slot["buffers"]


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

    def degenerate_groups(self) -> list[list[str]]:
        """Shapes the null space ties together, as connected components.

        A single unidentified shape is not actionable, but a group is: the fit
        can trade time between the members of one group and nowhere else, so a
        weighted sum over a whole group is often determined even though no
        member is. Reporting the group states exactly what the data does and
        does not separate.
        """
        if self.null.size == 0:
            return []
        load = np.abs(self.null) > self.tol
        parent = list(range(len(self.keys)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in load:
            idx = [int(i) for i in np.where(row)[0]]
            for j in idx[1:]:
                ra, rb = find(idx[0]), find(j)
                if ra != rb:
                    parent[rb] = ra
        groups = collections.defaultdict(list)
        for i in range(len(self.keys)):
            if load[:, i].any():
                groups[find(i)].append(self.keys[i])
        return sorted((sorted(v) for v in groups.values()), key=len, reverse=True)


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


def family_pure_rates(leg, phase, width):
    """Per-family ns per dispatch from buffers that hold ONE family only.

    This needs no fit and no null-space argument: a buffer whose every
    dispatch belongs to one family charges its whole GPU interval to that
    family. Coverage is partial, so the table also reports the share of the
    phase's GPU time these buffers carry. Applying the rate to the family's
    full dispatch count assumes a pure buffer prices a mixed one, which is an
    assumption, not a measurement, and is labelled as such.
    """
    pure = collections.defaultdict(
        lambda: {"gpu_ns": 0, "dispatches": 0, "buffers": 0,
                 "shape_dispatches": collections.Counter()})
    total_ns = 0
    for key, v in leg.signatures.items():
        w, _, rest = key.partition("|")
        ph, _, sig = rest.partition("|")
        if ph != phase or w != f"w{width}":
            continue
        total_ns += v["gpu_ns"]
        counts = parse_signature(sig)
        families = set()
        for shape in counts:
            parsed = parse_shape(shape)
            families.add(family_of(parsed["kernel"] if parsed else shape))
        if len(families) != 1:
            continue
        slot = pure[families.pop()]
        slot["gpu_ns"] += v["gpu_ns"]
        slot["dispatches"] += sum(counts.values()) * v["buffers"]
        slot["buffers"] += v["buffers"]
        for shape, c in counts.items():
            slot["shape_dispatches"][shape] += c * v["buffers"]
    return pure, total_ns


def width_tax_decomposition(leg, high_width, low_width=1,
                            high_phase="target_verify",
                            low_phase="target_forward", anchor=None,
                            leg_label="isolated"):
    """Rung 3. Splits `F(M) - F(1)` by family inside ONE leg.

    Every leg runs a serial reference pass, so `w1|target_forward` and
    `wM|target_verify` are measured in the same session, on the same build, at
    the same thermal state. That is a tighter comparison than E71's, which
    differenced two census blocks.
    """
    lo_times, lo_diag, lo_ident = fit_phase(leg, low_phase, {low_width})
    hi_times, hi_diag, hi_ident = fit_phase(leg, high_phase, {high_width})
    lo_rows, lo_n, lo_ms = build_rows(leg, low_width, low_phase, lo_times)
    hi_rows, hi_n, hi_ms = build_rows(leg, high_width, high_phase, hi_times)
    lo_fam = family_totals(lo_rows, lo_ident)
    hi_fam = family_totals(hi_rows, hi_ident)
    lo_unit = qmv_unit_totals(lo_rows, lo_ident)
    hi_unit = qmv_unit_totals(hi_rows, hi_ident)
    tax = hi_ms - lo_ms
    print(f"\n## rung 3: F({high_width}) - F({low_width}) in GPU time, "
          f"decomposed inside one leg\n")
    print(f"F({low_width}) = {lo_ms:.3f} ms/round over {lo_n} rounds "
          f"({low_phase}); F({high_width}) = {hi_ms:.3f} ms/round over "
          f"{hi_n} rounds ({high_phase}); tax = **{tax:.3f} ms/round**.\n")
    print(f"The split comes from the **{leg_label}** leg, whose design matrix "
          f"has the higher rank. Rows are fitted times; the two totals are "
          f"measured.\n")
    print("| side | phase | signatures | buffers | shapes | rank | "
          "unidentified | in degenerate ms/round | closure |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, d, i, ph, wd, tms in ((f"F({low_width})", lo_diag, lo_ident,
                                    low_phase, low_width, lo_times),
                                   (f"F({high_width})", hi_diag, hi_ident,
                                    high_phase, high_width, hi_times)):
        deg_ms = 0.0
        if i is not None:
            counts = leg.shape_dispatches[(wd, ph)]
            rounds = leg.round_count(wd, ph) or 1
            for g in i.degenerate_groups():
                deg_ms += sum(tms.get(s, 0.0) * counts.get(s, 0) / rounds
                              for s in g) / 1e6
        cl = d.get("closure")
        print(f"| {tag} | {ph} | {d['signatures']} | {d['buffers']} | "
              f"{d['kernels']} | {d['rank']} | "
              f"{len(d.get('unidentified_shapes') or [])} | {deg_ms:.3f} | "
              + (f"{cl:.4f} |" if cl else "n/a |"))
    print()
    if anchor is not None:
        a_lo = anchor.by_width_phase.get(f"w{low_width}|{low_phase}")
        a_hi = anchor.by_width_phase.get(f"w{high_width}|{high_phase}")
        a_lo_n = anchor.round_count(low_width, low_phase) or 1
        a_hi_n = anchor.round_count(high_width, high_phase) or 1
        if a_lo and a_hi and a_hi_n:
            a_lo_ms = a_lo["gpu_ns"] / 1e6 / a_lo_n
            a_hi_ms = a_hi["gpu_ns"] / 1e6 / a_hi_n
            a_tax = a_hi_ms - a_lo_ms
            print(f"Absolute anchor, measured with no fit on the **default** "
                  f"leg that the candidate actually runs: F({low_width}) = "
                  f"{a_lo_ms:.3f}, F({high_width}) = {a_hi_ms:.3f}, tax = "
                  f"**{a_tax:.3f} ms/round**. Isolating buffers inflates the "
                  f"tax by {100*(tax/a_tax - 1):+.1f} %, so read the shares "
                  f"below as shares and the anchor as milliseconds.\n")
            out_anchor = {"low_ms": a_lo_ms, "high_ms": a_hi_ms,
                          "tax_ms": a_tax, "inflation": tax / a_tax}
        else:
            out_anchor = None
    else:
        out_anchor = None
    print(f"| family | F({low_width}) ms | F({high_width}) ms | tax ms | "
          f"share of tax | both identified |")
    print("|---|---:|---:|---:|---:|---|")
    named = 0.0
    out = {"tax_ms": tax, f"F{low_width}_ms": lo_ms,
           f"F{high_width}_ms": hi_ms, "families": {}, "qmv_units": {},
           "fit_leg": leg_label, "default_leg_anchor": out_anchor}
    for name in sorted(set(lo_fam) | set(hi_fam),
                       key=lambda k: -(hi_fam.get(k, {}).get("ms_per_round", 0.0)
                                       - lo_fam.get(k, {}).get("ms_per_round", 0.0))):
        lo = lo_fam.get(name, {}).get("ms_per_round", 0.0)
        hi = hi_fam.get(name, {}).get("ms_per_round", 0.0)
        both = (bool(lo_fam.get(name, {}).get("identified", True))
                and bool(hi_fam.get(name, {}).get("identified", True)))
        delta = hi - lo
        named += delta
        out["families"][name] = {"low_ms": lo, "high_ms": hi, "tax_ms": delta,
                                 "both_identified": both}
        print(f"| {name} | {lo:8.3f} | {hi:8.3f} | {delta:8.3f} | "
              f"{100*delta/tax if tax else 0:6.2f}% | {both} |")
    print(f"\nnamed families sum to {named:.3f} ms, which closes "
          f"{100*named/tax if tax else 0:.2f} % of the tax.")
    print(f"\n| qmv unit | reachable by an E71 arm | F({low_width}) ms | "
          f"F({high_width}) ms | tax ms | share of tax |")
    print("|---|---|---:|---:|---:|---:|")
    for name in sorted(set(lo_unit) | set(hi_unit),
                       key=lambda k: -(hi_unit.get(k, {}).get("ms_per_round", 0.0)
                                       - lo_unit.get(k, {}).get("ms_per_round", 0.0))):
        lo = lo_unit.get(name, {}).get("ms_per_round", 0.0)
        hi = hi_unit.get(name, {}).get("ms_per_round", 0.0)
        reach = (hi_unit.get(name) or lo_unit.get(name) or {}).get(
            "e71_interceptable")
        delta = hi - lo
        out["qmv_units"][name] = {"low_ms": lo, "high_ms": hi, "tax_ms": delta,
                                  "e71_interceptable": reach}
        print(f"| {name} | {reach} | {lo:8.3f} | {hi:8.3f} | {delta:8.3f} | "
              f"{100*delta/tax if tax else 0:6.2f}% |")
    unreachable = sum(v["tax_ms"] for v in out["qmv_units"].values()
                      if v["e71_interceptable"] is False)
    out["qmv_unreachable_by_e71_ms"] = unreachable
    out["qmv_unreachable_share_of_tax"] = unreachable / tax if tax else None
    print(f"\nqmv units no E71 arm can intercept carry {unreachable:.3f} ms, "
          f"{100*unreachable/tax if tax else 0:.2f} % of the tax. E71 left "
          f"22.6 % unattributed.")
    return out


def h221_table(default, isolated):
    """Prices the host boundary instead of the kernel.

    `host_boundary_ms` is the advisor's closure gap: the round's wall clock
    minus every millisecond of GPU time the census can name. With
    `MLX_MAX_OPS_PER_BUFFER=1` one command buffer holds exactly one MLX
    primitive op, so the isolated leg's buffer count IS the MLX op count and
    the gap can be divided by it directly.

    The last column needs no attribution assumption at all. Both legs run the
    same ops in a different number of command buffers, so the wall difference
    over the commit difference is an upper bound on ONE command-buffer
    boundary.
    """
    rows = []
    print("\n## H-221: host boundary cost, wall clock minus named GPU time\n")
    print("| width | rounds | wall ms/round | GPU ms/round | "
          "`host_boundary_ms`/round | GPU/wall | commits/round | "
          "MLX ops/round | ms per MLX op, uniform | "
          "ms per MLX op, head path only |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for width in sorted(default.round_total):
        n = default.round_total[width]
        if not n:
            continue
        wall = default.round_wall_ns[width] / 1e6 / n
        gpu = sum(v["gpu_ns"] for k, v in default.by_width_phase.items()
                  if k.split("|", 1)[0] == f"w{width}") / 1e6 / n
        gap = wall - gpu
        row = {
            "width": width, "rounds": n,
            "wall_ms_per_round": wall, "gpu_ms_per_round": gpu,
            "host_boundary_ms_per_round": gap,
            "gpu_over_wall": gpu / wall if wall else None,
            "commits_per_round": default.round_commits[width] / n,
            "waits_per_round": default.round_waits[width] / n,
            "wait_ms_per_round": default.round_wait_ns[width] / 1e6 / n,
        }
        ops = head_ops = None
        if isolated is not None and isolated.round_total.get(width):
            m = isolated.round_total[width]
            ops = sum(v["buffers"] for k, v in isolated.by_width_phase.items()
                      if k.split("|", 1)[0] == f"w{width}") / m
            head_ops = sum(
                v["buffers"] for k, v in isolated.by_width_phase.items()
                if k == f"w{width}|draft_head") / m
            row["mlx_ops_per_round"] = ops
            row["mlx_ops_head_path_per_round"] = head_ops
            row["ms_per_mlx_op_uniform"] = gap / ops if ops else None
            row["ms_per_mlx_op_head_only"] = gap / head_ops if head_ops else None
            i_wall = isolated.round_wall_ns[width] / 1e6 / m
            i_commits = isolated.round_commits[width] / m
            row["isolated_wall_ms_per_round"] = i_wall
            row["isolated_commits_per_round"] = i_commits
            if i_commits > row["commits_per_round"]:
                row["ms_per_command_buffer_boundary"] = (
                    (i_wall - wall) / (i_commits - row["commits_per_round"]))
        rows.append(row)
        print(f"| {width} | {n} | {wall:.3f} | {gpu:.3f} | {gap:.3f} | "
              f"{row['gpu_over_wall']:.4f} | {row['commits_per_round']:.1f} | "
              f"{'n/a' if ops is None else f'{ops:.1f}'} | "
              f"{fmt_us(row.get('ms_per_mlx_op_uniform'))} | "
              f"{fmt_us(row.get('ms_per_mlx_op_head_only'))} |")
    if any(r.get("waits_per_round") for r in rows):
        print("\nHost synchronisation points. MLX blocks the host only in "
              "`CommandEncoder::synchronize()`, which ends encoding, commits "
              "and calls `waitUntilCompleted`. Dividing the closure gap by "
              "those calls prices one synchronisation instead of one op.\n")
        print("| width | sync points/round | blocked ms/round | "
              "`host_boundary_ms`/round | ms per sync point | "
              "gap left after blocked time |")
        print("|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            w = row.get("waits_per_round") or 0.0
            if not w:
                continue
            blocked = row["wait_ms_per_round"]
            gap = row["host_boundary_ms_per_round"]
            print(f"| {row['width']} | {w:.1f} | {blocked:.3f} | {gap:.3f} | "
                  f"{gap/w:.4f} | {gap-blocked:.3f} |")
    if isolated is not None:
        print("\nPacking bound. Both legs run the same MLX ops. Only the "
              "number of command buffers changes, so the wall difference over "
              "the commit difference is an upper bound on one command-buffer "
              "boundary.\n")
        print("| width | default commits/round | isolated commits/round | "
              "default wall ms | isolated wall ms | ms per command buffer |")
        print("|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            if "ms_per_command_buffer_boundary" not in row:
                continue
            print(f"| {row['width']} | {row['commits_per_round']:.1f} | "
                  f"{row['isolated_commits_per_round']:.1f} | "
                  f"{row['wall_ms_per_round']:.3f} | "
                  f"{row['isolated_wall_ms_per_round']:.3f} | "
                  f"{row['ms_per_command_buffer_boundary']:.5f} |")
    return rows


def fmt_ms(v):
    return "     n/a" if v is None else f"{v:8.3f}"


def fmt_us(v):
    return "n/a" if v is None else f"{v:.5f}"


def fmt_pct(v):
    return "    n/a" if v is None else f"{100 * v:6.2f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--default", type=pathlib.Path, required=True,
                    action="append")
    ap.add_argument("--isolated", type=pathlib.Path, action="append")
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

    h221 = h221_table(default, isolated)

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
            groups = ident.degenerate_groups() if ident is not None else []
            print(f"- {len(free)} of {diag['kernels']} shapes are not "
                  f"individually identified, in {len(groups)} degenerate "
                  f"groups. The fit can move time between the members of one "
                  f"group and nowhere else, so a sum over a whole group is "
                  f"often exact even though no member is. Every family, qmv "
                  f"unit and rider below carries its own verdict.")
            for gi, g in enumerate(groups, 1):
                per_round = {}
                for width in widths:
                    counts = leg.shape_dispatches[(width, phase)]
                    rounds = leg.round_count(width, phase) or 1
                    ns = sum(times.get(s, 0.0) * counts.get(s, 0) / rounds
                             for s in g)
                    if ns:
                        per_round[width] = ns / 1e6
                w = {}
                for width in widths:
                    counts = leg.shape_dispatches[(width, phase)]
                    rounds = leg.round_count(width, phase) or 1
                    for s in g:
                        w[s] = w.get(s, 0.0) + counts.get(s, 0) / rounds
                sum_ok = ident.identified(w) if ident is not None else False
                shown = "  ".join(f"w{k} {v:.3f} ms/round"
                                  for k, v in sorted(per_round.items()))
                shown = shown or "no dispatches at the reported widths"
                print(f"    - group {gi} ({len(g)} shapes): {shown}; "
                      f"group sum identified: {sum_ok}")
                for s in g:
                    print(f"        . {s}")
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
              "ms/round | share | id? | direct ns/disp |")
        print("|---|---|---|---|---:|---:|---:|---:|---|---:|")
        for r in rows:
            grid = "x".join(str(g) for g in r["grid"]) if r["grid"] else "?"
            ns = ("" if r["fitted_ns_per_dispatch"] is None
                  else str(int(r["fitted_ns_per_dispatch"])))
            solo = (primary_ident.identified({r["shape"]: 1.0})
                    if primary_ident else False)
            r["identified"] = bool(solo)
            direct, n_direct = leg.exclusive_ns(width, phase, r["shape"])
            r["direct_ns_per_dispatch"] = direct
            r["direct_buffers"] = n_direct
            direct_txt = "" if direct is None else f"{int(direct)} (n={n_direct})"
            print(f"| {r['kernel'][:52]} | {grid} | {r['unit'] or ''} | "
                  f"{r['family']} | {r['dispatches_per_round']:.1f} | {ns} | "
                  f"{fmt_ms(r['ms_per_round'])} | {fmt_pct(r['share'])} | "
                  f"{'yes' if solo else 'NO'} | {direct_txt} |")

        fam = family_totals(rows, primary_ident)
        print("\n| family | kernels | disp/round | ms/round | share | identified |")
        print("|---|---:|---:|---:|---:|---|")
        for name, slot in sorted(fam.items(),
                                 key=lambda kv: -kv[1]["ms_per_round"]):
            print(f"| {name} | {slot['kernels']} | "
                  f"{slot['dispatches_per_round']:.1f} | "
                  f"{slot['ms_per_round']:8.3f} | {100*slot['share']:6.2f}% | "
                  f"{slot['identified']} |")

        pure, pure_total_ns = family_pure_rates(leg, phase, width)
        if pure:
            covered = sum(s["gpu_ns"] for s in pure.values())
            print(f"\nFit-free cross-check. Buffers holding ONE family only "
                  f"carry {100*covered/pure_total_ns:.1f} % of this phase's "
                  f"signature GPU time. No fit, no null space.\n")
            print("The NNLS column reweights the fitted times to the exact "
                  "shape mix of the pure sample, so the two columns describe "
                  "the same dispatches. A ratio below one means the fit "
                  "charges a dispatch less than it costs alone, which is what "
                  "overlap inside a shared buffer looks like.\n")
            print("| family | pure buffers | pure disp | ns/disp (direct) | "
                  "ns/disp (NNLS, same mix) | ratio |")
            print("|---|---:|---:|---:|---:|---:|")
            for name, slot in sorted(pure.items(), key=lambda kv: -kv[1]["gpu_ns"]):
                direct = slot["gpu_ns"] / slot["dispatches"] if slot["dispatches"] else None
                mix = slot.get("shape_dispatches") or {}
                num = sum(times.get(s, 0.0) * c for s, c in mix.items())
                den = sum(c for s, c in mix.items() if s in times)
                fitted = (num / den) if den else None
                ratio = (fitted / direct) if (direct and fitted) else None
                print(f"| {name} | {slot['buffers']} | {slot['dispatches']} | "
                      f"{'n/a' if direct is None else int(direct)} | "
                      f"{'n/a' if fitted is None else int(fitted)} | "
                      f"{'n/a' if ratio is None else f'{ratio:.3f}'} |")

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

    # -- rung 3: the width tax, decomposed --------------------------------
    rung3 = {}
    if phase == "target_verify":
        for width in sorted(tables):
            if width <= 1:
                continue
            rung3[str(width)] = width_tax_decomposition(
                leg, width, anchor=(default if leg is not default else None),
                leg_label=primary)

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
            "h221_host_boundary": h221,
            "rung3_width_tax": rung3,
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
