#!/usr/bin/env python3
"""Zero-GPU transcription of the MLX SDPA dispatch law, used to predict and
validate the key-length-1024 residual band for E19.

Every rule below is a literal transcription of vendored source. Citations are
relative to the E19 base (BASE_SHA 1bb627ab9339fd17c7560bd3d1134dc40fbb5885):

  CPP = Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/
        scaled_dot_product_attention.cpp
  HDR = Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h
  FAST = Vendor/mlx-swift/Source/Cmlx/mlx/mlx/fast.cpp
  ATTN = Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift

Usage:
  python3 research/sdpa_keylen_band.py --report
  python3 research/sdpa_keylen_band.py --validate
  python3 research/sdpa_keylen_band.py --repair
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model geometry. Source: weights/config.json, fixtures/qwen3_6_27b_config.json
# (text_config), and the fail-closed check at
# Sources/MLXFastTrustedHarness/QwenRuntimeWorker.swift:699-741.
# ---------------------------------------------------------------------------
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
GQA_FACTOR = NUM_Q_HEADS // NUM_KV_HEADS  # 6

SEED_TOKENS = 512
DECODE_TOKENS = 512
MAX_KEY_LEN = SEED_TOKENS + DECODE_TOKENS  # 1024

FULL = "fallback-composite-bf16"
VEC1 = "sdpa_vector"
VEC2 = "sdpa_vector_2pass"


# ---------------------------------------------------------------------------
# Dispatch law
# ---------------------------------------------------------------------------
def supports_sdpa_vector(q_len: int, k_len: int) -> bool:
    """CPP:635-638."""
    return (
        q_len <= 8
        and q_len <= k_len
        and HEAD_DIM in (64, 96, 128, 256)
        and q_len * GQA_FACTOR <= 32
    )


def supports_sdpa_full(q_len: int, k_len: int, do_causal: bool) -> bool:
    """CPP:626-633. head_dim 256 is absent from the full-kernel list, so this is
    always False for this checkpoint."""
    full_head_dim = HEAD_DIM in (64, 80, 128)
    full_mask = q_len <= k_len and do_causal
    return q_len > 8 and full_mask and full_head_dim


def blocks_for(devc: str, k_len: int, q_len: int) -> int:
    """CPP:440-478. n_simds = gqa_factor * q.shape(2)."""
    n_simds = GQA_FACTOR * q_len
    n = k_len
    if devc == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            blocks = 128 if n <= 8192 else 256 if n <= 32768 else 512 if n <= 65536 else 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return blocks


@dataclass(frozen=True)
class Dispatch:
    family: str
    blocks: int | None
    q_len: int
    k_len: int
    do_causal: bool

    def usable_hi(self, q_seq_idx: int) -> int:
        """Highest key index this row may read.

        Single pass, HDR:99-102, group_dims(1024,1,1) / grid_dims(B*H, qL, 1)
        so tpg.y == q_len:      use_key = i <= (N - q_len + q_seq_idx)
        Two pass, HDR:~230,
        tptg == (32, gqa, qL):  use_key = i <= (N - q_len + q_seq_idx)
        do_causal is forced false at q_len == 1 (CPP:745), which yields the same
        set because N - 1 + 0 == N - 1.
        """
        if not self.do_causal:
            return self.k_len - 1
        return self.k_len - self.q_len + q_seq_idx

    def row_kernel(self, q_seq_idx: int) -> tuple:
        """Numerically relevant identity of one row's attention output.

        Both kernels stride the key axis with a compile-time constant stride
        (single pass `i += BN`, BN = 32, HDR:43,99; two pass `i += blocks`,
        HDR:~235) and skip masked keys instead of shortening the loop. So the
        set of usable keys assigned to each thread, and the order within a
        thread, depend only on (family, blocks, usable key set) -- never on the
        allocated k_len, and never on the round width q_len. That invariance is
        exactly why chunk A of a segmented call reproduces serial bit for bit,
        and why a two-pass row is width-invariant.
        """
        return (self.family, self.blocks, self.usable_hi(q_seq_idx))


def dispatch(devc: str, q_len: int, k_len: int, causal_requested: bool) -> Dispatch:
    do_causal = causal_requested and q_len > 1  # CPP:745
    if supports_sdpa_vector(q_len, k_len):
        # CPP:685 vector mode; CPP:748-751 two-pass gate. The second disjunct
        # needs k_len >= 4096, unreachable in a 1024-token window.
        two_pass = (devc in ("d", "s") and k_len >= 1024) or k_len >= 4096
        family = VEC2 if two_pass else VEC1
        blocks = blocks_for(devc, k_len, q_len) if two_pass else None
        return Dispatch(family, blocks, q_len, k_len, do_causal)
    if supports_sdpa_full(q_len, k_len, do_causal):
        raise AssertionError("unreachable for head_dim=256")
    # FAST:828-869 -> `return fallback(inputs)[0]`: bf16 matmul + precise
    # softmax + bf16 matmul. A different numerical path entirely.
    return Dispatch(FULL, None, q_len, k_len, do_causal)


# ---------------------------------------------------------------------------
# Call segmentation. Source: ATTN, the WIDE-DECODE chunk.
#
#   let (cachedKeys, cachedValues) = cache.update(keys:values:)   // all w rows
#   if queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL, case .causal = mask {
#       let split = 5
#       let kSplit = kL - (qL - split)
#       outA = SDPA(q[0..<5],  k[0..<kSplit], v[0..<kSplit], .causal)
#       outB = SDPA(q[5...],   cachedKeys,    cachedValues,  .causal)
#   }
#
# kSplit == p_last(A) + 1, i.e. causal-aligned exactly.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    rows: tuple[int, ...]  # 0-based absolute cache positions
    k_len: int


def segments_current(p0: int, width: int) -> list[Segment]:
    """Segmentation performed by the shipped AttentionUtils chunk."""
    k_len = p0 + width
    rows = tuple(range(p0, p0 + width))
    if 6 <= width <= 9 and k_len >= width:
        split = 5
        k_split = k_len - (width - split)
        return [Segment(rows[:split], k_split), Segment(rows[split:], k_len)]
    return [Segment(rows, k_len)]


def segments_repair(p0: int, width: int) -> list[Segment]:
    """Strict-boundary repair: every row whose causal key length would cross the
    1024 two-pass threshold is isolated into its own call at full k_len, and all
    earlier rows are served at k_len = p_last + 1 <= 1023 (single pass, matching
    serial bit for bit).

    Falls back to the shipped chunk whenever the boundary is not crossed, so the
    6..9 fallback-composite hazard stays covered.
    """
    k_len = p0 + width
    rows = tuple(range(p0, p0 + width))
    if k_len < 1024 or width == 1:
        return segments_current(p0, width)
    low = tuple(p for p in rows if p <= 1022)
    high = tuple(p for p in rows if p >= 1023)
    out: list[Segment] = []
    if low:
        # Causal alignment forces k_len == p_last + 1 exactly.
        out.extend(_split_for_vector_width(low, low[-1] + 1))
    if high:
        out.extend(_split_for_vector_width(high, k_len))
    return out


def _split_for_vector_width(rows: tuple[int, ...], k_len: int) -> list[Segment]:
    """Keep every emitted call inside supports_sdpa_vector (q_len <= 5 at
    gqa_factor 6), preserving causal alignment for each piece."""
    out: list[Segment] = []
    i = 0
    while i < len(rows):
        piece = rows[i : i + 5]
        piece_k = k_len if piece[-1] == rows[-1] else piece[-1] + 1
        out.append(Segment(piece, piece_k))
        i += 5
    return out


# ---------------------------------------------------------------------------
# Per-row plans
# ---------------------------------------------------------------------------
class CausalMisalignment(AssertionError):
    pass


def serial_row(devc: str, p: int) -> tuple:
    """Serial depth-0 decode: one row per forward pass, k_len = p + 1."""
    d = dispatch(devc, 1, p + 1, causal_requested=True)
    assert d.usable_hi(0) == p
    return d.row_kernel(0)


def candidate_rows(devc: str, p0: int, width: int, segmenter) -> dict[int, tuple]:
    """Per-row kernel identity, asserting each segment is causally aligned.

    A segment is aligned only when every row's usable key ceiling equals its own
    absolute cache position. This is the off-by-one audit for any proposed
    segmentation: k_len must equal p_last + 1 unless the segment ends the call.
    """
    out: dict[int, tuple] = {}
    for seg in segmenter(p0, width):
        d = dispatch(devc, len(seg.rows), seg.k_len, causal_requested=True)
        for j, p in enumerate(seg.rows):
            if d.usable_hi(j) != p:
                raise CausalMisalignment(
                    f"p0={p0} w={width} segment rows={seg.rows} k_len={seg.k_len}: "
                    f"row p={p} (q_seq_idx={j}) may read keys up to {d.usable_hi(j)}"
                )
            out[p] = d.row_kernel(j)
    return out


# ---------------------------------------------------------------------------
# Round schedule
# ---------------------------------------------------------------------------
@dataclass
class Round:
    index: int
    p0: int      # cache offset at round start (0-based position of primary row)
    depth: int   # trace `d`
    accepted: int  # trace `acc`

    @property
    def width(self) -> int:
        """Rows evaluated by the verify pass: primary + depth drafts."""
        return self.depth + 1

    @property
    def key_len(self) -> int:
        return self.p0 + self.width

    @property
    def committed(self) -> tuple[int, ...]:
        return tuple(range(self.p0, self.p0 + self.accepted + 1))


@dataclass
class Analysis:
    rounds: list[Round]
    injected: set[int] = field(default_factory=set)
    inherited: set[int] = field(default_factory=set)
    detail: list[str] = field(default_factory=list)

    @property
    def suspect(self) -> set[int]:
        return self.injected | self.inherited


def analyze(devc: str, rounds: list[Round], segmenter=segments_current) -> Analysis:
    """Classify every emitted row as clean, directly injected, or inherited.

    Direct injection: the candidate's kernel signature for that row differs from
    serial's, so the row's own attention output may differ.

    Inherited: the row's signature matches serial, but it reads at least one key
    position whose hidden state was itself perturbed. Because cache.update()
    appends all `width` rows before the SDPA call, a row can read a same-pass
    key; and a committed perturbed position stays in the cache for later rounds.
    """
    a = Analysis(rounds=rounds)
    tainted_keys: set[int] = set()
    for r in rounds:
        cand = candidate_rows(devc, r.p0, r.width, segmenter)
        round_injected: set[int] = set()
        for p in sorted(cand):
            if serial_row(devc, p) != cand[p]:
                round_injected.add(p)
        emitted = set(r.committed)
        a.injected |= round_injected & emitted
        for p in sorted(emitted):
            if p in round_injected:
                continue
            # Keys visible to row p are absolute positions 0..p. Positions in
            # [p0, p] were written by cache.update() earlier in this same forward
            # pass (ATTN, update before the SDPA call), so an injected row in the
            # same round contaminates a later row of that round immediately.
            visible = set(range(0, p + 1))
            if visible & (tainted_keys | round_injected):
                a.inherited.add(p)
        tainted_keys |= (round_injected | a.inherited) & emitted
        if round_injected or (a.inherited & emitted):
            segs = segmenter(r.p0, r.width)
            a.detail.append(
                f"round={r.index} p0={r.p0} w={r.width} acc={r.accepted} kL={r.key_len} "
                + "segs=["
                + ", ".join(
                    f"q{len(s.rows)}@p{s.rows[0]}..{s.rows[-1]}/kL{s.k_len}"
                    f"->{dispatch(devc, len(s.rows), s.k_len, True).family}"
                    f"/b{dispatch(devc, len(s.rows), s.k_len, True).blocks}"
                    for s in segs
                )
                + "]"
            )
    return a


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
ROW_RE = re.compile(r"mtp-row: pos=(\d+) ")


def parse_trace(path: str, seed: int = SEED_TOKENS) -> list[Round]:
    rounds: list[Round] = []
    off = seed
    with open(path) as fh:
        for line in fh:
            m = ROUND_RE.search(line)
            if not m:
                continue
            idx, depth, acc = (int(g) for g in m.groups())
            rounds.append(Round(index=idx, p0=off, depth=depth, accepted=acc))
            off += acc + 1
    return rounds


# ---------------------------------------------------------------------------
# Analytic band table (independent of any trace)
# ---------------------------------------------------------------------------
def band_table(devc: str) -> list[dict]:
    """Injection band for a final round of width w whose last row lands on the
    window end (0-based p 1023, trace pos 1024)."""
    out = []
    for w in range(1, 10):
        p0 = MAX_KEY_LEN - w
        rounds = [Round(index=0, p0=p0, depth=w - 1, accepted=w - 1)]
        cur = analyze(devc, rounds, segments_current)
        rep = analyze(devc, rounds, segments_repair)
        out.append(
            {
                "w": w,
                "p0": p0,
                "key_len": p0 + w,
                "injected_pos": sorted(p + 1 for p in cur.injected),
                "inherited_pos": sorted(p + 1 for p in cur.inherited),
                "repair_injected_pos": sorted(p + 1 for p in rep.injected),
                "repair_inherited_pos": sorted(p + 1 for p in rep.inherited),
                "segments": [
                    (len(s.rows), s.k_len, dispatch(devc, len(s.rows), s.k_len, True).family)
                    for s in segments_current(p0, w)
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Validation against measured PR #2 arms
# ---------------------------------------------------------------------------
ARMS = [
    ("runI-base-cap8-512", "research/analysis-runI.json", "research/trace-runI-base-cap8-512.log"),
    ("runJ-cap7-gate3-512", "research/analysis-runJ.json", "research/trace-runJ-cap7-512.log"),
    ("runK-gate2-cap8-512", "research/analysis-runK.json", "research/trace-runK-gate2-cap8-512.log"),
    ("runL-gate1-cap8-512", "research/analysis-runL.json", "research/trace-runL-gate1-cap8-512.log"),
    ("runM-gate0-cap8-512", "research/analysis-runM.json", "research/trace-runM-gate0-cap8-512.log"),
    ("runN-gate1-cap8-512-confirm", "research/analysis-runN.json", "research/trace-runN-gate1-cap8-512-confirm.log"),
    ("runO-cap7-gate3-512", "research/analysis-runO.json", "research/trace-runO-cap7-gate3-512.log"),
]


def validate(devc: str, root: str) -> int:
    failures = 0
    total_rows = 0
    print(f"# validation: predicted band vs measured rows  (devc={devc!r})\n")
    hdr = f"{'arm':<28} {'rows':>5} {'final':>6} {'w':>2} {'kL':>5} {'observed':<18} {'injected':<14} {'inherited':<10} ok"
    print(hdr)
    print("-" * len(hdr))
    for label, jpath, tpath in ARMS:
        jf, tf = os.path.join(root, jpath), os.path.join(root, tpath)
        if not (os.path.exists(jf) and os.path.exists(tf)):
            print(f"{label:<28} MISSING")
            failures += 1
            continue
        data = json.load(open(jf))
        gate = data["row_gate"]
        rounds = parse_trace(tf)
        a = analyze(devc, rounds, segments_current)
        observed = sorted({s["pos"] for s in gate.get("mismatch_samples", [])})
        suspect_pos = {p + 1 for p in a.suspect}
        out_of_band = [p for p in observed if p not in suspect_pos]
        last = rounds[-1]
        total_rows += gate["compared_rows"]
        ok = (
            not out_of_band
            and gate["unmatched_positions"] == 0
            and data["metrics"]["all_tokens_matched"] is True
            and data["metrics"]["residual_divergence_count"] == 0
        )
        failures += 0 if ok else 1
        print(
            f"{label:<28} {gate['compared_rows']:>5} {last.index:>6} {last.width:>2} "
            f"{last.key_len:>5} {str(observed):<18} "
            f"{str(sorted(p + 1 for p in a.injected)):<14} "
            f"{str(sorted(p + 1 for p in a.inherited)):<10} {'PASS' if ok else 'FAIL ' + str(out_of_band)}"
        )
        for d in a.detail:
            print(f"    {d}")
    print(f"\ntotal compared rows: {total_rows}")
    print(f"arms out of band:    {failures}")
    # Cross-check: every arm's committed positions must tile 1..512 decode slots
    # and no round may exceed the window.
    for label, _, tpath in ARMS:
        tf = os.path.join(root, tpath)
        if not os.path.exists(tf):
            continue
        rounds = parse_trace(tf)
        committed = [p for r in rounds for p in r.committed]
        assert committed == list(range(SEED_TOKENS, SEED_TOKENS + DECODE_TOKENS)), label
        assert max(r.key_len for r in rounds) == MAX_KEY_LEN, (label, max(r.key_len for r in rounds))
    print("schedule cross-check: committed positions tile the window exactly;")
    print(f"                      max key_len == {MAX_KEY_LEN} in every arm (no overshoot).")
    return failures


def repair_check(devc: str, root: str) -> int:
    print(f"\n# strict-boundary repair, replayed on the same measured schedules  (devc={devc!r})\n")
    hdr = f"{'arm':<28} {'shipped inj':<14} {'shipped inh':<12} {'repair inj':<11} {'repair inh':<11} {'extra calls':>11}"
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for label, _, tpath in ARMS:
        tf = os.path.join(root, tpath)
        if not os.path.exists(tf):
            continue
        rounds = parse_trace(tf)
        cur, rep = analyze(devc, rounds, segments_current), analyze(devc, rounds, segments_repair)
        extra = sum(
            len(segments_repair(r.p0, r.width)) - len(segments_current(r.p0, r.width)) for r in rounds
        )
        if rep.suspect:
            bad += 1
        print(
            f"{label:<28} {str(sorted(p + 1 for p in cur.injected)):<14} "
            f"{str(sorted(p + 1 for p in cur.inherited)):<12} "
            f"{str(sorted(p + 1 for p in rep.injected)):<11} "
            f"{str(sorted(p + 1 for p in rep.inherited)):<11} {extra:>11}"
        )
    print(f"\narms with residual suspect rows after repair: {bad}")
    return bad


def report(devc: str) -> None:
    print(f"# analytic band table, final round of width w ending at window end  (devc={devc!r})\n")
    hdr = f"{'w':>2} {'p0':>5} {'kL':>5} {'segments (qL,kL,family)':<62} {'injected pos':<20} {'inherited pos':<14} {'after repair':<12}"
    print(hdr)
    print("-" * len(hdr))
    for row in band_table(devc):
        segs = "; ".join(f"q{q}/k{k}/{f.replace('sdpa_vector','vec')}" for q, k, f in row["segments"])
        after = sorted(set(row["repair_injected_pos"]) | set(row["repair_inherited_pos"]))
        print(
            f"{row['w']:>2} {row['p0']:>5} {row['key_len']:>5} {segs:<62} "
            f"{str(row['injected_pos']):<20} {str(row['inherited_pos']):<14} {str(after):<12}"
        )
    print("\n# kernel family by round width (no boundary crossing, kL < 1024)\n")
    for w in range(1, 11):
        segs = segments_current(600, w)
        fams = "; ".join(
            f"q{len(s.rows)}/k{s.k_len}/{dispatch(devc, len(s.rows), s.k_len, True).family}" for s in segs
        )
        print(f"  w={w:>2}  {fams}")


def sweep(devc: str) -> int:
    """Exhaustive audit of both segmentations over every reachable (p0, width).

    Checks three properties per case: causal alignment of every emitted segment
    (the off-by-one audit), that no segment escapes supports_sdpa_vector into the
    bf16 composite fallback, and the resulting suspect set.
    """
    print(f"\n# exhaustive segmentation sweep  (devc={devc!r})\n")
    stats = {}
    for name, seg in (("shipped", segments_current), ("repair", segments_repair)):
        cases = misaligned = composite = dirty = 0
        for p0 in range(SEED_TOKENS, MAX_KEY_LEN):
            for w in range(1, 10):
                if p0 + w > MAX_KEY_LEN:
                    continue
                cases += 1
                r = [Round(index=0, p0=p0, depth=w - 1, accepted=w - 1)]
                try:
                    a = analyze(devc, r, seg)
                except CausalMisalignment as exc:
                    misaligned += 1
                    print(f"  {name}: MISALIGNED {exc}")
                    continue
                if any(
                    dispatch(devc, len(s.rows), s.k_len, True).family == FULL
                    for s in seg(p0, w)
                ):
                    composite += 1
                if a.suspect:
                    dirty += 1
        stats[name] = (cases, misaligned, composite, dirty)
        print(
            f"  {name:<8} cases={cases:<5} misaligned={misaligned:<3} "
            f"composite-fallback={composite:<3} cases-with-suspect-rows={dirty}"
        )
    return stats["repair"][1] + stats["repair"][2] + stats["repair"][3]


def discriminate(root: str) -> None:
    """Which device classes can explain the measured rows at all?

    The two-pass gate (CPP:748-751) fires only for 'd' and 's'. On any other
    architecture suffix every row in a 1024-key window stays single pass, the
    predicted band is empty, and the measured window-end deviations become
    inexplicable. So the PR #2 rows are themselves evidence about the class of
    the measuring host, and the same test transfers to the ranked M5: a 512-token
    candidate leg on a 'g'-class box must show zero deviation at pos 1022..1024.
    """
    print("\n# device-class discriminator against the measured rows\n")
    print(f"{'devc':<6} {'two-pass at kL=1024':<20} {'blocks':>7} {'predicted band':<22} {'arms unexplained':>17}")
    print("-" * 78)
    for devc in ("s", "d", "g", "p"):
        d = dispatch(devc, 3, 1024, True)
        band = sorted(p + 1 for p in analyze(devc, [Round(0, 1016, 7, 7)], segments_current).suspect)
        unexplained = 0
        for label, jpath, tpath in ARMS:
            jf, tf = os.path.join(root, jpath), os.path.join(root, tpath)
            if not (os.path.exists(jf) and os.path.exists(tf)):
                continue
            gate = json.load(open(jf))["row_gate"]
            a = analyze(devc, parse_trace(tf), segments_current)
            suspect = {p + 1 for p in a.suspect}
            if any(s["pos"] not in suspect for s in gate.get("mismatch_samples", [])):
                unexplained += 1
        print(
            f"{devc:<6} {str(d.family == VEC2):<20} {str(d.blocks):>7} {str(band):<22} {unexplained:>17}"
        )
    print("\nA measured host that shows window-end deviation is therefore in {'s','d'};")
    print("both classes yield an identical band table, so the E19 conclusion and the")
    print("repair are invariant over the remaining ambiguity about the ranked M5.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devc", default="s", choices=["s", "d", "g", "p"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--discriminate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not (args.report or args.validate or args.repair or args.discriminate or args.sweep):
        args.report = args.validate = args.repair = args.discriminate = args.sweep = True
    rc = 0
    if args.report:
        report(args.devc)
    if args.validate:
        print()
        rc |= 1 if validate(args.devc, root) else 0
    if args.repair:
        rc |= 1 if repair_check(args.devc, root) else 0
    if args.discriminate:
        discriminate(root)
    if args.sweep:
        for devc in ("s", "d"):
            rc |= 1 if sweep(devc) else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
