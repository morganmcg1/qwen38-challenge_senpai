#!/usr/bin/env python3
"""E123: the four instruction classes the E118 price ladder is missing.

    research/e123_arms.py --emit /tmp/e123-arms
    research/e123_arms.py --census /tmp/e123-arms --out research/e123-artifacts/census.json
    research/e123_arms.py --aircheck /tmp/e123-arms --out research/e123-artifacts/aircheck.json

E118 priced three instruction classes -- device load, ALU and `simd_shuffle` --
by injecting an exactly known instruction count into the wide affine-4 QMV inner
loop at a fixed register footprint, in one counterbalanced session. It has no
`threadgroup` load, no `threadgroup` store, no barrier and no bf16-to-float
conversion class, and those are exactly the classes the campaign now needs.

The method is E118's, unchanged, because the method is what made E118 credible:

  * `lanes` loop-carried accumulators and `depth` rounds of `lanes` injected
    operations, so the arm issues exactly `lanes * depth` extra instructions of
    ONE class per k-block iteration while allocating exactly `lanes` extra
    registers;
  * two rungs per class, and the price is the RUNG CONTRAST, which cancels the
    injection scaffold exactly instead of charging a share of it to every
    instruction;
  * every injected operation gets a distinct address, so no rung can collapse
    onto a lower one through common-subexpression elimination.

What is new here, and why:

  `k_tg0`      the threadgroup scaffold at ZERO injected instructions: the
               staging arrays, their data-dependent fill, one fence per k-block
               and the sink. Every threadgroup class is measured against
               `a_base`, so the scaffold has to be measured too, or a holdout
               prediction can only ever be a contrast and never an absolute.
  `k_tgld*`    conflict-free threadgroup loads: lane i reads element i + n, so
               the 32 lanes hit 32 distinct 4-byte banks.
  `k_tgldc*`   the same count of threadgroup loads at a stride of 4 floats, so
               lanes i and i+8 hit the same bank and every access is 4-way
               conflicted. `k_tgld` against `k_tgldc` is the bank-conflict
               price, which is what an exchange arm needs before it picks a
               layout.
  `k_tgst*`    threadgroup stores, conflict free, same design.
  `k_bar*`     1, 2, 4 and 8 `threadgroup_barrier(mem_flags::mem_threadgroup)`
               per k-block and NOTHING else -- no accumulators, no staging, no
               address arithmetic. Four rungs, because the published claim
               under test ("Apple barriers are about 2 cycles") predicts a
               slope of zero and a flat line is only convincing over a range.
  `k_sbar*`    `simdgroup_barrier(mem_flags::mem_none)`, the cheaper primitive
               the epilogue would need if the reduction were restructured.
  `k_cvt*`     bf16-to-float widening. A widening cannot be injected on its own:
               a register-resident bf16 source is loop invariant and gets
               hoisted, and a round trip through bf16 prices two conversions and
               a rounding. So `k_cvt` loads a bf16 from threadgroup memory and
               widens it, and its control is `k_tgld`, which loads a float from
               threadgroup memory at the SAME bank pattern and the same address
               arithmetic. The difference is one widening and one 2-byte load
               against one 4-byte load. Both AIR and machine text are censused
               so a measured zero can be shown to be a real zero.
  `k_ssum*`    `simd_sum`, the epilogue's reduction primitive, at 2 and 4 per
               k-block.
  `x_cvtshift` the static half of the conversion question, in situ and bit
               exact: `static_cast<float>(bf16)` is exactly a 16-bit left shift
               for every bit pattern, so this arm writes the shift by hand. If
               its machine text is byte-identical to `a_base` on both
               architectures, the compiler's conversion IS that shift and the
               census can price the group with certainty.

Holdouts, declared here and fitted nowhere -- the ladder predicts them before
the session and the report scores the prediction:

  `k_hold_mix`   8 threadgroup stores, 8 threadgroup loads, 2 extra barriers and
                 4 ALU operations in one block. This is the shape of an
                 exchange arm, so it is the holdout the campaign actually needs.
  `k_hold_alu12` 12 ALU operations: interpolation inside a fitted class.
  `k_hold_sl`    4 shuffles and 4 device loads: cross-class additivity between
                 two classes E118 fitted separately.

Anchors carried over from E118 so the extension is measured in the same session
as the classes it extends: `a_base`, `q_scaffold` (the byte-identical null and
therefore this session's noise floor), the six E118 calibration rungs, and
`n_nosums`, whose +7.518 % E118 predicted with no free parameter.

Research-only: nothing here is on the scored path and no submitted file changes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e118_arms as e118  # noqa: E402
from e118_arms import (  # noqa: E402
    BODY_BASE, BODY_NOSUMS, EPILOGUE, PROLOGUE, SIG_TAIL, WEIGHT_META_LOOP,
    WIDTHS, cal_decl, cal_plan, cal_sink, expect,
)

# --- the threadgroup staging every threadgroup-class arm shares ---------------
#
# One float array and one bf16 array, the same size in every arm that uses
# either, so the threadgroup footprint -- and therefore any occupancy effect --
# is identical across the whole threadgroup half of the ladder. 256 floats plus
# 512 bf16 is 2048 bytes against a 32768 byte budget, so occupancy is not
# limited by it at any width; the census prints the number rather than assuming
# it.
TG_FLOATS = 256
TG_BF16 = 512
TG_BYTES = TG_FLOATS * 4 + TG_BF16 * 2

SIG_TAIL_CAL_TG = ("    uint simd_lid,\n"
                   "    threadgroup float* tgf = nullptr,\n"
                   "    threadgroup T* tgb = nullptr) {\n")

ENTRY_DECL_TG = ("  threadgroup float tgf_store[%d];\n"
                 "  threadgroup bfloat16_t tgb_store[%d];\n"
                 % (TG_FLOATS, TG_BF16))

# Filled from `out_row`, which is a function of `tid.y` and `simd_gid`, so the
# contents are not compile-time constant and no load of them can be folded.
TG_INIT = """  for (int i = int(simd_lid); i < %d; i += 32) {
    tgf[i] = float(out_row + i);
  }
  for (int i = int(simd_lid); i < %d; i += 32) {
    tgb[i] = static_cast<T>(float(out_row + i) * 0.5f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

""" % (TG_FLOATS, TG_BF16)

# One fence per k-block iteration, ahead of the injected block. Without it the
# injected loads have loop-invariant addresses and the optimiser hoists every
# one of them out of the k loop, and the injected stores are dead until the very
# last iteration. It is part of the scaffold, so it cancels in the rung contrast
# and it is measured on its own by `k_tg0`.
TG_FENCE = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"

# Keeps the staged values live without letting them reach `y`: `in_vec_size` is
# a positive dispatch parameter, and the backend cannot prove the branch dead.
TG_SINK = """  if (in_vec_size < 0) {
    y[0] = static_cast<T>(tgf[simd_lid] + float(tgb[simd_lid]));
  }

"""

# --- the injected instruction classes -----------------------------------------
#
# `n` is the injected operation's index inside the block, so every operation of
# an arm addresses a distinct element; `j` is its accumulator lane.
#
#   tgld   lane i reads element (i + n) mod 32: 32 lanes, 32 distinct banks.
#   tgldc  lane i reads element (4i + n) mod 128: lanes i and i+8 collide, so
#          every access is 4-way conflicted on a 32-bank file.
#   tgst   the conflict-free pattern, written instead of read.
#   cvt    the conflict-free pattern over the bf16 array at a stride of two
#          elements, so the byte stride and therefore the bank pattern is the
#          same 4 bytes as `tgld`. The only differences from `tgld` are the
#          access width and the widening.
#   ssum   the epilogue's reduction primitive.
CAL_KINDS = {
    "tgld": "      cal[%(j)d] += tgf[(simd_lid + %(n)d) & 31];\n",
    "tgldc": "      cal[%(j)d] += tgf[((simd_lid * 4) + %(n)d) & 127];\n",
    "tgst": "      tgf[(simd_lid + %(n)d) & 31] = cal[%(j)d];\n",
    "cvt": "      cal[%(j)d] += static_cast<float>("
           "tgb[2 * ((simd_lid + %(n)d) & 31)]);\n",
    "ssum": "      cal[%(j)d] = simd_sum(cal[%(j)d]);\n",
    # the two barrier classes carry no accumulator and no address arithmetic.
    "bar": "    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
    "sbar": "    simdgroup_barrier(mem_flags::mem_none);\n",
    # the three E118 classes, so a holdout can mix them with the new ones. The
    # bodies are E118's, with the per-operation index rewritten to this
    # module's single `n` so one block can hold several classes and still give
    # every operation a distinct address.
    "alu": e118.CAL_KINDS["alu"].replace("%(jr)d", "0"),
    "shuf": e118.CAL_KINDS["shuf"].replace("%(rot)d", "%(n)d"),
    "ld": ("      cal[%(j)d] += float(scales[out_row * in_vec_size_g"
           " + k / 64 + ((simd_lid + %(n)d) & 7)]);\n"),
}
TG_KINDS = ("tgld", "tgldc", "tgst", "cvt")
BARRIER_KINDS = ("bar", "sbar")


def cal_step(kind: str, lanes: int, depth: int) -> str:
    """Exactly `lanes * depth` injected instructions of one class.

    A barrier carries no accumulator, so its arms pass `lanes = 0` and inject
    `depth` barriers with no other change at all.
    """
    template = CAL_KINDS[kind]
    if kind in BARRIER_KINDS:
        return template * depth
    out = ["    {\n"]
    for d in range(depth):
        for j in range(lanes):
            n = d * lanes + j
            out.append(template % {"j": j, "n": n})
    out.append("    }\n")
    return "".join(out)


def mix_step(spec: list[tuple[str, int]], lanes: int) -> str:
    """One injected block holding several classes, for the holdout arms.

    The classes are emitted in the order given, which is the order an exchange
    arm would write them: stage, fence, read back.
    """
    out = ["    {\n"]
    n = 0
    for kind, count in spec:
        for _ in range(count):
            if kind in BARRIER_KINDS:
                out.append(CAL_KINDS[kind].replace("    ", "      ", 1))
                continue
            out.append(CAL_KINDS[kind] % {"j": n % lanes, "n": n})
            n += 1
    out.append("    }\n")
    return "".join(out)


def with_tail(text: str, tail: str) -> str:
    expect(text, SIG_TAIL, 1, "prologue signature tail")
    return text.replace(SIG_TAIL, tail)


def build_prologue(step: str, lanes: int, threadgroup: bool) -> str:
    """The shipped prologue plus the injection scaffold and one injected block."""
    text = PROLOGUE
    head = ""
    if threadgroup:
        text = with_tail(text, SIG_TAIL_CAL_TG)
        head = TG_INIT
    site = "  VF acc[rows_per_simd];\n"
    expect(text, site, 1, "accumulator declaration")
    decl = cal_decl(lanes) if lanes else site
    text = text.replace(site, head + decl)
    expect(text, WEIGHT_META_LOOP, 1, "prologue weight+metadata loop")
    fence = TG_FENCE if threadgroup else ""
    return text.replace(WEIGHT_META_LOOP, WEIGHT_META_LOOP + fence + step)


def build_epilogue(lanes: int, threadgroup: bool) -> str:
    head = "  for (int r = 0; r < rows_per_simd; r++) {\n"
    expect(EPILOGUE, head, 1, "epilogue store loop")
    sink = (cal_sink(lanes) if lanes else "") + (TG_SINK if threadgroup else "")
    return EPILOGUE.replace(head, sink + head)


# The entry point owns the staging arrays and passes pointers to them, so this
# string is both the call-site argument list and the marker for "this arm needs
# the threadgroup declarations".
TG_ARGS = "tgf_store, tgb_store"


def plan(kind: str, lanes: int, depth: int):
    threadgroup = kind in TG_KINDS
    step = cal_step(kind, lanes, depth) if depth else ""
    return ((build_prologue(step, lanes, threadgroup), BODY_BASE,
             build_epilogue(lanes, threadgroup)),
            TG_ARGS if threadgroup else "")


def mix_plan(spec: list[tuple[str, int]], lanes: int = 2):
    threadgroup = any(k in TG_KINDS for k, _ in spec)
    return ((build_prologue(mix_step(spec, lanes), lanes, threadgroup),
             BODY_BASE, build_epilogue(lanes, threadgroup)),
            TG_ARGS if threadgroup else "")


# --- the in-situ static probe for the conversion question ---------------------
#
# bf16 to float32 is exactly a 16-bit left shift of the bit pattern for every
# finite value, every zero, every infinity and every NaN payload, so this arm is
# bit exact by construction on real activations. It exists to answer the
# conversion question STATICALLY: if `x_cvtshift` and `a_base` translate to
# byte-identical machine text, the compiler's `static_cast<float>` already is
# that shift and nothing about the group is in doubt.
CVT_SHIFT_HELPER = """static inline float e123_bf16_to_float(bfloat16_t v) {
  return as_type<float>(uint(as_type<ushort>(v)) << 16);
}

"""
expect(BODY_BASE, "a0[m] = static_cast<float>(xm[0]);", 1, "a0 conversion")
BODY_CVTSHIFT = BODY_BASE
for _slot in range(4):
    BODY_CVTSHIFT = BODY_CVTSHIFT.replace(
        "a%d[m] = static_cast<float>(xm[%d]);" % (_slot, _slot),
        "a%d[m] = e123_bf16_to_float(xm[%d]);" % (_slot, _slot))
expect(BODY_CVTSHIFT, "e123_bf16_to_float", 4, "shift-form conversions")


PLANS = {
    # anchors, carried over unchanged from E118
    "a_base": (None, ""),
    "q_scaffold": ((PROLOGUE, BODY_BASE, EPILOGUE), ""),
    "n_nosums": ((PROLOGUE, BODY_NOSUMS, EPILOGUE), ""),
    # the three E118 classes, re-measured in this session
    "k_alu8": cal_plan("alu", 2, 4),
    "k_alu16": cal_plan("alu", 2, 8),
    "k_ld8": cal_plan("ld", 2, 4),
    "k_ld16": cal_plan("ld", 2, 8),
    "k_shuf8": cal_plan("shuf", 2, 4),
    "k_shuf16": cal_plan("shuf", 2, 8),
    # the threadgroup scaffold at zero injected instructions
    "k_tg0": plan("tgld", 2, 0),
    # the four new classes
    "k_tgld8": plan("tgld", 2, 4),
    "k_tgld16": plan("tgld", 2, 8),
    "k_tgldc8": plan("tgldc", 2, 4),
    "k_tgldc16": plan("tgldc", 2, 8),
    "k_tgst8": plan("tgst", 2, 4),
    "k_tgst16": plan("tgst", 2, 8),
    "k_cvt8": plan("cvt", 2, 4),
    "k_cvt16": plan("cvt", 2, 8),
    "k_bar1": plan("bar", 0, 1),
    "k_bar2": plan("bar", 0, 2),
    "k_bar4": plan("bar", 0, 4),
    "k_bar8": plan("bar", 0, 8),
    "k_sbar2": plan("sbar", 0, 2),
    "k_sbar4": plan("sbar", 0, 4),
    "k_ssum2": plan("ssum", 2, 1),
    "k_ssum4": plan("ssum", 2, 2),
    # holdouts: predicted before the session, fitted never
    "k_hold_mix": mix_plan([("tgst", 8), ("bar", 2), ("tgld", 8), ("alu", 4)]),
    "k_hold_alu12": cal_plan("alu", 2, 6),
    "k_hold_sl": mix_plan([("shuf", 4), ("ld", 4)]),
    # the in-situ static probe
    "x_cvtshift": ((CVT_SHIFT_HELPER + PROLOGUE, BODY_CVTSHIFT, EPILOGUE), ""),
}

# Injected instruction count per k-block iteration, by class, for every arm the
# analysis prices or predicts. `bar` counts the fence the threadgroup scaffold
# already pays, so a holdout prediction is absolute and not a contrast.
ARM_INJECTION = {
    "a_base": {},
    "q_scaffold": {},
    "k_alu8": {"alu": 8}, "k_alu16": {"alu": 16},
    "k_ld8": {"ld": 8}, "k_ld16": {"ld": 16},
    "k_shuf8": {"shuf": 8}, "k_shuf16": {"shuf": 16},
    "k_tg0": {"tgscaffold": 1},
    "k_tgld8": {"tgscaffold": 1, "tgld": 8},
    "k_tgld16": {"tgscaffold": 1, "tgld": 16},
    "k_tgldc8": {"tgscaffold": 1, "tgldc": 8},
    "k_tgldc16": {"tgscaffold": 1, "tgldc": 16},
    "k_tgst8": {"tgscaffold": 1, "tgst": 8},
    "k_tgst16": {"tgscaffold": 1, "tgst": 16},
    "k_cvt8": {"tgscaffold": 1, "cvt": 8},
    "k_cvt16": {"tgscaffold": 1, "cvt": 16},
    "k_bar1": {"bar": 1}, "k_bar2": {"bar": 2},
    "k_bar4": {"bar": 4}, "k_bar8": {"bar": 8},
    "k_sbar2": {"sbar": 2}, "k_sbar4": {"sbar": 4},
    "k_ssum2": {"ssum": 2}, "k_ssum4": {"ssum": 4},
    "k_hold_mix": {"tgscaffold": 1, "tgst": 8, "bar": 2, "tgld": 8, "alu": 4},
    "k_hold_alu12": {"alu": 12},
    "k_hold_sl": {"shuf": 4, "ld": 4},
    "x_cvtshift": {},
    "n_nosums": {},
}

# class -> ((arm, injected count), ...). The price is the contrast between the
# two extreme rungs; the intermediate rungs test linearity.
CAL_LADDER = {
    "alu": (("k_alu8", 8), ("k_alu16", 16)),
    "ld": (("k_ld8", 8), ("k_ld16", 16)),
    "shuf": (("k_shuf8", 8), ("k_shuf16", 16)),
    "tgld": (("k_tgld8", 8), ("k_tgld16", 16)),
    "tgldc": (("k_tgldc8", 8), ("k_tgldc16", 16)),
    "tgst": (("k_tgst8", 8), ("k_tgst16", 16)),
    "cvt": (("k_cvt8", 8), ("k_cvt16", 16)),
    "bar": (("k_bar1", 1), ("k_bar2", 2), ("k_bar4", 4), ("k_bar8", 8)),
    "sbar": (("k_sbar2", 2), ("k_sbar4", 4)),
    "ssum": (("k_ssum2", 2), ("k_ssum4", 4)),
}
# Classes whose rungs also carry the threadgroup scaffold, so the zero rung is
# `k_tg0` and not `a_base`.
TG_CLASSES = ("tgld", "tgldc", "tgst", "cvt")
HOLDOUT_ARMS = ("k_hold_mix", "k_hold_alu12", "k_hold_sl")
# Not required to reproduce `a_base` bit for bit.
DIAGNOSTIC_ARMS = ("n_nosums",)

ARMS = tuple(PLANS)


def install() -> None:
    """Point the E118 emitter and census at this experiment's arm set."""
    e118.PLANS.clear()
    e118.PLANS.update(PLANS)
    e118.ALL_PLANS.clear()
    e118.ALL_PLANS.update(PLANS)
    e118.ARMS = ARMS
    e118.CENSUS_ARMS = ()
    e118.DIAGNOSTIC_ARMS = DIAGNOSTIC_ARMS
    e118.PROMOTION_ARMS = ()
    e118.ROWS_PER_SIMD = {}
    e118.ENTRY_BUFFER = {}
    e118.ENTRY_DECL = {a: ENTRY_DECL_TG for a, (_, extra) in PLANS.items()
                       if extra == TG_ARGS}
    e118.THREADGROUP_BYTES = {a: {na: TG_BYTES for na in WIDTHS}
                              for a in e118.ENTRY_DECL}


# --- the AIR check: did the compiler emit what the arm claims to inject? -------
#
# The rung contrast is only a price if the rungs really differ by the instruction
# count the source says. Four specific ways this can fail, all of them silent:
# adjacent barriers merge, a threadgroup load is hoisted out of the k loop, a
# threadgroup store is eliminated as dead, and a conversion is folded away. This
# pass reads the AIR the front end hands the backend and counts each class per
# entry point, so every one of those failures is visible BEFORE any GPU time is
# spent.
KERNEL_RE = re.compile(r"e118_iso_na(\d+)$")
COUNTERS = {
    "tg_loads": re.compile(r"=\s*load\s.*addrspace\(3\)"),
    "tg_stores": re.compile(r"^\s+store\s.*addrspace\(3\)"),
    "device_loads": re.compile(r"=\s*load\s.*addrspace\(1\)"),
    "barriers": re.compile(r"air\.wg\.barrier|threadgroup_barrier"),
    "simd_barriers": re.compile(r"air\.simdgroup\.barrier|simdgroup_barrier"),
    "shuffles": re.compile(r"simd_shuffle"),
    "simd_sums": re.compile(r"air\.simd_sum|simd_sum"),
    "fpext": re.compile(r"=\s*fpext\s"),
    "fptrunc": re.compile(r"=\s*fptrunc\s"),
    "sitofp": re.compile(r"=\s*(?:sitofp|uitofp)\s"),
    "fma": re.compile(r"llvm\.fmuladd|air\.fma"),
}


def air_counts(source: pathlib.Path, workdir: pathlib.Path) -> dict:
    ll = workdir / "air.ll"
    done = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(source), "-o", str(ll)],
        capture_output=True, text=True)
    if done.returncode != 0:
        return {"error": done.stderr.strip().splitlines()[-8:]}
    found: dict[str, dict] = {}
    name, body = None, []
    for line in ll.read_text().splitlines():
        if line.startswith("define "):
            hit = re.search(r"@([\w.]+)\(", line)
            name, body = (hit.group(1) if hit else None), []
        elif line == "}" and name is not None:
            hit = KERNEL_RE.search(name)
            if hit:
                found[hit.group(1)] = {
                    key: sum(1 for row in body if pattern.search(row))
                    for key, pattern in COUNTERS.items()}
                found[hit.group(1)]["air_lines"] = len(body)
            name = None
        elif name is not None:
            body.append(line)
    return found


def aircheck(directory: pathlib.Path, out: pathlib.Path | None) -> int:
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for arm in ARMS:
            rows[arm] = air_counts(directory / ("arm_%s.metal" % arm), workdir)
            print("aircheck %s" % arm)

    keys = ("tg_loads", "tg_stores", "barriers", "simd_barriers", "shuffles",
            "simd_sums", "fpext", "device_loads")
    print("\nAIR counts per entry point, NA=4")
    print("  %-14s %s" % ("arm", " ".join("%9s" % k for k in keys)))
    for arm in ARMS:
        cell = rows[arm].get("4", {})
        print("  %-14s %s" % (arm, " ".join("%9s" % cell.get(k, "?")
                                            for k in keys)))

    # The contrast the price depends on: rung-to-rung deltas must equal the
    # injected count in the injected class and zero everywhere else.
    print("\nRung contrast, NA=4 (delta must match the injected count)")
    failures = []
    for klass, rungs in CAL_LADDER.items():
        (lo_arm, lo_n), (hi_arm, hi_n) = rungs[0], rungs[-1]
        lo = rows[lo_arm].get("4", {})
        hi = rows[hi_arm].get("4", {})
        key = {"tgld": "tg_loads", "tgldc": "tg_loads", "tgst": "tg_stores",
               "cvt": "tg_loads", "bar": "barriers", "sbar": "simd_barriers",
               "ssum": "simd_sums", "shuf": "shuffles", "ld": "device_loads",
               "alu": "fma"}[klass]
        seen = hi.get(key, 0) - lo.get(key, 0)
        want = hi_n - lo_n
        ok = seen == want
        if not ok:
            failures.append("%s: %s delta %d, injected %d"
                            % (klass, key, seen, want))
        print("  %-6s %-14s -> %-14s %-12s delta=%-4d injected=%-4d %s"
              % (klass, lo_arm, hi_arm, key, seen, want,
                 "ok" if ok else "MISMATCH"))
        if klass == "cvt":
            widen = hi.get("fpext", 0) - lo.get("fpext", 0)
            print("       %-14s widening delta (fpext) = %d, injected %d %s"
                  % ("", widen, want, "ok" if widen == want else "MISMATCH"))
            if widen != want:
                failures.append("cvt: fpext delta %d, injected %d"
                                % (widen, want))

    if failures:
        print("\nAIRCHECK FAILURES")
        for line in failures:
            print("  " + line)
    else:
        print("\nAIRCHECK: every rung contrast matches its injected count")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arms": rows, "failures": failures,
                                   "injection": ARM_INJECTION,
                                   "ladder": {k: list(v) for k, v
                                              in CAL_LADDER.items()}},
                                  indent=2) + "\n")
        print("\nwrote %s" % out)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--aircheck", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--arm-list", action="store_true")
    args = ap.parse_args()
    install()
    if args.arm_list:
        print(",".join(a + (":diag" if a in DIAGNOSTIC_ARMS else "")
                       for a in ARMS))
        return 0
    if args.emit is not None:
        e118.emit(args.emit)
    if args.aircheck is not None:
        return aircheck(args.aircheck, args.out)
    if args.census is not None:
        return e118.census(args.census, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
