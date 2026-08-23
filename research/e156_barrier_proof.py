#!/usr/bin/env python3
"""E156 R1: the offline barrier proof for the double-buffered NAX k-loop.

WHY THE PROOF IS OFFLINE.

`is_nax_available()` (`backend/metal/device.cpp:913-931`) needs
`get_architecture_gen() >= 17`. Every Mac this campaign owns is
`applegpu_g16s`, gen 16, so `backend/metal/quantized.cpp:697` never routes to
`qmm_nax` and `affine_qmm_t_nax` executes zero times here. Deleting a barrier
from the pipelined k-loop changes no locally observable byte. A local
`--local-submit` with the barrier removed would PASS, and reporting that pass as
a race control would be worse than reporting no control at all.

The risk this file exists to close is a missing or misplaced
`threadgroup_barrier(mem_flags::mem_threadgroup)`: a race that is silent on one
host and wrong on another, on a kernel we cannot run.

WHAT IS PROVED, IN THREE INDEPENDENT LAYERS.

  1. SOURCE FIDELITY. The event program is extracted FROM the shipped
     runtime-effective twin, not hand-transcribed. Every barrier, staged write,
     staging-half switch and staged read inside the `kDoubleBuffer` branch is
     read off the source in order and must equal the declared program. An edit
     to the loop that the model does not know about turns this red.
  2. EXHAUSTIVE ENUMERATION. Every read and every write to each staging half is
     enumerated over barrier epochs, for every k-tile count from 1 to 64, for 1
     to 4 grid-stride tiles per threadgroup, both k-tile parities. Two accesses
     to one half inside one epoch, at least one of them a write, is a race. A
     read that finds a half holding a k tile other than the one it wants is a
     stale read. Both must be zero everywhere.
  3. FAILING CONTROLS, Rule 101. Six separately defeated orderings, including
     the three the assignment names -- delete the prologue-publishing barrier,
     delete the steady-state barrier, swap the parity of the half index -- must
     each be CAUGHT. A proof that cannot fail is not a proof.

WHAT IS NOT PROVED. SIMD divergence, the memory model of
`mem_flags::mem_threadgroup` itself, and the arithmetic. The arithmetic is a
separate argument: `e156_k_order_preserved` below shows the inner k accumulation
is byte-identical between the two loops except for the staging base pointer, so
every output element still sums the same partial products in the same order.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "research/e156-barrier-proof.json"

HEADER = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
TWIN = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp"

BK = 64

# From research/e151-r0-safety-case.json, e151_retile_shape_table: every affine
# group-64 4-bit QMM the scored seed prefill evaluates, at M = 512.
SCORED_SHAPES = [
    ("gdn.in_proj", 512, 5120, 16480),
    ("fa.qkv", 512, 5120, 14336),
    ("mlp.gate_up", 512, 5120, 34816),
    ("lm_head", 512, 5120, 248320),
    ("gdn.out_proj", 512, 6144, 5120),
    ("fa.o_proj", 512, 6144, 5120),
    ("mlp.down", 512, 17408, 5120),
]

HOST_BM, HOST_BN = 64, 64
ARM_BM, ARM_BN = 128, 32


# ---------------------------------------------------------------- layer 1


# The statements the model claims to mirror, in the order the model runs them.
# `READ` is the only consumer of a staging half; `WRITE` is the only producer.
STATEMENTS = [
    (re.compile(r"threadgroup_barrier\(mem_flags::mem_threadgroup\);"), "BARRIER"),
    (re.compile(r"loader_w\.load_unsafe\(\);"), "WRITE"),
    (re.compile(r"loader_w\.load_safe\("), "WRITE"),
    (re.compile(r"loader_w\.next\(\);"), "ADVANCE"),
    (re.compile(r"loader_w\.shift_dst\("), "SHIFT"),
    (re.compile(r"Btile\.template load<T, BK_padded, 1>\(Wk \+"), "READ"),
]

# The declared program of the shipped `kDoubleBuffer` branch. Two WRITEs appear
# per staging step because `load_unsafe` and `load_safe` are the two arms of one
# `if constexpr`, and exactly one of them is compiled into any instantiation.
EXPECTED_PROGRAM = [
    # prologue, guarded by `if (K > 0)`
    "WRITE", "WRITE", "ADVANCE", "SHIFT",
    # steady state, one iteration of `for (int k = 0; k < K; k += BK)`
    "BARRIER",
    "WRITE", "WRITE", "ADVANCE", "SHIFT",
    "READ",
]

DBUF_OPEN = "if constexpr (kDoubleBuffer) {"
DBUF_ELSE = "} else {"


def dbuf_branch(text: str) -> str:
    """The `kDoubleBuffer == true` branch body, by brace matching."""
    start = text.index(DBUF_OPEN)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise SystemExit("e156_barrier_proof: unbalanced kDoubleBuffer branch")


def single_branch(text: str) -> str:
    """The `else` branch body: the unmodified single-buffered k-loop.

    `dbuf_branch` consumes the `}` of `} else {`, so the `else` must be the
    very next token. Requiring that, rather than searching forward, stops this
    helper from silently binding to an unrelated `else` later in the file.
    """
    dbuf = dbuf_branch(text)
    rest = text[text.index(dbuf) + len(dbuf) :]
    if not rest.lstrip().startswith("else {"):
        raise SystemExit(
            "e156_barrier_proof: the kDoubleBuffer branch is not followed by `else {`"
        )
    start = rest.index("else {") + len("else ")
    depth = 0
    for i in range(start, len(rest)):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                return rest[start : i + 1]
    raise SystemExit("e156_barrier_proof: unbalanced else branch")


def event_program(body: str) -> list[str]:
    hits = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        for pattern, name in STATEMENTS:
            if pattern.search(stripped):
                hits.append(name)
                break
    return hits


def source_fidelity() -> dict:
    """Bind the model to the shipped source, in both files, so an edit reds it."""
    out = {}
    for label, path in (("header", HEADER), ("twin", TWIN)):
        text = path.read_text()
        program = event_program(dbuf_branch(text))
        out[label] = {
            "path": str(path.relative_to(ROOT)),
            "extracted_program": program,
            "matches_model": program == EXPECTED_PROGRAM,
            "barriers_in_branch": program.count("BARRIER"),
            "staged_reads_in_branch": program.count("READ"),
        }
    out["header_twin_programs_agree"] = (
        out["header"]["extracted_program"] == out["twin"]["extracted_program"]
    )
    out["expected_program"] = EXPECTED_PROGRAM
    return out


INNER_LOOP_OPEN = "for (int kk1 = 0; kk1 < BK; kk1 += SK) {"


def inner_loop(body: str) -> str:
    start = body.index(INNER_LOOP_OPEN)
    depth = 0
    for i in range(start, len(body)):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return body[start : i + 1]
    raise SystemExit("e156_barrier_proof: unbalanced kk1 loop")


def k_order_preserved() -> dict:
    """The two k-loops must differ only by the staging base pointer.

    If the `kk1` nest, the `SK` step, the tile loads and the `tile_matmad_nax`
    call are identical after rewriting `Wk` to `Ws`, then every output element
    accumulates the same partial products in the same order, and the pipelined
    loop is bit-identical to the single-buffered one. That is the strongest
    exactness argument available for a kernel that cannot be executed here.
    """
    text = HEADER.read_text()
    dbuf_inner = inner_loop(dbuf_branch(text))
    single_inner = inner_loop(single_branch(text))
    normalised = re.sub(r"\bWk\b", "Ws", dbuf_inner)
    normalised = "\n".join(x.strip() for x in normalised.splitlines())
    reference = "\n".join(x.strip() for x in single_inner.splitlines())

    outer_dbuf = re.findall(r"for \(int k = 0; k < K; k \+= BK\)", dbuf_branch(text))
    outer_single = re.findall(
        r"for \(int k = 0; k < K; k \+= BK\)", single_branch(text)
    )
    return {
        "inner_loops_identical_modulo_staging_pointer": normalised == reference,
        "outer_k_loop_header_unchanged": (
            len(outer_dbuf) == 1 and len(outer_single) == 1
        ),
        "sk_step_unchanged": "constexpr short SK = 32;" in text,
        "bk_padded_unchanged": "constexpr int BK_padded = (BK + 16 / sizeof(T));" in text,
        "only_difference": "the B tile is read from `Wk = Ws + cur * Ws_tile` "
        "instead of from `Ws`; K, BK, SK, TK and the k/kk1 nest are untouched",
        "citations": [
            "quantized_nax.h: `constexpr short SK = 32;`",
            "quantized_nax.h: `for (int k = 0; k < K; k += BK)` in both branches",
            "quantized_nax.h: `for (int kk1 = 0; kk1 < BK; kk1 += SK)` in both branches",
            "quantized_nax.h: `tile_matmad_nax(Dtile, Atile, ..., Btile, ...)` in both",
        ],
    }


# ---------------------------------------------------------------- layer 2


@dataclass
class Trace:
    """An ordered access log over barrier epochs for one threadgroup."""

    epoch: int = 0
    accesses: dict = field(default_factory=dict)
    staged: dict = field(default_factory=dict)
    races: list = field(default_factory=list)
    stale_reads: list = field(default_factory=list)
    writes: int = 0
    reads: int = 0

    def barrier(self) -> None:
        self.epoch += 1

    def _touch(self, half: int, op: str, k_tile: int) -> None:
        bucket = self.accesses.setdefault((self.epoch, half), [])
        for prior_op, prior_k in bucket:
            if op == "W" or prior_op == "W":
                self.races.append(
                    {
                        "epoch": self.epoch,
                        "half": half,
                        "first": f"{prior_op}(k_tile={prior_k})",
                        "second": f"{op}(k_tile={k_tile})",
                    }
                )
        bucket.append((op, k_tile))

    def write(self, half: int, k_tile: int) -> None:
        self.writes += 1
        self._touch(half, "W", k_tile)
        self.staged[half] = k_tile

    def read(self, half: int, k_tile: int) -> None:
        self.reads += 1
        self._touch(half, "R", k_tile)
        if self.staged.get(half) != k_tile:
            self.stale_reads.append(
                {
                    "epoch": self.epoch,
                    "half": half,
                    "wanted_k_tile": k_tile,
                    "found_k_tile": self.staged.get(half),
                }
            )


DEFECTS = [
    "none",
    # the three the assignment names
    "drop_prologue_publish_barrier",
    "drop_steady_state_barrier",
    "swap_half_parity",
    # three more the mechanism can plausibly get wrong
    "defeat_double_buffer",
    "drop_pre_store_barrier",
    "prologue_before_last_read",
]

NAMED_CONTROLS = DEFECTS[1:4]


def pipelined_tile(
    tr: Trace, n_k: int, defect: str, emit_prologue: bool, next_tile_follows: bool
) -> None:
    """The shipped `kDoubleBuffer == true` k-loop for one output tile.

    Mirrors quantized_nax.h. The prologue stages k tile 0 into half 0 and moves
    `dst` to half 1. Iteration i stages k tile i+1 into half `1 - cur` and reads
    k tile i out of half `cur`. There is exactly one barrier per iteration and
    it does both jobs: it publishes the half about to be read, and it retires
    every reader of the half about to be overwritten.
    """
    if emit_prologue and n_k > 0:
        tr.write(half=0, k_tile=0)
    dst = 1
    cur = 0
    for i in range(n_k):
        first = i == 0
        skip = (defect == "drop_prologue_publish_barrier" and first) or (
            defect == "drop_steady_state_barrier" and not first
        )
        if not skip:
            tr.barrier()
        if i + 1 < n_k:
            target = cur if defect == "defeat_double_buffer" else dst
            tr.write(half=target, k_tile=i + 1)
            dst = 1 - dst
        if (
            defect == "prologue_before_last_read"
            and next_tile_follows
            and i + 1 == n_k
        ):
            tr.write(half=0, k_tile=0)
        read_half = (1 - cur) if defect == "swap_half_parity" else cur
        tr.read(half=read_half, k_tile=i)
        cur ^= 1


def single_buffered_tile(tr: Trace, n_k: int, defect: str) -> None:
    """The unmodified `else` k-loop, used as a detector sanity check."""
    for i in range(n_k):
        tr.barrier()
        tr.write(half=0, k_tile=i)
        if defect != "drop_publish_barrier":
            tr.barrier()
        tr.read(half=0, k_tile=i)


def run_tile_loop(n_k: int, n_tiles: int, pipelined: bool, defect: str) -> Trace:
    """The grid-stride tile loop around `compute_tile`."""
    tr = Trace()
    hoisted = defect == "prologue_before_last_read"
    for t in range(n_tiles):
        if pipelined:
            pipelined_tile(
                tr,
                n_k,
                defect,
                emit_prologue=not (hoisted and t > 0),
                next_tile_follows=t + 1 < n_tiles,
            )
        else:
            single_buffered_tile(tr, n_k, defect)
        # compute_tile's pre-store barrier. `Dtile.store` writes device memory
        # and never touches Ws, so it raises no threadgroup hazard itself.
        if defect != "drop_pre_store_barrier":
            tr.barrier()
    return tr


def tiles_per_threadgroup(m: int, n: int) -> tuple[int, int, int]:
    tiles_x_host = (n + HOST_BN - 1) // HOST_BN
    tiles_x = (n + ARM_BN - 1) // ARM_BN
    tiles_y = (m + ARM_BM - 1) // ARM_BM
    launched = ((m + HOST_BM - 1) // HOST_BM) * tiles_x_host
    required = tiles_y * tiles_x
    worst = (required + launched - 1) // launched if launched else 0
    return required, launched, worst


MAX_K_TILES = 64
MAX_TILES_PER_THREADGROUP = 4

# A control that is not caught on a cell has NOT found a hole in the proof: the
# enumerator reports a catch exactly when the defeated ordering creates a real
# read/write overlap, so "not caught" means the shipped barrier is redundant on
# that cell. Each supplementary control therefore carries a closed form for the
# cells where its defeated ordering is genuinely safe. The closed form is
# CHECKED against the enumeration, not asserted: if the uncaught set differs by
# one cell the proof goes red, so this cannot be used to explain away a hole.
SAFE_CELL_PREDICATES = {
    # With one k tile the loop stages nothing after the prologue, so collapsing
    # the two halves onto one cannot overlap anything.
    "defeat_double_buffer": (
        lambda n_k, n_tiles: n_k == 1,
        "the defeated ordering is safe exactly when k_tiles == 1, because the "
        "steady state never stages a second k tile",
    ),
    # `compute_tile` rebuilds `loader_w` at `Ws`, so every tile's prologue
    # writes half 0. Tile n's last Ws touch is half ((n_k - 1) mod 2). For even
    # n_k that is half 1, which the next prologue does not touch.
    "drop_pre_store_barrier": (
        lambda n_k, n_tiles: n_k % 2 == 0,
        "the defeated ordering is safe exactly when k_tiles is even, because "
        "the tile's last staging touch is then half 1 and the next tile's "
        "prologue writes half 0",
    ),
    "prologue_before_last_read": (
        lambda n_k, n_tiles: n_k % 2 == 0,
        "same parity argument as drop_pre_store_barrier: for even k_tiles the "
        "hoisted prologue writes half 0 while the last read is from half 1",
    ),
}


def exhaustive() -> dict:
    """Every (k-tile count, tiles per threadgroup) the loop structure admits.

    The loop body does not depend on K, N or M except through these two counts,
    so enumerating them enumerates the reachable orderings. Both k-tile
    parities are covered because the range includes odd and even counts.
    """
    clean = True
    caught = {d: 0 for d in DEFECTS if d != "none"}
    reachable = {d: 0 for d in DEFECTS if d != "none"}
    uncaught_cells = {d: [] for d in DEFECTS if d != "none"}
    total_reads = total_writes = 0
    first_failure = None
    cells = 0
    for n_k in range(1, MAX_K_TILES + 1):
        for n_tiles in range(1, MAX_TILES_PER_THREADGROUP + 1):
            cells += 1
            tr = run_tile_loop(n_k, n_tiles, pipelined=True, defect="none")
            total_reads += tr.reads
            total_writes += tr.writes
            if tr.races or tr.stale_reads:
                clean = False
                if first_failure is None:
                    first_failure = {
                        "k_tiles": n_k,
                        "tiles": n_tiles,
                        "race": tr.races[:1],
                        "stale": tr.stale_reads[:1],
                    }
            for defect in caught:
                bad = run_tile_loop(n_k, n_tiles, pipelined=True, defect=defect)
                # A control is "reachable" on a cell when the defect can express
                # itself there at all: a tile-boundary defect needs two tiles.
                if defect in ("drop_pre_store_barrier", "prologue_before_last_read"):
                    if n_tiles < 2:
                        continue
                if defect == "drop_steady_state_barrier" and n_k < 2:
                    continue
                reachable[defect] += 1
                if bad.races or bad.stale_reads:
                    caught[defect] += 1
                else:
                    uncaught_cells[defect].append((n_k, n_tiles))

    # Every uncaught cell must be a cell the closed form already calls safe.
    # A control with no closed form must be caught everywhere it is reachable.
    safe_forms = {}
    for defect, cells_ in uncaught_cells.items():
        entry = SAFE_CELL_PREDICATES.get(defect)
        if entry is None:
            safe_forms[defect] = {
                "has_closed_form": False,
                "uncaught_cells": len(cells_),
                "closed_form_matches_enumeration": not cells_,
                "explanation": "no closed form: this control must be caught on "
                "every reachable cell",
            }
            continue
        predicate, explanation = entry
        predicted = {
            (n_k, n_tiles)
            for n_k in range(1, MAX_K_TILES + 1)
            for n_tiles in range(1, MAX_TILES_PER_THREADGROUP + 1)
            if predicate(n_k, n_tiles)
            and not (
                defect in ("drop_pre_store_barrier", "prologue_before_last_read")
                and n_tiles < 2
            )
            and not (defect == "drop_steady_state_barrier" and n_k < 2)
        }
        safe_forms[defect] = {
            "has_closed_form": True,
            "uncaught_cells": len(cells_),
            "predicted_safe_cells": len(predicted),
            "closed_form_matches_enumeration": set(cells_) == predicted,
            "explanation": explanation,
        }

    return {
        "controls_uncaught_cell_counts": {
            d: len(v) for d, v in uncaught_cells.items()
        },
        "controls_safe_cell_closed_forms": safe_forms,
        "every_uncaught_cell_has_a_verified_closed_form": all(
            v["closed_form_matches_enumeration"] for v in safe_forms.values()
        ),
        "k_tile_counts": [1, MAX_K_TILES],
        "tiles_per_threadgroup_counts": [1, MAX_TILES_PER_THREADGROUP],
        "cells_enumerated": cells,
        "staging_reads_enumerated": total_reads,
        "staging_writes_enumerated": total_writes,
        "shipped_ordering_clean_everywhere": clean,
        "first_failure": first_failure,
        "controls_reachable_cells": reachable,
        "controls_caught_cells": caught,
        "controls_caught_on_every_reachable_cell": {
            d: reachable[d] > 0 and caught[d] == reachable[d] for d in caught
        },
    }


def scored_rows() -> list[dict]:
    rows = []
    for name, m, k, n in SCORED_SHAPES:
        n_k = (k + BK - 1) // BK
        required, launched, worst = tiles_per_threadgroup(m, n)
        row = {
            "shape": name,
            "M": m,
            "K": k,
            "N": n,
            "k_tiles": n_k,
            "k_tiles_parity": "even" if n_k % 2 == 0 else "odd",
            "required_tiles": required,
            "launched_threadgroups": launched,
            "tiles_per_threadgroup_max": worst,
            "defects": {},
        }
        for defect in DEFECTS:
            tr = run_tile_loop(n_k, max(worst, 1), pipelined=True, defect=defect)
            row["defects"][defect] = {
                "races": len(tr.races),
                "stale_reads": len(tr.stale_reads),
                "first_race": tr.races[0] if tr.races else None,
                "first_stale_read": tr.stale_reads[0] if tr.stale_reads else None,
            }
        rows.append(row)
    return rows


def main() -> int:
    fidelity = source_fidelity()
    korder = k_order_preserved()
    enumeration = exhaustive()
    rows = scored_rows()

    scored_clean = all(
        r["defects"]["none"]["races"] == 0 and r["defects"]["none"]["stale_reads"] == 0
        for r in rows
    )
    named_caught_on_scored = {
        d: all(
            r["defects"][d]["races"] > 0 or r["defects"][d]["stale_reads"] > 0
            for r in rows
        )
        for d in NAMED_CONTROLS
    }

    sb_clean_trace = run_tile_loop(37, 2, pipelined=False, defect="none")
    sb_defect_trace = run_tile_loop(37, 2, pipelined=False, defect="drop_publish_barrier")

    proof_exact = bool(
        fidelity["header"]["matches_model"]
        and fidelity["twin"]["matches_model"]
        and fidelity["header_twin_programs_agree"]
        and enumeration["shipped_ordering_clean_everywhere"]
        and scored_clean
    )
    # The assignment names three controls. Each must be caught on every cell
    # where it is reachable AND on every scored shape. The three supplementary
    # controls are not part of this count; they are reported with the verified
    # closed form for the cells where their defeated ordering is safe.
    named_controls_caught = sum(
        1
        for d in NAMED_CONTROLS
        if enumeration["controls_caught_on_every_reachable_cell"][d]
        and named_caught_on_scored[d]
    )
    detector_works = bool(
        not sb_clean_trace.races
        and not sb_clean_trace.stale_reads
        and sb_defect_trace.races
    )
    controls_caught = bool(
        named_controls_caught == len(NAMED_CONTROLS)
        and detector_works
        and enumeration["every_uncaught_cell_has_a_verified_closed_form"]
    )

    out = {
        "experiment": "E156",
        "rung": "R1",
        "harness": "offline",
        "why_not_executed": (
            "is_nax_available() needs architecture gen >= 17 "
            "(backend/metal/device.cpp:913-931); every campaign Mac is "
            "applegpu_g16s, gen 16, so affine_qmm_t_nax executes zero times "
            "locally and no local run can observe a barrier removed from this "
            "k-loop"
        ),
        "e156_barrier_proof_exact": proof_exact,
        "e156_barrier_failing_controls_caught": named_controls_caught,
        "failing_controls_expected": len(NAMED_CONTROLS),
        "e156_barrier_all_named_controls_caught": controls_caught,
        "e156_barrier_detector_positive_control_works": detector_works,
        "e156_barrier_named_controls": NAMED_CONTROLS,
        "e156_barrier_named_controls_caught_on_scored_shapes": named_caught_on_scored,
        "e156_barrier_all_controls": [d for d in DEFECTS if d != "none"],
        "e156_k_order_preserved": bool(
            korder["inner_loops_identical_modulo_staging_pointer"]
            and korder["outer_k_loop_header_unchanged"]
            and korder["sk_step_unchanged"]
            and korder["bk_padded_unchanged"]
        ),
        "e156_k_order_detail": korder,
        "e156_barrier_source_fidelity": fidelity,
        "e156_barrier_exhaustive": enumeration,
        "e156_barrier_scored_rows": rows,
        "e156_barrier_scored_shapes_clean": scored_clean,
        "e156_barrier_single_buffered_model_clean": not sb_clean_trace.races,
        "e156_barrier_single_buffered_control_caught": bool(sb_defect_trace.races),
        "barriers_per_k_iteration": 1,
        "note_on_the_prologue_barrier": (
            "The shipped loop has no separate prologue barrier. The barrier at "
            "the top of iteration 0 publishes the prologue's staged tile, so "
            "the control that deletes it is `drop_prologue_publish_barrier`, "
            "and it is caught on every shape with at least one k tile."
        ),
    }
    out["e156_barrier_proof_pass"] = bool(
        proof_exact and controls_caught and out["e156_k_order_preserved"]
    )
    out["configurations_enumerated"] = enumeration["cells_enumerated"]
    out["orderings_checked"] = len(DEFECTS)
    out["source_binding_ok"] = bool(
        fidelity["header"]["matches_model"]
        and fidelity["twin"]["matches_model"]
        and fidelity["header_twin_programs_agree"]
    )

    safe_forms = enumeration["controls_safe_cell_closed_forms"]
    supplementary = ", ".join(
        f"{d} caught {enumeration['controls_caught_cells'][d]}/"
        f"{enumeration['controls_reachable_cells'][d]}, "
        f"{safe_forms[d]['uncaught_cells']} uncaught cells all matched by its "
        f"closed form ({safe_forms[d]['closed_form_matches_enumeration']})"
        for d in DEFECTS[4:]
    )
    out["evidence"] = (
        f"The event program of the shipped `kDoubleBuffer` branch was extracted "
        f"from BOTH {fidelity['header']['path']} and {fidelity['twin']['path']} "
        f"and both match the model "
        f"({fidelity['header_twin_programs_agree']} agree). The enumeration "
        f"covers every reachable loop shape: k tiles 1 to {MAX_K_TILES} crossed "
        f"with 1 to {MAX_TILES_PER_THREADGROUP} output tiles per threadgroup, "
        f"{enumeration['cells_enumerated']} cells, "
        f"{enumeration['staging_reads_enumerated']} staged reads and "
        f"{enumeration['staging_writes_enumerated']} staged writes. The shipped "
        f"ordering has zero races and zero stale reads on every cell "
        f"({enumeration['shipped_ordering_clean_everywhere']}) and on every "
        f"scored shape ({scored_clean}). All "
        f"{named_controls_caught} of {len(NAMED_CONTROLS)} named failing "
        f"controls are caught on every reachable cell and on every scored "
        f"shape. The detector itself is positive-controlled: the unmodified "
        f"single-buffered loop is clean and deleting its publish barrier is "
        f"caught ({detector_works}). Supplementary controls: {supplementary}. "
        f"An uncaught cell is not a hole: the enumerator flags a catch exactly "
        f"when the defeated ordering overlaps a read and a write, so an "
        f"uncaught cell proves the barrier is redundant there, and every such "
        f"cell is matched exactly by a checked closed form "
        f"({enumeration['every_uncaught_cell_has_a_verified_closed_form']})."
    )
    out["k_order_evidence"] = (
        f"The pipelined inner `kk1` loop is textually identical to the "
        f"single-buffered one after rewriting the staging base pointer `Wk` to "
        f"`Ws`: {korder['inner_loops_identical_modulo_staging_pointer']}. "
        f"The outer k loop header is unchanged in both branches: "
        f"{korder['outer_k_loop_header_unchanged']}. SK is unchanged: "
        f"{korder['sk_step_unchanged']}. BK_padded is unchanged: "
        f"{korder['bk_padded_unchanged']}. {korder['only_difference']}. Every "
        f"output element therefore accumulates the same K partial products in "
        f"the same order, so the pipelined loop is bit-identical to the "
        f"single-buffered one."
    )

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    for key in (
        "e156_barrier_proof_exact",
        "e156_barrier_failing_controls_caught",
        "failing_controls_expected",
        "e156_barrier_all_named_controls_caught",
        "e156_barrier_detector_positive_control_works",
        "e156_k_order_preserved",
        "e156_barrier_scored_shapes_clean",
        "e156_barrier_single_buffered_model_clean",
        "e156_barrier_single_buffered_control_caught",
        "e156_barrier_proof_pass",
    ):
        print(f"{key:<52} {out[key]}")
    print()
    print(
        f"  enumerated {enumeration['cells_enumerated']} loop shapes, "
        f"{enumeration['staging_reads_enumerated']} staged reads, "
        f"{enumeration['staging_writes_enumerated']} staged writes"
    )
    for defect, ok in enumeration["controls_caught_on_every_reachable_cell"].items():
        kind = "NAMED" if defect in NAMED_CONTROLS else "suppl"
        form = safe_forms[defect]
        note = (
            "caught everywhere reachable"
            if ok
            else f"{form['uncaught_cells']} safe cells, closed form verified="
            f"{form['closed_form_matches_enumeration']}"
        )
        print(
            f"  {kind} {defect:<32} "
            f"{enumeration['controls_caught_cells'][defect]:>4}/"
            f"{enumeration['controls_reachable_cells'][defect]:<4} {note}"
        )
    print()
    for r in rows:
        flags = " ".join(
            f"{d}={r['defects'][d]['races']}/{r['defects'][d]['stale_reads']}"
            for d in NAMED_CONTROLS
        )
        print(f"  {r['shape']:<16} k_tiles={r['k_tiles']:<4} tpt="
              f"{r['tiles_per_threadgroup_max']}  {flags}")
    print(f"\nwrote {OUT}")
    return 0 if out["e156_barrier_proof_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
