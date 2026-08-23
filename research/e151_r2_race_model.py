#!/usr/bin/env python3
"""E151 rung R2: the threadgroup-ordering positive control F4 section 4 asked for.

WHY THIS IS A MODEL AND NOT A RUN.

F4 asked me to defeat the ordering and show the exactness check fails, and said
it would be cheap because I already need the exactness harness. It is not
cheap, it is impossible, and the reason is Finding 250: `is_nax_available()` is
false on every Mac this campaign owns, so `affine_qmm_t_nax` never executes
here. Removing a `threadgroup_barrier` from the pipelined loop changes no
locally observable byte, because no local thread ever reaches that loop. A
512-token local-submit with the barrier deleted would pass, and it would prove
nothing. Reporting that pass as a control would be worse than reporting no
control at all.

So the control moves offline, to the same place the R0.2 coverage proof lives.
This file models the threadgroup memory ordering of `qmm_t_nax_tgp_impl` as an
event trace over barrier epochs, and asserts BOTH polarities:

  positive: the shipped pipelined loop is race-free and stages the correct k
            tile for every read, on every scored shape;
  negative: four separately defeated orderings are each CAUGHT, by name.

The model is at half-tile granularity. Inside one barrier epoch every thread of
the threadgroup runs concurrently, so two accesses to the same staging half in
the same epoch conflict when at least one is a write. That is exactly the
hazard a `threadgroup_barrier(mem_flags::mem_threadgroup)` exists to prevent,
and it is the whole of what the double buffer changes.

WHAT THE MODEL DOES NOT COVER. It does not model SIMD divergence, the memory
consistency of `mem_flags::mem_threadgroup` itself, or the correctness of the
arithmetic. R0.3 owns the arithmetic argument (K order unchanged, so
bit-identical) and the compile gate owns buildability.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

BK = 64
HOST_BM = 64
HOST_BN = 64
ARM_BM = 128
ARM_BN = 32

# From research/e151-r0-safety-case.json, e151_retile_shape_table.
SCORED_SHAPES = [
    ("gdn.in_proj", 512, 5120, 16480, True),
    ("fa.qkv", 512, 5120, 14336, True),
    ("mlp.gate_up", 512, 5120, 34816, True),
    ("lm_head", 512, 5120, 248320, False),
    ("gdn.out_proj", 512, 6144, 5120, True),
    ("fa.o_proj", 512, 6144, 5120, True),
    ("mlp.down", 512, 17408, 5120, True),
]

# A threadgroup only reaches a second grid-stride tile when required > launched,
# which needs ceil(M/128) == ceil(M/64), that is M <= 64. Every scored prefill
# shape is M = 512 or M = 511 and therefore single-tile (R0.1:
# tiles_per_threadgroup_max == 1 everywhere). The retile arm is a compile-time
# choice, not gated on runtime M, so a small-M call through this kernel WOULD
# take the multi-tile path. These shapes exercise it, one with an even k-tile
# count and one odd, so the model can separate "the barrier protects it" from
# "the halves happen not to alias".
SYNTHETIC_SHAPES = [
    ("synthetic.even_k.multitile.M64", 64, 5120, 512, False),
    ("synthetic.odd_k.multitile.M64", 64, 192, 512, False),
]


@dataclass
class Trace:
    """An ordered event log over barrier epochs for one threadgroup."""

    epoch: int = 0
    # (epoch, half) -> list of (op, k_tile)
    accesses: dict = field(default_factory=dict)
    # half -> k_tile currently staged, as the model believes it
    staged: dict = field(default_factory=dict)
    races: list = field(default_factory=list)
    stale_reads: list = field(default_factory=list)

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
        self._touch(half, "W", k_tile)
        self.staged[half] = k_tile

    def read(self, half: int, k_tile: int) -> None:
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


def pipelined_tile(
    tr: Trace, n_k: int, defect: str, emit_prologue: bool, next_tile_follows: bool
) -> None:
    """`qmm_t_nax_tgp_impl`'s kDoubleBuffer=true k-loop for one output tile.

    Mirrors quantized_nax.h: prologue stages k tile 0 into half 0 and shifts
    `dst` to half 1; iteration i stages k tile i+1 into half 1-cur and reads
    k tile i out of half cur.
    """
    if emit_prologue and n_k > 0:
        tr.write(half=0, k_tile=0)
    dst = 1
    cur = 0
    for i in range(n_k):
        if defect != "drop_per_iteration_barrier":
            tr.barrier()
        if i + 1 < n_k:
            # `defeat_double_buffer` keeps every barrier but stages the next k
            # tile on top of the one the mma is about to read. It is the
            # control that fires on the scored cell itself.
            target = cur if defect == "defeat_double_buffer" else dst
            tr.write(half=target, k_tile=i + 1)
            dst = 1 - dst
        if (
            defect == "prologue_before_last_read"
            and next_tile_follows
            and i + 1 == n_k
        ):
            # Hoist the next tile's prologue above this tile's final read, so
            # it shares an epoch with it instead of following the pre-store
            # barrier.
            tr.write(half=0, k_tile=0)
        tr.read(half=cur, k_tile=i)
        cur ^= 1


def single_buffered_tile(tr: Trace, n_k: int, defect: str) -> None:
    """The unmodified kDoubleBuffer=false k-loop, used as a model sanity check."""
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
        # compute_tile's pre-store barrier, then Dtile.store to device memory.
        # The store never touches Ws, so it raises no threadgroup hazard.
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


DEFECTS = [
    "none",
    "drop_per_iteration_barrier",
    "defeat_double_buffer",
    "drop_pre_store_barrier",
    "prologue_before_last_read",
]


def main() -> int:
    shapes = [(*s, True) for s in SCORED_SHAPES] + [
        (*s, False) for s in SYNTHETIC_SHAPES
    ]
    rows = []
    for name, m, k, n, scored, is_scored_shape in shapes:
        n_k = (k + BK - 1) // BK
        required, launched, worst = tiles_per_threadgroup(m, n)
        row = {
            "shape": name,
            "M": m,
            "K": k,
            "N": n,
            "evaluated_in_scored_seed_prefill": scored,
            "from_scored_shape_table": is_scored_shape,
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

    scored_rows = [r for r in rows if r["from_scored_shape_table"]]
    multitile_rows = [r for r in rows if r["tiles_per_threadgroup_max"] > 1]
    odd_multitile = [r for r in multitile_rows if r["k_tiles"] % 2 == 1]

    def clean(rs, defect):
        return all(
            r["defects"][defect]["races"] == 0
            and r["defects"][defect]["stale_reads"] == 0
            for r in rs
        )

    def caught(rs, defect):
        return bool(rs) and all(
            r["defects"][defect]["races"] > 0
            or r["defects"][defect]["stale_reads"] > 0
            for r in rs
        )

    # Model sanity: the unmodified single-buffered loop must also come out
    # clean, and dropping ITS publish barrier must be caught. A detector that
    # only ever fires on the new code is not measuring the new code.
    sb_clean = clean(
        [
            {
                "defects": {
                    "none": {
                        "races": len(
                            run_tile_loop(80, 1, pipelined=False, defect="none").races
                        ),
                        "stale_reads": 0,
                    }
                }
            }
        ],
        "none",
    )
    sb_defect = run_tile_loop(80, 1, pipelined=False, defect="drop_publish_barrier")

    out = {
        "experiment": "E151",
        "rung": "R2",
        "harness": "offline",
        "model": "threadgroup staging-half access trace over barrier epochs",
        "why_not_executed": (
            "is_nax_available() is false on every campaign Mac (Finding 250), so "
            "affine_qmm_t_nax never runs locally and no local exactness leg can "
            "observe a barrier removed from its k-loop"
        ),
        "e151_r2_race_model_rows": rows,
        "e151_r2_shipped_ordering_race_free": clean(rows, "none"),
        "e151_r2_shipped_staging_values_correct": all(
            r["defects"]["none"]["stale_reads"] == 0 for r in rows
        ),
        "e151_r2_scored_shapes_single_tile_per_threadgroup": all(
            r["tiles_per_threadgroup_max"] == 1 for r in scored_rows
        ),
        "e151_r2_control_drop_per_iteration_barrier_caught": caught(
            rows, "drop_per_iteration_barrier"
        ),
        "e151_r2_control_defeat_double_buffer_caught": caught(
            rows, "defeat_double_buffer"
        ),
        "e151_r2_control_drop_pre_store_barrier_caught": caught(
            odd_multitile, "drop_pre_store_barrier"
        ),
        "e151_r2_control_prologue_before_last_read_caught": caught(
            odd_multitile, "prologue_before_last_read"
        ),
        "e151_r2_single_buffered_model_race_free": sb_clean,
        "e151_r2_single_buffered_control_caught": len(sb_defect.races) > 0,
        "tile_boundary_note": (
            "On every scored shape the grid-stride body runs at most once per "
            "threadgroup, so the tile-boundary hazard is unreachable there. The "
            "two tile-boundary controls therefore fire only on a synthetic "
            "multi-tile shape, and only when the k-tile count is odd. With an "
            "even k-tile count the staging halves at a tile boundary do not "
            "alias, so the pre-store barrier is not the only thing protecting "
            "it. All seven scored shapes have an even k-tile count."
        ),
    }
    required_true = [k for k in out if k.startswith("e151_r2_") and k != "e151_r2_race_model_rows"]
    out["e151_r2_race_model_required_fields"] = required_true
    out["e151_r2_race_model_pass"] = all(out[k] is True for k in required_true)

    dest = pathlib.Path(__file__).resolve().parent / "e151-r2-race-model.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    for r in rows:
        flags = " ".join(
            f"{d}={r['defects'][d]['races']}/{r['defects'][d]['stale_reads']}"
            for d in DEFECTS
        )
        print(
            f"{r['shape']:<30} k_tiles={r['k_tiles']:<4} "
            f"tpt={r['tiles_per_threadgroup_max']}  {flags}"
        )
    print()
    for k in required_true:
        print(f"{k:<62} {out[k]}")
    print(f"\ne151_r2_race_model_pass = {out['e151_r2_race_model_pass']}")
    print(f"wrote {dest}")
    return 0 if out["e151_r2_race_model_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
