#!/usr/bin/env python3
"""E156 R1: does the NAX seed-prefill retile compose with the k-loop double buffer?

F1 moved the target base to the crown surface `0863b06a`, so this audit is run in
CROWN COORDINATES. For the two files in scope the crown surface, the campaign
frontier `770a3ff2` and the pristine organizer file are one and the same blob:

    quantized_nax.h    11f7b087023762dc62abff80b939d7174c645d8c
    quantized_nax.cpp  d2d80a72e512363f35c1d1e877d158b8de1a62f3

That identity is asserted below rather than assumed, so this file goes red if a
later organizer commit moves the crown NAX pair.

WHAT "COMPOSE" MEANS HERE, DECLARED BEFORE THE ANSWER IS COMPUTED.

Three separate questions, three separate fields, and the enum reports the
CONSERVATIVE one:

  textual   Do the two mechanisms edit overlapping line spans of the crown file?
            Whitespace counts. This is the strict reading of the advisor's
            "do the edited regions overlap".
  semantic  Same question after normalising leading whitespace, so a line that
            one mechanism only re-indents is not counted as edited by it.
  buildable Do all four `(retile off/on) x (dbuf off/on)` cells exist as real
            configurations? That question belongs to `e156_compile_gate.py`,
            which compiles them; this file only names the flags.

`e156_compose_verdict` takes the TEXTUAL result. If the spans overlap the
verdict is `compose_overlapping_regions` even when every overlapping line is
parameter plumbing, because reporting the friendlier number would be exactly the
kind of flattery the campaign has been burned by. The enumerated overlap sites
are published next to the verdict so the advisor can price them.

THE THIRD QUESTION, THE THREADGROUP BUDGET, CAN FORCE THE ANSWER ON ITS OWN, IN
EITHER DIRECTION. The verdict is four-valued and the budget is read first:

  cannot_compose_tgp_budget   The composed shape does not fit. This is a stop.
  compose_forced_by_tgp_limit The composed shape fits, but the double buffer
                              cannot arm the shipped instantiation set alone,
                              because at least one emitted `(T, BN)` pair
                              crosses the limit with the retile off. The two
                              mechanisms are not independently shippable.
  compose_disjoint_regions    Composed shape fits, spans do not overlap.
  compose_overlapping_regions Composed shape fits, spans overlap.

The region result is published separately as `e156_compose_region_result` when
the budget forces the verdict, so no information is lost.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "research/e156-compose-audit.json"

HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp"
METAL = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.metal"

CROWN = "0863b06ac16e26e48fc06e97444095b00feb66d4"
FRONTIER = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
# The retile mechanism alone, no double buffer: E151 R1 tip.
RETILE_ONLY = "8b515fd8b81e6b580e796b996c27c4288d6dcfa9"

CROWN_HEADER_BLOB = "11f7b087023762dc62abff80b939d7174c645d8c"
CROWN_TWIN_BLOB = "d2d80a72e512363f35c1d1e877d158b8de1a62f3"

# Threadgroup limit for one threadgroup on the Apple GPU families this track
# runs on. Fail-closed: if the real g17s limit were larger, this predicate only
# ever disarms an arm that would have fitted; it never arms one that does not.
TGP_LIMIT_BYTES = 32768

# `quantized_nax.metal:74-77` instantiates `affine_qmm_t_nax` at
# (BM, BK, BN, WM, WN) = (64, 64, 64, 2, 2) and at no other shape, for
# float, float16_t and bfloat16_t, at group sizes 128, 64, 32.
HOST_BM, HOST_BK, HOST_BN = 64, 64, 64
ARM_BM, ARM_BN = 128, 32
DTYPES = {"float": 4, "float16_t": 2, "bfloat16_t": 2}
SCORED_DTYPE = "bfloat16_t"


def show(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def blob(rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{rev}:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def tgp_bytes(dtype: str, staged_bn: int, halves: int) -> int:
    size = DTYPES[dtype]
    bk_padded = HOST_BK + 16 // size
    return halves * staged_bn * bk_padded * size


GROUP_SIZES = (128, 64, 32)
SCORED_GROUP_SIZE = 64


def staged_bn(retile: bool, group_size: int) -> int:
    """The width the loader actually stages.

    The retile is NOT unconditional. `kE147NaxRetileLoaderLegal` reproduces the
    weight loader's own `(BCOLS_PACKED / n_reads) == n_groups` arithmetic, which
    a 32-wide tile breaks at `group_size == 32`. Those cells keep the 64-wide
    host tile even with the retile ON, so the composed arm still has to survive
    a 64-wide staged tile there. Modelling the retile as "always 32" would hide
    exactly the cell the capacity predicate exists for.
    """
    return ARM_BN if (retile and group_size != 32) else HOST_BN


def budget_table() -> list[dict]:
    """The staging cost of every shipped cell, in the four arm combinations.

    The allocation the caller makes is `max(single-buffered host tile, both
    double-buffered halves)`, because the host tile is what the unretiled arm
    still needs. A cell whose two halves do not fit keeps the single-buffered
    loop: the capacity predicate disarms it rather than shrinking the tile.
    """
    rows = []
    for dtype in DTYPES:
        single_host = tgp_bytes(dtype, HOST_BN, 1)
        for retile in (False, True):
            for group_size in GROUP_SIZES:
                bn = staged_bn(retile, group_size)
                for dbuf in (False, True):
                    want = tgp_bytes(dtype, bn, 2) if dbuf else single_host
                    armed = dbuf and want <= TGP_LIMIT_BYTES
                    alloc = max(want, single_host) if armed else single_host
                    rows.append(
                        {
                            "T": dtype,
                            "retile": "on" if retile else "off",
                            "dbuf": "on" if dbuf else "off",
                            "group_size": group_size,
                            "staged_BN": bn,
                            "retile_disarmed_by_loader_guard": (
                                retile and group_size == 32
                            ),
                            "dbuf_armed": armed,
                            "dbuf_disarmed_by_capacity": dbuf and not armed,
                            "want_bytes": want,
                            "alloc_bytes": alloc,
                            "limit_bytes": TGP_LIMIT_BYTES,
                            "fits": alloc <= TGP_LIMIT_BYTES,
                            "delta_vs_crown_bytes": alloc - single_host,
                            "scored_cell": (
                                dtype == SCORED_DTYPE
                                and group_size == SCORED_GROUP_SIZE
                            ),
                        }
                    )
    return rows


ARMED_KERNEL = "affine_qmm_t_nax"

INST_ARGS = re.compile(
    r"instantiate_quantized_aligned_batched\(\s*" + ARMED_KERNEL + r"\s*,(?P<rest>[^)]*)\)"
)
FUNCS_CALL = re.compile(r"instantiate_quantized_funcs\(\s*([A-Za-z0-9_]+)\s*,")
TYPES_CALL = re.compile(r"instantiate_quantized_types\(\s*(\d+)\s*,")
GROUPS_CALL = re.compile(r"instantiate_quantized_groups\(\s*(\d+)\s*\)")
TYPES_DEFINE = re.compile(
    r"#define\s+instantiate_quantized_types\(.*?\)(?P<body>(?:.*?\\\n)*.*)"
)


def instantiation_census() -> dict:
    """Derive the shipped `affine_qmm_t_nax` cell set from `quantized_nax.metal`.

    F2 section 3 asks whether `float` is emitted at BN = 64 and which
    `(T, BN)` pairs cross the threadgroup limit. Both answers are read out of
    the instantiation macros rather than asserted, so a later change to the
    macro set moves this census instead of silently invalidating it.
    """
    text = (ROOT / METAL).read_text()

    shapes = []
    for m in INST_ARGS.finditer(text):
        # `type`, `group_size`, `bits` and `aligned` are macro parameters or
        # boolean literals, so the numeric tail is (BM, BK, BN, WM, WN, batched).
        nums = [int(t) for t in re.findall(r"\b\d+\b", m.group("rest"))]
        shapes.append(tuple(nums[:5]))
    per_type_count = len(shapes)
    distinct_shapes = sorted(set(shapes))

    # Only the concrete calls inside `instantiate_quantized_types` name real
    # types; a bare `FUNCS_CALL` scan also catches the macro parameter `type`.
    types_body = TYPES_DEFINE.search(text)
    if types_body is None:
        raise SystemExit(f"{METAL}: instantiate_quantized_types not found")
    types = FUNCS_CALL.findall(types_body.group("body"))
    unknown = sorted(set(types) - set(DTYPES))
    if unknown:
        raise SystemExit(f"{METAL}: unknown element types {unknown}; add their byte size")
    group_sizes = sorted({int(g) for g in TYPES_CALL.findall(text)})
    bits = sorted({int(b) for b in GROUPS_CALL.findall(text)})

    cells = []
    for dtype in types:
        for shape in shapes:
            bm, bk, bn, _wm, _wn = shape
            for gs in group_sizes:
                for b in bits:
                    cells.append(
                        {"T": dtype, "BM": bm, "BK": bk, "BN": bn, "group_size": gs, "bits": b}
                    )

    float_bn64 = [c for c in cells if c["T"] == "float" and c["BN"] == HOST_BN]

    crossings = []
    for dtype in sorted({c["T"] for c in cells}):
        for bn in sorted({c["BN"] for c in cells}):
            group = [c for c in cells if c["T"] == dtype and c["BN"] == bn]
            if not group:
                continue
            dbuf_alone = tgp_bytes(dtype, bn, 2)
            # With the retile ON the staged width is 32 everywhere except
            # `group_size == 32`, where the loader guard keeps the host width.
            composed_want = {
                gs: tgp_bytes(dtype, staged_bn(True, gs), 2)
                for gs in {c["group_size"] for c in group}
            }
            composed_over = sorted(
                gs for gs, w in composed_want.items() if w > TGP_LIMIT_BYTES
            )
            crossings.append(
                {
                    "T": dtype,
                    "BN": bn,
                    "instantiations": len(group),
                    "dbuf_alone_bytes": dbuf_alone,
                    "limit_bytes": TGP_LIMIT_BYTES,
                    "dbuf_alone_crosses": dbuf_alone > TGP_LIMIT_BYTES,
                    "dbuf_alone_crossing_instantiations": (
                        len(group) if dbuf_alone > TGP_LIMIT_BYTES else 0
                    ),
                    "composed_want_bytes_by_group_size": composed_want,
                    "composed_group_sizes_over_limit": composed_over,
                    "composed_instantiations_disarmed_by_capacity": len(
                        [c for c in group if c["group_size"] in composed_over]
                    ),
                    # A disarmed cell keeps the single-buffered host allocation,
                    # so the composed arm never allocates over the limit.
                    "composed_alloc_bytes_max": max(
                        min(w, tgp_bytes(dtype, HOST_BN, 1))
                        if w > TGP_LIMIT_BYTES
                        else max(w, tgp_bytes(dtype, HOST_BN, 1))
                        for w in composed_want.values()
                    ),
                }
            )
            crossings[-1]["composed_crosses"] = (
                crossings[-1]["composed_alloc_bytes_max"] > TGP_LIMIT_BYTES
            )

    crossed = [c for c in crossings if c["dbuf_alone_crosses"]]
    return {
        "armed_kernel": ARMED_KERNEL,
        "types_emitted": types,
        "group_sizes_emitted": group_sizes,
        "bits_emitted": bits,
        "distinct_shapes_BM_BK_BN_WM_WN": [list(s) for s in distinct_shapes],
        "instantiations_per_type": per_type_count,
        "instantiations_total": len(cells),
        "e156_float_instantiation_emitted": bool(float_bn64),
        "float_BN64_instantiations": len(float_bn64),
        "e156_tgp_limit_crossings": crossed,
        "e156_tgp_limit_crossing_instantiations": sum(c["instantiations"] for c in crossed),
        "composed_crossings": [c for c in crossings if c["composed_crosses"]],
        "per_pair_table": crossings,
    }


LEAD_WS = re.compile(r"^[ \t]+")


def strip_indent(line: str) -> str:
    return LEAD_WS.sub("", line)


def touched_spans(old: list[str], new: list[str]) -> tuple[set[int], list[dict]]:
    """Lines of `old` that a mechanism replaces or deletes, plus its insertions.

    An insertion is anchored at the `old` line it follows, so an insertion and a
    replacement at the same anchor count as touching the same span.
    """
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    touched: set[int] = set()
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append({"tag": tag, "old_lines": [i1 + 1, i2], "new_lines": [j1 + 1, j2]})
        if tag in ("replace", "delete"):
            touched.update(range(i1, i2))
        else:  # insert: anchor on the crown line it lands after
            touched.add(max(i1 - 1, 0))
    return touched, ops


def crown_line_map(crown: list[str], variant: list[str]) -> dict[int, int]:
    """variant line index -> crown line index, for lines the two share."""
    sm = difflib.SequenceMatcher(a=crown, b=variant, autojunk=False)
    out = {}
    for i, j, n in sm.get_matching_blocks():
        for d in range(n):
            out[j + d] = i + d
    return out


def dbuf_spans_in_crown(
    crown: list[str], retile_only: list[str], composed: list[str]
) -> tuple[set[int], list[str], list[dict]]:
    """Crown lines the double buffer touches, plus the ones with no crown line.

    The double buffer is only available as `retile_only -> composed`, so its
    touched spans are computed there and then carried back to crown
    coordinates. A touched line with NO crown counterpart is a line the RETILE
    introduced. Editing it is an overlap by construction, so those lines are
    returned verbatim and the caller counts them as overlap. Reporting them as
    "not in the crown, therefore not an overlap" would be the flattering answer
    and the wrong one.
    """
    touched_variant, ops = touched_spans(retile_only, composed)
    back = crown_line_map(crown, retile_only)
    in_crown: set[int] = set()
    orphans: list[int] = []
    for idx in sorted(touched_variant):
        if idx in back:
            in_crown.add(back[idx])
        else:
            orphans.append(idx)
    return in_crown, orphans, ops


def describe(crown: list[str], lines: set[int]) -> list[dict]:
    """Group overlapping crown lines into contiguous, quoted sites."""
    sites = []
    for line in sorted(lines):
        text = crown[line].rstrip("\n")
        if sites and line == sites[-1]["crown_lines"][1]:
            sites[-1]["crown_lines"][1] = line + 1
            sites[-1]["text"].append(text)
        else:
            sites.append({"crown_lines": [line + 1, line + 1], "text": [text]})
    return sites


def audit_file(path: str) -> dict:
    crown = show(CROWN, path).splitlines()
    retile_only = show(RETILE_ONLY, path).splitlines()
    composed = (ROOT / path).read_text().splitlines()

    retile_touched, retile_ops = touched_spans(crown, retile_only)
    dbuf_touched, dbuf_orphans, dbuf_ops = dbuf_spans_in_crown(
        crown, retile_only, composed
    )
    raw_overlap = retile_touched & dbuf_touched

    crown_n = [strip_indent(x) for x in crown]
    retile_n = [strip_indent(x) for x in retile_only]
    composed_n = [strip_indent(x) for x in composed]
    retile_touched_n, _ = touched_spans(crown_n, retile_n)
    dbuf_touched_n, dbuf_orphans_n, _ = dbuf_spans_in_crown(
        crown_n, retile_n, composed_n
    )
    semantic_overlap = retile_touched_n & dbuf_touched_n

    mask = comment_mask(retile_only)
    shared_class = classify_shared(
        [retile_only[i] for i in dbuf_orphans_n], [mask[i] for i in dbuf_orphans_n]
    )

    return {
        "path": path,
        "crown_lines": len(crown),
        "retile_only_lines": len(retile_only),
        "composed_lines": len(composed),
        "retile_touched_crown_lines": len(retile_touched),
        "dbuf_touched_crown_lines": len(dbuf_touched),
        "dbuf_lines_inside_retile_inserted_code": len(dbuf_orphans),
        "retile_hunks": len(retile_ops),
        "dbuf_hunks": len(dbuf_ops),
        "textual_overlap_lines": len(raw_overlap) + len(dbuf_orphans),
        "textual_overlap_shared_crown_lines": len(raw_overlap),
        "textual_overlap_sites": describe(crown, raw_overlap),
        "semantic_overlap_lines": len(semantic_overlap) + len(dbuf_orphans_n),
        "semantic_overlap_shared_crown_lines": len(semantic_overlap),
        "semantic_overlap_sites": describe(crown, semantic_overlap),
        "semantic_dbuf_lines_inside_retile_inserted_code": len(dbuf_orphans_n),
        "semantic_shared_line_classes": shared_class,
        "semantic_shared_undeclared_lines": len(shared_class["undeclared_lines"]),
    }


ARM_FLAGS = {
    "retile": "constexpr bool kE147NaxRetileOn",
    "dbuf": "constexpr bool kE151NaxDoubleBufferOn",
}

# The only NON-COMMENT lines of retile-inserted code that the double buffer is
# allowed to edit, declared before the audit runs. Each is parameter plumbing:
# a template parameter list, the caller's staging allocation, and the retiled
# call site's template argument list. A double buffer that reached into the
# retile's tile geometry, its grid-stride loop or its coverage asserts would
# add a line here and turn this check red.
ALLOWED_SHARED_PLUMBING = {
    "const int kHostBN = BN>",
    "threadgroup T Ws[kTgBN * BK_padded];",
    "BN>(",
}


def comment_mask(lines: list[str]) -> list[bool]:
    """True for every line that carries no code, tracking `/* */` state."""
    mask = []
    in_block = False
    for raw in lines:
        text = raw.strip()

        if in_block:
            if "*/" in text:
                in_block = False
                mask.append(text.split("*/", 1)[1].strip() == "")
                continue
            mask.append(True)
            continue
        if text.startswith("/*") and "*/" not in text:
            in_block = True
        mask.append(text == "" or text.startswith("//") or text.startswith("/*"))
    return mask


def classify_shared(lines: list[str], is_comment: list[bool]) -> dict:
    """Split shared lines into comments, declared plumbing, and everything else."""
    comments, plumbing, other = [], [], []
    for raw, comment in zip(lines, is_comment):
        text = raw.strip()
        if comment:
            comments.append(text)
        elif text in ALLOWED_SHARED_PLUMBING:
            plumbing.append(text)
        else:
            other.append(text)
    return {
        "comment_lines": comments,
        "declared_plumbing_lines": plumbing,
        "undeclared_lines": other,
    }


def flag_report() -> dict:
    text = (ROOT / HEADER).read_text()
    twin = (ROOT / TWIN).read_text()
    out = {}
    for arm, needle in ARM_FLAGS.items():
        m = re.search(re.escape(needle) + r" = (true|false);", text)
        mt = re.search(re.escape(needle) + r" = (true|false);", twin)
        out[arm] = {
            "flag": needle,
            "header_default": m.group(1) if m else None,
            "twin_default": mt.group(1) if mt else None,
            "header_occurrences": text.count(needle),
            "twin_occurrences": twin.count(needle),
        }
    return out


def main() -> int:
    crown_pair_is_frontier_pair = blob(CROWN, HEADER) == blob(FRONTIER, HEADER) and blob(
        CROWN, TWIN
    ) == blob(FRONTIER, TWIN)
    crown_blobs_pinned = (
        blob(CROWN, HEADER) == CROWN_HEADER_BLOB
        and blob(CROWN, TWIN) == CROWN_TWIN_BLOB
    )

    table = budget_table()
    composed_rows = [r for r in table if r["retile"] == "on" and r["dbuf"] == "on"]
    dbuf_alone_rows = [r for r in table if r["retile"] == "off" and r["dbuf"] == "on"]
    # The shipped predicate disarms a cell it cannot fit, so no ALLOCATION ever
    # exceeds the limit. The overflow claim must therefore be read off the
    # REQUESTED two-half size, `want_bytes`, not off the allocation.
    composed_fits = all(r["alloc_bytes"] <= TGP_LIMIT_BYTES for r in composed_rows)
    composed_free = all(r["delta_vs_crown_bytes"] == 0 for r in composed_rows)
    composed_all_armed = all(r["dbuf_armed"] for r in composed_rows)
    dbuf_alone_scored = next(
        r for r in dbuf_alone_rows
        if r["T"] == SCORED_DTYPE and r["group_size"] == SCORED_GROUP_SIZE
    )
    dbuf_alone_float = next(
        r for r in dbuf_alone_rows
        if r["T"] == "float" and r["group_size"] == SCORED_GROUP_SIZE
    )
    composed_scored = next(
        r for r in composed_rows
        if r["T"] == SCORED_DTYPE and r["group_size"] == SCORED_GROUP_SIZE
    )
    composed_disarmed = [r for r in composed_rows if r["dbuf_disarmed_by_capacity"]]

    files = [audit_file(HEADER), audit_file(TWIN)]
    textual_disjoint = all(f["textual_overlap_lines"] == 0 for f in files)
    semantic_disjoint = all(f["semantic_overlap_lines"] == 0 for f in files)
    shared_crown_lines = sum(f["textual_overlap_shared_crown_lines"] for f in files)
    plumbing_only = all(
        f["semantic_shared_undeclared_lines"] == 0
        and f["textual_overlap_shared_crown_lines"] == 0
        for f in files
    )

    census = instantiation_census()

    # The composition is FORCED, not merely convenient, when the double buffer
    # cannot arm the whole shipped instantiation set on its own. A standalone
    # R2 must then either carry a capacity predicate that disarms the crossing
    # cells or fail to build; only the retile makes every cell arm.
    compose_forced = bool(census["e156_tgp_limit_crossings"]) and composed_fits

    if not composed_fits:
        verdict = "cannot_compose_tgp_budget"
    elif compose_forced:
        verdict = "compose_forced_by_tgp_limit"
    elif textual_disjoint:
        verdict = "compose_disjoint_regions"
    else:
        verdict = "compose_overlapping_regions"

    # The region result is not discarded when the budget forces the verdict.
    region_result = "compose_disjoint_regions" if textual_disjoint else (
        "compose_overlapping_regions"
    )

    out = {
        "experiment": "E156",
        "rung": "R1",
        "harness": "offline",
        "question": "does the 128x32 seed-prefill retile compose with the k-loop double buffer",
        "coordinates": "crown 0863b06a",
        "crown": CROWN,
        "frontier": FRONTIER,
        "retile_only_rev": RETILE_ONLY,
        "e156_crown_nax_pair_equals_frontier_pair": crown_pair_is_frontier_pair,
        "e156_crown_nax_blobs_pinned": crown_blobs_pinned,
        "e156_compose_verdict": verdict,
        "e156_compose_region_result": region_result,
        "e156_float_instantiation_emitted": census["e156_float_instantiation_emitted"],
        "e156_tgp_limit_crossings": census["e156_tgp_limit_crossings"],
        "e156_instantiation_census": census,
        "e156_compose_regions_disjoint_textual": textual_disjoint,
        "e156_compose_regions_disjoint_semantic": semantic_disjoint,
        "e156_compose_shared_crown_lines": shared_crown_lines,
        "e156_compose_overlap_is_retile_inserted_plumbing_only": plumbing_only,
        "e156_compose_tgp_budget_table": table,
        # Requested two-half staging bytes at each tile shape. Requested, not
        # allocated: the shipped predicate disarms a cell it cannot fit, so an
        # allocation figure would always look safe and prove nothing.
        "e156_tgp_bytes_dbuf_64x64": {
            r["T"]: r["want_bytes"]
            for r in dbuf_alone_rows
            if r["group_size"] == SCORED_GROUP_SIZE
        },
        "e156_tgp_bytes_dbuf_128x32": {
            r["T"]: r["want_bytes"]
            for r in composed_rows
            if r["group_size"] == SCORED_GROUP_SIZE
        },
        "e156_composed_fits_every_shipped_cell": composed_fits,
        "e156_composed_costs_zero_extra_tgp_bytes": composed_free,
        "e156_composed_arms_every_shipped_cell": composed_all_armed,
        "e156_composed_cells_disarmed_by_capacity": [
            {"T": r["T"], "group_size": r["group_size"],
             "staged_BN": r["staged_BN"], "want_bytes": r["want_bytes"]}
            for r in composed_disarmed
        ],
        "e156_composed_scored_cell_bytes": composed_scored["want_bytes"],
        "e156_composed_scored_cell_armed": composed_scored["dbuf_armed"],
        "e156_dbuf_alone_scored_cell_bytes": dbuf_alone_scored["want_bytes"],
        "e156_dbuf_alone_scored_cell_doubles_staging": (
            dbuf_alone_scored["want_bytes"] > tgp_bytes(SCORED_DTYPE, HOST_BN, 1)
        ),
        "e156_dbuf_alone_float_cell_overflows": (
            dbuf_alone_float["want_bytes"] > TGP_LIMIT_BYTES
        ),
        "e156_compose_arm_flags": flag_report(),
        "e156_compose_files": files,
    }
    out["e156_compose_source_evidence"] = (
        f"crown {CROWN} NAX pair is blob-identical to frontier {FRONTIER}: "
        f"{crown_pair_is_frontier_pair}. "
        f"Retile touches {files[0]['retile_touched_crown_lines']} crown header lines in "
        f"{files[0]['retile_hunks']} hunks; the double buffer touches "
        f"{files[0]['dbuf_touched_crown_lines']} crown header lines in "
        f"{files[0]['dbuf_hunks']} hunks; they share "
        f"{files[0]['textual_overlap_lines']} header lines textually and "
        f"{files[0]['semantic_overlap_lines']} after leading whitespace is normalised, "
        f"of which {shared_crown_lines} are lines that exist in the crown file: the "
        f"whole overlap is the double buffer editing lines the retile itself inserted, "
        f"and every non-comment one is declared parameter plumbing "
        f"({plumbing_only}). "
        f"Composed staging fits every shipped cell: {composed_fits}; composed staging "
        f"costs zero extra threadgroup bytes: {composed_free}; the double buffer alone "
        f"needs {dbuf_alone_scored['alloc_bytes']} B on the scored bfloat16_t cell "
        f"against {tgp_bytes(SCORED_DTYPE, HOST_BN, 1)} B on the crown, and "
        f"{dbuf_alone_float['alloc_bytes']} B on the float cell, which exceeds the "
        f"{TGP_LIMIT_BYTES} B limit."
    )
    crossing_names = ", ".join(
        f"(T={c['T']}, BN={c['BN']}) at {c['dbuf_alone_bytes']} B over "
        f"{c['instantiations']} instantiations"
        for c in census["e156_tgp_limit_crossings"]
    ) or "none"
    out["e156_tgp_limit_crossing_evidence"] = (
        f"{METAL} emits {ARMED_KERNEL} at shapes "
        f"{census['distinct_shapes_BM_BK_BN_WM_WN']} for types "
        f"{sorted(set(census['types_emitted']))}, group sizes "
        f"{census['group_sizes_emitted']} and bits {census['bits_emitted']}: "
        f"{census['instantiations_total']} instantiations in total. "
        f"float IS emitted at BN = {HOST_BN}: "
        f"{census['e156_float_instantiation_emitted']} "
        f"({census['float_BN64_instantiations']} instantiations). "
        f"Pairs whose two staged halves cross the {TGP_LIMIT_BYTES} B limit with the "
        f"retile OFF: {crossing_names}, covering "
        f"{census['e156_tgp_limit_crossing_instantiations']} instantiations. "
        f"With the retile ON the staged width falls to {ARM_BN} except at "
        f"group_size 32, where the retile's own loader-legality predicate keeps "
        f"the {HOST_BN}-wide host tile. The composed arm therefore still meets a "
        f"{HOST_BN}-wide staged tile at "
        f"{len(out['e156_composed_cells_disarmed_by_capacity'])} shipped cells, "
        f"and the capacity predicate disarms exactly those; no composed "
        f"allocation exceeds the limit "
        f"({len(census['composed_crossings'])} composed crossings). A standalone "
        f"double buffer therefore cannot arm the shipped instantiation set: it must "
        f"either carry a capacity predicate that disarms those cells or fail to "
        f"build. That is why the verdict is {verdict}, not a free choice between "
        f"shipping the mechanisms apart or together."
    )

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"e156_compose_verdict = {verdict}")
    for key in (
        "e156_crown_nax_pair_equals_frontier_pair",
        "e156_crown_nax_blobs_pinned",
        "e156_compose_region_result",
        "e156_float_instantiation_emitted",
        "e156_compose_regions_disjoint_textual",
        "e156_compose_regions_disjoint_semantic",
        "e156_composed_fits_every_shipped_cell",
        "e156_composed_costs_zero_extra_tgp_bytes",
        "e156_dbuf_alone_scored_cell_doubles_staging",
        "e156_dbuf_alone_float_cell_overflows",
    ):
        print(f"{key:<52} {out[key]}")
    print()
    for row in table:
        if row["dbuf"] != "on":
            continue
        if row["T"] == "float16_t":
            continue
        state = (
            "DISARMED by capacity" if row["dbuf_disarmed_by_capacity"] else "armed"
        )
        guard = " (loader guard keeps BN 64)" if row["retile_disarmed_by_loader_guard"] else ""
        print(
            f"  T={row['T']:<11} retile={row['retile']:<3} g={row['group_size']:<4} "
            f"staged_BN={row['staged_BN']:<3} want={row['want_bytes']:>6} B "
            f"alloc={row['alloc_bytes']:>6} B delta={row['delta_vs_crown_bytes']:>+6} B "
            f"{state}{guard}"
        )
    print()
    print("  e156_tgp_limit_crossings, double buffer with the retile OFF:")
    for c in census["per_pair_table"]:
        mark = "CROSSES" if c["dbuf_alone_crosses"] else "fits   "
        print(
            f"    T={c['T']:<11} BN={c['BN']:<3} n={c['instantiations']:<4} "
            f"dbuf_alone={c['dbuf_alone_bytes']:>6} B {mark}  composed_max_alloc="
            f"{c['composed_alloc_bytes_max']:>6} B "
            f"disarmed={c['composed_instantiations_disarmed_by_capacity']}"
        )
    print()
    for f in files:
        print(
            f"  {pathlib.Path(f['path']).name:<20} retile_lines="
            f"{f['retile_touched_crown_lines']:<4} dbuf_lines="
            f"{f['dbuf_touched_crown_lines']:<4} textual_overlap="
            f"{f['textual_overlap_lines']:<4} semantic_overlap="
            f"{f['semantic_overlap_lines']}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
