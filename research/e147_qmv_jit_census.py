#!/usr/bin/env python3
"""E147 hypothesis A1: did the `quantized.h` edits perturb the decode kernels?

F8 and F9 ask whether rung A and rung B changed register allocation, spill tier
or occupancy for `qmv_fast_impl` and `qmv_fast_crossrow_affine4_g64`, which run
in every decode round. `research/e129_entry_point_census.py` cannot answer that
question: it censuses the Route B QMV entry point lifted out of `Qwen35.swift`,
and neither rung touched that file, so its diff is null by construction.

This is the instrument that can answer it. It reproduces what MLX actually
compiles.

`jit_kernels.cpp:915-932` `get_quantized_kernel` builds ONE Metal library per
kernel name, from

    utils() + gemm() + quantized_utils() + quantized() + template_def

where `template_def` instantiates exactly one `[[kernel]]` entry point
(`kernels.h:403-423`). So each entry point is compiled in its own translation
unit that happens to carry the whole `quantized` twin as uninstantiated
templates. This script rebuilds that exact library at several git revisions,
translates it with the real AGX backend for both generations, and diffs
registers, spill and the machine-code digest per entry point.

`affine_qmm_t` is included as a Rule 101 positive control: it is the kernel the
rungs edited, so it MUST move. A census where nothing moves anywhere is a
broken census, not a clean result.

Zero GPU. Offline compilation only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import agx_crossarch as agx  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
OUT = ROOT / "research/e147-qmv-jit-census.json"

# The concatenation order in `get_quantized_kernel`, mode == "affine".
TWINS = ["utils.cpp", "gemm.cpp", "quantized_utils.cpp", "quantized.cpp"]

# (label, git rev). `base` is the parent of the rung A commit.
DEFAULT_REVS = [
    ("base", "01ed58d5^"),
    ("rungA", "01ed58d5"),
    ("rungB", "302a44df"),
    ("head", "HEAD"),
]

# (host_name, template instantiation, role). Names and argument orders come
# from `backend/metal/quantized.cpp`: qmv at :256-278, qmm_t at :725-756,
# qmm_t_splitk at :832-846, gather_qmm_t at :920-941. `get_template_definition`
# streams a C++ bool without boolalpha, so a flag renders as 0 or 1.
KERNELS = [
    (
        "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
        "affine_qmv_fast<bfloat16_t, 64, 4, 0>",
        "decode: every draft round",
    ),
    (
        "affine_qmv_bfloat16_t_gs_64_b_4_batch_0",
        "affine_qmv<bfloat16_t, 64, 4, 0>",
        "decode: unaligned fallback",
    ),
    (
        "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
        "affine_qmm_t<bfloat16_t, 64, 4, 1, 0>",
        "prefill: EDITED, Rule 101 positive control",
    ),
    (
        "affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true",
        "affine_qmm_t_splitk<bfloat16_t, 64, 4, 1>",
        "prefill: split-K, not retiled",
    ),
    (
        "affine_gather_qmm_t_bfloat16_t_gs_64_b_4_alN_true",
        "affine_gather_qmm_t<bfloat16_t, 64, 4, 1>",
        "gather: not retiled",
    ),
]

ARCHES = [agx.LOCAL_ARCH, agx.RANKED_ARCH]

RAW = re.compile(r'R"([A-Za-z_]*)\(\n?(.*)\)\1"', re.DOTALL)


def twin_source(rev: str, name: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{rev}:{GEN}/{name}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m = RAW.search(blob)
    if not m:
        raise SystemExit(f"no raw string literal in {name} at {rev}")
    return m.group(2)


def library_source(rev: str, template_def: str) -> str:
    parts = [twin_source(rev, n) for n in TWINS]
    parts.append(
        f'\ntemplate [[host_name("{template_def[0]}")]] [[kernel]] '
        f"decltype({template_def[1]}) {template_def[1]};\n"
    )
    return "".join(parts)


def census_one(rev: str, host_name: str, instantiation: str, workdir: pathlib.Path):
    source = library_source(rev, (host_name, instantiation))
    lib = agx.build_metallib(source, workdir)
    out = {}
    for arch in ARCHES:
        records = agx.translate(lib, arch, workdir, select=lambda n: n == host_name)
        if host_name not in records:
            raise SystemExit(
                f"{rev} {host_name}: entry point missing from the translated archive"
            )
        out[arch] = records[host_name]
    return out, len(source)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rev",
        action="append",
        default=None,
        help="label=rev, repeatable; defaults to base/rungA/rungB/head",
    )
    ap.add_argument("--kernel", action="append", default=None,
                    help="restrict to these host names")
    args = ap.parse_args()

    revs = DEFAULT_REVS
    if args.rev:
        revs = []
        for spec in args.rev:
            label, _, rev = spec.partition("=")
            revs.append((label, rev or label))

    kernels = KERNELS
    if args.kernel:
        kernels = [k for k in KERNELS if k[0] in set(args.kernel)]

    resolved = {}
    for label, rev in revs:
        resolved[label] = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    records = {}
    source_bytes = {}
    with tempfile.TemporaryDirectory() as td:
        for label, rev in revs:
            for host_name, instantiation, _role in kernels:
                wd = pathlib.Path(td) / f"{label}_{host_name}"
                rec, nbytes = census_one(rev, host_name, instantiation, wd)
                records[(label, host_name)] = rec
                source_bytes[(label, rev, host_name)] = nbytes
                print(f"  censused {label:<6} {host_name}", flush=True)

    labels = [lbl for lbl, _ in revs]
    baseline = labels[0]
    moves = []
    rows = []
    for host_name, _inst, role in kernels:
        base_rec = records[(baseline, host_name)]
        for label in labels:
            rec = records[(label, host_name)]
            for arch in ARCHES:
                a, b = base_rec[arch], rec[arch]
                changed = (
                    a["registers"] != b["registers"]
                    or a["spill_bytes"] != b["spill_bytes"]
                    or a["text_sha8"] != b["text_sha8"]
                )
                rows.append(
                    {
                        "kernel": host_name,
                        "role": role,
                        "rev": label,
                        "arch": arch,
                        "registers": b["registers"],
                        "spill_bytes": b["spill_bytes"],
                        "text_bytes": b["text_bytes"],
                        "text_sha8": b["text_sha8"],
                        "changed_vs_base": changed,
                    }
                )
                if changed and label != baseline:
                    moves.append(
                        {
                            "kernel": host_name,
                            "role": role,
                            "rev": label,
                            "arch": arch,
                            "registers": [a["registers"], b["registers"]],
                            "spill_bytes": [a["spill_bytes"], b["spill_bytes"]],
                            "text_sha8": [a["text_sha8"], b["text_sha8"]],
                        }
                    )

    decode = [
        m
        for m in moves
        if m["kernel"].startswith(("affine_qmv_fast", "affine_qmv_b"))
    ]
    control_moved = any(
        m["kernel"].startswith("affine_qmm_t_bfloat16") for m in moves
    )

    report = {
        "experiment": "e147",
        "hypothesis": "A1: the quantized.h edits perturbed the decode kernels",
        "instrument": (
            "rebuild the exact per-kernel JIT library MLX compiles "
            "(jit_kernels.cpp:915-932) and translate it with xcrun metal-tt"
        ),
        "why_e129_cannot_answer_this": (
            "research/e129_entry_point_census.py censuses the Route B QMV "
            "entry point lifted out of Qwen35.swift, which neither rung "
            "touched, so its cross-revision diff is null by construction"
        ),
        "revisions": {lbl: resolved[lbl] for lbl in labels},
        "baseline_revision_label": baseline,
        "arches": ARCHES,
        "rows": rows,
        "moves": moves,
        "e147_rungA_decode_census_delta": (
            [m for m in decode if m["rev"] == "rungA"] or "none"
        ),
        "e147_rungAB_full_census_delta": moves or "none",
        "decode_entry_points_moved": bool(decode),
        "rule_101_positive_control_moved": control_moved,
        "verdict": (
            "A1 REFUTED"
            if (not decode and control_moved)
            else ("A1 CONFIRMED" if decode else "CENSUS INVALID: control did not move")
        ),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print()
    hdr = (
        f"{'kernel':<52} {'rev':<6} {'arch':<16} {'regs':>5} "
        f"{'spill':>6} {'text B':>7} {'sha8':>9} {'moved':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['kernel']:<52} {r['rev']:<6} {r['arch']:<16} "
            f"{str(r['registers']):>5} {str(r['spill_bytes']):>6} "
            f"{r['text_bytes']:>7} {r['text_sha8']:>9} "
            f"{str(r['changed_vs_base']):>6}"
        )

    print()
    print(f"decode entry points moved      : {report['decode_entry_points_moved']}")
    print(f"Rule 101 control moved         : {report['rule_101_positive_control_moved']}")
    print(f"verdict                        : {report['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if report["verdict"] != "CENSUS INVALID: control did not move" else 1


if __name__ == "__main__":
    sys.exit(main())
