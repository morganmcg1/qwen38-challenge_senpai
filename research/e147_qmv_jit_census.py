#!/usr/bin/env python3
"""JIT entry-point census: what did an edit do to the compiled Metal kernels?

This is a campaign tool. The `e147` prefix is historical; the file keeps its
original path so existing references stay valid, and the advisor should rename
it when wiring it into the mandated pre-submit chain.

WHY THIS TOOL EXISTS (HARNESS DEFECT 45). The chain previously used
`research/e129_entry_point_census.py`, which censuses the Route B QMV entry
point lifted out of `Qwen35.swift`. An edit to a Metal kernel header does not
touch `Qwen35.swift`, so that census returns a null diff by construction: it
reports "no change" for every kernel-header experiment and cannot fail. A gate
that cannot fail is not a gate.

WHAT THIS TOOL MEASURES. MLX does not ship these kernels in `mlx.metallib`. It
JIT-compiles one Metal library per entry point at runtime
(`jit_kernels.cpp:915-932`, `get_quantized_kernel`), from

    utils() + gemm() + quantized_utils() + quantized() + template_def

where `template_def` instantiates exactly one `[[kernel]]` entry point
(`kernels.h:403-423`). Each entry point is therefore its own translation unit
that happens to carry the whole twin as uninstantiated templates. This tool
rebuilds that exact library at each requested git revision, translates it with
the real AGX backend for both GPU generations, and reports registers, spill
bytes, machine-text bytes and a machine-text digest per entry point per arch.

Zero GPU. Offline compilation only.

--------------------------------------------------------------------------
HOW TO NAME A KERNEL SET
--------------------------------------------------------------------------
`--kernel-set NAME` selects one entry in `KERNEL_SETS`, repeatable. Each set
names the JIT builder it models, the twin concatenation order that builder
uses, and one row per entry point: the `host_name` MLX assigns, the C++
instantiation, and the role that entry point plays in the scored run. Run
`--list` to print the sets, their entry points and their roles.

`--kernel HOST_NAME` narrows the selection to named entry points, repeatable.
An unknown name is an error, not a silent empty census.

To add a set, read the builder in `jit_kernels.cpp`, copy its twin order, and
take each `host_name` and instantiation from the matching launcher in
`backend/metal/quantized.cpp`. `get_template_definition` streams a C++ bool
without `boolalpha`, so a flag renders as `0` or `1` in the host name.

--------------------------------------------------------------------------
HOW TO NAME REVISIONS
--------------------------------------------------------------------------
`--rev LABEL=REV` is repeatable and ordered. The FIRST `--rev` is the baseline
that every later revision is compared against. `REV` is anything `git rev-parse`
accepts, so `--rev base=$BASE_SHA --rev candidate=HEAD` is the normal
pre-submit form. `--rev LABEL` on its own uses the label as the revision.

Revisions are read with `git show`, never from the working tree, so a dirty
checkout cannot contaminate a census. Commit before censusing a candidate.

--------------------------------------------------------------------------
WHAT THE POSITIVE CONTROL IS
--------------------------------------------------------------------------
The census answers "did anything move?". A census whose instrument is blind
answers "no" for the same reason a correct result answers "no", so the two are
indistinguishable without a control (Rule 101).

The control is synthetic and mandatory. For every selected entry point the tool
compiles the BASELINE source a second time with the quantization group size
changed from 64 to 32 in the instantiation, and requires that perturbed build to
differ from the unperturbed baseline in registers, spill or machine-text digest,
on every arch. The perturbation never ships; it exists only to prove the
instrument can see a change in that specific entry point on that specific arch.

This control does not depend on what the candidate edited. The earlier E147 form
of this check required `affine_qmm_t` to move because the E147 rungs edited it,
which silently turns into a false alarm for any experiment that edits something
else. If the control does not fire, the census is INVALID and the tool exits
non-zero, whatever the candidate did.

--------------------------------------------------------------------------
HOW A CALLER READS THE VERDICT
--------------------------------------------------------------------------
Exit code 0 means the census is valid and complete: every selected entry point
compiled, translated on both arches, and its control fired. Exit code 1 means
the census is INVALID and its rows must not be quoted. Movement is never an
error by itself; movement is the measurement.

The JSON report carries:

    census_valid            bool, mirrors the exit code
    invalid_reasons         list, empty when valid
    revisions               label -> resolved commit
    baseline_revision_label the label every comparison uses
    rows                    one record per (entry point, revision, arch)
    moves                   the subset of rows that differ from the baseline
    moved_kernels           sorted unique entry points that moved anywhere
    unchanged_kernels       sorted unique entry points that moved nowhere
    control                 per (entry point, arch): fired or not

A caller that wants "the candidate perturbed no decode kernel" reads
`moved_kernels` and checks that no decode-role entry point is in it. Read the
role strings from `--list`, not from the kernel names.

--------------------------------------------------------------------------
WHICH ENTRY POINTS THIS COVERS, AND WHICH IT DOES NOT
--------------------------------------------------------------------------
COVERS
  * Entry points JIT-compiled through a builder modelled in `KERNEL_SETS`.
    Today that is `get_quantized_kernel`, mode "affine": the affine QMV decode
    kernels, the affine `qmm_t` prefill kernel, its split-K sibling and the
    gather variant.
  * Both GPU generations: `applegpu_g16s` (local) and `applegpu_g17s` (the
    ranked M5), through the real AGX backend, not a simulator.
  * Registers, spill bytes, machine-text bytes and machine-text digest.

DOES NOT COVER
  * The NAX quantized family, `get_qmm_nax_kernel`. The `quantized-nax-affine`
    set exists and compiles, but this toolchain's offline translator refuses
    every real NAX GEMM shape on BOTH arches with a byte-identical diagnostic
    (rung E-2), so the census reports it untranslatable instead of clean. Never
    read a NAX silence as a NAX result.
  * Anything loaded from `mlx.metallib` rather than JIT-compiled from a twin.
    Use `tools/build-mlx-metallib.sh` and inspect that library instead.
  * Swift code. A Swift symbol never reaches the Metal string table. Use
    `senpai/rebuild-and-assert-worker.sh --require-symbol`.
  * The host launcher `backend/metal/quantized.cpp`. It picks the grid and the
    entry point; it is not editable and it is not a Metal kernel. A census
    cannot see a dispatch change.
  * Occupancy, scheduling, thermal behaviour and runtime. Register count is an
    input to occupancy, not occupancy, and never a substitute for a timed run.
  * Numerical exactness. A moved digest is not a fidelity failure and an
    unmoved digest is not a fidelity proof. Use the exact-token gate for that.
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

ARCHES = [agx.LOCAL_ARCH, agx.RANKED_ARCH]
RAW = re.compile(r'R"([A-Za-z_]*)\(\n?(.*)\)\1"', re.DOTALL)

# The synthetic Rule 101 perturbation. Every instantiation below passes the
# quantization group size as its second template argument, so this one
# substitution reaches every entry point in every set.
CONTROL = ("<bfloat16_t, 64, 4", "<bfloat16_t, 32, 4")

# name -> {builder, twins, kernels: [(host_name, instantiation, role)]}.
# Host names and argument orders come from `backend/metal/quantized.cpp`:
# qmv at :256-278, qmm_t at :725-756, qmm_t_splitk at :832-846,
# gather_qmm_t at :920-941, qmm_nax at :526-541.
KERNEL_SETS = {
    "quantized-affine": {
        "builder": "jit_kernels.cpp get_quantized_kernel, mode affine",
        "twins": ["utils.cpp", "gemm.cpp", "quantized_utils.cpp",
                  "quantized.cpp"],
        "translatable": True,
        "kernels": [
            ("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
             "affine_qmv_fast<bfloat16_t, 64, 4, 0>",
             "decode: every draft round"),
            ("affine_qmv_bfloat16_t_gs_64_b_4_batch_0",
             "affine_qmv<bfloat16_t, 64, 4, 0>",
             "decode: unaligned fallback"),
            ("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
             "affine_qmm_t<bfloat16_t, 64, 4, 1, 0>",
             "prefill: the non-NAX GEMM, live only where NAX is absent"),
            ("affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true",
             "affine_qmm_t_splitk<bfloat16_t, 64, 4, 1>",
             "prefill: split-K, never selected at the scored shapes"),
            ("affine_gather_qmm_t_bfloat16_t_gs_64_b_4_alN_true",
             "affine_gather_qmm_t<bfloat16_t, 64, 4, 1>",
             "gather: not on the scored prefill path"),
        ],
    },
    "quantized-nax-affine": {
        "builder": "jit_kernels.cpp get_qmm_nax_kernel, mode affine",
        "twins": ["utils.cpp", "gemm_nax.cpp", "quantized_utils.cpp",
                  "quantized_nax.cpp"],
        # Compiles, but `metal-tt` refuses every real NAX GEMM shape on both
        # arches (rung E-2). Selecting this set makes the census INVALID, which
        # is the honest outcome: no record exists to compare.
        "translatable": False,
        "kernels": [
            ("affine_qmm_t_nax_bfloat16_t_gs_64_b_4_alN_true_batch_0",
             "affine_qmm_t_nax<bfloat16_t, 64, 4, 1, 0, 64, 64, 64, 2, 2>",
             "prefill: the scored GEMM on the ranked M5"),
        ],
    },
}


def show(rev: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


def twin_source(rev: str, name: str) -> str:
    m = RAW.search(show(rev, f"{GEN}/{name}"))
    if not m:
        raise SystemExit(f"no raw string literal in {name} at {rev}")
    return m.group(2)


def library_source(rev: str, twins: list[str], host_name: str,
                   instantiation: str) -> str:
    parts = [twin_source(rev, n) for n in twins]
    parts.append(f'\ntemplate [[host_name("{host_name}")]] [[kernel]] '
                 f"decltype({instantiation}) {instantiation};\n")
    return "".join(parts)


def census_one(rev: str, twins: list[str], host_name: str, instantiation: str,
               workdir: pathlib.Path) -> tuple[dict, int]:
    source = library_source(rev, twins, host_name, instantiation)
    workdir.mkdir(parents=True, exist_ok=True)
    lib = agx.build_metallib(source, workdir)
    out = {}
    for arch in ARCHES:
        try:
            records = agx.translate(lib, arch, workdir,
                                    select=lambda n: n == host_name)
        except SystemExit as exc:
            out[arch] = {"translated": False,
                         "error_body": str(exc).split("\n", 1)[-1].strip()}
            continue
        if host_name not in records:
            out[arch] = {"translated": False,
                         "error_body": "entry point missing from the archive"}
            continue
        out[arch] = dict(records[host_name], translated=True)
    return out, len(source)


def differs(a: dict, b: dict) -> bool:
    if not (a.get("translated") and b.get("translated")):
        return False
    return (a["registers"] != b["registers"]
            or a["spill_bytes"] != b["spill_bytes"]
            or a["text_sha8"] != b["text_sha8"])


def select_kernels(set_names: list[str], only: list[str] | None):
    chosen = []
    for name in set_names:
        if name not in KERNEL_SETS:
            raise SystemExit(f"unknown kernel set {name!r}; try --list")
        spec = KERNEL_SETS[name]
        for host, inst, role in spec["kernels"]:
            chosen.append((name, spec["twins"], host, inst, role))
    if only:
        known = {c[2] for c in chosen}
        unknown = sorted(set(only) - known)
        if unknown:
            raise SystemExit(f"unknown entry point(s): {', '.join(unknown)}")
        chosen = [c for c in chosen if c[2] in set(only)]
    if not chosen:
        raise SystemExit("no entry points selected")
    return chosen


def do_list() -> int:
    for name, spec in KERNEL_SETS.items():
        print(f"{name}  ({spec['builder']})")
        print(f"  twins: {' + '.join(spec['twins'])}")
        print(f"  translatable: {spec['translatable']}")
        for host, inst, role in spec["kernels"]:
            print(f"  - {host}\n      {inst}\n      role: {role}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="census JIT-compiled Metal entry points across revisions")
    ap.add_argument("--kernel-set", action="append", default=None,
                    help="named entry-point set; repeatable; see --list")
    ap.add_argument("--kernel", action="append", default=None,
                    help="restrict to these host names; repeatable")
    ap.add_argument("--rev", action="append", default=None,
                    help="LABEL=REV; repeatable; the first is the baseline")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--list", action="store_true",
                    help="print the kernel sets and exit")
    args = ap.parse_args()

    if args.list:
        return do_list()

    if not args.rev or len(args.rev) < 2:
        raise SystemExit(
            "give at least two revisions, baseline first, for example: "
            "--rev base=$BASE_SHA --rev candidate=HEAD")
    revs = []
    for spec in args.rev:
        label, _, rev = spec.partition("=")
        revs.append((label, rev or label))
    if len({lbl for lbl, _ in revs}) != len(revs):
        raise SystemExit("revision labels must be unique")

    chosen = select_kernels(args.kernel_set or ["quantized-affine"],
                            args.kernel)
    resolved = {lbl: subprocess.run(["git", "rev-parse", rev], cwd=ROOT,
                                    capture_output=True, text=True,
                                    check=True).stdout.strip()
                for lbl, rev in revs}
    labels = [lbl for lbl, _ in revs]
    baseline = labels[0]

    records, control = {}, {}
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for label, rev in revs:
            for _set, twins, host, inst, _role in chosen:
                rec, _n = census_one(rev, twins, host, inst,
                                     root / f"{label}_{host}")
                records[(label, host)] = rec
                print(f"  censused {label:<10} {host}", flush=True)
        for _set, twins, host, inst, _role in chosen:
            perturbed = inst.replace(*CONTROL)
            if perturbed == inst:
                raise SystemExit(
                    f"the control perturbation does not apply to {inst!r}")
            rec, _n = census_one(revs[0][1], twins, host, perturbed,
                                 root / f"control_{host}")
            control[host] = rec
            print(f"  control    {host}", flush=True)

    rows, moves, invalid = [], [], []
    for _set, _twins, host, _inst, role in chosen:
        base_rec = records[(baseline, host)]
        for arch in ARCHES:
            if not base_rec[arch].get("translated"):
                invalid.append(
                    f"{host} did not translate on {arch}: "
                    f"{base_rec[arch].get('error_body', '')[:160]}")
                continue
            if not differs(base_rec[arch], control[host][arch]):
                invalid.append(
                    f"the control did not fire for {host} on {arch}")
        for label in labels:
            rec = records[(label, host)]
            for arch in ARCHES:
                if not rec[arch].get("translated"):
                    rows.append({"kernel": host, "role": role, "rev": label,
                                 "arch": arch, "translated": False,
                                 "error_body": rec[arch].get("error_body")})
                    continue
                changed = differs(base_rec[arch], rec[arch])
                rows.append({"kernel": host, "role": role, "rev": label,
                             "arch": arch, "translated": True,
                             "registers": rec[arch]["registers"],
                             "spill_bytes": rec[arch]["spill_bytes"],
                             "text_bytes": rec[arch]["text_bytes"],
                             "text_sha8": rec[arch]["text_sha8"],
                             "changed_vs_base": changed})
                if changed and label != baseline:
                    moves.append({
                        "kernel": host, "role": role, "rev": label,
                        "arch": arch,
                        "registers": [base_rec[arch]["registers"],
                                      rec[arch]["registers"]],
                        "spill_bytes": [base_rec[arch]["spill_bytes"],
                                        rec[arch]["spill_bytes"]],
                        "text_sha8": [base_rec[arch]["text_sha8"],
                                      rec[arch]["text_sha8"]]})

    moved_kernels = sorted({m["kernel"] for m in moves})
    report = {
        "tool": "jit entry-point census",
        "instrument": ("rebuild the exact per-kernel JIT library MLX compiles "
                       "and translate it with xcrun metal-tt"),
        "kernel_sets": args.kernel_set or ["quantized-affine"],
        "arches": ARCHES,
        "revisions": resolved,
        "baseline_revision_label": baseline,
        "control": {"perturbation": f"{CONTROL[0]} -> {CONTROL[1]}",
                    "fired": {h: {a: differs(records[(baseline, h)][a],
                                             control[h][a])
                                  for a in ARCHES}
                              for _s, _t, h, _i, _r in chosen}},
        "rows": rows,
        "moves": moves,
        "moved_kernels": moved_kernels,
        "unchanged_kernels": sorted({c[2] for c in chosen}
                                    - set(moved_kernels)),
        "census_valid": not invalid,
        "invalid_reasons": invalid,
    }
    # Compatibility fields for the E147 result artifact. They are only
    # meaningful when a revision carries the E147 `rungA` label.
    decode = [m for m in moves if m["role"].startswith("decode")]
    report["e147_rungA_decode_census_delta"] = (
        [m for m in decode if m["rev"] == "rungA"] or "none")
    report["e147_rungAB_full_census_delta"] = moves or "none"
    report["decode_entry_points_moved"] = bool(decode)
    report["rule_101_positive_control_moved"] = not invalid

    pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    hdr = (f"{'kernel':<52} {'rev':<10} {'arch':<16} {'regs':>5} "
           f"{'spill':>6} {'text B':>7} {'sha8':>9} {'moved':>6}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r["translated"]:
            print(f"{r['kernel']:<52} {r['rev']:<10} {r['arch']:<16} "
                  f"{'NOT TRANSLATED':>36}")
            continue
        print(f"{r['kernel']:<52} {r['rev']:<10} {r['arch']:<16} "
              f"{r['registers']:>5} {r['spill_bytes']:>6} "
              f"{r['text_bytes']:>7} {r['text_sha8']:>9} "
              f"{str(r['changed_vs_base']):>6}")
    print()
    print(f"moved_kernels={moved_kernels or 'none'}")
    print(f"census_valid={report['census_valid']}")
    for reason in invalid:
        print(f"  INVALID: {reason}", file=sys.stderr)
    return 0 if report["census_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
