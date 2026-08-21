#!/usr/bin/env python3
"""E108: does NON-EXECUTED text inside `affine_qmv_fast` cost time?

E104 finding 36 showed that compiled ISA text bytes track measured time on arms
that change the EXECUTED body, while AIR operation counts do not. E102 built two
arms that change only the NON-executed text of the same entry point and never
timed them. This module rebuilds those arms, adds the control and extent arms
E108 needs, and censuses each one on both GPU generations.

Arms:

  a_base         E102 `A_shipped`: the shipped entry point, unmodified.
  h_prunenarrow  E102 `H_prune_narrow`: the `1024 <= out_vec_size < 4096`
                 switch loses cases 3..9. Dead: the smallest scored quantised
                 projection is n = 5120.
  i_pruneall     E102 `I_prune_all_dead`: `h_prunenarrow` plus the wide
                 `case 9`. Dead: M = min(8, 7) + 1 = 8.
  p_misprune5    the mandatory positive control. `i_pruneall` with the LIVE
                 wide `case 5` removed as well, so M = 5 falls through to the
                 generic `qmv_fast_impl`. If the bit-for-bit comparison cannot
                 see this, it cannot see a real mis-prune either.

Extent arms (`--set extent`) locate the live case inside the compiled text.
`i_minus_case{m}` removes exactly one live wide case from `i_pruneall`, so the
byte range where the two compiled texts differ is that case's own extent, and
the common-prefix length is its distance from the entry point.

    research/e108_arms.py --emit /tmp/e108-arms --set prune
    research/e108_arms.py --census /tmp/e108-arms --out research/out/e108/x.json

Research-only: nothing here is on the scored path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jit_string_compile import PREAMBLES, host_name, preamble  # noqa: E402
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, find_mach_headers, section,
    translate,
)
from e102_register_reconcile import (  # noqa: E402
    ARMS as E102_ARMS, NARROW_CALL, WIDE_CALL, arm_source, drop_case,
)

ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
ENTRY = host_name(ENTRY_CELL)

# The wide switch as the current tree ships it. E102's own table predates the
# E100 collapse and still says M = 5 uses IPG 3, so E108 carries its own copy;
# every use goes through `drop_case`, which fails loudly if an anchor moved.
WIDE_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
LIVE_WIDTHS = (2, 3, 4, 5, 6, 7, 8)

# The `out_vec_size >= 4096` arm of the dispatcher. Case 2 calls the same pair
# kernel in both arms of the `>= 4096` test, so a live case is located inside
# this span rather than by a text search over the whole string.
WIDE_SWITCH_OPEN = "    if (out_vec_size >= 4096) {\n"
WIDE_SWITCH_CLOSE = "    } else {\n"


def wide_case_call(m: int) -> str:
    if m == 2:
        return NARROW_CALL.format(m=2, ipg=0)
    return WIDE_CALL.format(m=m, ipg=WIDE_IPG[m])


def e102_arm(name: str) -> str:
    """Rebuild one E102 arm through E102's own code path."""
    bound, table, drops = E102_ARMS[name]
    return arm_source(bound, table, drops)


def minus_live_case(m: int) -> str:
    """`i_pruneall` with one LIVE wide case removed, located inside the span."""
    text = e102_arm("I_prune_all_dead")
    open_at = text.index(WIDE_SWITCH_OPEN)
    close_at = text.index(WIDE_SWITCH_CLOSE, open_at)
    span = drop_case(text[open_at:close_at], wide_case_call(m),
                     f"live_M{m}", False)
    return text[:open_at] + span + text[close_at:]


SETS = {
    "prune": ("a_base", "h_prunenarrow", "i_pruneall"),
    "exact": ("a_base", "h_prunenarrow", "i_pruneall", "p_misprune5"),
    "extent": ("i_pruneall",) + tuple(f"i_minus_case{m}" for m in LIVE_WIDTHS),
}

NAMES = {
    "a_base": lambda: e102_arm("A_shipped"),
    "h_prunenarrow": lambda: e102_arm("H_prune_narrow"),
    "i_pruneall": lambda: e102_arm("I_prune_all_dead"),
    "p_misprune5": lambda: minus_live_case(5),
    **{f"i_minus_case{m}": (lambda m=m: minus_live_case(m))
       for m in LIVE_WIDTHS},
}


def emit(outdir: pathlib.Path, names: tuple[str, ...]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    for name in names:
        text = NAMES[name]()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            raise SystemExit(
                f"e108: arms {seen[digest]} and {name} are byte-identical")
        seen[digest] = name
        (outdir / f"arm_{name}.metal").write_text(text)
        print(f"{name:<16}{len(text):>9} bytes  sha {digest[:8]}  "
              f"arm_{name}.metal")


def text_blob(metallib: pathlib.Path, arch: str,
              workdir: pathlib.Path) -> bytes:
    """The raw `__TEXT,__text` of the single translated kernel."""
    script = workdir / f"blob_{arch}.mtlp-json"
    script.write_text(json.dumps({"pipelines": {"compute_pipelines": [
        {"compute_function": ENTRY}]}}))
    out = workdir / f"blob_{arch}.mtlp"
    done = subprocess.run(
        ["xcrun", "metal-tt", "-arch", arch, str(metallib), str(script),
         "-o", str(out)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"metal-tt failed for {arch}:\n{done.stderr}")
    compute = section(out.read_bytes(), "__TEXT,__compute")
    if compute is None:
        raise SystemExit("translated archive has no __TEXT,__compute section")
    for start in find_mach_headers(compute):
        inner = compute[start:]
        if section(inner, "__GPU_METADATA,__compute") is None:
            continue
        return section(inner, "__TEXT,__text") or b""
    raise SystemExit("no GPU object in the translated archive")


def edit_range(ref: bytes, arm: bytes) -> dict:
    """Where two compiled texts differ, as one contiguous byte range."""
    limit = min(len(ref), len(arm))
    prefix = 0
    while prefix < limit and ref[prefix] == arm[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < limit - prefix
           and ref[len(ref) - 1 - suffix] == arm[len(arm) - 1 - suffix]):
        suffix += 1
    return {
        "common_prefix_bytes": prefix,
        "common_suffix_bytes": suffix,
        "ref_changed_from": prefix,
        "ref_changed_to": len(ref) - suffix,
        "ref_changed_bytes": len(ref) - suffix - prefix,
        "arm_changed_bytes": len(arm) - suffix - prefix,
    }


def census(srcdir: pathlib.Path, names: tuple[str, ...], arches: list[str],
           reference: str | None, out_path: pathlib.Path) -> None:
    result = {"entry_cell": ENTRY_CELL, "architectures": arches,
              "reference_arm": reference, "arms": {}}
    blobs: dict[str, dict[str, bytes]] = {}
    with tempfile.TemporaryDirectory(prefix="e108-") as tmp:
        root = pathlib.Path(tmp)
        for name in names:
            source = (srcdir / f"arm_{name}.metal").read_text()
            work = root / name
            work.mkdir()
            lib = build_metallib(source, work)
            record = {"source_bytes": len(source),
                      "source_sha8": hashlib.sha256(
                          source.encode()).hexdigest()[:8]}
            blobs[name] = {}
            for arch in arches:
                record[arch] = translate(lib, arch, work)[ENTRY]
                if not reference:
                    continue
                blobs[name][arch] = text_blob(lib, arch, work)
                if len(blobs[name][arch]) != record[arch]["text_bytes"]:
                    raise SystemExit(f"{name} {arch}: text blob length "
                                     "disagrees with the census record")
            result["arms"][name] = record
            print(f"[arm] {name} " + "  ".join(
                f"{arch.replace('applegpu_', '')}="
                f"{record[arch]['registers']}reg/"
                f"{record[arch]['spill_bytes']}spill/"
                f"{record[arch]['text_bytes']}B/{record[arch]['text_sha8']}"
                for arch in arches), flush=True)

        if reference:
            for name in names:
                if name == reference:
                    continue
                result["arms"][name]["vs_reference"] = {
                    arch: edit_range(blobs[reference][arch], blobs[name][arch])
                    for arch in arches}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="prune", choices=sorted(SETS))
    ap.add_argument("--emit", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--arch", nargs="+", default=[LOCAL_ARCH, RANKED_ARCH])
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()

    names = SETS[args.which]
    if args.emit:
        emit(args.emit, names)
    if args.census:
        if not args.out:
            raise SystemExit("--census needs --out")
        census(args.census, names, args.arch, args.reference, args.out)
    if not args.emit and not args.census:
        raise SystemExit("nothing to do: pass --emit and/or --census")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
