#!/usr/bin/env python3
"""Read `MLXFast.metalKernel` Metal sources out of Swift, at any revision.

`MLXFast.metalKernel` hands MLX a body string and MLX generates the
`[[kernel]]` signature around it at dispatch time from the dtypes and rank of
the arrays actually passed, so nothing writes the compiled source to disk.
`research/e120_g17s_census.py` reproduces the generation rule; this module adds
the two things a standing gate needs on top of it:

  * the Swift literals are read at a named git revision, so a base and a
    candidate are both available to one process;
  * the interpolated E120 QMV body is rebuilt FROM the Swift text rather than
    from a copy of it, so a width added to the dispatch list, or a line added
    to the body, is picked up instead of silently censusing last month's
    kernel. Any interpolation this module does not understand raises rather
    than compiling a wrong kernel.

Nothing here writes to the tracked tree.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
QWEN35 = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

# `metal_kernel.cpp:19`, max_constant_array_size == 8 elements: every array in
# these kernels is far larger, so every one binds as `device`.
ROUTE_B_ROLES = {
    "qwen35_custom_affine4_g64_qmv_wide_v1":
        "Route B replica wide QMV, no chunk-sum table",
    "qwen35_custom_affine4_g64_qmv_wide_sums_v1":
        "Route B shipped wide QMV, USE_TABLE=true (the .sumtable arm)",
    "qwen35_custom_affine4_g64_xsums_v1":
        "Route B activation chunk-sum table fill",
}

# Binding order and element type at the shipped call sites
# (`Qwen35.swift` Qwen35CustomQMV.matmul, matmulWithTable and xsumsTable).
QMV_INPUTS = [("w", "uint32_t"), ("scales", "bfloat16_t"),
              ("biases", "bfloat16_t"), ("x", "bfloat16_t")]
QMV_OUTPUTS = [("y", "bfloat16_t")]
XSUMS_INPUTS = [("x", "bfloat16_t")]
XSUMS_OUTPUTS = [("xsums", "float")]


class SourceUnavailable(RuntimeError):
    """The Swift text does not carry a literal this module knows how to read."""


def swift_text(relative: str, rev: str | None) -> str:
    if rev is None:
        return (ROOT / relative).read_text()
    return subprocess.run(["git", "show", "%s:%s" % (rev, relative)],
                          cwd=str(ROOT), capture_output=True, text=True,
                          check=True).stdout


def multiline_literal(text: str, at: int) -> tuple[str, int]:
    """The Swift multi-line string literal that opens at `at`, dedented.

    Swift strips the indentation of the closing delimiter from every line, so
    the closing delimiter's column is read rather than assumed.
    """
    start = text.index('"""', at) + 3
    start = text.index("\n", start) + 1
    end = text.index('"""', start)
    close_line = text.rindex("\n", start, end) + 1
    indent = text[close_line:end]
    if indent.strip():
        raise SourceUnavailable("closing delimiter is not on its own line")
    body = text[start:close_line]
    lines = []
    for line in body.split("\n"):
        lines.append(line[len(indent):] if line.startswith(indent) else line)
    return "\n".join(lines).rstrip("\n"), end + 3


def named_literal(text: str, name: str) -> str:
    marker = "let %s = \"\"\"" % name
    at = text.find(marker)
    if at < 0:
        raise SourceUnavailable("no multi-line literal named %s" % name)
    return multiline_literal(text, at)[0]


def ternary(text: str, name: str) -> tuple[str, str]:
    """`let NAME = cond ? "a" : "b"` as (true value, false value)."""
    match = re.search(r'let %s = \w+ \? (".*?") : (".*?")\n' % name, text)
    if not match:
        raise SourceUnavailable("no string ternary named %s" % name)
    return (match.group(1)[1:-1].encode().decode("unicode_escape"),
            match.group(2)[1:-1].encode().decode("unicode_escape"))


def e120_qmv_body(text: str, table: bool) -> str:
    """Rebuild `qwen35E120QMVSource(table:)` from the Swift text.

    Only the interpolation semantics are known here. The dispatch width list,
    the per-case template and the body itself are all read out of Swift, so a
    change to any of them reaches the census.
    """
    at = text.find("func qwen35E120QMVSource(table: Bool) -> String {")
    if at < 0:
        raise SourceUnavailable("qwen35E120QMVSource is gone")
    end = text.index("\n}\n", at)
    func = text[at:end]

    sums = ternary(func, "sums")[0 if table else 1]
    flag = ternary(func, "flag")[0 if table else 1]
    null_decl = ternary(func, "nullDecl")[0 if table else 1]

    widths = re.search(r"let cases = \[(.*?)\]\n", func, re.DOTALL)
    if not widths:
        raise SourceUnavailable("no dispatch width list in qwen35E120QMVSource")
    pairs = re.findall(r"\((\d+),\s*(\d+)\)", widths.group(1))
    if not pairs:
        raise SourceUnavailable("empty dispatch width list")

    case_at = func.index(".map {")
    case_template, _ = multiline_literal(func, case_at)
    return_at = func.index("return \"\"\"")
    body_template, _ = multiline_literal(func, return_at)

    cases = "\n".join(
        case_template.replace("\\(m)", m).replace("\\(ipg)", ipg)
        .replace("\\(flag)", flag).replace("\\(sums)", sums)
        for m, ipg in pairs)
    body = (body_template.replace("\\(nullDecl)", null_decl)
            .replace("\\(cases)", cases))
    if "\\(" in body:
        raise SourceUnavailable(
            "unresolved Swift interpolation in the QMV body: %s"
            % body[body.index("\\("):][:48])
    return body


def route_b_library(text: str) -> str:
    """One Metal source holding all three shipped Route B kernels."""
    import e120_g17s_census as e120

    header = named_literal(text, "qwen35E120QMVHeader")
    xsums_at = text.find("name: \"qwen35_custom_affine4_g64_xsums_v1\"")
    if xsums_at < 0:
        raise SourceUnavailable("the xsums kernel is gone")
    xsums_body = multiline_literal(text, text.index("source:", xsums_at))[0]

    parts = [e120.PRELUDE, header, ""]
    parts.append(e120.generate(
        "qwen35_custom_affine4_g64_qmv_wide_v1", QMV_INPUTS, QMV_OUTPUTS,
        e120_qmv_body(text, table=False)))
    parts.append(e120.generate(
        "qwen35_custom_affine4_g64_qmv_wide_sums_v1",
        QMV_INPUTS + [("xsums", "float")], QMV_OUTPUTS,
        e120_qmv_body(text, table=True),
        template=[("bool", "USE_TABLE", "true")]))
    parts.append(e120.generate(
        "qwen35_custom_affine4_g64_xsums_v1", XSUMS_INPUTS, XSUMS_OUTPUTS,
        xsums_body))
    return "\n".join(parts) + "\n"
