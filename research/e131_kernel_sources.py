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
#
# E129 replaced the two fixed Route B QMV entry points with one entry point per
# distinct `ipg`, so the shipped name set is no longer a constant. It is read
# out of `qwen35E120QMVName` and the dispatch plan instead, by `route_b_route`.
XSUMS_ROLE = "Route B activation chunk-sum table fill"

# The two custom affine-2 cluster kernels the frontier import added. They share
# `qwen35ClusterAffine2QMVHeader` and both run on every draft step, so the gate
# has to see them.
CLUSTER_QMV_ROLES = {
    "qwen_mtp_cluster_centroid_qmv_a2g64_v1":
        "dense centroid 2-bit QMV over 12,292 leaves, every draft step",
    "qwen_mtp_cluster_row_qmv_a2g64_v1":
        "gathered leaf 2-bit QMV, one threadgroup per probed cluster",
}

# Binding order and element type at the shipped call sites
# (`Qwen35.swift` Qwen35CustomQMV.matmul, matmulWithTable and xsumsTable).
QMV_INPUTS = [("w", "uint32_t"), ("scales", "bfloat16_t"),
              ("biases", "bfloat16_t"), ("x", "bfloat16_t")]
QMV_OUTPUTS = [("y", "bfloat16_t")]
XSUMS_INPUTS = [("x", "bfloat16_t")]
XSUMS_OUTPUTS = [("xsums", "float")]

# `qwen35ClusterCentroidQMV` and `qwen35ClusterRowQMV` call sites. `probed` is
# guarded to `.uint32` at the call site, so the generated signature is fixed.
CLUSTER_CENTROID_INPUTS = [("x", "bfloat16_t"), ("w", "uint32_t"),
                           ("scales", "bfloat16_t"), ("biases", "bfloat16_t")]
CLUSTER_ROW_INPUTS = CLUSTER_CENTROID_INPUTS + [("probed", "uint32_t")]
CLUSTER_OUTPUTS = [("y", "bfloat16_t")]


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


def qmv_source_func(text: str) -> str:
    """The text of `qwen35E120QMVSource`, whose signature this module pins.

    An unknown parameter list raises. E129 added `tier:` and the reader that
    matched only `(table:)` reported the generator as gone, which made the
    census blind to every Route B QMV entry point.
    """
    match = re.search(
        r"func qwen35E120QMVSource\(([^)]*)\) -> String \{", text)
    if not match:
        raise SourceUnavailable("qwen35E120QMVSource is gone")
    params = [p.strip() for p in match.group(1).split(",")]
    if params != ["table: Bool", "tier: Int?"]:
        raise SourceUnavailable(
            "qwen35E120QMVSource has an unknown signature (%s)"
            % match.group(1))
    return text[match.start():text.index("\n}\n", match.start())]


def _env_default(text: str, decl: str, what: str) -> str:
    """The case a `ProcessInfo`-backed selector falls back to when unset.

    The ranked runner sets no `MLX_E120_*` environment, so the guard's `else`
    branch is the route it takes.
    """
    at = text.find(decl)
    if at < 0:
        raise SourceUnavailable("no %s selector" % what)
    block = text[at:text.index("\n    }()", at)]
    match = re.search(r"else \{ return \.(\w+) \}", block)
    if not match:
        raise SourceUnavailable("no unset-environment default for %s" % what)
    return match.group(1)


def _compiled_default(text: str, kind: str) -> str:
    match = re.search(
        r"public static let compiledDefault = %s\.(\w+)" % kind, text)
    if not match:
        raise SourceUnavailable("no %s.compiledDefault" % kind)
    return match.group(1)


def default_entry(text: str) -> str:
    """`shared` or `tiered`: the entry-point layout an unset environment takes."""
    entry = _env_default(text, "public static let entry: Entry = {", "entry")
    return _compiled_default(text, "Entry") if entry == "compiledDefault" \
        else entry


def default_table(text: str) -> str:
    """The `Table` case an unset environment takes."""
    table = _env_default(text, "public static let table: Table = {", "table")
    return _compiled_default(text, "Table") if table == "compiledDefault" \
        else table


def default_arm(text: str) -> str:
    return _env_default(text, "public static let arm: Arm = {", "arm")


def width_plan(text: str, case_name: str) -> list[tuple[int, int, int]]:
    """`(m, ipg, rps)` for every routed width of one `Table` case.

    The plan is checked against that case's own `witness` literal, which is
    the string the built worker carries, so a plan and its witness cannot
    drift apart inside this reader.
    """
    at = text.find("public var plan: [(m: Int, ipg: Int, rps: Int)] {")
    if at < 0:
        raise SourceUnavailable("no QMV width-plan table")
    region = text[at:text.index("public var witness: String {", at)]
    case_at = region.find("case .%s:" % case_name)
    if case_at < 0:
        raise SourceUnavailable("no width plan for table .%s" % case_name)
    literal = region[region.index("[", case_at):region.index("]", case_at) + 1]
    plan = [(int(m), int(ipg), int(rps))
            for m, ipg, rps in re.findall(
                r"\((\d+),\s*(\d+),\s*(\d+)\)", literal)]
    if not plan:
        raise SourceUnavailable("empty width plan for table .%s" % case_name)
    if render_plan(plan) != plan_witness(text, case_name):
        raise SourceUnavailable(
            "the .%s width plan and its witness literal disagree" % case_name)
    return plan


def plan_witness(text: str, case_name: str) -> str:
    at = text.find("public var witness: String {")
    if at < 0:
        raise SourceUnavailable("no QMV width-plan witness table")
    region = text[at:text.index("\n    }\n", at)]
    case_at = region.find("case .%s:" % case_name)
    if case_at < 0:
        raise SourceUnavailable("no witness literal for table .%s" % case_name)
    match = re.search(r'return "([^"]+)"', region[case_at:])
    if not match:
        raise SourceUnavailable("no witness string for table .%s" % case_name)
    return match.group(1)


def render_plan(plan: list[tuple[int, int, int]]) -> str:
    """`Qwen35CustomQMV.renderPlan`."""
    return "e120_width_plan/" + ",".join(
        "%d:%d:%d" % entry for entry in plan)


def qmv_names(text: str) -> dict[tuple[bool, int | None], str]:
    """`(table, tier)` to pipeline name, from `qwen35E120QMVName`."""
    at = text.find("func qwen35E120QMVName(table: Bool, tier: Int?) -> String {")
    if at < 0:
        raise SourceUnavailable("qwen35E120QMVName is gone")
    body = text[at:text.index("\n}\n", at)]
    names = {
        (flag == "true", None if tier == "nil" else int(tier)): name
        for flag, tier, name in re.findall(
            r'case \((true|false), (nil|\d+)\): return "([^"]+)"', body)}
    if not names:
        raise SourceUnavailable("no pipeline names in qwen35E120QMVName")
    return names


def route_b_route(text: str) -> dict:
    """The Route B QMV entry point the default route dispatches per width.

    This reproduces `Qwen35CustomQMV.matmul` and `.matmulWithTable` on an
    unset environment: the sum-table arm takes every width at or above
    `minimumTableWidth`, and the tiered layout sends a width to the entry
    point of its own `ipg`.
    """
    entry = default_entry(text)
    if entry not in ("shared", "tiered"):
        raise SourceUnavailable("unknown QMV entry layout %s" % entry)
    table_case = default_table(text)
    if table_case != "shipped" and entry != "tiered":
        raise SourceUnavailable(
            "a one-pass table is only legal on tiered entry points")
    arm = default_arm(text)
    plan = width_plan(text, table_case)
    names = qmv_names(text)

    minimum = re.search(r"public static let minimumTableWidth = (\d+)", text)
    if not minimum:
        raise SourceUnavailable("no minimumTableWidth")
    minimum = int(minimum.group(1))
    routed = re.search(r"public static let widths = (\d+) \.\.\. (\d+)", text)
    if not routed:
        raise SourceUnavailable("no routed width range")
    widths = range(int(routed.group(1)), int(routed.group(2)) + 1)
    if sorted(m for m, _, _ in plan) != list(widths):
        raise SourceUnavailable(
            "the .%s width plan does not cover the routed widths %s"
            % (table_case, list(widths)))

    serving: dict[int, dict] = {}
    for m, ipg, rps in plan:
        use_table = arm == "sumTable" and m >= minimum
        tier = ipg if entry == "tiered" else None
        if (use_table, tier) not in names:
            raise SourceUnavailable(
                "no pipeline name for (table=%s, tier=%s)" % (use_table, tier))
        serving[m] = {"name": names[(use_table, tier)], "table": use_table,
                      "tier": tier, "ipg": ipg, "rps": rps}
    return {"entry": entry, "table": table_case, "arm": arm,
            "witness": plan_witness(text, table_case), "serving": serving}


def route_b_serving(text: str) -> dict[int, str]:
    """Routed width to the name of the entry point that serves it."""
    return {m: cell["name"]
            for m, cell in route_b_route(text)["serving"].items()}


def route_b_present(text: str) -> bool:
    """Does this revision carry the Route B custom QMV family at all?

    A revision from before the family landed has nothing for the census to
    read, which is a different fact from a revision whose family is there but
    unreadable. The caller must fail on the second and may skip the first.
    """
    return "Qwen35CustomQMV" in text or "qwen35E120QMV" in text


def cluster_qmv_present(text: str) -> bool:
    return "qwen35ClusterAffine2QMVHeader" in text


def e120_qmv_body(text: str, table: bool, tier: int | None = None,
                  plan: list[tuple[int, int, int]] | None = None) -> str:
    """Rebuild `qwen35E120QMVSource(table:tier:)` from the Swift text.

    Only the interpolation semantics are known here. The dispatch width plan,
    the per-case template and the body itself are all read out of Swift, so a
    change to any of them reaches the census. `tier` keeps only the widths
    whose `ipg` equals it, exactly as the Swift `filter` does.

    `plan` replaces the shipped dispatch table with a hypothetical one, which
    is what a plan-surface census needs: the case template, the kernel body and
    every interpolation rule still come from live Swift, so the compiler sees
    the shipped kernel instantiated at another `(m, ipg, rps)`. The emitted
    witness comment then renders the hypothetical plan rather than the shipped
    one, so a censused source cannot claim to be the shipped route.
    """
    func = qmv_source_func(text)
    sums = ternary(func, "sums")[0 if table else 1]
    flag = ternary(func, "flag")[0 if table else 1]
    null_decl = ternary(func, "nullDecl")[0 if table else 1]

    table_case = default_table(text)
    witness = plan_witness(text, table_case) if plan is None \
        else render_plan(plan)
    source_plan = width_plan(text, table_case) if plan is None else plan
    plan = [entry for entry in source_plan
            if tier is None or entry[1] == tier]
    if not plan:
        raise SourceUnavailable(
            "no width in the .%s plan has ipg %s" % (table_case, tier))

    case_at = func.index(".map {")
    case_template, _ = multiline_literal(func, case_at)
    return_at = func.index("return \"\"\"")
    body_template, _ = multiline_literal(func, return_at)

    cases = "\n".join(
        case_template.replace("\\(plan.m)", str(m))
        .replace("\\(plan.ipg)", str(ipg))
        .replace("\\(2 * plan.rps)", str(2 * rps))
        .replace("\\(plan.rps)", str(rps))
        .replace("\\(flag)", flag).replace("\\(sums)", sums)
        for m, ipg, rps in plan)
    body = (body_template
            .replace("\\(Qwen35CustomQMV.planWitness)", witness)
            .replace("\\(nullDecl)", null_decl)
            .replace("\\(cases)", cases))
    if "\\(" in body:
        raise SourceUnavailable(
            "unresolved Swift interpolation in the QMV body: %s"
            % body[body.index("\\("):][:48])
    return body


def route_b_library(text: str) -> tuple[str, dict[str, str]]:
    """The Metal library holding every Route B kernel the default route runs.

    Returns the source and the role of each entry point in it. The set is the
    dispatch plan's own, so a table or layout change moves the census with it.
    """
    import e120_g17s_census as e120

    route = route_b_route(text)
    header = named_literal(text, "qwen35E120QMVHeader")
    xsums_at = text.find("name: \"qwen35_custom_affine4_g64_xsums_v1\"")
    if xsums_at < 0:
        raise SourceUnavailable("the xsums kernel is gone")
    xsums_body = multiline_literal(text, text.index("source:", xsums_at))[0]

    parts = [e120.PRELUDE, header, ""]
    roles: dict[str, str] = {}
    emitted: dict[str, tuple[bool, int | None]] = {}
    for width in sorted(route["serving"]):
        cell = route["serving"][width]
        key = (cell["table"], cell["tier"])
        if cell["name"] in emitted:
            continue
        emitted[cell["name"]] = key
        parts.append(e120.generate(
            cell["name"],
            QMV_INPUTS + ([("xsums", "float")] if cell["table"] else []),
            QMV_OUTPUTS,
            e120_qmv_body(text, table=cell["table"], tier=cell["tier"]),
            template=[("bool", "USE_TABLE", "true")] if cell["table"]
            else None))
    for name, (use_table, tier) in emitted.items():
        served = sorted(w for w, c in route["serving"].items()
                        if c["name"] == name)
        roles[name] = (
            "Route B %s QMV, %s, widths %s"
            % ("sum-table" if use_table else "no-table",
               "ipg %d" % tier if tier is not None else "shared switch",
               ",".join(str(w) for w in served)))
    parts.append(e120.generate(
        "qwen35_custom_affine4_g64_xsums_v1", XSUMS_INPUTS, XSUMS_OUTPUTS,
        xsums_body))
    roles["qwen35_custom_affine4_g64_xsums_v1"] = XSUMS_ROLE
    return "\n".join(parts) + "\n", roles


def kernel_body(text: str, name: str) -> str:
    """The `source:` literal of the `MLXFast.metalKernel` called `name`."""
    at = text.find("name: \"%s\"" % name)
    if at < 0:
        raise SourceUnavailable("no MLXFast.metalKernel named %s" % name)
    return multiline_literal(text, text.index("source:", at))[0]


def cluster_qmv_library(text: str) -> tuple[str, dict[str, str]]:
    """One Metal source holding both shipped affine-2 cluster QMV kernels."""
    import e120_g17s_census as e120

    header = named_literal(text, "qwen35ClusterAffine2QMVHeader")
    parts = [e120.PRELUDE, header, ""]
    for name, inputs in (
            ("qwen_mtp_cluster_centroid_qmv_a2g64_v1",
             CLUSTER_CENTROID_INPUTS),
            ("qwen_mtp_cluster_row_qmv_a2g64_v1", CLUSTER_ROW_INPUTS)):
        parts.append(e120.generate(name, inputs, CLUSTER_OUTPUTS,
                                   kernel_body(text, name)))
    return "\n".join(parts) + "\n", dict(CLUSTER_QMV_ROLES)
