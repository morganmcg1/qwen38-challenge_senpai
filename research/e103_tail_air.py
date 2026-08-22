#!/usr/bin/env python3
"""Per-entry-point AIR instruction accounting for the E103 SDPA tail arms.

Askeladd's `n_nosums` result (PR #120, +7.60 %) says the binding resource in
this kernel family is total instruction issue, not load-issue port and not
memory latency. A static AIR line count cannot answer that on its own, because
Metal emits AIR with every loop still rolled: a body that runs eight times and
a body that runs twice look the same to `grep`.

So this tool builds the control-flow graph, finds the natural loops with a real
dominator computation, reads the trip count out of the latch comparison when the
bound is a literal, and reports both the static count and the count each
instruction class actually issues once per thread.

  python3 research/e103_tail_air.py ARMS.ll [--match SUBSTR] [--json OUT]

The AIR must come from a `-fno-fast-math` compile. The shipped kernels are built
with that flag (`kernels/CMakeLists.txt:18`) and `xcrun metal` defaults to the
opposite, which rewrites `x / y` into `x * (1 / y)` and hides the divide this
experiment is about.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

LABEL = re.compile(r"^(\d+):")
BR = re.compile(r"^\s*br\s+(?:label %(\d+)|i1 [^,]+, label %(\d+), label %(\d+))")
LOOP_MD = re.compile(r"!llvm\.loop")
LATCH_ICMP = re.compile(r"=\s*icmp\s+\w+\s+i\d+\s+%[\w.]+,\s+(-?\d+)\b")

CLASSES = (
    ("barrier", re.compile(r"@air\.wg\.barrier")),
    ("simd_reduce", re.compile(r"@air\.simd_(sum|max|min|prod|xor|and|or)\.")),
    ("tg_load", re.compile(r"=\s*load\s.*addrspace\(3\)")),
    ("tg_store", re.compile(r"^\s*store\s.*addrspace\(3\)")),
    ("dev_load", re.compile(r"=\s*load\s.*addrspace\(1\)")),
    ("dev_store", re.compile(r"^\s*store\s.*addrspace\(1\)")),
    ("fdiv", re.compile(r"=\s*fdiv\s")),
    ("fmul", re.compile(r"=\s*fmul\s")),
    ("fadd", re.compile(r"=\s*fadd\s")),
    ("fsub", re.compile(r"=\s*fsub\s")),
    ("fma", re.compile(r"@(llvm|air)\.fma\.")),
    ("fast_exp", re.compile(r"@air\.fast_exp\.")),
    ("select", re.compile(r"=\s*select\s")),
    ("getelementptr", re.compile(r"=\s*getelementptr")),
    ("phi", re.compile(r"=\s*phi\s")),
)

SKIP = re.compile(r"^\s*(;|$)|@llvm\.lifetime\.")


def entry_points(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        if line.startswith("define "):
            match = re.search(r"@([\w.]+)\(", line)
            name = match.group(1) if match else None
            if name:
                out[name] = []
        elif line == "}":
            name = None
        elif name is not None:
            out[name].append(line)
    return out


def blocks(body: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Split a function body into labelled basic blocks, in listing order."""
    order: list[str] = ["entry"]
    found: dict[str, list[str]] = {"entry": []}
    current = "entry"
    for line in body:
        match = LABEL.match(line)
        if match:
            current = match.group(1)
            order.append(current)
            found[current] = []
        else:
            found[current].append(line)
    return order, found


def successors(lines: list[str]) -> list[str]:
    for line in lines:
        match = BR.match(line)
        if match:
            if match.group(1) is not None:
                return [match.group(1)]
            return [match.group(2), match.group(3)]
    return []


def dominators(order: list[str], succ: dict[str, list[str]]) -> dict[str, set[str]]:
    pred: dict[str, list[str]] = {b: [] for b in order}
    for b in order:
        for s in succ[b]:
            if s in pred:
                pred[s].append(b)
    everything = set(order)
    dom = {b: (set([b]) if b == order[0] else set(everything)) for b in order}
    changed = True
    while changed:
        changed = False
        for b in order[1:]:
            new = set(everything)
            for p in pred[b]:
                new &= dom[p]
            new.add(b)
            if new != dom[b]:
                dom[b] = new
                changed = True
    return dom


def natural_loops(
    order: list[str], succ: dict[str, list[str]], dom: dict[str, set[str]]
) -> list[tuple[str, str, set[str]]]:
    """(header, latch, body) for every backedge latch -> header."""
    pred: dict[str, list[str]] = {b: [] for b in order}
    for b in order:
        for s in succ[b]:
            if s in pred:
                pred[s].append(b)

    loops = []
    for latch in order:
        for header in succ[latch]:
            if header not in dom.get(latch, ()):
                continue
            # A self-loop is its own body. Seeding the walk with the latch would
            # step straight past the header into the whole preceding function.
            body = {header, latch}
            stack = [] if latch == header else [latch]
            while stack:
                node = stack.pop()
                for p in pred[node]:
                    if p not in body:
                        body.add(p)
                        stack.append(p)
            loops.append((header, latch, body))
    return loops


BR_COND = re.compile(r"^\s*br\s+i1 (%[\w.]+), label %(\d+), label %(\d+)")
BOOL_PHI = re.compile(r"(%[\w.]+)\s*=\s*phi i1 \[ (true|false), [^]]+\], \[ (true|false), ")
IV_PHI = re.compile(r"(%[\w.]+)\s*=\s*phi i32 (.*)$")
PHI_ARM = re.compile(r"\[ (%[\w.]+|-?\d+), %[\w.]+ \]")
ADD_STEP = re.compile(r"(%[\w.]+)\s*=\s*add nuw nsw i32 (%[\w.]+), (-?\d+)")
ICMP_LIT = re.compile(r"(%[\w.]+)\s*=\s*icmp (\w+) i32 (%[\w.]+), (-?\d+)")


def trip_count(
    latch: list[str], header: list[str], body: list[str]
) -> int | None:
    """Literal trip count of a counted loop.

    Three forms appear in this file, and reading only the bound constant gets
    two of them wrong.

    A counted loop compares either the induction variable itself or its
    incremented value against the bound. `icmp ult %iv, 6` on a loop that steps
    by two runs four times, not six and not three, because the test uses the
    value before the increment. The step therefore has to be read as well.

    A two-iteration loop is rotated instead: the compiler drops the comparison
    and controls the back edge with an `i1` phi that is true on entry and false
    on the back edge.

    Returns None for the key loop, whose bound is the runtime N.
    """
    terminator = next((i for i, l in enumerate(latch) if BR.match(l)), None)
    if terminator is None or not LOOP_MD.search(latch[terminator]):
        return None
    cond = BR_COND.match(latch[terminator])
    if not cond:
        return None
    name = cond.group(1)

    for line in header:
        phi = BOOL_PHI.search(line)
        if phi and phi.group(1) == name and phi.group(2) != phi.group(3):
            return 2

    compare = next(
        (m for l in body if (m := ICMP_LIT.search(l)) and m.group(1) == name), None
    )
    if compare is None:
        return None
    _, predicate, operand, bound_text = compare.groups()
    bound = int(bound_text)

    # The incoming arms of the induction phi appear in either order, so take the
    # literal one as the start value and the defined one as the increment.
    variable = start = increment = None
    for line in header:
        candidate = IV_PHI.search(line)
        if not candidate:
            continue
        arms = PHI_ARM.findall(candidate.group(2))
        literals = [a for a in arms if not a.startswith("%")]
        defined = [a for a in arms if a.startswith("%")]
        if len(arms) == 2 and len(literals) == 1 and len(defined) == 1:
            variable, start, increment = (
                candidate.group(1),
                int(literals[0]),
                defined[0],
            )
            step_match = next(
                (
                    m
                    for l in body
                    if (m := ADD_STEP.search(l))
                    and m.group(1) == increment
                    and m.group(2) == variable
                ),
                None,
            )
            if step_match is not None and operand in (variable, increment):
                break
    else:
        return None
    if variable is None or step_match is None:
        return None
    step = int(step_match.group(3))
    if step <= 0 or predicate not in ("eq", "ne", "ult", "slt"):
        return None

    if operand == increment:
        trips = (bound - start) // step
    elif operand == variable:
        trips = (bound - start) // step + 1
    else:
        return None
    return trips if trips > 0 else None


def classify(lines: list[str]) -> dict[str, int]:
    counts = {name: 0 for name, _ in CLASSES}
    counts["instructions"] = 0
    for line in lines:
        if SKIP.match(line) or LABEL.match(line):
            continue
        counts["instructions"] += 1
        for name, pattern in CLASSES:
            if pattern.search(line):
                counts[name] += 1
                break
    return counts


def analyse(body: list[str]) -> dict:
    order, found = blocks(body)
    succ = {b: successors(found[b]) for b in order}
    dom = dominators(order, succ)
    loops = natural_loops(order, succ, dom)

    # Innermost first, so a nested loop multiplies its own trip count onto the
    # enclosing one instead of being counted flat.
    loops.sort(key=lambda item: len(item[2]))
    depth: dict[str, int] = {b: 1 for b in order}
    symbolic: set[str] = set()
    detail = []
    for header, latch, nodes in loops:
        inside = [l for n in sorted(nodes, key=order.index) for l in found[n]]
        trips = trip_count(found[latch], found[header], inside)
        detail.append(
            {
                "header": header,
                "latch": latch,
                "blocks": sorted(nodes, key=order.index),
                "trips": trips,
                "has_barrier": any(
                    "@air.wg.barrier" in l for n in nodes for l in found[n]
                ),
            }
        )
        for node in nodes:
            if trips is None:
                symbolic.add(node)
            else:
                depth[node] *= trips

    # The tail region is everything after the last block of the key loop. That
    # boundary is what the experiment moves; the prologue and the key loop are
    # byte for byte identical across these arms and would only dilute the count.
    key_blocks = [b for b in order if b in symbolic]
    tail_from = max(order.index(b) for b in key_blocks) + 1 if key_blocks else 0

    static = classify(body)
    dynamic = {k: 0 for k in static}
    sym = {k: 0 for k in static}
    tail = {k: 0 for k in static}
    for b in order:
        counts = classify(found[b])
        target = sym if b in symbolic else dynamic
        for k, v in counts.items():
            target[k] += v * depth[b]
            if order.index(b) >= tail_from:
                tail[k] += v * depth[b]

    return {
        "static": static,
        "dynamic_fixed_trip": dynamic,
        "dynamic_tail_only": tail,
        "per_key_loop_iteration": sym,
        "tail_first_block": order[tail_from] if tail_from < len(order) else None,
        "loops": detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("air_ll", type=pathlib.Path)
    ap.add_argument("--match", default="")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    report = {}
    for name, body in entry_points(args.air_ll.read_text()).items():
        if args.match and args.match not in name:
            continue
        report[name] = analyse(body)

    columns = (
        "instructions",
        "barrier",
        "tg_store",
        "tg_load",
        "simd_reduce",
        "fdiv",
        "fmul",
        "fadd",
        "select",
        "getelementptr",
    )
    print(f"{'entry':16s} " + " ".join(f"{c[:6]:>6s}" for c in columns))
    for scope in (
        "static",
        "dynamic_tail_only",
        "dynamic_fixed_trip",
        "per_key_loop_iteration",
    ):
        print(f"-- {scope}")
        for name, data in report.items():
            row = " ".join(f"{data[scope][c]:6d}" for c in columns)
            print(f"{name:16s} {row}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
