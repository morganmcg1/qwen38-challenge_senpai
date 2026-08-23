#!/usr/bin/env python3
"""E152 Step 1b: per-cell AIR digest diff between two `metal -S` outputs.

F5 asks whether our `quantized_nax.h`, which carries E147's disarmed retile
scaffolding, generates the same code as the frontier's unmodified header. The
decision must be made on codegen, not on source text.

Two spellings must be normalised before the comparison means anything.

Metadata node IDs, attribute-group IDs and identified-struct-type names are all
numbered per module, so two modules built from sources of different length can
carry the same instruction stream under different `!N`, `#N` and
`%class.anon.N` labels. Attribute groups and struct types are resolved to their
own text, so a real attribute or layout change is still visible.

Our `qmm_t_nax_tgp_impl` and `qmm_n_nax_tgp_impl` take two extra template
parameters, so every mangled symbol that mentions them differs by name even
when the body is identical. Comparing raw text would therefore report a
difference for the name alone. The digest is instead recursive: a call to an
internally defined function contributes that callee's own digest, so two entry
points match when the whole reachable code is the same, whatever the callees
are called. References to undefined symbols, which are the AIR intrinsics, keep
their names because those names are the semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re

DEFINE = re.compile(r"^define\b.*?@([\w.$]+)\(")
ATTR_GROUP = re.compile(r"^attributes #(\d+) = \{(.*)\}\s*$")
TYPE_DEF = re.compile(r'^(%(?:"[^"]+"|[\w.$]+)) = type (.*)$')
TYPE_REF = re.compile(r'%(?:"[^"]+"|[A-Za-z_.$][\w.$]*)')
ATTR_REF = re.compile(r"#(\d+)\b")
META_REF = re.compile(r"!(\d+)\b")
SYMBOL = re.compile(r"@([\w.$]+)")


class Module:
    """The parts of one textual AIR module the comparison needs."""

    def __init__(self, path: pathlib.Path) -> None:
        self.functions: dict[str, list[str]] = {}
        self.attrs: dict[str, str] = {}
        self.types: dict[str, str] = {}
        name = None
        for line in path.read_text().splitlines():
            match = DEFINE.match(line)
            if match:
                name = match.group(1)
                self.functions[name] = []
                continue
            group = ATTR_GROUP.match(line)
            if group:
                self.attrs[group.group(1)] = group.group(2).strip()
                continue
            typedef = TYPE_DEF.match(line)
            if typedef and name is None:
                self.types[typedef.group(1)] = typedef.group(2).strip()
                continue
            if line == "}":
                name = None
            elif name is not None:
                self.functions[name].append(line)
        self._type_text: dict[str, str] = {}

    def type_text(self, name: str, depth: int = 0) -> str:
        """A named struct type expanded to its layout, so its number cannot matter."""
        if name in self._type_text:
            return self._type_text[name]
        body = self.types.get(name)
        if body is None or depth > 6:
            return name
        self._type_text[name] = name  # cycle guard for a self-referential type
        expanded = TYPE_REF.sub(
            lambda m: self.type_text(m.group(0), depth + 1)
            if m.group(0) in self.types and m.group(0) != name
            else m.group(0),
            body,
        )
        self._type_text[name] = "type{%s}" % expanded
        return self._type_text[name]

    def normalize(self, body: list[str]) -> list[str]:
        seen: dict[str, int] = {}

        def meta(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in seen:
                seen[key] = len(seen)
            return "!m%d" % seen[key]

        def attr(match: re.Match[str]) -> str:
            return "#attr{%s}" % self.attrs.get(match.group(1), "?" + match.group(1))

        def named_type(match: re.Match[str]) -> str:
            token = match.group(0)
            return self.type_text(token) if token in self.types else token

        return [
            TYPE_REF.sub(named_type, ATTR_REF.sub(attr, META_REF.sub(meta, line.rstrip())))
            for line in body
        ]


def recursive_digests(module: Module) -> dict[str, str]:
    """sha256 of each function with internal callee names replaced by their digest."""
    functions = module.functions
    normalized = {name: module.normalize(body) for name, body in functions.items()}
    cache: dict[str, str] = {}
    active: set[str] = set()

    def digest(name: str) -> str:
        if name in cache:
            return cache[name]
        if name in active:
            # Metal forbids recursion; a cycle here means the parse is wrong.
            raise RuntimeError("call cycle through %s" % name)
        active.add(name)

        def sub(match: re.Match[str]) -> str:
            callee = match.group(1)
            if callee == name or callee not in normalized:
                return "@" + callee
            return "@fn:" + digest(callee)

        text = "\n".join(SYMBOL.sub(sub, line) for line in normalized[name])
        active.discard(name)
        cache[name] = hashlib.sha256(text.encode()).hexdigest()
        return cache[name]

    return {name: digest(name) for name in normalized}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("left_ll")
    ap.add_argument("right_ll")
    ap.add_argument("--left-label", default="left")
    ap.add_argument("--right-label", default="right")
    ap.add_argument(
        "--entry-prefix",
        default="",
        help="compare only functions whose name starts with this",
    )
    ap.add_argument("--show", default="", help="print every compared cell matching this")
    args = ap.parse_args()

    def entries(path: str) -> dict[str, str]:
        full = recursive_digests(Module(pathlib.Path(path)))
        return {
            name: value
            for name, value in full.items()
            if name.startswith(args.entry_prefix)
        }

    left = entries(args.left_ll)
    right = entries(args.right_ll)

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    shared = sorted(set(left) & set(right))
    differ = [n for n in shared if left[n] != right[n]]

    print("compared cells  %s=%d  %s=%d  shared=%d"
          % (args.left_label, len(left), args.right_label, len(right), len(shared)))
    print("only in %s: %d" % (args.left_label, len(only_left)))
    print("only in %s: %d" % (args.right_label, len(only_right)))
    print("shared cells with a different recursive AIR digest: %d" % len(differ))
    print()

    for name in only_left[:20]:
        print("  ONLY-%s  %s" % (args.left_label.upper(), name))
    for name in only_right[:20]:
        print("  ONLY-%s  %s" % (args.right_label.upper(), name))
    for name in differ[:40]:
        print("  DIFFER   %-78s %s != %s"
              % (name, left[name][:16], right[name][:16]))

    if args.show:
        print()
        print("%-72s %-18s %-18s %s"
              % ("cell", args.left_label, args.right_label, "verdict"))
        for name in shared:
            if args.show not in name:
                continue
            same = "identical" if left[name] == right[name] else "DIFFERENT"
            print("%-72s %-18s %-18s %s"
                  % (name, left[name][:16], right[name][:16], same))

    identical = not (only_left or only_right or differ)
    print()
    print("air_identical=%s" % ("true" if identical else "false"))
    raise SystemExit(0 if identical else 1)


if __name__ == "__main__":
    main()
