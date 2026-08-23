"""E144 R-D: the head tree digest rule has two implementations, and they differ.

R-D rehearses an in-branch head declaration end to end. On the way it found a
divergence in the one number a candidate is asked to declare.

The rule is stated once and implemented three times:

  * `computeQwenMTPHeadProvenance` skips `relative == "README.md"` -- the
    TOP-LEVEL file only;
  * the ranked workflow's resolve step, and the equivalent shell printed in
    `mtp-head/README.md`, use `find . -type f ! -name README.md` -- any depth;
  * the workflow's pre-timing re-scrub refuses a `README.md` at any depth, then
    digests the survivor with no exclusion at all.

This file reproduces the divergence in pure Python on a scratch tree, and
locates each enforcing predicate in the live sources so the citation cannot go
stale silently. It touches no submitted path and runs outside every timed
window.

The divergence opens no benchmark escape: the pre-timing re-scrub fails the job
before a nested README can reach a timed round. It is a consistency and
diagnosability defect.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SITES = [
    {
        "site": "trusted CLI provenance sealer",
        "path": "Sources/MLXFastTrustedHarness/QwenMTPHeadDeclaration.swift",
        "needle": 'if relative == "README.md" { continue }',
        "rule": "top-level README.md only",
    },
    {
        "site": "ranked workflow, resolve the declared head",
        "path": ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        "needle": "&& find . -type f ! -name README.md \\",
        "rule": "README.md at any depth",
    },
    {
        "site": "participant-facing equivalent shell",
        "path": "mtp-head/README.md",
        "needle": "find . -type f ! -name README.md \\",
        "rule": "README.md at any depth",
    },
    {
        "site": "ranked workflow, pre-timing re-scrub",
        "path": ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        "needle": '"${declared_head}" -name README.md -print -quit',
        "rule": "refuse README.md at any depth",
    },
]


def locate(site):
    """Find a predicate in the live source and return its line number."""
    path = os.path.join(ROOT, site["path"])
    with open(path, errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if site["needle"].rstrip("\\").strip() in line:
                return {**site, "line": number, "text": line.strip(), "found": True}
    return {**site, "line": None, "text": None, "found": False}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk(root):
    for directory, _, files in os.walk(root):
        for name in files:
            absolute = os.path.join(directory, name)
            yield os.path.relpath(absolute, root), absolute


def tree_digest(root, skip):
    """The shared digest shape; `skip(relative)` is the only thing that varies."""
    entries = sorted(
        (relative, file_sha256(absolute), os.path.getsize(absolute))
        for relative, absolute in walk(root)
        if not skip(relative)
    )
    hasher = hashlib.sha256()
    total = 0
    for relative, digest, size in entries:
        hasher.update(f"{digest}  {relative}\n".encode())
        total += size
    return {"sha256": hasher.hexdigest(), "bytes": total, "file_count": len(entries)}


def sealer_digest(root):
    return tree_digest(root, lambda relative: relative == "README.md")


def workflow_digest(root):
    return tree_digest(root, lambda relative: os.path.basename(relative) == "README.md")


def scratch_case(nested_readme):
    root = tempfile.mkdtemp(prefix="e144-rd-")
    try:
        with open(os.path.join(root, "model.safetensors"), "w") as handle:
            handle.write("weights")
        with open(os.path.join(root, "README.md"), "w") as handle:
            handle.write("top level docs")
        if nested_readme:
            os.makedirs(os.path.join(root, "shard"))
            with open(os.path.join(root, "shard", "README.md"), "w") as handle:
                handle.write("nested docs")
        sealed = sealer_digest(root)
        workflow = workflow_digest(root)
        return {
            "nested_readme": nested_readme,
            "sealed": sealed,
            "workflow": workflow,
            "digests_agree": sealed["sha256"] == workflow["sha256"],
            "bytes_agree": sealed["bytes"] == workflow["bytes"],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="e144-rd.json")
    parser.add_argument(
        "--shipped-head",
        default=os.path.expanduser(
            "~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
        ),
    )
    arguments = parser.parse_args()

    sites = [locate(site) for site in SITES]
    flat = scratch_case(nested_readme=False)
    nested = scratch_case(nested_readme=True)

    exposure = {"checked": os.path.isdir(arguments.shipped_head)}
    if exposure["checked"]:
        readmes = [
            relative
            for relative, _ in walk(arguments.shipped_head)
            if os.path.basename(relative) == "README.md"
        ]
        exposure["path"] = arguments.shipped_head
        exposure["readme_files"] = readmes
        exposure["shipped_head_is_exposed"] = bool(readmes)

    report = {
        "experiment": "e144",
        "rung": "R-D",
        "harness": "local",
        "finding": "two readings of one rule",
        "enforcing_sites": sites,
        "all_sites_located": all(site["found"] for site in sites),
        "cases": {"flat_tree": flat, "nested_readme_tree": nested},
        "e144_digest_rules_agree_without_nested_readme": flat["digests_agree"],
        "e144_digest_rules_agree_with_nested_readme": nested["digests_agree"],
        "e144_nested_readme_splits_the_rule": flat["digests_agree"]
        and not nested["digests_agree"],
        "benchmark_escape": False,
        "escape_closed_by": "the workflow's pre-timing re-scrub refuses a "
        "README.md at any depth in the resolved declared head, so a nested "
        "README fails the job instead of riding into a timed round as "
        "digest-exempt payload.",
        "shipped_head_exposure": exposure,
        "swift_evidence": "Tests/MLXFastTests/E144HeadDeclarationTests.swift, "
        "run with: swift test --force-resolved-versions --filter "
        "E144HeadDeclarationTests",
    }

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    for site in sites:
        print(
            "%-42s %s:%s  %s"
            % (site["site"], site["path"], site["line"], site["rule"])
        )
    print()
    for label, case in report["cases"].items():
        print(
            "%-20s sealed %s (%d files, %d B)  workflow %s (%d files, %d B)  agree %s"
            % (
                label,
                case["sealed"]["sha256"][:16],
                case["sealed"]["file_count"],
                case["sealed"]["bytes"],
                case["workflow"]["sha256"][:16],
                case["workflow"]["file_count"],
                case["workflow"]["bytes"],
                case["digests_agree"],
            )
        )
    print()
    print("nested README splits the rule: %s" % report["e144_nested_readme_splits_the_rule"])
    print("shipped head exposure: %s" % json.dumps(exposure))


if __name__ == "__main__":
    main()
