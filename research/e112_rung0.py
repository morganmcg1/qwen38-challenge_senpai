#!/usr/bin/env python3
"""E112 rung 0. Price two single-hunk board mechanisms on the TARGET probe.

Mechanisms
----------
Q1  the kL=1025 128-block SDPA compile-warm family inside
    `Sources/MLXFastModel/Qwen36MTPBlockSession.swift`. The proposed edit
    DELETES it, so the mechanism direction is present -> absent.

Q2  one `asyncEval(outA)` between the split-5 SDPA chunks inside
    `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift`. The
    proposed edit INSERTS it, so the mechanism direction is absent -> present.

Tree identity
-------------
`submissionCommitSha` is not identity: the same tree ships under many commits.
Identity here is the CODE the compiler sees. Every submission is diffed against
one anchor tree over the scored surface, each changed file is canonicalised by
removing comments, blank lines and trailing whitespace, and the identity is the
set of files whose canonical form differs from the anchor. Both board pairs the
advisor cited differ from their partner by the mechanism plus a comment block,
so a comment-sensitive digest would have missed them.

The canonicaliser and the measured floor both come from
`research/board_prompt_instrument.py`. This file used to hold its own copies.
The stripper copy was a regex that deleted `//` inside the Metal kernel source
strings in `Vendor/.../mlx-generated/*.cpp`, which merged trees that
JIT-compile different kernels, and the floor copy was 0.0431 %, which was the
spread of one narrow replicate class rather than the measurement floor. Both
copies are gone. Never reintroduce a second one.

An ISOLATED PAIR for a mechanism is two submissions whose identities are equal
after the mechanism text itself is also canonicalised away, and which disagree
on whether the mechanism is present. Nothing else in the compiled candidate
differs.

Sign convention matches research/board_pair_decompose.py and
research/board_prompt_instrument.py: a POSITIVE percentage means the second
tree decoded FASTER. Pairs are ordered so positive means the mechanism HELPED.

Usage
-----
    python3 research/e112_rung0.py                      # both mechanisms
    python3 research/e112_rung0.py --mech q1
    python3 research/e112_rung0.py --neighbours e72058d7    # T1 support
"""

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# One canonicaliser and one floor for the whole campaign. The copies that used
# to live here held a naive comment regex that ate `//` inside the Metal kernel
# source strings in `mlx-generated/*.cpp`, and a TARGET floor of 0.0431 % that
# the corrected canonicalisation retracted.
from board_prompt_instrument import (  # noqa: E402
    CONSERVATIVE, canon_code as _canon_code_aware, strip_comments_aware)

BOARD_JSON = os.environ.get("YUKON_BOARD_JSON", "/tmp/yukon-board/full.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.environ.get("E112_ANCHOR",
                        "b129f202fc25413015463da559777aaa59534065")

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
PROMPT_ORDER = ["plutarch", "drama", "travel", "beagle", "medicine",
                "republic", "essays", "botany"]
TARGET_PROBE = "plutarch"
DRAFT_PROBES = ["beagle", "medicine", "republic", "essays", "botany"]

# The conservative TARGET floor is the widest single replicate class, from
# research/board_prompt_instrument.py. Re-measure it with `--noise` there.
RES_TARGET_SAME_MODE = CONSERVATIVE["target_per_run"]
MODE_DRAFT_SHIFT = 0.60

BUILD_PATHS = ["Sources", "Vendor", "Package.swift", "Package.resolved",
               "tools", "mtp-head.manifest.json", "mtp-head"]

Q1_PATH = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
Q2_PATH = "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift"


def strip_comments(text):
    """Comment-free, blank-free text. Comments cannot cost decode time."""
    stripped = strip_comments_aware(text)
    return "\n".join(ln.rstrip() for ln in stripped.split("\n") if ln.strip())


def canon_code(text, path=""):
    return _canon_code_aware(text, path), False


def canon_q1(text):
    """Strip the kL=1025 128-block warm family; report whether it was there."""
    lines = strip_comments(text).split("\n")
    present = False
    out = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*if\s+.*\{\s*$", lines[i]):
            depth = 0
            j = i
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and j > i:
                    break
                j += 1
            body = "\n".join(lines[i:j + 1])
            if "1025" in body and "scaledDotProductAttention" in body:
                present = True
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(ln for ln in out if "1025" not in ln), present


ASYNC_OUTA = re.compile(r"^\s*(MLX\.)?asyncEval\(\s*outA\s*\)\s*$")


def canon_q2(text):
    """Strip a standalone `asyncEval(outA)` statement; report presence."""
    present = False
    kept = []
    for line in strip_comments(text).split("\n"):
        if ASYNC_OUTA.match(line):
            present = True
            continue
        kept.append(line)
    return "\n".join(kept), present


MECHANISMS = {
    "q1": {
        "path": Q1_PATH, "canon": canon_q1,
        "label": "Q1 kL=1025 128-block SDPA compile-warm family",
        "edit": "DELETE the family",
        "helped_when": "absent",
        "prediction": "largest on plutarch, small on the wide prompts",
    },
    "q2": {
        "path": Q2_PATH, "canon": canon_q2,
        "label": "Q2 asyncEval(outA) between the split-5 SDPA chunks",
        "edit": "INSERT the call",
        "helped_when": "present",
        "prediction": "near zero on plutarch, rising with mean draft width, "
                      "largest on botany",
    },
}


# --- git plumbing -----------------------------------------------------------

def git(args):
    return subprocess.run(["git"] + args, cwd=REPO, capture_output=True,
                          text=True, errors="replace")


_blob_cache = {}


def blob_text(oid):
    if oid not in _blob_cache:
        proc = git(["cat-file", "blob", oid])
        _blob_cache[oid] = proc.stdout if proc.returncode == 0 else ""
    return _blob_cache[oid]


_canon_cache = {}


def canon_digest(oid, mech_key, path):
    """(digest, mechanism_present) for one blob under one canonicalisation."""
    key = (oid, mech_key, path)
    if key not in _canon_cache:
        if mech_key and path == MECHANISMS[mech_key]["path"]:
            canon, present = MECHANISMS[mech_key]["canon"](blob_text(oid))
        else:
            canon, present = canon_code(blob_text(oid), path)
        _canon_cache[key] = (hashlib.sha256(canon.encode()).hexdigest()[:16],
                             present)
    return _canon_cache[key]


ZERO = "0" * 40


def changed_blobs(ref):
    """{path: (anchor_oid, ref_oid)} over the scored surface, vs the anchor."""
    proc = git(["diff", "--raw", "-z", ANCHOR, ref, "--"] + BUILD_PATHS)
    if proc.returncode != 0:
        return None
    fields = proc.stdout.split("\0")
    out = {}
    i = 0
    while i < len(fields):
        meta = fields[i]
        if not meta.startswith(":"):
            i += 1
            continue
        parts = meta.split()
        old_oid, new_oid, status = parts[2], parts[3], parts[4]
        nfiles = 2 if status[0] in ("R", "C") else 1
        path = fields[i + nfiles] if i + nfiles < len(fields) else ""
        out[path] = (old_oid, new_oid)
        i += 1 + nfiles
    return out


_anchor_cache = {}


def anchor_blob_oid(path):
    if path not in _anchor_cache:
        proc = git(["rev-parse", f"{ANCHOR}:{path}"])
        _anchor_cache[path] = (proc.stdout.strip()
                               if proc.returncode == 0 else None)
    return _anchor_cache[path]


def identity(ref, mech_key):
    """(identity_set, mechanism_present) for one submission."""
    changed = changed_blobs(ref)
    if changed is None:
        return None, None
    mech_path = MECHANISMS[mech_key]["path"] if mech_key else None
    items = []
    present = None
    for path, (old_oid, new_oid) in sorted(changed.items()):
        if new_oid == ZERO:
            items.append((path, "DELETED"))
            if path == mech_path:
                present = False
            continue
        new_digest, new_present = canon_digest(new_oid, mech_key, path)
        if path == mech_path:
            present = new_present
        if old_oid != ZERO:
            old_digest, _ = canon_digest(old_oid, mech_key, path)
            if old_digest == new_digest:
                continue  # comment-only or mechanism-only difference
        items.append((path, new_digest))
    if mech_path is not None and present is None:
        oid = anchor_blob_oid(mech_path)
        present = canon_digest(oid, mech_key, mech_path)[1] if oid else False
    return frozenset(items), present


# --- board ------------------------------------------------------------------

def load_rows(path=BOARD_JSON):
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        for key in ("submissions", "rows", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    return [r for r in payload if isinstance(r, dict)]


def prompt_map(row):
    metrics = row.get("officialMetrics") or {}
    per_prompt = metrics.get("per_prompt")
    if not isinstance(per_prompt, list) or len(per_prompt) != 8:
        return None
    out = {}
    for entry in per_prompt:
        if not isinstance(entry, dict):
            return None
        name = PROMPT_NAMES.get((entry.get("prompt_sha256") or "")[:8])
        if name is None or not entry.get("mtp_seconds_per_token_mean"):
            return None
        if entry.get("effective_mean_draft_len") is None:
            return None
        out[name] = entry
    return out if len(out) == 8 else None


def schedule_signature(pmap):
    return tuple(repr(pmap[n]["effective_mean_draft_len"]) for n in PROMPT_ORDER)


def cand_pct(pmap_a, pmap_b, name):
    return 100.0 * math.log(pmap_a[name]["mtp_seconds_per_token_mean"]
                            / pmap_b[name]["mtp_seconds_per_token_mean"])


def collect(rows):
    refs = git(["for-each-ref", "--format=%(refname:short)",
                "refs/remotes/upstream/submissions/"]).stdout.split()
    by_id = {r.rsplit("/", 1)[-1]: r for r in refs}
    recs = []
    for row in rows:
        ref = by_id.get(row.get("id") or "")
        pmap = prompt_map(row)
        if ref is None or pmap is None:
            continue
        recs.append({
            "id8": (row.get("id") or "")[:8], "ref": ref, "row": row,
            "pmap": pmap, "sig": schedule_signature(pmap),
            "solver": row.get("solverUsername") or "",
            "score": row.get("officialScore"),
            "status": row.get("status") or "",
            "date": (row.get("createdAt") or "")[:16].replace("T", " "),
        })
    return recs


# --- reporting --------------------------------------------------------------

def report_mechanism(recs, mech_key, require_schedule=True):
    mech = MECHANISMS[mech_key]
    print("=" * 78)
    print(f"MECHANISM {mech_key.upper()}: {mech['label']}")
    print(f"  file       {mech['path']}")
    print(f"  proposal   {mech['edit']}")
    print(f"  helps if   the mechanism ends up {mech['helped_when']}")
    print("=" * 78)

    groups = defaultdict(lambda: {"present": [], "absent": []})
    n_present = 0
    usable = 0
    for rec in recs:
        key, present = identity(rec["ref"], mech_key)
        if key is None:
            continue
        usable += 1
        n_present += bool(present)
        groups[key]["present" if present else "absent"].append(rec)

    print(f"\n{usable} scored board rows carry a readable tree; "
          f"{n_present} contain the mechanism, {usable - n_present} do not.")
    print(f"{len(groups)} distinct residual code identities.")

    pairs = []
    dropped_sched = 0
    for side in groups.values():
        if not side["present"] or not side["absent"]:
            continue
        for a in side["present"]:
            for b in side["absent"]:
                first, second = ((a, b) if mech["helped_when"] == "absent"
                                 else (b, a))
                if first["sig"] != second["sig"]:
                    dropped_sched += 1
                    if require_schedule:
                        continue
                target = cand_pct(first["pmap"], second["pmap"], TARGET_PROBE)
                draft = statistics.fmean(
                    cand_pct(first["pmap"], second["pmap"], n)
                    for n in DRAFT_PROBES)
                pairs.append({
                    "a": first, "b": second,
                    "same_sched": first["sig"] == second["sig"],
                    "target": target, "draft": draft,
                    "per_prompt": {n: cand_pct(first["pmap"], second["pmap"], n)
                                   for n in PROMPT_ORDER},
                    "mode_flip": abs(draft) > MODE_DRAFT_SHIFT,
                })

    if dropped_sched:
        print(f"{dropped_sched} isolated pairs had a schedule mismatch "
              f"({'dropped' if require_schedule else 'kept'}).")
    if not pairs:
        print("\nNO isolated schedule-matched pair exists for this mechanism.")
        return None

    print(f"\nISOLATED SCHEDULE-MATCHED PAIRS: {len(pairs)}")
    print("A lacks the favoured state, B has it. Positive TARGET means the "
          "mechanism\nmade plutarch FASTER.\n")
    print(f"{'A':>8} {'B':>8} {'A solver':>14} {'B solver':>14} "
          f"{'B date':>16} {'TARGET%':>9} {'sigma':>7} {'DRAFT%':>9} "
          f"{'mode':>5}")
    for p in sorted(pairs, key=lambda q: q["target"]):
        print(f"{p['a']['id8']:>8} {p['b']['id8']:>8} "
              f"{p['a']['solver'][:14]:>14} {p['b']['solver'][:14]:>14} "
              f"{p['b']['date']:>16} {p['target']:+9.4f} "
              f"{p['target'] / RES_TARGET_SAME_MODE:+7.2f} "
              f"{p['draft']:+9.4f} {'FLIP' if p['mode_flip'] else '-':>5}")

    runs_a = {p["a"]["id8"] for p in pairs}
    runs_b = {p["b"]["id8"] for p in pairs}
    pooled = statistics.fmean(p["target"] for p in pairs)
    naive_res = RES_TARGET_SAME_MODE / math.sqrt(len(pairs))
    proper_res = RES_TARGET_SAME_MODE * math.sqrt(
        1.0 / len(runs_a) + 1.0 / len(runs_b))
    pos = sum(1 for p in pairs if p["target"] > 0)
    neg = len(pairs) - pos

    print(f"\nPOOLED TARGET effect   {pooled:+.4f} %")
    print(f"  distinct runs         {len(runs_a)} without the favoured state, "
          f"{len(runs_b)} with it")
    floor = RES_TARGET_SAME_MODE
    print(f"  naive resolution      {naive_res:.4f} %  "
          f"({floor:.4f} / sqrt({len(pairs)}))   "
          f"sigma {pooled / naive_res:+.2f}")
    print(f"  independent-run res   {proper_res:.4f} %  "
          f"({floor:.4f} * sqrt(1/{len(runs_a)} + 1/{len(runs_b)}))   "
          f"sigma {pooled / proper_res:+.2f}")
    print(f"  signs                 {pos} positive, {neg} negative "
          f"({neg / len(pairs):.0%} opposite)")

    print(f"\nPER-PROMPT PROFILE over {len(pairs)} pairs")
    print(f"  predicted if the mechanism is real: {mech['prediction']}\n")
    print(f"{'prompt':>9} {'mean draft len':>15} {'effect %':>10} "
          f"{'sd over pairs':>14}")
    members = list(runs_pmaps(pairs))
    for name in PROMPT_ORDER:
        vals = [p["per_prompt"][name] for p in pairs]
        drafts = [r["pmap"][name]["effective_mean_draft_len"] for r in members]
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        print(f"{name:>9} {statistics.fmean(drafts):15.3f} "
              f"{statistics.fmean(vals):+10.4f} {sd:14.4f}")
    return {"mech": mech_key, "pairs": pairs, "pooled": pooled,
            "naive_res": naive_res, "proper_res": proper_res,
            "pos": pos, "neg": neg}


def runs_pmaps(pairs):
    seen = {}
    for p in pairs:
        seen[p["a"]["id8"]] = p["a"]
        seen[p["b"]["id8"]] = p["b"]
    return seen.values()


def report_near(recs, mech_key, max_extra=3):
    """Pairs that differ by the mechanism plus at most `max_extra` files."""
    mech = MECHANISMS[mech_key]
    by_present = {True: [], False: []}
    for rec in recs:
        _, present = identity(rec["ref"], mech_key)
        if present is None:
            continue
        by_present[bool(present)].append(rec)
    print(f"NEAR-ISOLATED PAIRS for {mech_key.upper()} "
          f"(mechanism plus at most {max_extra} other scored files)\n")
    favoured = mech["helped_when"] == "present"
    rows = []
    for a in by_present[not favoured]:
        for b in by_present[favoured]:
            if a["sig"] != b["sig"]:
                continue
            proc = git(["diff", "--name-only", a["ref"], b["ref"], "--"]
                       + BUILD_PATHS)
            names = [n for n in proc.stdout.splitlines() if n.strip()]
            extra = [n for n in names if n != mech["path"]]
            if len(extra) > max_extra:
                continue
            rows.append((a, b, extra,
                         cand_pct(a["pmap"], b["pmap"], TARGET_PROBE)))
    if not rows:
        print("none")
        return
    for a, b, extra, target in sorted(rows, key=lambda r: r[3]):
        print(f"{a['id8']} {a['solver']:>14} -> {b['id8']} {b['solver']:>14} "
              f"TARGET {target:+.4f} %")
        for n in extra:
            print(f"      also differs: {n}")


def identity_excluding(ref, drop_path):
    """Code identity of everything except one file, plus that file's digest."""
    changed = changed_blobs(ref)
    if changed is None:
        return None, None
    items = []
    own = None
    for path, (old_oid, new_oid) in sorted(changed.items()):
        if new_oid == ZERO:
            if path == drop_path:
                own = "DELETED"
            else:
                items.append((path, "DELETED"))
            continue
        new_digest, _ = canon_digest(new_oid, None, path)
        if path == drop_path:
            own = new_digest
            continue
        if old_oid != ZERO and canon_digest(old_oid, None, path)[0] == new_digest:
            continue
        items.append((path, new_digest))
    if own is None:
        oid = anchor_blob_oid(drop_path)
        own = canon_digest(oid, None, drop_path)[0] if oid else "MISSING"
    return frozenset(items), own


def canon_lines(ref, path):
    proc = git(["show", f"{ref}:{path}"])
    if proc.returncode != 0:
        return []
    return strip_comments(proc.stdout).split("\n")


WARM_QL = re.compile(r"for qL in (\[[^\]]*\])")


def signature(diff):
    """One-line description of what a canonical single-file diff changed."""
    tags = []
    added = [ln[1:] for ln in diff if ln.startswith("+")
             and not ln.startswith("+++")]
    removed = [ln[1:] for ln in diff if ln.startswith("-")
               and not ln.startswith("---")]
    add_1025 = any("k1025" in ln for ln in added)
    del_1025 = any("k1025" in ln for ln in removed)
    if add_1025 and not del_1025:
        tags.append("+1025warm")
    if del_1025 and not add_1025:
        tags.append("-1025warm")
    add_ql = {m.group(1) for ln in added for m in [WARM_QL.search(ln)] if m}
    del_ql = {m.group(1) for ln in removed for m in [WARM_QL.search(ln)] if m}
    if add_ql != del_ql:
        tags.append(f"qL {sorted(del_ql)}->{sorted(add_ql)}")
    other = len([ln for ln in added + removed
                 if "1025" not in ln and not WARM_QL.search(ln)])
    if other:
        tags.append(f"{other} other lines")
    return "; ".join(tags) or "no classified change"


def report_single_file(recs, path, grep=None, max_diff=40, compact=False):
    """Every schedule-matched pair whose only compiled difference is one file.

    This is the strongest controlled instrument the board can give for a
    one-file mechanism: everything else in the binary is identical and the
    draft schedule is bit-identical, so the plutarch contrast prices exactly
    the printed code diff.
    """

    groups = defaultdict(list)
    for rec in recs:
        key, own = identity_excluding(rec["ref"], path)
        if key is None:
            continue
        rec["_own"] = own
        groups[(key, rec["sig"])].append(rec)

    print(f"CONTROLLED SINGLE-FILE PAIRS on {path}")
    if grep:
        print(f"filtered to diffs mentioning {grep!r}")
    print()
    shown = 0
    for members in groups.values():
        variants = defaultdict(list)
        for m in members:
            variants[m["_own"]].append(m)
        if len(variants) < 2:
            continue
        keys = list(variants)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = variants[keys[i]][0]
                b = variants[keys[j]][0]
                diff = list(difflib.unified_diff(
                    canon_lines(a["ref"], path), canon_lines(b["ref"], path),
                    lineterm="", n=0))
                body = "\n".join(diff)
                if grep and grep not in body:
                    continue
                if len(diff) > max_diff:
                    continue
                target = cand_pct(a["pmap"], b["pmap"], TARGET_PROBE)
                draft = statistics.fmean(cand_pct(a["pmap"], b["pmap"], n)
                                         for n in DRAFT_PROBES)
                shown += 1
                if compact:
                    hunks = sum(1 for ln in diff if ln.startswith("@@"))
                    adds = sum(1 for ln in diff if ln.startswith("+")
                               and not ln.startswith("+++"))
                    dels = sum(1 for ln in diff if ln.startswith("-")
                               and not ln.startswith("---"))
                    print(f"{a['id8']} {b['id8']} {target:+9.4f} "
                          f"{draft:+9.4f} {hunks:3d} hunk {adds:3d}+ {dels:3d}- "
                          f"{a['solver'][:12]:>12}/{b['solver'][:12]:<12} "
                          f"{signature(diff)}")
                    continue
                print("-" * 74)
                print(f"A {a['id8']} {a['solver']:>14} {a['date']} "
                      f"({len(variants[keys[i]])} runs)")
                print(f"B {b['id8']} {b['solver']:>14} {b['date']} "
                      f"({len(variants[keys[j]])} runs)")
                print(f"TARGET {target:+.4f} %   DRAFT {draft:+.4f} %   "
                      f"B faster when positive")
                for line in diff[2:]:
                    print(f"  {line}")
    print("-" * 74)
    print(f"{shown} controlled single-file pairs")


def report_neighbours(recs, prefix, max_files=4):
    """Closest schedule-matched trees to one submission, with the file diffs."""
    hits = [r for r in recs if r["id8"].startswith(prefix)]
    if not hits:
        raise SystemExit(f"no scored board row with a fetched tree for {prefix}")
    anchor = hits[0]
    print("=" * 78)
    print(f"NEIGHBOURS of {anchor['id8']} {anchor['solver']} "
          f"published {anchor['score']} ({anchor['status']})")
    print("=" * 78)
    found = 0
    for rec in recs:
        if rec["id8"] == anchor["id8"] or rec["sig"] != anchor["sig"]:
            continue
        proc = git(["diff", "--numstat", anchor["ref"], rec["ref"], "--"]
                   + BUILD_PATHS)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) > max_files:
            continue
        found += 1
        target = cand_pct(anchor["pmap"], rec["pmap"], TARGET_PROBE)
        draft = statistics.fmean(cand_pct(anchor["pmap"], rec["pmap"], n)
                                 for n in DRAFT_PROBES)
        print(f"\n{rec['id8']} {rec['solver']:>14} {rec['date']} "
              f"published {rec['score']} ({rec['status']})")
        print(f"  TARGET {target:+.4f} % "
              f"({target / RES_TARGET_SAME_MODE:+.2f} sigma)   "
              f"DRAFT {draft:+.4f} %")
        for ln in lines:
            print(f"    {ln}")
        if not lines:
            print("    byte-identical scored surface")
    if not found:
        print("\nno schedule-matched tree within the file-count limit")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=BOARD_JSON)
    ap.add_argument("--mech", choices=["q1", "q2", "both"], default="both")
    ap.add_argument("--allow-schedule-mismatch", action="store_true")
    ap.add_argument("--neighbours", metavar="ID8")
    ap.add_argument("--max-files", type=int, default=4)
    ap.add_argument("--near", action="store_true")
    ap.add_argument("--max-extra", type=int, default=3)
    ap.add_argument("--single-file", metavar="PATH")
    ap.add_argument("--grep")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--max-diff", type=int, default=40)
    args = ap.parse_args(argv)

    recs = collect(load_rows(args.board))
    print(f"anchor {ANCHOR[:8]}   {len(recs)} scored submissions with a "
          f"fetched tree\n")
    if args.neighbours:
        report_neighbours(recs, args.neighbours, args.max_files)
        return 0
    if args.single_file:
        report_single_file(recs, args.single_file, args.grep,
                           max_diff=args.max_diff, compact=args.compact)
        return 0
    if args.near:
        for key in (["q1", "q2"] if args.mech == "both" else [args.mech]):
            report_near(recs, key, args.max_extra)
            print()
        return 0
    for key in (["q1", "q2"] if args.mech == "both" else [args.mech]):
        report_mechanism(recs, key,
                         require_schedule=not args.allow_schedule_mismatch)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
