"""Read an official receipt with the per-prompt candidate-leg instrument.

The problem this solves
-----------------------
The published score is `median_p( serial_p / candidate_p )`. Two independent
noise sources sit on top of any mechanism:

  * the serial lottery (Finding 20), an independent per-run draw on the
    denominator, worth about 0.277 % on the published median; and
  * the FACT-2 measurement mode, a binary run-level state worth about
    0.601 ms per DRAFTING round.

Both are large compared with almost every mechanism anyone has shipped. A board
rank is therefore a poor read on engineering, and even the serial-free rescoring
used by `research/board_pair_decompose.py` still carries the FACT-2 mode.

The instrument
--------------
Plutarch draws only 38 drafting rounds out of 487. Every other prompt drafts on
most rounds. So the FACT-2 mode, which lives in the drafting path, barely
touches plutarch's candidate leg and dominates everyone else's.

That makes two orthogonal probes out of one receipt:

    TARGET  = plutarch candidate leg          -> target runtime and kernels
    DRAFT   = mean of the five G=2 prompts    -> proposal head, selection,
                                                 schedule, and drafting cost

The replicate rule (Finding 46)
-------------------------------
A replicate pair is two runs of THE SAME COMPILED CODE. The default rule here
is comment-insensitive CODE IDENTITY, not the byte digest of the scored tree.

Each submission is diffed against one anchor tree over the scored surface. Each
changed file is canonicalised by removing comments, blank lines and trailing
whitespace, and the submission's identity is the set of files whose canonical
form still differs from the anchor. Two submissions are replicates when those
sets are equal. Anchor choice does not affect the comparison: two canonically
equal trees produce equal identity sets against any anchor.

Why the byte digest is the wrong default. Solvers resubmit identical code with
a comment added and label the row a resample. Those rows are deliberate
independent measurements of the same binary, and the byte digest throws every
one of them away. What survives the byte rule is a narrower and measurably
quieter population, so the byte rule reports a TARGET resolution about 3.0x too
tight.

Two exclusions, and both are load bearing.

1. JSON manifests are never comment-stripped. `mtp-head.manifest.json` has no
   comment syntax, and a `//` inside a declared head URL would be treated as a
   line comment, which would merge two submissions that declare DIFFERENT
   proposal heads into one identity.

2. The stripper is STRING-LITERAL AWARE, not a regex. This is not a
   precaution; the naive regex was measurably wrong on this corpus.
   `Vendor/mlx-swift/Source/Cmlx/mlx-generated/*.cpp` carries the Metal kernel
   source as a C++ string literal, and that literal contains `//` comments that
   are part of the JIT source string. The regex deleted them, which merged
   trees that JIT-compile DIFFERENT Metal source. Over 746 submissions the two
   strippers disagreed on 1477 of 6735 changed source blobs and on 744 of 746
   identities, and the regex manufactured 7 false replicate pairs out of 75.
   Reproduce with `--validate-canon`.

The rule is NOT "comments never matter". One case breaks that claim, and it is
the same file family as exclusion 2. A `//` comment inside the Metal preamble
string literal in `mlx-generated/*.cpp` is part of the text MLX hands to the
JIT: `mlx-generated/quantized.cpp:3` returns that literal, and
`jit_kernels.cpp` passes the built string to `Device::get_library`, which calls
`newLibrary(source, ...)` at `metal/device.cpp:622`. So the comment reaches the
Metal compiler even though the compiled GPU code is identical.

Two halves of that, and only the first is sourced here.

  * SOURCED. MLX's own in-process cache is keyed by the library NAME, not by
    the source text: `Device::get_library` looks up `library_map_.find(name)`
    at `metal/device.cpp:770-788`. So inside ONE worker process a comment
    change in the literal cannot cause an extra MLX-level compile.
  * NOT SOURCED FROM THIS CHECKOUT. Apple's driver-level compiled-shader cache
    is keyed on the source text the driver receives, and that cache lives
    outside MLX. If so, two trees differing only by such a comment miss each
    other's cached library and pay a real recompile on first use in a fresh
    process, which lands in warm-up rather than steady-state decode. Treat this
    as plausible and unproven; do not quote a magnitude for it.

The practical rule does not depend on settling the second half. The
string-aware stripper KEEPS those comments, so two such trees get DIFFERENT
identities and are never pooled as replicates, which is the safe behaviour
either way. A comment outside the literal in the same file, and a comment in a
`.metal` source compiled ahead of time into `mlx.metallib`, are both inert. Do
not "simplify" the stripper back to a regex: that is precisely the merge this
rule exists to prevent.

Measured resolution. Run `--noise` to reproduce all three replicate classes,
the class-by-time-gap split and the stratified variance-ratio tests.

    per-run candidate-leg sd, same measurement mode, in percent

    class                 pairs   same-mode   TARGET     DRAFT
    byte-identical           40          18   0.0431    0.1139
    comment-only diff        28          17   0.1281    0.0702
    all code-identical       68          35   0.0945    0.0952

    published median floor, for comparison:   0.2770 %

These constants were fitted on a 746-submission board. They were re-measured on
a later 764-submission board, 711 groups and 79 code-identical pairs, and every
one reproduced: TARGET 0.0916 against 0.0945, DRAFT 0.0931 against 0.0952,
TARGET-all 0.1047 against 0.1100, DRAFT-all 0.6393 against 0.6687. All four
agree within 5 %.

The `--noise` table prints four measurement columns, `T same`, `D same`,
`T all` and `D all`. Only the two `same` columns are resolutions. `D all` is
about 0.64 % because it CONTAINS the mode, and quoting it as a resolution
inflates the DRAFT floor about sevenfold. Read the `same` columns, and do not
transpose the two probes.

The widest class differs between the probes, so the conservative floor is taken
per probe: TARGET 0.1281 % per run and 0.1812 % per pair, DRAFT 0.1139 % per run
and 0.1611 % per pair. Use the conservative floor when a claim is about to spend
GPU or a submission slot. Use the all-code-identical point estimate, 0.0945 %
TARGET and 0.0952 % DRAFT, when reporting a measurement. State which one a
number used.

What the class split is and is not. Conditioning on the time gap does not
remove it: inside the well-populated under-3-hour stratum the two classes still
differ by a variance ratio of 11.7 at p = 0.0001. The reverse is not true.
Inside the byte-identical class the time gap does nothing, F = 0.76 at
p = 0.64, so the marginal "pairs over 3 hours apart are quieter" reading is an
artefact of class composition. The CAUSE of the class split is still open. It
is not a resample population: `--provenance` shows 39 of the 40 byte-identical
pairs are two DIFFERENT solvers submitting the same scored tree, with different
`submissionCommitSha` on every side.

Two things follow.

1. Plutarch is the sharpest target-path instrument on the board, and it is
   nearly immune to the mode: the mode inflates the drafting probe 8.57x but
   plutarch only 2.04x.

2. The mode is DETECTABLE inside a single pair, and the two states do not
   overlap. Across 68 replicate pairs the largest same-mode |DRAFT| is
   0.4907 % and the smallest cross-mode |DRAFT| is 0.9031 %, an empty band
   0.41 % wide. The 0.60 % cut therefore classifies every pair without
   ambiguity, and it sits 4.5 same-mode pair sd from zero, so conditioning on
   it truncates a negligible share of the same-mode population. When two runs
   share a mode the drafting probe is a 0.0952 % instrument, 2.9x sharper than
   the published median.

The predicted plutarch mode shift is 38 drafting rounds x 0.601 ms over a
15.5 s plutarch leg, or 0.147 %. The measured cross-mode plutarch pair RMS is
0.1244 %. That agreement, from an entirely independent direction, is the
strongest confirmation of FACT 2 the campaign has.

Why the TARGET mode cut is 0.30 and not 0.15
--------------------------------------------
The mode classifier is a conjunction: a pair is a mode flip when the DRAFT
probe exceeds `MODE_DRAFT_SHIFT` AND the TARGET probe stays under
`MODE_TARGET_SHIFT`. The DRAFT half is safe, because the same-mode and
cross-mode DRAFT populations are separated by an empty 0.41 % band. The TARGET
half was not.

The same-mode TARGET sd is 0.0945 % per run, so a PAIR of runs carries
0.0945 x sqrt(2) = 0.1336 %. A 0.15 cut therefore sat at 1.12 pair sd from
zero. A two-sided normal at 1.12 sd leaves about 26 % of genuine same-mode mass
outside the cut, so roughly one mode flip in four was silently reclassified as
"mode flip plus a target mechanism" purely from TARGET noise. The `xv4` receipt
landed at 1.20 and 1.28 pair sd against its two references and was misread on
exactly this margin.

The cut is now `MODE_TARGET_SHIFT = 0.30`, which is 2.25 pair sd and leaves
about 2.5 % of same-mode mass outside. The band between the two values is not
discarded: `MODE_TARGET_AMBIGUOUS = 0.15` marks it, and a pair that lands there
is reported as `mode_ambiguous`. An ambiguous pair is not evidence of a target
mechanism and it is not evidence against one. It needs a second receipt or the
decomposition below.

The decomposition (CAMPAIGN RULE 60)
------------------------------------
A single mode-flipped receipt cannot separate a small target mechanism from the
mode, because both land on the same eight numbers. What CAN separate them is
that the two effects have different SHAPES across the prompts. The mode is paid
per drafting round, so it scales with `drafting_rounds_p / leg_p`. A mechanism
inside the target path is paid on every round, so to first order it is a flat
percentage. That gives a two-parameter ordinary least squares fit:

    slower_p (%) = 100 * m * drafting_rounds_p / leg_p + c

`m` is milliseconds per drafting round and should land near the FACT-2 value of
0.601 ms when a mode flip is present. `c` is the flat, prompt-independent
component, which is the only part attributable to a mechanism.

`mode_decompose()` returns `m, se_m, c, se_c, r2, rms` and the per-prompt
residuals. `--read` prints it for every pair.

Read `c` against ZERO, with a +/- 0.2335 % systematic band.

An earlier version of this file told you to read `c` against a single
calibration pair, `51b9bf85` / `097991a0`, whose fit returns `c = +0.2255`. That
instruction was wrong and is withdrawn. Over the 26 byte-identical cross-mode
pairs on the board, where the true `c` is exactly 0 by construction, the fitted
mean is `c = -0.0576 +/- 0.0444 %`, which is consistent with zero. The 0.2255
figure was one draw from a zero-mean distribution, not a bias.

What IS real is the scatter. Across those 26 pairs the fitted `c` has an rms of
0.2335 % (sd 0.2263, range -0.53 to +0.38). So:

    score c against 0, and treat |c| under about 0.47 % (2 sigma) as
    consistent with no mechanism at all

The corresponding code-identical cross-mode class gives `c = -0.1285 +/- 0.0400`
with rms 0.2808 %, so a 2 sigma band of about 0.56 % is the conservative read.

Two robustness results that fix the fit's form:

* The regressor MUST be drafting rounds. Using total ranked rounds instead
  returns `c = +1.998 +/- 0.100 %` on a null population, which would make every
  mode-flipped pair look like a two-percent mechanism.
* Weighted least squares tightens the scatter to 0.2123 % but introduces a
  -0.139 +/- 0.032 % bias. Keep ordinary least squares.

KNOWN MISSPECIFICATION AT PLUTARCH. The linear-in-drafting-rounds model predicts
a cross-mode plutarch shift of +0.147 % at `m = 0.601` and +0.215 % at the
fitted `m`. The measured shift is +0.0056 +/- 0.0263 %, which rejects those
predictions at 5.6 and 8 sigma. The other seven prompts shift by +1.19 to
+2.17 % as the model expects. Plutarch has high leverage in this regressor, so
its unmodelled near-immunity is the mechanical origin of the 0.2335 % scatter
above. Until a better regressor is fitted, prefer plutarch ALONE as the
mode-immune probe wherever the mechanism reaches the target path: its cross-mode
pair sd is 0.1643 % against the decomposition's 0.2808 %, and its bias is
+0.006 % against -0.129 %. Plutarch is target-path only, so a drafting-side
mechanism still needs the DRAFT probe or this decomposition.

The regressor needs a per-prompt drafting-round count. The board's `per_prompt`
block does NOT carry one: it has `non_drafting_round_count` and
`effective_mean_draft_len` but no total round count. `RANKED_ROUNDS` below
supplies the Finding 12 counts. Those counts were measured on one ranked run,
so they set the SHAPE of the regressor and not its absolute level; `m` is
correspondingly a relative quantity and should not be quoted as a hardware
constant. On the byte-identical cross-mode population it fits to
+0.8769 +/- 0.0482 ms per drafting round, above the FACT-2 value of 0.601 ms,
which is the same misspecification seen from the other side.

Measured resolution of one receipt (2 sigma, conservative replicate class)
--------------------------------------------------------------------------
These replace the campaign's earlier guessed 0.5 % submission bar. A uniform
x % candidate speedup moves every candidate leg, every `raw_p` and the median
by x %, so all estimators share one scale.

    mode SHARED, read on the candidate mean of 8 legs      0.216 %
    mode SHARED, read on the published median              0.698 %
    mode UNKNOWN, read on plutarch alone                   0.387 %
    mode UNKNOWN, read on the published median             2.104 %
    mode REMOVED by this decomposition                     0.702 %
    mode FLIPPED, read on plutarch alone                   0.450 %

Empirical false-positive rates at a 0.5 % decision threshold, measured on null
pairs: the published median exceeds 0.5 % on 9.8 % of same-mode nulls and on
53.8 % of unknown-mode nulls. The candidate mean-of-8 under a shared mode, and
plutarch alone under an unknown mode, exceed 0.5 % on 0 % of null pairs.

Never decide a mechanism from the published median.

Sign convention
---------------
A POSITIVE percentage means B is FASTER than A, matching
`research/board_pair_decompose.py`.

Usage
-----
    python3 research/board_prompt_instrument.py --noise
        Re-measure both probe resolutions over all three replicate classes,
        the class-by-time-gap and class-by-solver splits, the stratified
        variance-ratio tests, and the mode separation. Add `--byte-digest` to
        fall back to the old, narrower replicate rule.
        Needs `git fetch upstream 'refs/heads/submissions/*:...'` first.

    python3 research/board_prompt_instrument.py --validate-canon
        Compare the string-aware stripper with the naive regex and report
        every blob, identity and replicate pair they disagree on.

    python3 research/board_prompt_instrument.py --provenance
        Print `submissionCommitSha`, `createdAt`, `solverUsername`,
        `promotionStatus` and `status` for the byte-identical pairs, which is
        how you check what that replicate class actually contains.

    python3 research/board_prompt_instrument.py --read <a_prefix> <b_prefix>
        Read one pair through the instrument: mode classification, target-path
        effect, drafting-path effect, each in resolution units, and the
        two-parameter mode/mechanism decomposition required by CAMPAIGN
        RULE 60.

    python3 research/board_prompt_instrument.py --rank --min-score 3.30
        Rank the largest schedule-matched cohort on each probe separately.

Reads /tmp/yukon-board/full.json, or $YUKON_BOARD_JSON.
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

BOARD_JSON = os.environ.get("YUKON_BOARD_JSON", "/tmp/yukon-board/full.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.environ.get("BOARD_ANCHOR",
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

# Measured per-run candidate-leg resolution, in percent, over ALL
# code-identical replicate pairs. See the module docstring for provenance.
# Re-measure with --noise after the board grows.
RESOLUTION = {
    "target_all": 0.1100,
    "target_same_mode": 0.0945,
    "draft_all": 0.6687,
    "draft_same_mode": 0.0952,
}
# The CONSERVATIVE floor is the WIDEST single replicate class, per probe, not
# the pooled point estimate. The widest class differs between the two probes:
# comment-only for TARGET, byte-identical for DRAFT. Spend GPU or a submission
# slot against these numbers. The advisor adopted 0.1196 / 0.1691 for TARGET
# before the string-aware canonicalisation removed seven false replicates.
CONSERVATIVE = {
    "target_per_run": 0.1281, "target_per_pair": 0.1812,
    "draft_per_run": 0.1139, "draft_per_pair": 0.1611,
}
# A mode flip moves the drafting probe by about this much and plutarch by far
# less. Anything above the first number is a flip, not a mechanism.
MODE_DRAFT_SHIFT = 0.60
# MODE_TARGET_SHIFT was 0.15 and misclassified the `7bef7d4c` receipt.
# The same-mode TARGET floor is 0.0945 % per RUN, so a PAIR difference carries
# sqrt(2) times that, 0.1336 %. A threshold of 0.15 therefore sat at 1.12 pair
# sigma, and any true mode flip whose plutarch reading landed beyond 1.1 sigma
# of zero was silently reported as "no flip". `7bef7d4c` landed at 1.20 and
# 1.28 sigma against its two references and was missed by 0.01 and 0.02
# percentage points. The threshold is now 2.25 pair sigma, and readings between
# the two constants are reported as AMBIGUOUS rather than as a clean negative.
MODE_TARGET_SHIFT = 0.30
MODE_TARGET_AMBIGUOUS = 0.15

# Ranked round count per prompt, from FINDING 12's M5 cost curve. Used only to
# weight the two regressors in `mode_decompose` relative to each other. Two
# rows with a bit-identical schedule signature share these counts exactly, so
# an error in the absolute value shifts the units of `m` without disturbing the
# separation of `m` from `c`. `report_read` warns when the schedules differ.
RANKED_ROUNDS = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
                 "republic": 93, "essays": 92, "medicine": 90, "botany": 81}

# Comment stripping applies only to languages that have `//` or `/* */`.
# A JSON manifest has neither, and a `//` inside a declared proposal-head URL
# would be eaten as a line comment, merging two DIFFERENT heads into one
# identity. See the module docstring.
SOURCE_SUFFIXES = (".swift", ".h", ".hpp", ".cpp", ".c", ".cc", ".metal",
                   ".m", ".mm")
BUILD_PATHS = ["Sources", "Vendor", "Package.swift", "Package.resolved",
               "tools", "mtp-head.manifest.json", "mtp-head"]
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//.*$", re.M)
ZERO = "0" * 40


# --- replicate identity -----------------------------------------------------

def git(args):
    return subprocess.run(["git"] + args, cwd=REPO, capture_output=True,
                          text=True, errors="replace")


_blob_cache = {}


def blob_text(oid):
    if oid not in _blob_cache:
        proc = git(["cat-file", "blob", oid])
        _blob_cache[oid] = proc.stdout if proc.returncode == 0 else ""
    return _blob_cache[oid]


def strip_comments_naive(text):
    """Regex stripper. Wrong inside a string literal that contains `//`."""
    text = BLOCK_COMMENT.sub("", text)
    text = LINE_COMMENT.sub("", text)
    return text


def strip_comments_aware(text):
    """Scanner that never strips inside a Swift, C or Metal string literal.

    A `//` inside `MLXFast.metalKernel(source: "...")` is kernel source, not a
    comment. The naive regex would delete the rest of that line including the
    closing quote, merging two trees that compile different kernels. Handles
    `"..."`, Swift `\"\"\"..."\"\"`, Swift raw `#"..."#`, and `'...'`.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if ch == "#" and text.startswith('#"', i):
            hashes = 1
            while i - hashes >= 0 and text[i - hashes] == "#":
                hashes += 1
            close = '"' + "#" * hashes
            j = text.find(close, i + 1 + hashes)
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            i = j
            continue
        if text.startswith('"""', i):
            j = text.find('"""', i + 3)
            j = n if j < 0 else j + 3
            out.append(text[i:j])
            i = j
            continue
        if ch in ('"', "'"):
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(text[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def canon_code(text, path, aware=True):
    """Remove the text the compiler discards. Comments cannot cost time."""
    if not path.endswith(SOURCE_SUFFIXES):
        return text
    stripped = (strip_comments_aware(text) if aware
                else strip_comments_naive(text))
    return "\n".join(ln.rstrip() for ln in stripped.split("\n") if ln.strip())


_canon_cache = {}


def canon_digest(oid, path, aware=True):
    key = (oid, path, aware)
    if key not in _canon_cache:
        canon = canon_code(blob_text(oid), path, aware)
        _canon_cache[key] = hashlib.sha256(canon.encode()).hexdigest()[:16]
    return _canon_cache[key]


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


def code_identity(ref, aware=True):
    """Comment-insensitive identity of one submission's scored surface.

    The identity is the set of (path, canonical digest) entries that still
    differ from the anchor. Two canonically equal trees produce equal sets
    against any anchor, so the anchor choice does not affect the comparison.
    """
    changed = changed_blobs(ref)
    if changed is None:
        return None
    items = []
    for path, (old_oid, new_oid) in sorted(changed.items()):
        if new_oid == ZERO:
            items.append((path, "DELETED"))
            continue
        new_digest = canon_digest(new_oid, path, aware)
        if old_oid != ZERO and canon_digest(old_oid, path, aware) == new_digest:
            continue  # comment-only difference
        items.append((path, new_digest))
    return frozenset(items)


def anchor_canon_lines(path):
    """Canonical lines of one path in the campaign anchor tree."""
    proc = git(["show", f"{ANCHOR}:{path}"])
    if proc.returncode != 0:
        return []
    return canon_code(proc.stdout, path).split("\n")


def byte_identity(ref):
    """Byte digest of the scored surface. The narrower, stricter old rule."""
    listing = git(["ls-tree", ref] + BUILD_PATHS)
    if listing.returncode != 0:
        return None
    return listing.stdout.strip()


def load_rows(path=BOARD_JSON):
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        for key in ("submissions", "rows", "data", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"unexpected board payload shape in {path}")
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


def probes(pmap_a, pmap_b):
    target = cand_pct(pmap_a, pmap_b, TARGET_PROBE)
    draft = statistics.fmean(cand_pct(pmap_a, pmap_b, n) for n in DRAFT_PROBES)
    return target, draft


def rms(values):
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def per_run_sd(values, floor=3):
    """Per-run sd from paired differences: a pair carries twice the variance."""
    if len(values) < floor:
        return float("nan")
    return rms(values) / math.sqrt(2)


def collect(rows):
    """Board rows that have a usable per-prompt block and a public branch."""
    refs = git(["for-each-ref", "--format=%(refname:short)",
                "refs/remotes/upstream/submissions/"]).stdout.split()
    by_id = {r.rsplit("/", 1)[-1]: r for r in refs}
    recs = []
    for row in rows:
        ref = by_id.get(row.get("id") or "")
        pmap = prompt_map(row)
        if ref is None or pmap is None or row.get("officialScore") is None:
            continue
        recs.append({
            "id8": (row.get("id") or "")[:8], "ref": ref, "row": row,
            "pmap": pmap, "sig": schedule_signature(pmap),
            "solver": row.get("solverUsername") or "",
            "score": row.get("officialScore"),
            "date": (row.get("createdAt") or "")[:16].replace("T", " "),
        })
    return recs


def hours_apart(rec_a, rec_b):
    fmt = "%Y-%m-%d %H:%M"
    ta = _dt.datetime.strptime(rec_a["date"], fmt)
    tb = _dt.datetime.strptime(rec_b["date"], fmt)
    return abs((tb - ta).total_seconds()) / 3600.0


def replicate_pairs(recs, byte_digest=False):
    """Every within-group pair under the chosen replicate rule.

    Grouping also requires a bit-identical eight-prompt schedule, so a pair can
    never mix two drafting policies.
    """
    identity = byte_identity if byte_digest else code_identity
    groups = defaultdict(list)
    for rec in recs:
        key = identity(rec["ref"])
        if key is None:
            continue
        groups[(key, rec["sig"])].append(rec)

    pairs = []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                target, draft = probes(a["pmap"], b["pmap"])
                bytes_equal = not git(
                    ["diff", "--name-only", a["ref"], b["ref"], "--"]
                    + BUILD_PATHS).stdout.split()
                pairs.append({
                    "a": a, "b": b, "target": target, "draft": draft,
                    "bytes_equal": bytes_equal,
                    "same_solver": a["solver"] == b["solver"],
                    "gap": hours_apart(a, b),
                    "same_mode": abs(draft) <= MODE_DRAFT_SHIFT,
                })
    return groups, pairs


def _betacf(a, b, x, iters=200):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / (c if abs(c) > tiny else tiny)
            h *= d * c
    return h


def betainc(a, b, x):
    """Regularised incomplete beta, so an F tail needs no SciPy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_two_sided_p(var1, df1, var2, df2):
    """Two-sided p for var1 == var2. Sums of squares carry full df here.

    These per-run sd figures are RMS with no mean removed, so a class of n
    pairs contributes n degrees of freedom rather than n-1.
    """
    if not (var1 > 0 and var2 > 0 and df1 > 0 and df2 > 0):
        return float("nan")
    f = var1 / var2
    upper = betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))
    return min(1.0, 2.0 * min(upper, 1.0 - upper))


FLOOR_HEADER = (f"{'group':>26} {'pairs':>6} {'same':>5} "
                f"{'T same':>8} {'D same':>8} {'T all':>8} {'D all':>8}")


def _floor_row(label, sub):
    same = [p for p in sub if p["same_mode"]]
    print(f"{label:>26} {len(sub):6d} {len(same):5d} "
          f"{per_run_sd([p['target'] for p in same]):8.4f} "
          f"{per_run_sd([p['draft'] for p in same]):8.4f} "
          f"{per_run_sd([p['target'] for p in sub]):8.4f} "
          f"{per_run_sd([p['draft'] for p in sub]):8.4f}")


def report_noise(rows, byte_digest=False):
    recs = collect(rows)
    groups, pairs = replicate_pairs(recs, byte_digest)
    rule = ("byte digest of the scored surface" if byte_digest
            else "comment-insensitive code identity")
    replicated = sum(1 for m in groups.values() if len(m) > 1)
    print(f"replicate rule: {rule}")
    print(f"{len(recs)} usable board rows, {len(groups)} (identity, schedule) "
          f"groups, {replicated} replicated, {len(pairs)} pairs\n")
    if len(pairs) < 4:
        print("not enough replicate pairs to measure resolution")
        return

    print("Per-run candidate-leg sd in percent. A pair difference carries "
          "twice the\nvariance of one run, so every cell is pairRMS / sqrt(2). "
          "TARGET is plutarch\nalone; DRAFT is the mean of the five G=2 "
          "prompts. `same` counts same-mode pairs.\n")
    byte_pairs = [p for p in pairs if p["bytes_equal"]]
    comment_pairs = [p for p in pairs if not p["bytes_equal"]]
    classes = [("byte-identical", byte_pairs),
               ("comment-only diff", comment_pairs),
               ("all code-identical", pairs)]

    print("--- replicate class")
    print(FLOOR_HEADER)
    for label, sub in classes:
        _floor_row(label, sub)

    print("\n--- 2x2: replicate class by time gap")
    print(FLOOR_HEADER)
    for label, sub in classes:
        for gap_label, keep in (("< 3 h", lambda p: p["gap"] < 3),
                                (">= 3 h", lambda p: p["gap"] >= 3)):
            _floor_row(f"{label}, {gap_label}", [p for p in sub if keep(p)])

    print("\n--- 2x2: replicate class by solver identity")
    print(FLOOR_HEADER)
    for label, sub in classes:
        for s_label, keep in (("same solver", lambda p: p["same_solver"]),
                              ("diff solver", lambda p: not p["same_solver"])):
            _floor_row(f"{label}, {s_label}", [p for p in sub if keep(p)])

    print("\n--- marginals this replaces")
    print(FLOOR_HEADER)
    for label, keep in (("same solver", lambda p: p["same_solver"]),
                        ("diff solver", lambda p: not p["same_solver"]),
                        ("< 3 h", lambda p: p["gap"] < 3),
                        (">= 3 h", lambda p: p["gap"] >= 3)):
        _floor_row(label, [p for p in pairs if keep(p)])

    print("\n--- is the split replicate class or time gap? stratified "
          "variance-ratio tests")
    print("Each test holds one factor fixed and varies the other, on the "
          "same-mode\nTARGET probe. F is the variance ratio, p is two-sided.\n")
    print(f"{'held fixed':>20} {'contrast':>28} {'n1':>4} {'n2':>4} "
          f"{'F':>7} {'p':>8}")
    tests = [
        ("gap < 3 h", "comment-only vs byte-identical",
         [p for p in comment_pairs if p["gap"] < 3],
         [p for p in byte_pairs if p["gap"] < 3]),
        ("gap >= 3 h", "comment-only vs byte-identical",
         [p for p in comment_pairs if p["gap"] >= 3],
         [p for p in byte_pairs if p["gap"] >= 3]),
        ("byte-identical", "< 3 h vs >= 3 h",
         [p for p in byte_pairs if p["gap"] < 3],
         [p for p in byte_pairs if p["gap"] >= 3]),
        ("comment-only", "< 3 h vs >= 3 h",
         [p for p in comment_pairs if p["gap"] < 3],
         [p for p in comment_pairs if p["gap"] >= 3]),
        ("nothing", "comment-only vs byte-identical",
         comment_pairs, byte_pairs),
        ("nothing", "< 3 h vs >= 3 h",
         [p for p in pairs if p["gap"] < 3],
         [p for p in pairs if p["gap"] >= 3]),
    ]
    for held, contrast, sub1, sub2 in tests:
        v1 = [p["target"] for p in sub1 if p["same_mode"]]
        v2 = [p["target"] for p in sub2 if p["same_mode"]]
        if not v1 or not v2:
            continue
        s1, s2 = per_run_sd(v1, 1), per_run_sd(v2, 1)
        p_val = f_two_sided_p(s1 * s1, len(v1), s2 * s2, len(v2))
        print(f"{held:>20} {contrast:>28} {len(v1):4d} {len(v2):4d} "
              f"{(s1 * s1) / (s2 * s2):7.2f} {p_val:8.4f}")

    print("\n--- mode separation, which decides whether `D same` is usable")
    mags = sorted(abs(p["draft"]) for p in pairs)
    below = [v for v in mags if v <= MODE_DRAFT_SHIFT]
    above = [v for v in mags if v > MODE_DRAFT_SHIFT]
    print(f"|DRAFT| below the {MODE_DRAFT_SHIFT} % cut, largest five: "
          + ", ".join(f"{v:.4f}" for v in below[-5:]))
    print(f"|DRAFT| above the cut, smallest five:      "
          + ", ".join(f"{v:.4f}" for v in above[:5]))
    if below and above:
        sd_below = per_run_sd(below, 1) * math.sqrt(2)
        print(f"gap across the cut {above[0] - below[-1]:.4f} %; the cut sits "
              f"{MODE_DRAFT_SHIFT / sd_below:.1f} same-mode pair sd from zero")
        print("A cut that far into the tail truncates a negligible share of "
              "the same-mode\npopulation, so `D same` is a usable resolution "
              "and not an artefact of the cut.")

    print("\nCaveat on the DRAFT columns. The same-mode filter is a cut on the "
          "DRAFT\nprobe itself, so `D same` is conditioned on its own value. "
          "The separation\ncheck above bounds that bias. `D all` mixes in the "
          "FACT-2 mode flip, which is\na real run-level state rather than "
          "measurement noise.")
    print("\n--- floors, measured now against the constants in use")
    print(f"{'probe':>8} {'basis':>22} {'per run':>9} {'per pair':>9} "
          f"{'in use':>9}")
    live = {
        ("TARGET", "conservative"): max(
            per_run_sd([p["target"] for p in sub if p["same_mode"]], 1)
            for _, sub in classes[:2]),
        ("TARGET", "point estimate"): per_run_sd(
            [p["target"] for p in pairs if p["same_mode"]], 1),
        ("DRAFT", "conservative"): max(
            per_run_sd([p["draft"] for p in sub if p["same_mode"]], 1)
            for _, sub in classes[:2]),
        ("DRAFT", "point estimate"): per_run_sd(
            [p["draft"] for p in pairs if p["same_mode"]], 1),
    }
    in_use = {
        ("TARGET", "conservative"): CONSERVATIVE["target_per_run"],
        ("TARGET", "point estimate"): RESOLUTION["target_same_mode"],
        ("DRAFT", "conservative"): CONSERVATIVE["draft_per_run"],
        ("DRAFT", "point estimate"): RESOLUTION["draft_same_mode"],
    }
    for key, value in live.items():
        basis = f"{key[1]}, same mode"
        print(f"{key[0]:>8} {basis:>22} {value:9.4f} "
              f"{value * math.sqrt(2):9.4f} {in_use[key]:9.4f}")
    print("\nconservative = the widest single replicate class for that probe.")
    print("Use the conservative floor to decide whether to spend GPU or a "
          "submission\nslot. Use the point estimate when reporting a "
          "measurement. Always state\nwhich one a number used. If `in use` "
          "has drifted from the measured column,\nupdate RESOLUTION and "
          "CONSERVATIVE at the top of this file.")


def report_validate_canon(rows):
    """Prove the comment stripper never eats a string literal on this corpus.

    A `//` inside a kernel source string is code. The naive regex stripper
    would delete it and merge two trees that compile different kernels. The
    string-aware scanner cannot. If the two strippers partition the corpus
    identically, the risk is absent here rather than merely unlikely.
    """
    recs = collect(rows)
    print(f"{len(recs)} submissions; comparing the string-aware stripper with "
          f"the naive regex\n")

    blob_diff, blobs_seen = [], 0
    for rec in recs:
        changed = changed_blobs(rec["ref"]) or {}
        for path, (_, new_oid) in changed.items():
            if new_oid == ZERO or not path.endswith(SOURCE_SUFFIXES):
                continue
            blobs_seen += 1
            if canon_digest(new_oid, path, True) != canon_digest(new_oid, path,
                                                                 False):
                blob_diff.append((rec["id8"], path))

    print(f"changed source blobs inspected        {blobs_seen}")
    print(f"blobs where the two strippers differ  {len(blob_diff)}")
    for id8, path in blob_diff[:20]:
        print(f"  {id8}  {path}")

    ident_diff = sum(1 for rec in recs
                     if code_identity(rec["ref"], True)
                     != code_identity(rec["ref"], False))
    print(f"submissions whose identity differs    {ident_diff}")

    _, aware_pairs = replicate_pairs(recs)
    naive_groups = defaultdict(list)
    for rec in recs:
        key = code_identity(rec["ref"], False)
        if key is not None:
            naive_groups[(key, rec["sig"])].append(rec)
    naive_n = sum(len(m) * (len(m) - 1) // 2 for m in naive_groups.values())
    print(f"replicate pairs, string-aware rule    {len(aware_pairs)}")
    print(f"replicate pairs, naive regex rule     {naive_n}")
    verdict = ("IDENTICAL. The naive rule was safe on this corpus, and the "
               "aware rule\nremoves the risk for future trees."
               if len(aware_pairs) == naive_n and not blob_diff
               else "DIFFERENT. Use the string-aware rule; the naive rule "
                    "merged distinct trees.")
    print(f"\nverdict: {verdict}")


def report_provenance(rows, limit=12):
    """How the byte-identical replicate pairs were actually created.

    A byte-identical pair is two board rows whose scored surfaces have the same
    content. This prints the fields that show whether they are two runs of one
    commit or two commits with the same content.
    """
    recs = collect(rows)
    _, pairs = replicate_pairs(recs)
    byte_pairs = [p for p in pairs if p["bytes_equal"]]
    print(f"{len(byte_pairs)} byte-identical replicate pairs; "
          f"showing the first {min(limit, len(byte_pairs))} by date\n")

    def field(rec, key):
        return rec["row"].get(key)

    for pair in sorted(byte_pairs, key=lambda p: p["a"]["date"])[:limit]:
        print(f"pair  target {pair['target']:+.4f} %  draft "
              f"{pair['draft']:+.4f} %  gap {pair['gap']:.2f} h")
        for side in ("a", "b"):
            rec = pair[side]
            sha = field(rec, "submissionCommitSha")
            print(f"  {side.upper()} {rec['id8']}  {rec['date']}  "
                  f"{rec['solver']:>14}  score {rec['score']:.8f}\n"
                  f"     submissionCommitSha {sha}\n"
                  f"     promotionStatus     {field(rec, 'promotionStatus')}  "
                  f"status {field(rec, 'status')}")
        sha_a = field(pair["a"], "submissionCommitSha")
        sha_b = field(pair["b"], "submissionCommitSha")
        if sha_a is None or sha_b is None:
            verdict = "UNKNOWN, the field is absent on one side"
        else:
            verdict = "YES" if sha_a == sha_b else "NO"
        print(f"     same submissionCommitSha: {verdict}\n")

    shas = defaultdict(int)
    for pair in byte_pairs:
        for side in ("a", "b"):
            shas[pair[side]["row"].get("submissionCommitSha")] += 1
    missing = shas.get(None, 0)
    print(f"submissionCommitSha present on "
          f"{sum(v for k, v in shas.items() if k is not None)} pair sides, "
          f"absent on {missing}")

    same_solver = sum(1 for p in byte_pairs
                      if p["a"]["solver"] == p["b"]["solver"])
    both_shas = [p for p in byte_pairs
                 if field(p["a"], "submissionCommitSha")
                 and field(p["b"], "submissionCommitSha")]
    same_sha = sum(1 for p in both_shas
                   if field(p["a"], "submissionCommitSha")
                   == field(p["b"], "submissionCommitSha"))
    mixed_outcome = sum(
        1 for p in byte_pairs
        if field(p["a"], "promotionStatus") != field(p["b"], "promotionStatus"))
    print(f"\nover all {len(byte_pairs)} byte-identical pairs\n"
          f"  same solver on both sides          {same_solver}\n"
          f"  both sides carry a commit sha      {len(both_shas)}\n"
          f"  of those, the same commit sha      {same_sha}\n"
          f"  sides disagree on promotionStatus  {mixed_outcome}\n"
          "\nA byte-identical pair is therefore not a solver resample. It is\n"
          "usually two different solvers submitting the same scored tree from\n"
          "different commits, so the pair is two independent measurements.")


def find_row(rows, prefix):
    hits = [r for r in rows if (r.get("id") or "").startswith(prefix)]
    if not hits:
        raise SystemExit(f"no board row with id prefix {prefix}")
    if len(hits) > 1:
        raise SystemExit(f"{prefix} is ambiguous over {len(hits)} rows")
    return hits[0]


def mode_decompose(pmap_a, pmap_b):
    """Separate the FACT-2 measurement mode from a uniform mechanism.

        slower_p (%) = 100 * m * drafting_rounds_p / leg_p  +  c

    `m` is the FACT-2 cost per drafting round in seconds; `c` is a uniform
    percentage that scales with leg time, which is what a kernel mechanism
    looks like. The two regressors separate because their shapes differ across
    the eight prompts: plutarch runs about 38 drafting rounds in a 15.5 s leg
    while botany runs 81 in a 5.6 s leg.

    The `c` estimate carries a systematic offset of roughly +0.22 % whenever a
    mode flip is present, measured on the byte-identical `51b9bf85`/`097991a0`
    pair whose true `c` is exactly zero. Read `c` against that offset, not
    against zero.
    """
    xs, ys = [], []
    for name in PROMPT_ORDER:
        pa, pb = pmap_a[name], pmap_b[name]
        ta = pa["mtp_seconds_per_token_mean"]
        tb = pb["mtp_seconds_per_token_mean"]
        drafting = RANKED_ROUNDS[name] - (pa.get("non_drafting_round_count") or 0)
        xs.append(100.0 * drafting / (tb * 512.0))
        ys.append(100.0 * (ta / tb - 1.0))

    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    m = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    c = mean_y - m * mean_x
    resid = [y - (m * x + c) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - mean_y) ** 2 for y in ys)
    s2 = sse / (n - 2)
    return {
        "m": m, "se_m": (s2 / sxx) ** 0.5,
        "c": c, "se_c": (s2 * (1.0 / n + mean_x ** 2 / sxx)) ** 0.5,
        "r2": 1.0 - sse / sst if sst else float("nan"),
        "rms": (sse / n) ** 0.5,
        "resid": dict(zip(PROMPT_ORDER, resid)),
    }


def report_read(rows, prefix_a, prefix_b):
    row_a, row_b = find_row(rows, prefix_a), find_row(rows, prefix_b)
    pmap_a, pmap_b = prompt_map(row_a), prompt_map(row_b)
    if pmap_a is None or pmap_b is None:
        raise SystemExit("one of the rows has no usable per-prompt block")

    same_schedule = schedule_signature(pmap_a) == schedule_signature(pmap_b)
    target, draft = probes(pmap_a, pmap_b)
    draft_moved = abs(draft) > MODE_DRAFT_SHIFT
    mode_flip = draft_moved and abs(target) < MODE_TARGET_AMBIGUOUS
    mode_ambiguous = (draft_moved
                      and MODE_TARGET_AMBIGUOUS <= abs(target)
                      < MODE_TARGET_SHIFT)

    print(f"A {(row_a.get('id') or '')[:8]}  {row_a.get('solverUsername')}  "
          f"published {row_a.get('officialScore')}")
    print(f"B {(row_b.get('id') or '')[:8]}  {row_b.get('solverUsername')}  "
          f"published {row_b.get('officialScore')}")
    print(f"\nschedule bit-identical on all eight prompts: "
          f"{'YES' if same_schedule else 'NO'}")
    if not same_schedule:
        print("  A drafting-policy difference is present. The probes below mix\n"
              "  the schedule change with everything else and are NOT a clean\n"
              "  mechanism measurement.")

    print(f"\n{'prompt':>9} {'A s/tok':>12} {'B s/tok':>12} {'B faster %':>11}")
    for name in PROMPT_ORDER:
        print(f"{name:>9} {pmap_a[name]['mtp_seconds_per_token_mean']:12.8f} "
              f"{pmap_b[name]['mtp_seconds_per_token_mean']:12.8f} "
              f"{cand_pct(pmap_a, pmap_b, name):+11.4f}")

    key = "same_mode" if not (mode_flip or mode_ambiguous) else "all"
    t_res = RESOLUTION[f"target_{key}"]
    d_res = RESOLUTION[f"draft_{key}"]
    if mode_flip:
        verdict = "FLIPPED between the two runs"
    elif mode_ambiguous:
        verdict = "AMBIGUOUS, a flip and a mechanism both fit"
    else:
        verdict = "no flip detected"
    print(f"\nFACT-2 measurement mode: {verdict}")
    if mode_flip:
        print("  The drafting probe moved more than "
              f"{MODE_DRAFT_SHIFT} % while plutarch stayed under "
              f"{MODE_TARGET_AMBIGUOUS} %.\n"
              "  Treat the drafting probe as uninformative and read plutarch "
              "only.")
    if mode_ambiguous:
        print("  AMBIGUOUS. The drafting probe moved more than "
              f"{MODE_DRAFT_SHIFT} % and plutarch landed between "
              f"{MODE_TARGET_AMBIGUOUS} % and {MODE_TARGET_SHIFT} %, which is "
              "1.1 to 2.25\n  pair sigma of the measured TARGET floor. A mode "
              "flip and a mechanism both fit. Read the decomposition below "
              "and do not\n  attribute the drafting move to a mechanism on "
              "the strength of this pair alone.")

    print(f"\n{'probe':>8} {'effect %':>10} {'resolution':>11} {'sigma':>8}")
    print(f"{'TARGET':>8} {target:+10.4f} {t_res:11.4f} {target / t_res:+8.2f}")
    print(f"{'DRAFT':>8} {draft:+10.4f} {d_res:11.4f} {draft / d_res:+8.2f}")
    print("\nTARGET is plutarch alone: target runtime, kernels, weight "
          "streaming.\nDRAFT is the five G=2 prompts: proposal head, selection "
          "chain, schedule.")

    fit = mode_decompose(pmap_a, pmap_b)
    if fit is not None:
        print("\n--- mode and mechanism decomposition (CAMPAIGN RULE 60)")
        if not same_schedule:
            print("  Schedules differ, so the ranked round table is only "
                  "approximate here.")
        print(f"  FACT-2 mode cost m = {fit['m'] * 1000:+.4f} ms per drafting "
              f"round (se {fit['se_m'] * 1000:.4f})")
        print(f"  uniform mechanism c = {fit['c']:+.4f} %  "
              f"(se {fit['se_c']:.4f});  positive means A is slower")
        print(f"  R2 = {fit['r2']:.4f}   rms residual {fit['rms']:.4f} %")
        print("  Judge c against ZERO. Over 26 byte-identical cross-mode pairs, "
              "where true c is\n  exactly zero, the fitted mean is "
              "-0.0576 (se 0.0444) with rms scatter 0.2335 %.\n"
              "  |c| under 0.47 % is not a mechanism; under 0.56 % is not a "
              "mechanism on the\n  conservative code-identical class.")
        print("  The model is MISSPECIFIED at plutarch: it predicts a "
              "+0.147 to +0.215 % cross-mode\n  shift there and the measured "
              "shift is +0.0056 (se 0.0263). Plutarch alone is the\n  better "
              "mode-immune probe for a target-path mechanism: pair sd 0.1643 % "
              "against\n  0.2808 % here, 2 sigma MDE 0.450 % against 0.702 %.")


def report_rank(rows, min_score, min_members):
    cohorts = defaultdict(list)
    for row in rows:
        score = row.get("officialScore")
        if score is None or score < min_score:
            continue
        pmap = prompt_map(row)
        if pmap is None:
            continue
        cohorts[schedule_signature(pmap)].append((row, pmap))
    if not cohorts:
        raise SystemExit("no scored rows at that threshold")
    sig, members = max(cohorts.items(), key=lambda kv: len(kv[1]))
    if len(members) < min_members:
        raise SystemExit(f"largest cohort has only {len(members)} runs")

    recs = []
    for row, pmap in members:
        recs.append((row,
                     100.0 * math.log(pmap[TARGET_PROBE]["mtp_seconds_per_token_mean"]),
                     statistics.fmean(
                         100.0 * math.log(pmap[n]["mtp_seconds_per_token_mean"])
                         for n in DRAFT_PROBES)))
    t_centre = statistics.fmean(r[1] for r in recs)
    d_centre = statistics.fmean(r[2] for r in recs)

    print(f"largest schedule-matched cohort: {len(members)} runs at "
          f">= {min_score}")
    print("Values are percent relative to the cohort mean. NEGATIVE is FASTER.")
    print(f"target resolution {RESOLUTION['target_all']:.4f} %, "
          f"drafting resolution {RESOLUTION['draft_all']:.4f} %\n")

    for label, index in (("TARGET PATH (plutarch)", 1),
                         ("DRAFTING PATH (five G=2 prompts)", 2)):
        centre = t_centre if index == 1 else d_centre
        print(f"--- {label}, fastest first")
        ordered = sorted(recs, key=lambda r: r[index])[:12]
        print(f"{'id':>8} {'solver':>15} {'published':>11} {'rel %':>8}")
        for row, tv, dv in ordered:
            value = (tv if index == 1 else dv) - centre
            print(f"{(row.get('id') or '')[:8]:>8} "
                  f"{(row.get('solverUsername') or '')[:15]:>15} "
                  f"{row.get('officialScore'):11.8f} {value:+8.4f}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", default=BOARD_JSON)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--byte-digest", action="store_true",
                        help="use the old, narrower byte-digest replicate rule")
    parser.add_argument("--validate-canon", action="store_true",
                        help="compare the string-aware and naive strippers")
    parser.add_argument("--provenance", action="store_true",
                        help="show how the byte-identical pairs were created")
    parser.add_argument("--read", nargs=2, metavar=("A", "B"))
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--min-score", type=float, default=3.30)
    parser.add_argument("--min-members", type=int, default=10)
    args = parser.parse_args(argv)

    rows = load_rows(args.board)
    if args.noise:
        report_noise(rows, args.byte_digest)
    elif args.validate_canon:
        report_validate_canon(rows)
    elif args.provenance:
        report_provenance(rows)
    elif args.read:
        report_read(rows, args.read[0], args.read[1])
    elif args.rank:
        report_rank(rows, args.min_score, args.min_members)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
