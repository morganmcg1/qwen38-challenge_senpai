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

Measured resolution, from 39 byte-identical replicate pairs with matching
schedules (same scored-surface tree digest, so any difference is pure
measurement):

    probe            all pairs   same-mode   cross-mode
    plutarch          0.0709 %    0.0431 %     0.0880 %
    drafting mean     0.7205 %    0.1139 %     0.9762 %
    all-8 mean        0.7091 %    0.0793 %     0.9636 %

    published median floor, for comparison:   0.2770 %

Two things follow.

1. Plutarch is a 0.043 % target-path instrument, about 6x sharper than the
   published median, and it is nearly immune to the mode: the mode inflates the
   drafting probe 8.57x but plutarch only 2.04x.

2. The mode is DETECTABLE inside a single pair. A mode flip moves the drafting
   probe by about 1 % while leaving plutarch under about 0.15 %. When two runs
   are in the same mode, the drafting probe itself becomes a 0.114 %
   instrument, which is 2.4x sharper than the published median.

The predicted plutarch mode shift is 38 drafting rounds x 0.601 ms over a
15.5 s plutarch leg, or 0.147 %. The measured cross-mode plutarch pair RMS is
0.1244 %. That agreement, from an entirely independent direction, is the
strongest confirmation of FACT 2 the campaign has.

Sign convention
---------------
A POSITIVE percentage means B is FASTER than A, matching
`research/board_pair_decompose.py`.

Usage
-----
    python3 research/board_prompt_instrument.py --noise
        Re-measure the resolution from byte-identical replicate pairs.
        Needs `git fetch upstream 'refs/heads/submissions/*:...'` first.

    python3 research/board_prompt_instrument.py --read <a_prefix> <b_prefix>
        Read one pair through the instrument: mode classification, target-path
        effect, drafting-path effect, and each in resolution units.

    python3 research/board_prompt_instrument.py --rank --min-score 3.30
        Rank the largest schedule-matched cohort on each probe separately.

Reads /tmp/yukon-board/full.json, or $YUKON_BOARD_JSON.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict

BOARD_JSON = os.environ.get("YUKON_BOARD_JSON", "/tmp/yukon-board/full.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT_NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
PROMPT_ORDER = ["plutarch", "drama", "travel", "beagle", "medicine",
                "republic", "essays", "botany"]

TARGET_PROBE = "plutarch"
DRAFT_PROBES = ["beagle", "medicine", "republic", "essays", "botany"]

# Measured per-run candidate-leg resolution, in percent. See the module
# docstring for provenance. Re-measure with --noise after the board grows.
RESOLUTION = {
    "target_all": 0.0709,
    "target_same_mode": 0.0431,
    "draft_all": 0.7205,
    "draft_same_mode": 0.1139,
}
# A mode flip moves the drafting probe by about this much and plutarch by far
# less. Anything above the first number is a flip, not a mechanism.
MODE_DRAFT_SHIFT = 0.60
MODE_TARGET_SHIFT = 0.15


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


def tree_digests():
    """Scored-surface content digest for every public submission branch.

    `git ls-tree` returns the subtree object ids, which digest the whole scored
    surface in constant time. Equal digests mean byte-identical scored code.
    """
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)",
         "refs/remotes/upstream/submissions/"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for ref in refs:
        listing = subprocess.run(
            ["git", "ls-tree", ref, "Sources", "Vendor",
             "mtp-head.manifest.json"],
            cwd=REPO, capture_output=True, text=True)
        if listing.returncode != 0:
            continue
        out[ref.rsplit("/", 1)[-1][:8]] = listing.stdout.strip()
    return out


def rms(values):
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def report_noise(rows):
    digests = tree_digests()
    print(f"{len(digests)} submission branches digested", file=sys.stderr)
    groups = defaultdict(list)
    for row in rows:
        if row.get("officialScore") is None:
            continue
        pmap = prompt_map(row)
        if pmap is None:
            continue
        digest = digests.get((row.get("id") or "")[:8])
        if digest is None:
            continue
        groups[(digest, schedule_signature(pmap))].append(pmap)

    per_probe = defaultdict(list)
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                for name in PROMPT_ORDER:
                    per_probe[name].append(cand_pct(members[i], members[j], name))
                target, draft = probes(members[i], members[j])
                per_probe["_target"].append(target)
                per_probe["_draft"].append(draft)

    npairs = len(per_probe["_target"])
    replicated = sum(1 for m in groups.values() if len(m) > 1)
    print(f"{len(groups)} (tree, schedule) groups, {replicated} replicated, "
          f"{npairs} byte-identical pairs\n")
    if npairs < 4:
        print("not enough replicate pairs to measure resolution")
        return

    same = [k for k in range(npairs) if abs(per_probe["_draft"][k]) <= MODE_DRAFT_SHIFT]
    cross = [k for k in range(npairs) if abs(per_probe["_draft"][k]) > MODE_DRAFT_SHIFT]

    print(f"{'probe':>10} {'all':>9} {'same-mode':>10} {'cross-mode':>11}")
    for name in PROMPT_ORDER + ["_draft", "_target"]:
        vals = per_probe[name]
        row = [rms(vals) / math.sqrt(2)]
        for idx in (same, cross):
            sub = [vals[k] for k in idx]
            row.append(rms(sub) / math.sqrt(2) if len(sub) >= 3 else float("nan"))
        print(f"{name:>10} {row[0]:9.4f} {row[1]:10.4f} {row[2]:11.4f}")
    print(f"\nsame-mode pairs {len(same)}, cross-mode pairs {len(cross)}")
    print("Per-run candidate-leg standard deviation in percent. A pair "
          "difference\ncarries twice the variance of one run, so each column "
          "is pairRMS / sqrt(2).")


def find_row(rows, prefix):
    hits = [r for r in rows if (r.get("id") or "").startswith(prefix)]
    if not hits:
        raise SystemExit(f"no board row with id prefix {prefix}")
    if len(hits) > 1:
        raise SystemExit(f"{prefix} is ambiguous over {len(hits)} rows")
    return hits[0]


def report_read(rows, prefix_a, prefix_b):
    row_a, row_b = find_row(rows, prefix_a), find_row(rows, prefix_b)
    pmap_a, pmap_b = prompt_map(row_a), prompt_map(row_b)
    if pmap_a is None or pmap_b is None:
        raise SystemExit("one of the rows has no usable per-prompt block")

    same_schedule = schedule_signature(pmap_a) == schedule_signature(pmap_b)
    target, draft = probes(pmap_a, pmap_b)
    mode_flip = abs(draft) > MODE_DRAFT_SHIFT and abs(target) < MODE_TARGET_SHIFT

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

    key = "same_mode" if not mode_flip else "all"
    t_res = RESOLUTION[f"target_{key}"]
    d_res = RESOLUTION[f"draft_{key}"]
    print(f"\nFACT-2 measurement mode: "
          f"{'FLIPPED between the two runs' if mode_flip else 'no flip detected'}")
    if mode_flip:
        print("  The drafting probe moved more than "
              f"{MODE_DRAFT_SHIFT} % while plutarch stayed under "
              f"{MODE_TARGET_SHIFT} %.\n"
              "  Treat the drafting probe as uninformative and read plutarch "
              "only.")

    print(f"\n{'probe':>8} {'effect %':>10} {'resolution':>11} {'sigma':>8}")
    print(f"{'TARGET':>8} {target:+10.4f} {t_res:11.4f} {target / t_res:+8.2f}")
    print(f"{'DRAFT':>8} {draft:+10.4f} {d_res:11.4f} {draft / d_res:+8.2f}")
    print("\nTARGET is plutarch alone: target runtime, kernels, weight "
          "streaming.\nDRAFT is the five G=2 prompts: proposal head, selection "
          "chain, schedule.")


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
    parser.add_argument("--read", nargs=2, metavar=("A", "B"))
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--min-score", type=float, default=3.30)
    parser.add_argument("--min-members", type=int, default=10)
    args = parser.parse_args(argv)

    rows = load_rows(args.board)
    if args.noise:
        report_noise(rows)
    elif args.read:
        report_read(rows, args.read[0], args.read[1])
    elif args.rank:
        report_rank(rows, args.min_score, args.min_members)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
