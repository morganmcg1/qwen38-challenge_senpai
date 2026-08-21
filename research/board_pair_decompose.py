"""Decompose the published gap between two ranked submissions.

The published score is `median_p( serial_p / candidate_p )`, so a gap between
two rows mixes two independent things: how fast the candidate decoded, and
which serial denominator the runner happened to draw that day (Finding 20).

This tool separates them. It reports, per prompt, whether the two rows ran the
same draft schedule, the candidate-leg delta, the serial-leg delta, and then
the two aggregate statistics:

  published gap    - what the board shows
  serial-free gap  - the same rows rescored against the board-mean serial time
                     for each prompt, which removes the serial lottery

A large published gap with a near-zero serial-free gap is a resample, not a
mechanism. Identical per-prompt `effective_mean_draft_len` to full precision
additionally proves the two rows ran the same schedule, which is the signature
of a byte-identical scored surface.

Usage:
    python3 research/board_pair_decompose.py <id_a_prefix> <id_b_prefix>
    python3 research/board_pair_decompose.py f04b102e 276aa2c2

Reads the board dump written by the campaign refresh step at
/tmp/yukon-board/full.json, or the path in $YUKON_BOARD_JSON.
"""

import json
import os
import sys

BOARD_JSON = os.environ.get("YUKON_BOARD_JSON", "/tmp/yukon-board/full.json")

# Mean serial seconds per token per prompt over the whole board. Reproduces
# every published score to 3.98e-11 when substituted for the per-run serial
# draw, which is what makes it a valid common denominator.
BOARD_MEAN_SERIAL = {
    "919318e1": 0.037990260,
    "192fb621": 0.037996402,
    "4b9e88cd": 0.037994712,
    "a2ea8b60": 0.037997448,
    "00142a44": 0.037994720,
    "c1ec5866": 0.037993427,
    "ea82dcb5": 0.037993760,
    "3b10cb4d": 0.038002089,
}

PROMPT_NAME = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def load_rows(path=BOARD_JSON):
    payload = json.load(open(path))
    rows = payload
    if isinstance(payload, dict):
        for key in ("submissions", "rows", "data", "items"):
            if key in payload:
                rows = payload[key]
                break
    return [r for r in rows if isinstance(r, dict)]


def find_row(rows, prefix):
    hits = [r for r in rows if r.get("id", "").startswith(prefix)]
    if not hits:
        raise SystemExit("no submission id starts with %r" % prefix)
    if len(hits) > 1:
        raise SystemExit("ambiguous prefix %r matches %d rows" % (prefix, len(hits)))
    return hits[0]


def per_prompt(row):
    entries = row.get("officialMetrics", {}).get("per_prompt") or []
    if not entries:
        raise SystemExit("row %s has no per_prompt rows" % row["id"][:8])
    return {PROMPT_NAME[e["prompt_sha256"][:8]]: e for e in entries}


def median_of_eight(values):
    ordered = sorted(values)
    if len(ordered) != 8:
        raise SystemExit("expected 8 prompts, got %d" % len(ordered))
    return 0.5 * (ordered[3] + ordered[4])


def serial_free_score(row):
    ratios = []
    for entry in row["officialMetrics"]["per_prompt"]:
        key = entry["prompt_sha256"][:8]
        ratios.append(BOARD_MEAN_SERIAL[key] / entry["mtp_seconds_per_token_mean"])
    return median_of_eight(ratios)


def pct(new, old):
    return 100.0 * (new - old) / old


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    rows = load_rows()
    a = find_row(rows, argv[1])
    b = find_row(rows, argv[2])

    for tag, row in (("A", a), ("B", b)):
        print(
            "%s  %s  %-14s  published %.8f  %s"
            % (
                tag,
                row["id"][:8],
                row.get("solverUsername"),
                row.get("officialScore") or float("nan"),
                row.get("createdAt"),
            )
        )
        print("     commit %s" % row.get("submissionCommitSha"))
    print()

    pa, pb = per_prompt(a), per_prompt(b)
    order = sorted(pa, key=lambda n: pa[n]["effective_mean_draft_len"])

    print(
        "%-9s %-16s %-16s %-6s %11s %11s %9s %9s"
        % (
            "prompt",
            "draft_len A",
            "draft_len B",
            "same",
            "cand A",
            "cand B",
            "cand %",
            "serial %",
        )
    )
    cand_deltas, serial_deltas, same_all = [], [], True
    for name in order:
        ea, eb = pa[name], pb[name]
        same = ea["effective_mean_draft_len"] == eb["effective_mean_draft_len"]
        same_all = same_all and same
        # Positive percentages mean B is faster / slower-serial than A.
        dc = pct(ea["mtp_seconds_per_token_mean"], eb["mtp_seconds_per_token_mean"])
        ds = pct(eb["serial_seconds_per_token_mean"], ea["serial_seconds_per_token_mean"])
        cand_deltas.append(dc)
        serial_deltas.append(ds)
        print(
            "%-9s %-16.10f %-16.10f %-6s %.9f %.9f %+9.4f %+9.4f"
            % (
                name,
                ea["effective_mean_draft_len"],
                eb["effective_mean_draft_len"],
                "YES" if same else "no",
                ea["mtp_seconds_per_token_mean"],
                eb["mtp_seconds_per_token_mean"],
                dc,
                ds,
            )
        )

    def mean(xs):
        return sum(xs) / len(xs)

    def sd(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    sfa, sfb = serial_free_score(a), serial_free_score(b)
    print()
    print("schedule identical on all eight prompts   %s" % ("YES" if same_all else "NO"))
    print("candidate leg   B faster by  %+.4f %%   sd %.4f" % (mean(cand_deltas), sd(cand_deltas)))
    print("serial leg      B slower by  %+.4f %%   sd %.4f" % (mean(serial_deltas), sd(serial_deltas)))
    print()
    print("serial-free   A %.8f   B %.8f   gap %+.4f %%" % (sfa, sfb, pct(sfb, sfa)))
    if a.get("officialScore") and b.get("officialScore"):
        print(
            "published     A %.8f   B %.8f   gap %+.4f %%"
            % (a["officialScore"], b["officialScore"], pct(b["officialScore"], a["officialScore"]))
        )
    print()
    oma, omb = a["officialMetrics"], b["officialMetrics"]
    for field in (
        "baseline_serial_seconds_per_token_mean",
        "candidate_mtp_seconds_per_token_mean",
        "decode_tokens",
        "mtp_max_draft_depth",
        "qwen_mtp_weights_hash",
    ):
        va, vb = oma.get(field), omb.get(field)
        print("%-42s A %-66s %s" % (field, va, "SAME" if va == vb else "B %s" % vb))
    ha = pa[order[0]]["head_provenance_sha256"]
    hb = pb[order[0]]["head_provenance_sha256"]
    print("%-42s A %-66s %s" % ("head_provenance_sha256", ha, "SAME" if ha == hb else "B %s" % hb))


if __name__ == "__main__":
    main(sys.argv)
