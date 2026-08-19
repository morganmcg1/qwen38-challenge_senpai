#!/usr/bin/env python3
"""Per-leg decomposition of the LIVE FRONTIER (upstream/main) against our
scored tree and the shared submit base, straight from the ranked corpus.

Why this exists
---------------
Item 161 established that `upstream/main` carries a promoted memory-policy
change ("the crown") that our tree reverts, and that our own overlay measured
-0.3316 % against the same base. Both of those are SCORE deltas. A score delta
cannot tell you which leg moved, and `raw_p = serial / mtp` means a change that
*slows the serial leg* raises the score exactly like one that speeds the MTP
leg. Those two have opposite research value: only the second is a real speedup.

It also answers, from ranked data rather than prose, whether our E26
stop-token-continuation fix perturbs ranked behaviour at all. The ranked run
decodes 512 tokens; E26's local control died at emitted index 302. If the
defect fired on any ranked prompt, the round structure would differ and
`effective_mean_draft_len` could not agree to 16 digits between a tree that
HAS the early exit (base, crown) and one that does not (ours).

Every number this prints is derived here; nothing is typed in from a brief.
Run with --selftest to check the derivations against officialScore.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROWS = Path("/tmp/rows_probe.json")

# The three trees under comparison, by submission commit SHA prefix.
# base : shared ancestor of our submission and the crown's
# ours : our best scored submission
# crown: the tree that is upstream/main right now
#
# NOTE these are `submissionCommitSha` values -- commits in the SUBMITTER's own
# repo -- and they are NOT the organizer's snapshot commits on
# refs/remotes/upstream/submissions/<uuid>. The base tree is
# upstream/submissions snapshot 5068eb8d but submissionCommitSha 0cbaf6a7;
# confusing the two costs an hour.
TREES = {
    "base": "0cbaf6a7",
    "ours": "2b0c36a0",
    "crown": "ef42e043",
}

# Prompt sha256 prefix -> short name, so per-leg rows are readable.
PROMPTS = {
    "c1ec5866": "plutarch",
    "4b9e88cd": "drama",
    "3b10cb4d": "travel",
    "919318e1": "beagle",
    "00142a44": "medicine",
    "a2ea8b60": "essays",
    "192fb621": "republic",
    "ea82dcb5": "botany",
}

# Only compare rows measured against the same MTP head.
HEAD_PROVENANCE = "559b24eb"

# The API serialises `per_prompt.raw_ratio_of_means` to ~10 significant decimals
# while `officialScore` carries full double precision, so the reconstructed
# score agrees only to ~1e-11. That is the floor on anything derived from
# per-prompt fields here. It is ~1e-10 % in dln terms, i.e. nine orders of
# magnitude below sigma_score = 0.0978 %, so it constrains nothing we conclude
# -- but the reconstruction test must not pretend to more precision than the
# wire format has.
SCORE_TOL = 1e-9


def load_rows(path: Path = ROWS) -> list[dict]:
    with path.open() as fh:
        doc = json.load(fh)
    return doc["submissions"] if isinstance(doc, dict) else doc


def pick(rows: list[dict], sha_prefix: str) -> dict:
    hits = [
        r
        for r in rows
        if (r.get("submissionCommitSha") or "").startswith(sha_prefix)
        and isinstance(r.get("officialMetrics"), dict)
        and r["officialMetrics"].get("per_prompt")
    ]
    if not hits:
        raise SystemExit(f"no scored row for commit prefix {sha_prefix}")
    # Newest first; identical trees re-scored would be a replicate, not a bug.
    hits.sort(key=lambda r: r["createdAt"], reverse=True)
    return hits[0]


def prompt_table(row: dict) -> dict[str, dict]:
    """Per-prompt metrics for one row, keyed by prompt name. Fails closed.

    Audited after two fail-open defects were found elsewhere in this campaign's
    tooling. The load-bearing protection is downstream -- main() re-derives the
    score from these rows and compares it against officialScore -- so a
    truncated prompt set is caught there. But two gaps were open here:
    an unrecognised prompt hash was silently keyed by its own hash, and the
    prompt count was never asserted, so a row measured on a DIFFERENT corpus
    could be tabulated without complaint. Both now raise.
    """
    out = {}
    for p in row["officialMetrics"]["per_prompt"]:
        key = (p.get("prompt_sha256") or "")[:8]
        if key not in PROMPTS:
            raise SystemExit(
                f"row {row['id']} carries unknown prompt {key!r}: the corpus "
                f"differs from the 8 prompts this decomposition is defined "
                f"over, so no comparison is valid"
            )
        name = PROMPTS[key]
        head = (p.get("head_provenance_sha256") or "")[:8]
        if head != HEAD_PROVENANCE:
            raise SystemExit(
                f"row {row['id']} prompt {name} ran head {head}, "
                f"not {HEAD_PROVENANCE}: not comparable"
            )
        out[name] = p
    if len(out) != len(PROMPTS):
        missing = sorted(set(PROMPTS.values()) - set(out))
        raise SystemExit(
            f"row {row['id']} has {len(out)} of {len(PROMPTS)} prompts "
            f"(missing {missing}). score_from_rows() takes the 4th and 5th "
            f"order statistics by INDEX, so a short table would silently "
            f"return the wrong score rather than fail"
        )
    return out


def score_from_rows(table: dict[str, dict]) -> float:
    """Score = mean of the 4th and 5th order statistics of raw_ratio_of_means
    over the 8 prompts (established over 94 rows, 0 mismatches)."""
    ratios = sorted(p["raw_ratio_of_means"] for p in table.values())
    return (ratios[3] + ratios[4]) / 2.0


def dln(new: float, old: float) -> float:
    """Percent log-change, so leg deltas add to the ratio delta exactly."""
    return 100.0 * math.log(new / old)


def main() -> int:
    rows = load_rows()
    picked = {name: pick(rows, sha) for name, sha in TREES.items()}
    tables = {name: prompt_table(row) for name, row in picked.items()}

    print("=" * 78)
    print("TREES")
    print("=" * 78)
    for name, row in picked.items():
        print(
            f"  {name:6s} {row['submissionCommitSha'][:12]}  "
            f"score={row['officialScore']!r}  "
            f"{row['solverUsername']}  {row['createdAt'][:19]}  "
            f"{row['status']}/{row['promotionStatus']}"
        )
        derived = score_from_rows(tables[name])
        official = row["officialScore"]
        resid = abs(derived - official)
        agree = resid < SCORE_TOL
        print(
            f"         derived 4th/5th-order score={derived!r} "
            f"{'MATCHES' if agree else 'MISMATCH'} officialScore "
            f"(residual {resid:.2e})"
        )
        if not agree:
            return 2

    # ---------------------------------------------------------------- E26 test
    print()
    print("=" * 78)
    print("E26 STOP-TOKEN NEUTRALITY ON RANKED (decode_tokens=%d)"
          % picked["crown"]["officialMetrics"]["decode_tokens"])
    print("=" * 78)
    print("  ours REMOVES the early exit; base and crown BOTH CARRY it.")
    print("  If it fired inside 512 tokens the round structure would differ.")
    print()
    hdr = f"  {'prompt':10s} {'draftlen ours':>22s} {'crown':>22s} {'base':>22s}  ident"
    print(hdr)
    all_ident = True
    for name in PROMPTS.values():
        vals = {t: tables[t][name]["effective_mean_draft_len"] for t in TREES}
        ident = repr(vals["ours"]) == repr(vals["crown"]) == repr(vals["base"])
        all_ident &= ident
        print(
            f"  {name:10s} {vals['ours']!r:>22} {vals['crown']!r:>22} "
            f"{vals['base']!r:>22}  {'YES' if ident else 'NO'}"
        )
    print()
    print(
        "  VERDICT: "
        + (
            "every prompt agrees to full repr -> the stop-token early exit "
            "never fires\n           on ranked at 512 tokens, so E26 is "
            "score-NEUTRAL there and its cost is\n           zero. It remains "
            "insurance against a window that does contain one."
            if all_ident
            else "at least one prompt DIFFERS -> E26 changes ranked behaviour "
            "and its\n           score effect is NOT zero. Re-open the "
            "attribution of our overlay."
        )
    )

    # -------------------------------------------------------- leg decomposition
    for new, old in (("crown", "base"), ("ours", "base"), ("crown", "ours")):
        print()
        print("=" * 78)
        print(f"PER-LEG DECOMPOSITION: {new} vs {old}   (dln %, so legs subtract)")
        print("=" * 78)
        print(
            f"  {'prompt':10s} {'width':>6s} {'dln serial':>11s} "
            f"{'dln mtp':>9s} {'dln raw_p':>10s} {'check':>8s}"
        )
        legs = []
        for name in PROMPTS.values():
            a, b = tables[new][name], tables[old][name]
            ds = dln(
                a["serial_seconds_per_token_mean"], b["serial_seconds_per_token_mean"]
            )
            dm = dln(a["mtp_seconds_per_token_mean"], b["mtp_seconds_per_token_mean"])
            dr = dln(a["raw_ratio_of_means"], b["raw_ratio_of_means"])
            resid = abs((ds - dm) - dr)
            legs.append((name, a["effective_mean_draft_len"], ds, dm, dr))
            print(
                f"  {name:10s} {a['effective_mean_draft_len']:6.3f} "
                f"{ds:+11.4f} {dm:+9.4f} {dr:+10.4f} "
                f"{'ok' if resid < 1e-6 else 'RESID!':>8s}"
            )
        ds_mean = statistics.fmean(x[2] for x in legs)
        dm_mean = statistics.fmean(x[3] for x in legs)
        print(
            f"  {'MEAN':10s} {'':6s} {ds_mean:+11.4f} {dm_mean:+9.4f}"
        )
        s_new = score_from_rows(tables[new])
        s_old = score_from_rows(tables[old])
        print(
            f"  SCORE {s_old!r} -> {s_new!r}  "
            f"= {dln(s_new, s_old):+.4f} %"
        )
        # Which leg does the score move actually come from? Only the 4th/5th
        # order statistics matter, so report those two prompts separately.
        ranked = sorted(
            PROMPTS.values(), key=lambda n: tables[old][n]["raw_ratio_of_means"]
        )
        central = ranked[3:5]
        print(f"  scoring prompts at {old}: {central[0]}, {central[1]}")
        for name in central:
            a, b = tables[new][name], tables[old][name]
            print(
                f"    {name:10s} serial {dln(a['serial_seconds_per_token_mean'], b['serial_seconds_per_token_mean']):+8.4f} "
                f"  mtp {dln(a['mtp_seconds_per_token_mean'], b['mtp_seconds_per_token_mean']):+8.4f} "
                f"  raw_p {dln(a['raw_ratio_of_means'], b['raw_ratio_of_means']):+8.4f}"
            )

    print()
    print("=" * 78)
    print("READING")
    print("=" * 78)
    print(
        """  A positive dln serial means the leg got SLOWER. Because
  raw_p = serial / mtp, slowing serial and speeding mtp are
  indistinguishable in the score but not in value: only a negative dln mtp
  is a decode speedup. Check the sign of the crown's gain before copying its
  mechanism as though it were an optimisation."""
    )
    return 0


def selftest() -> int:
    rows = load_rows()
    # 1. dln is additive across the ratio, exactly, on every scored row we use.
    for sha in TREES.values():
        t = prompt_table(pick(rows, sha))
        for name, p in t.items():
            lhs = math.log(
                p["serial_seconds_per_token_mean"] / p["mtp_seconds_per_token_mean"]
            )
            rhs = math.log(p["raw_ratio_of_means"])
            assert abs(lhs - rhs) < 1e-9, (sha, name, lhs, rhs)
    # 2. The 4th/5th order-statistic rule reproduces officialScore.
    for sha in TREES.values():
        row = pick(rows, sha)
        d = score_from_rows(prompt_table(row))
        assert abs(d - row["officialScore"]) < SCORE_TOL, (sha, d, row["officialScore"])
    # 3. dln sign convention: a slower leg is positive.
    assert dln(2.0, 1.0) > 0

    # 4. NEGATIVE CONTROLS on the fail-closed guards. A guard that has never
    #    been made to fire is an assumption, not a check -- and the score rule
    #    indexes order statistics 3 and 4 positionally, so a short table returns
    #    a WRONG NUMBER rather than raising.
    import copy

    good = pick(rows, next(iter(TREES.values())))

    dropped = copy.deepcopy(good)
    dropped["officialMetrics"]["per_prompt"] = \
        dropped["officialMetrics"]["per_prompt"][:5]
    try:
        prompt_table(dropped)
    except SystemExit:
        pass
    else:
        raise AssertionError("a 5-prompt row was accepted; score_from_rows "
                             "would have returned the 4th/5th of FIVE")

    # Show the wrong answer the guard prevents, so the guard's value is a
    # measured quantity rather than an argument.
    short = {k: v for k, v in list(prompt_table(good).items())[:5]}
    wrong = score_from_rows(short)
    right = score_from_rows(prompt_table(good))
    assert abs(wrong - right) > SCORE_TOL, (
        "expected the truncated score to differ materially", wrong, right)

    alien = copy.deepcopy(good)
    alien["officialMetrics"]["per_prompt"][0]["prompt_sha256"] = "dead" * 16
    try:
        prompt_table(alien)
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown prompt hash was accepted")

    foreign = copy.deepcopy(good)
    foreign["officialMetrics"]["per_prompt"][0]["head_provenance_sha256"] = \
        "beef" * 16
    try:
        prompt_table(foreign)
    except SystemExit:
        pass
    else:
        raise AssertionError("a foreign MTP head was accepted")

    print("selftest: OK (ratio identity, score rule, dln sign, 3 negative "
          "controls; truncation would have shifted the score by %.4f)"
          % abs(wrong - right))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
