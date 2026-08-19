#!/usr/bin/env python3
"""Authoritative noise floors for this campaign, each tagged with the variance
component it was estimated from.

WHY THIS FILE EXISTS
--------------------
For most of this campaign the advisor divided effect sizes by
`SIGMA_SCORE_PCT = 0.0978 %` and reported the quotient as "sigmas". That number
is a **within-run replicate** sd. Almost every question it was used to answer
("will this tree outrank that tree when the organizer measures it?", "is the
crown gap real?", "should we submit?") is a **between-submission** question,
whose sd is 0.7678 % -- larger by 7.9x. Every probability and power figure
computed that way was wrong in the optimistic direction. See ledger 166.

The same mistake was then made a second time, per leg, from a single pair of
rows (ledger 160(D) -> corrected in ledger 172): the MTP leg was quoted at
0.0995 % when the pooled between-submission figure is 0.8040 %, and the serial
leg at 0.3475 % when it is 0.2110 %. Both legs wrong, in OPPOSITE directions,
which inverted the advisor's standing instruction about which leg to trust.

Neither original number was a computational error. Both were correct estimates
of the wrong variance component. So the fix is NOT to overwrite them -- it is to
make every constant carry its component in its NAME, and to make every call site
say which question it is asking.

RULE
----
    Before dividing an effect by a sigma, ask which variance component that
    sigma was estimated from, and how many independent units went into it.

HOW THE BETWEEN-SUBMISSION FLOORS WERE MEASURED
-----------------------------------------------
Group the ranked board's submission refs by WHOLE TREE:

    git rev-parse refs/remotes/upstream/submissions/<uuid>^{tree}

Keep trees that more than one solver submitted. Within such a set the submitted
code is byte-identical, so ALL spread between the rows is measurement noise.
Pool the within-set relative sd with dof = n-1 per cell.

This is a whole-tree key, deliberately NOT a content fingerprint: a fingerprint
that excludes some files (e.g. the QMV kernel) admits sets whose members are not
actually identical, and its spread is then an upper bound contaminated by real
effects rather than a floor.

Covariate control (edward, 13/13 of his sets): members of a set agree on
head_provenance_sha256 for all 8 prompts, on qwen_mtp_weights_hash,
effective_mean_draft_len, non_drafting_round_count, and the whole scoring
policy. Only the commit differs.

Two attempts to explain the floor away, both FAILED -- which is the evidence FOR
it, not a gap in it:
  * time/instance proximity: quiet sets sit at gaps of 0.79/1.56/2.77/8.21 h
    while the NOISIEST set sits at 2.06 h. No back-to-back escape.
  * promotion-status artefact: the quiet sets are one both-promoted, one
    both-rejected and two mixed -- the same mixture as the noisy ones. Excluding
    them moves the floor AGAINST us, to 0.8438 %.

Run `python3 research/noise_floors.py selftest` to check the internal
consistency of what is recorded here, and `... audit` to find stale citations.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Floor:
    """One noise floor, inseparable from the component it was estimated from."""

    def __init__(self, name, pct, component, n_units, dof, source, note=""):
        self.name = name
        self.pct = pct
        self.component = component      # "within-run" | "between-submission"
        self.n_units = n_units          # independent units behind the estimate
        self.dof = dof
        self.source = source
        self.note = note

    def sigmas(self, effect_pct):
        return effect_pct / self.pct

    def __repr__(self):
        return (f"<{self.name} {self.pct:.4f}% {self.component} "
                f"n={self.n_units} dof={self.dof}>")


# --------------------------------------------------------------------------
# BETWEEN-SUBMISSION floors. Use these for ANY question of the form
# "will the organizer's measurement of tree A beat its measurement of tree B".
# That includes: should we submit, is the crown gap real, did a rival's change
# help, is our roadmap item detectable on the board.
# --------------------------------------------------------------------------

SCORE_BETWEEN_SUBMISSION = Floor(
    "SCORE_BETWEEN_SUBMISSION", 0.7678, "between-submission",
    n_units=17, dof=23,
    source="advisor, 17 whole-tree sets, N=40 rows, ledger 166",
    note="edward independently got 0.7353 % from 13 sets / 31 rows. "
         "Excluding the 4 anomalously quiet sets moves it to 0.8438 %.")

MTP_LEG_BETWEEN_SUBMISSION = Floor(
    "MTP_LEG_BETWEEN_SUBMISSION", 0.8040, "between-submission",
    n_units=17, dof=184,
    source="advisor, 17 whole-tree sets, per-prompt pooled, ledger 172",
    note="edward independently got 0.7875 % from 13 sets. Per-set rms spans "
         "0.0683 %-1.1635 %, a 17x spread: a single pair is worthless here.")

SERIAL_LEG_BETWEEN_SUBMISSION = Floor(
    "SERIAL_LEG_BETWEEN_SUBMISSION", 0.2110, "between-submission",
    n_units=17, dof=184,
    source="advisor, 17 whole-tree sets, per-prompt pooled, ledger 172",
    note="edward independently got 0.2063 %. Spans only 0.1344 %-0.2952 % "
         "across 17 independent trees -- this is the REPRODUCIBLE leg, and its "
         "tightness independently confirms item 153's prompt-independence.")

RATIO_BETWEEN_SUBMISSION = Floor(
    "RATIO_BETWEEN_SUBMISSION", 0.7945, "between-submission",
    n_units=17, dof=184,
    source="advisor, 17 whole-tree sets, per-prompt pooled, ledger 172")

# --------------------------------------------------------------------------
# WITHIN-RUN floors. These are CORRECT, and they are the right denominator for
# "did this number move between two replicates inside one session". They are
# the WRONG denominator for anything involving the board.
# --------------------------------------------------------------------------

SCORE_WITHIN_RUN = Floor(
    "SCORE_WITHIN_RUN", 0.0978, "within-run",
    n_units=1, dof=None,
    source="the value formerly exported as SIGMA_SCORE_PCT",
    note="RETIRED as a between-submission yardstick by ledger 166. Correct for "
         "within-run questions; wrong by 7.9x for board questions.")

MTP_LEG_WITHIN_RUN_N1 = Floor(
    "MTP_LEG_WITHIN_RUN_N1", 0.0995, "within-run",
    n_units=1, dof=7,
    source="ledger 160(D), the single pair b8642b81f72f",
    note="n=1, and that one pair is RANK 1 OF 17 on quietness. Do not divide "
         "by this. Ledger 172.")

SERIAL_LEG_WITHIN_RUN_N1 = Floor(
    "SERIAL_LEG_WITHIN_RUN_N1", 0.3475, "within-run",
    n_units=1, dof=7,
    source="ledger 160(D), the single pair b8642b81f72f",
    note="n=1, and OVERSTATES the pooled figure by 1.6x. Ledger 172.")

ALL_FLOORS = [SCORE_BETWEEN_SUBMISSION, MTP_LEG_BETWEEN_SUBMISSION,
              SERIAL_LEG_BETWEEN_SUBMISSION, RATIO_BETWEEN_SUBMISSION,
              SCORE_WITHIN_RUN, MTP_LEG_WITHIN_RUN_N1,
              SERIAL_LEG_WITHIN_RUN_N1]

# --------------------------------------------------------------------------
# Effects the campaign cares about, priced against the floor that answers the
# question actually being asked. All are score-relative percentages.
# --------------------------------------------------------------------------

EFFECTS = {
    "crown gap over base":        0.5193,
    "engineerable gap":           0.2586,
    "our next submission's gain": 0.0283,
    "E27 score cost (measured)": -0.3321,
    "E40 width-8 stream cost":    0.4910,
    "alphonse E44 predicted":    -0.17,
    "local A/B MDE (pairs=5)":    0.5040,
}


def _phi(z):
    """Standard normal CDF via erf, no scipy dependency."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_beats_zero(effect_pct, floor=SCORE_BETWEEN_SUBMISSION):
    """P(a single ranked submission of this effect scores above the reference).

    This is the number that should have been quoted all campaign. With the
    within-run floor our +0.0283 % looked like P=0.61; against the real
    between-submission floor it is a coin flip.
    """
    return _phi(effect_pct / floor.pct)


# --------------------------------------------------------------------------
# audit: find citations of the retired literals anywhere in the tree
# --------------------------------------------------------------------------

STALE_LITERALS = {
    "0.0978": "score within-run sd quoted as a board yardstick (ledger 166)",
    "0.0995": "MTP leg n=1 from the quietest set of 17 (ledger 172)",
    "0.3475": "serial leg n=1, overstated 1.6x (ledger 172)",
    "0.0923": "an earlier score sd, same within-run category",
    "0.078":  "organizer paired figure, within-run",
}

# A citation is EXEMPT if its line, or the line above it, declares scope.
SCOPE_TAGS = re.compile(
    r"within[- ]run|WITHIN[- ]RUN|historical|HISTORICAL|retired|RETIRED|"
    r"superseded|SUPERSEDED|ledger 166|ledger 172|noise_floors")


def audit(paths=None):
    """Report every stale-literal citation that does not declare its scope.

    This module is ALWAYS excluded, whether scanning by default or by explicit
    path: it is the authority that defines the retired literals, so it must
    quote them. Applying the exclusion in only one of the two entry paths is
    how a scanner disagrees with itself about its own scope.
    """
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = sorted((ROOT / "research").rglob("*.py"))
        files += sorted((ROOT / "senpai").rglob("*.sh"))
    files = [f for f in files if f.resolve() != Path(__file__).resolve()]

    undeclared, declared = [], 0
    for f in files:
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            for lit, why in STALE_LITERALS.items():
                if lit not in line:
                    continue
                context = line + "\n" + (lines[i - 1] if i else "")
                if SCOPE_TAGS.search(context):
                    declared += 1
                else:
                    # A path outside ROOT is legitimate -- auditing a student
                    # checkout or a temp fixture. relative_to() RAISES there,
                    # so it must not be on the reporting path.
                    try:
                        shown = f.relative_to(ROOT)
                    except ValueError:
                        shown = f
                    undeclared.append(
                        (shown, i + 1, lit, why, line.strip()[:88]))
                break

    print(f"scanned {len(files)} files")
    print(f"citations declaring their scope : {declared}")
    print(f"citations NOT declaring scope   : {len(undeclared)}")
    if undeclared:
        print("\nUNDECLARED CITATIONS -- each must say which variance component "
              "it means,\nor import from research/noise_floors.py:\n")
        for rel, ln, lit, why, text in undeclared:
            print(f"  {rel}:{ln}\n      literal {lit}: {why}\n      {text}")
    return 1 if undeclared else 0


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def selftest():
    errs = []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    # 1. every floor declares a component we recognise
    for f in ALL_FLOORS:
        check(f.component in ("within-run", "between-submission"),
              f"{f.name}: unknown component {f.component!r}")
        check(f.pct > 0, f"{f.name}: non-positive sd")
        check(bool(f.source), f"{f.name}: no provenance")

    # 2. the headline error factor is what the ledger claims
    ratio = SCORE_BETWEEN_SUBMISSION.pct / SCORE_WITHIN_RUN.pct
    check(7.8 < ratio < 8.0, f"score floor ratio {ratio:.2f}, expected ~7.9")

    # 3. the per-leg inversion: MTP is the NOISY leg between submissions,
    #    which is the opposite of what the n=1 pair said.
    check(MTP_LEG_BETWEEN_SUBMISSION.pct > SERIAL_LEG_BETWEEN_SUBMISSION.pct,
          "between submissions the MTP leg must be the noisier one")
    check(MTP_LEG_WITHIN_RUN_N1.pct < SERIAL_LEG_WITHIN_RUN_N1.pct,
          "the retired n=1 pair must show the OPPOSITE ordering; that "
          "inversion is the whole point of ledger 172")

    # 4. the score inherits nearly all of the candidate leg's noise, so
    #    pinned-serial normalisation cannot rescue us
    inherit = RATIO_BETWEEN_SUBMISSION.pct / MTP_LEG_BETWEEN_SUBMISSION.pct
    check(inherit > 0.95,
          f"ratio/mtp = {inherit:.3f}; if this were small the noise would be "
          "common-mode and differencing would help. It is not.")

    # 5. n=1 floors are labelled as such so they cannot be pooled by accident
    for f in (MTP_LEG_WITHIN_RUN_N1, SERIAL_LEG_WITHIN_RUN_N1):
        check(f.n_units == 1, f"{f.name}: must record n_units == 1")

    # 6. the retraction that matters: our next submission is a coin flip
    p = p_beats_zero(EFFECTS["our next submission's gain"])
    check(0.50 < p < 0.53,
          f"P(next submission wins) = {p:.4f}; ledger 166 says ~0.515")
    p_old = p_beats_zero(EFFECTS["our next submission's gain"],
                         SCORE_WITHIN_RUN)
    check(p_old > 0.60,
          f"against the retired floor it should look like ~0.61, got {p_old:.4f}")

    # 7. EVERY effect the campaign cares about sits below the board floor.
    #    If this ever fails, something genuinely board-detectable has appeared
    #    and the campaign's strategy should change.
    for label, eff in EFFECTS.items():
        check(abs(eff) < SCORE_BETWEEN_SUBMISSION.pct,
              f"{label} = {eff:+.4f} % EXCEEDS the board floor -- if this is "
              "real, re-read the strategy, do not silence this check")

    # 8. audit's exemption regex must actually exempt, and must not exempt a
    #    bare citation. Constructed inputs, not whatever happens to be on disk.
    check(bool(SCOPE_TAGS.search("SIGMA = 0.0978  # within-run only")),
          "a declared citation must be exempt")
    check(not SCOPE_TAGS.search("SIGMA = 0.0978"),
          "a bare citation must NOT be exempt")

    # 9. audit must FLAG a bare citation and PASS a declared one. Constructed
    #    inputs, so this tests coverage rather than whatever is on disk today.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "bare.py"
        bare.write_text("SIGMA = 0.0978\nprint(SIGMA)\n")
        check(audit([str(bare)]) == 1, "audit must FLAG an undeclared citation")

        good = Path(td) / "good.py"
        good.write_text("# within-run only, see ledger 166\nSIGMA = 0.0978\n")
        check(audit([str(good)]) == 0, "audit must PASS a declared citation")

        clean = Path(td) / "clean.py"
        clean.write_text("x = 1\n")
        check(audit([str(clean)]) == 0, "audit must PASS a file with no hits")

    # 10. this module is excluded from BOTH audit entry paths. It quotes every
    #     retired literal by necessity, so an audit that flagged it would be
    #     permanently red and would therefore stop being run (lesson 13).
    check(audit([str(Path(__file__))]) == 0,
          "audit must exclude its own authority file when given it explicitly")

    if errs:
        print(f"SELFTEST FAIL ({len(errs)})")
        for e in errs:
            print("  -", e)
        return 1
    print("SELFTEST PASS (9 groups)")
    return 0


def report():
    print("NOISE FLOORS, by variance component\n")
    for comp in ("between-submission", "within-run"):
        print(f"  {comp.upper()}")
        for f in ALL_FLOORS:
            if f.component != comp:
                continue
            n = f"n={f.n_units}" + (f" dof={f.dof}" if f.dof else "")
            print(f"    {f.name:<30} {f.pct:>7.4f} %   {n}")
            print(f"        {f.source}")
            if f.note:
                print(f"        {f.note}")
        print()

    fl = SCORE_BETWEEN_SUBMISSION
    print(f"EFFECTS priced against {fl.name} = {fl.pct:.4f} %\n")
    print(f"    {'effect':<32}{'pct':>10}{'sd':>8}{'P(win)':>9}")
    for label, eff in sorted(EFFECTS.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {label:<32}{eff:>+9.4f} %{eff/fl.pct:>8.3f}"
              f"{p_beats_zero(eff):>9.3f}")
    print("\n  Everything on the roadmap is below the floor. The board cannot")
    print("  adjudicate any of it. Local pre-registered ABBA with a null arm is")
    print("  the only instrument that can. Board silence is not evidence.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "selftest":
        sys.exit(selftest())
    if mode == "audit":
        sys.exit(audit(sys.argv[2:] or None))
    if mode == "report":
        report()
        sys.exit(0)
    print(f"usage: {sys.argv[0]} [report|selftest|audit [PATH...]]")
    sys.exit(2)
