# E122 rung 0 pre-registration

Written and committed **before** any discrimination statistic was computed on the
forced-depth arm. The only forced-arm quantities inspected before this file was
committed are validity fields, listed in "already inspected" below. None of them
is a discrimination statistic.

`harness=local`, `timing_valid=false`, `official_or_ranked_score=false`.

Requested by the advisor in
<https://github.com/morganmcg1/qwen38-challenge_senpai/pull/123#issuecomment-5377783987>.

## Already inspected on the forced arm

Validity only:

- `all_tokens_matched=true`, `parity_all_ok=true`, `residual_divergence_count=0`
  on all nine legs;
- `effective_max_draft_len=7` on all nine legs, which is the behavioural proof
  that the width clamp is live and the arm never entered verify width 9;
- round counts and mean drafted depth per leg.

## Primary gate statistic

One prompt-stratified concordance statistic pooled over draft positions 2 to 5
on the **forced** arm.

Construction, exactly as the advisor specified:

1. For each `(prompt, position)` cell, count concordant, discordant and tied
   pairs between the target top-2 margin and the binary conditional acceptance
   of that draft position.
2. Aggregate the raw counts across all cells.
3. Form the ratio once, at the end. Ties score one half.

A pair is only formed inside one `(prompt, position)` cell, so neither a
between-prompt difference in margin scale nor a between-position difference in
acceptance rate can enter the statistic.

Interval: cluster bootstrap resampling rounds within prompt, 2000 replicates,
seed 122.

Pre-registered thresholds, unchanged from the assignment:

| pooled concordance | verdict |
| --- | --- |
| at or below 0.55 | kill the axis, report the null, no timed session |
| 0.55 to 0.65 exclusive | stop and ask the advisor before rung 1 |
| at or above 0.65 | rung 1 is licensed |

## Pre-registered predictions

These are my expectations before looking. They are recorded so that a surprise
is visible as a surprise.

- **Pooled concordance: 0.55, plausible range 0.53 to 0.58.** Basis: the shipped
  arm's per-position stratified AUC at positions 2 to 5 was 0.5252, 0.5803,
  0.6235 and 0.5234, and the pair counts there are dominated by position 2. The
  forced arm removes range restriction, which should raise the deep positions
  slightly, but position 1 in the shipped arm was already unselected and read
  0.5420, so I do not expect the pool to clear 0.60.
- **Implied Somers' D = 2 x AUC - 1: 0.10, plausible range 0.06 to 0.16.**
- **Within-prompt rank correlation of margin against the round's accept run
  length: 0.15, plausible range 0.05 to 0.25.**

## Agreement rule

The advisor's rule: the rank correlation and the pooled concordance statistic
must agree under `D = 2 x AUC - 1`, and a disagreement above 0.10 in `D` means
one of them is wrong and must be reported rather than resolved by preference.

I accept the rule and add one correction to make it exact. Spearman's rho
between a continuous margin and an ordinal run length is **not** algebraically
equal to the Somers' D of a pooled binary concordance; the two coincide only
under assumptions this data does not guarantee. Reporting only Spearman would
therefore let a definitional gap look like a measurement fault.

So I will report **both**:

- `spearman_margin_accepted`, the quantity the advisor named; and
- **`somers_d_margin_runlength`**, the within-prompt Somers' D of the accept run
  length given the margin, computed from concordant and discordant pair counts
  in exactly the same way as the primary statistic.

The second is the algebraically comparable one. The agreement check is applied
to it. If Spearman and Somers' D themselves differ materially, that difference
is a property of the two definitions and I will say so.

## `natural_history` is reported separately and can overrule the pool

`natural_history` is the local hard-end proxy. It is **not** beagle and will
never be labelled beagle. It is the local prose prompt with the lowest accepted
draft rate, which makes it the closest available analogue to the prompt that
carries half the published score.

Its own pooled concordance statistic, computed over positions 2 to 5 within that
prompt alone, is reported as its own row with its own verdict. Per the advisor:
if the pooled statistic and the `natural_history` statistic disagree in
direction, `natural_history` wins.

Direction here means which side of 0.55 the value falls on.

## Secondary, not gated on

- per-position stratified AUC with `n`, rejection count and `not_drafted`;
- raw pooled AUC and within-prompt standardised AUC;
- margin distribution per prompt, and whether a scale-free threshold is possible;
- the modelled value of a margin-conditioned depth schedule from
  `research/e122_value.py`, which prices the axis in decode time rather than in
  discrimination. Its `--self-test` control passes: an uninformative margin
  reports a 0.00 % held-out pool and a margin that determines the run length
  reports 23.90 % of a 32.66 % oracle pool.

The value model is reported because a concordance statistic near 0.55 still
leaves open how much time a perfectly exploited weak signal would buy. It is a
model evaluated on traced acceptance data, not a measurement, and it ignores
rejected-work cache traffic, rollback and replay, so it flatters any deep policy
and should be read as an upper bound.
