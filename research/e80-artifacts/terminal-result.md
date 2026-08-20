# E80 terminal result — submission handoff

The E80 experiment is complete. The terminal submission could not be published
because every typed GitHub operation returned `HTTP 403` at the end of the
session:

```
submit_experiment_result -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/83 returned HTTP 403
post_assignment_comment  -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/83 returned HTTP 403
get_prs                  -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/83 returned HTTP 403
curl with $GITHUB_TOKEN  -> HTTP 401 Bad credentials
git ls-remote origin     -> works
```

`post_assignment_comment` succeeded three times earlier in the same session, so
the credential expired mid-session. This file records the exact payload so a
retry needs no re-derivation.

## Submission identity

| field | value |
|---|---|
| `branch` | `qwen-edward/e80-per-kernel-gpu-time-census` |
| `remote_branch_sha_before_push` | `84dc616db9aaede7bdd78241bc758baddd1121d5` |
| `assignment.repo` | `morganmcg1/qwen38-challenge_senpai` |
| `assignment.pr_number` | `83` |
| `assignment.assignment_id` | `qwen38-r1-e80-per-kernel-gpu-time-census` |
| `assignment.revision_id` | `r1` |
| `assignment.student` | `qwen-edward` |
| `status` | `succeeded` |
| pinned base | `2eec73d352af2e689c91236e8eac89413797a19d` |

`expected_head_sha` and `commit_sha` must both equal the commit that adds this
file. Resolve it with `git rev-parse HEAD` and re-read
`git ls-remote origin refs/heads/qwen-edward/e80-per-kernel-gpu-time-census`
before the retry, because the remote head may have moved.

## Primary metric

| field | value |
|---|---|
| `name` | `verify_width_tax_attributed_fraction` |
| `direction` | `maximize` |
| `baseline` | `0.774` (E71: 44.416 of 57.404 ms) |
| `candidate` | `0.9996` (E80: 57.229 of 57.253 ms) |
| `delta` | `0.2256` |

## W&B runs

All in `wandb-applied-ai-team/qwen38-mlx-challenge-senpai`, group
`e80-per-kernel-gpu-time-census`, all `finished`. URL pattern
`https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/<id>`.

| leg | gate | widths | run id |
|---|---|---|---|
| `e80-census-w6-default` | gated | 1, 6 | `azgwyge5` |
| `e80-census-w6-isolated` | gated | 1, 6 | `y0gdzebh` |
| `e80-census-w5-isolated` | gated | 1, 5 | `xe118mah` |
| `e80-census-w1-default` | gated | 1 | `gcdjmemd` |
| `e80-census-w1-isolated` | gated | 1 | `cowjscbn` |
| `e80-hot-w4-default` | ungated | 1, 4 | `jok8s3qh` |
| `e80-hot-w4-isolated` | ungated | 1, 4 | `nr4q5wpn` |
| `e80-hot-w5-default` | ungated | 1, 5 | `d0kmcp1q` |
| `e80-hot-w5-isolated` | ungated | 1, 5 | `kbgk2bs9` |
| `e80-hot-w9-default` | ungated | 1, 9 | `ul97w9b4` |
| `e80-hot-w9-isolated` | ungated | 1, 9 | `xav95tv4` |
| `e80-rung1-gate` | gated | 1, 6 | `ws1e4j5m` |
| `e80-rung1-control` | gated | 1, 6 | `blld7vtb` |

## Hypothesis

A per-kernel GPU-time census inside the scored worker can attribute the
verify-width tax to named kernel dispatches with at least 95 percent closure,
and can name the 22.6 percent that E71's arm-based decomposition left
unattributed.

## Summary

E80 closed the verify-width tax to 99.96 percent at width 6. E71 attributed
44.42 of 57.40 ms (77.4 percent). This census names 57.229 of 57.253 ms.

THE MISSING 22.6 PERCENT is two raw `quantizedMM` calls that no E71 arm could
reach: `gdn_in_proj_fused` (`qmv Mx2060x1`, 7.820 ms) and `fa_qkv_gate_fused`
(`qmv Mx1792x1`, 2.343 ms), plus 1.789 ms of non-qmv drafting-only work and
0.70 ms of E71 undercount in the pooled `Mx640x1` row. Total 12.65 ms = 22.10
percent of the tax, against an E71 residual of 12.609 ms (0.3 percent
agreement). Both call `quantizedMM` directly in `Qwen35GatedDeltaNet` and
`Qwen35Attention` (`Qwen35.swift:677-689`, `:1707-1712`), never through a child
`Linear`, which is why no arm reached them.

The naming holds across the histogram: those two rows are 18.7 percent (w4),
18.9 (w5), 17.8 (w6), 19.4 (w9) of each width's own tax. Closure is 99.63 (w4),
99.81 (w5), 99.98 (w9). `unclassified_kernels` = 0 at every width.

THREE FINDINGS BEYOND THE ASSIGNMENT.

1. The proposal head costs a flat 4.810 ms per draft token with no fixed cost:
   `draft_head_ms = 4.8098 * drafts - 0.1130`, R2 = 0.999972 over ungated widths
   4/5/9. Gated w5 and w6 land on that line un-pooled (-0.81, +0.88 percent).
   The resident head is 849,398,784 bytes of bf16 read at 245 GB/s in five
   `gemv` rows. The declared `mtp-head.manifest.json` is 427,742,600 bytes =
   50.36 percent of that, so local measurement is pessimistic by 5-7 percent of
   whole-round GPU time, and that error cannot cancel in the local
   serial-to-MTP ratio because only the MTP leg drafts. Falsifiable
   discriminator: the ranked candidate head phase should dispatch `qmv`, not
   `gemv`.

2. The ceiling is flat while the break-even rises. Ceiling = M * F(1) / round =
   2.708 (w4), 2.817 (w5), 2.683 (w6), 2.850 (w9), a 6.2 percent span.
   Break-even acceptance climbs 15.9 -> 19.4 -> 24.7 -> 27.0 percent. Replacing
   the head with the declared artifact lowers every break-even by 2.5-2.7 points
   and raises every ceiling by 0.14-0.20. That is larger and more certain than
   any depth change.

3. H-221 is dead at every width. Host synchronisation waits are 0.00 per round
   on 18 of 20 leg-and-width rows and 0.01 on the other two. Blocked time is
   0.000 ms everywhere. In default packing the unexplained host gap is
   0.375-0.632 ms per round at w1-w6 and negative at w4/w5/w9, against H-221's
   claimed 0.35 ms. Per-dispatch cost is 0.66-1.55 us; per-commit cost is
   13.5-17.6 us.

RIDERS. The `gemv` rider FAILED at 10.30 (w4), 11.56 (w5), 13.39 percent (w9)
against a <2 percent bound. The cause is benign: `gemv` is the bf16 proposal
head, not a target family. The `copy` rider passes at 0.19-0.42 percent but
rises monotonically with width at about 20x the expected 0.02 percent. Ledger
218 stays closed; the trend is recorded.

SCOPE. The instrument is fully reverted.
`git diff BASE -- Sources/ Vendor/ Tests/ mtp-head/ mtp-head.manifest.json` is
empty. `check-editable-budget.sh` reports `growth=0/262144`,
`source=2469371/3000000`. `swift build -c release --force-resolved-versions`
passes in 39 s. The instrument is preserved as
`research/e80-artifacts/gputime-census.patch`, verified byte-identical to the
live diff before the revert.

CAVEATS. Host is Apple M4 Pro (`applegpu_g16s`), 20 GPU cores, not the ranked
M5; all numbers are directional. Sweep 2 ran ungated under the standing
counterbalanced exception: `cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`, ABBA order D I I D D I, entry spread 10.4 C.
The cool gate is worth +0.12 percent here, below the +/-0.25 ms noise floor.
Two sweep-2 legs recorded `dirty=1` and `dirty=6`; the dirty files are
research-only Python, all six legs record `candidate_sha=cdf33bb6`, and the
worker binary mtime (18:37:10Z) precedes the first leg (18:37:19Z) and never
changed. Census legs force the draft count, so break-even rates are a
requirement, not an observation. The GPU-time window covers decode only; the
ranked leg also times an ~8.0 s seed.

Full report: `research/e80-results.md`.

## Base drift

The advisor branch `senpai/qwen38-mtp-r1` moved to
`7fc6453c1ec047ba94e4946d8089c6e1d8735e8e` after this experiment pinned
`2eec73d352af2e689c91236e8eac89413797a19d`. The move does not invalidate the
result: the instrument is reverted, the candidate-surface diff against the
pinned base is empty, and the finding is a measurement of the scored path
rather than a code change to compose.
