# E83 r2 terminal result — submission payload

This file records the exact payload the r2 submission carries.

## The GitHub 403 window, for the record

Every typed GitHub operation returned `HTTP 403` from about 20:27Z, on seven
attempts across a controller-turn boundary:

```
get_prs                  -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/85 returned HTTP 403
post_assignment_comment  -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/85 returned HTTP 403
submit_experiment_result -> GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/85 returned HTTP 403
curl with $GITHUB_TOKEN  -> HTTP 401 Bad credentials
git ls-remote origin     -> works
```

`get_prs` succeeded at 20:13Z at the start of that session, so the credential
expired mid-session and recovered later without any action here. This is the
same failure qwen-edward recorded for PR #83 in
`research/e80-artifacts/terminal-result.md`. No shell push and no shell GitHub
mutation was attempted during the window.

## Submission identity

| field | value |
|---|---|
| `branch` | `qwen-thorfinn/e83-prefill-decomposition` |
| `remote_branch_sha_before_push` | `8fb1a9bf7feddf4acd989a58fe1336d264173efe` |
| `assignment.repo` | `morganmcg1/qwen38-challenge_senpai` |
| `assignment.pr_number` | `85` |
| `assignment.assignment_id` | `qwen38-r1-e83-decompose-the-untouched-prefill-leg` |
| `assignment.revision_id` | `r2` |
| `assignment.student` | `qwen-thorfinn` |
| `status` | `succeeded` |
| base | `07c75a708c2347021d3148d7bc87b246ba2aec73` |

`expected_head_sha` and `commit_sha` both equal the final commit of this
branch. Resolve it with `git rev-parse HEAD` and re-read
`git ls-remote origin refs/heads/qwen-thorfinn/e83-prefill-decomposition`
before any retry, because the remote head may move.

## Primary metric

| field | value |
|---|---|
| `name` | `seed_prefill_begin_ms_local_m4pro` |
| `direction` | `minimize` |
| `baseline` | 4042.9 |
| `candidate` | 4036.5 |
| `delta` | -6.4 |

## Runs

| run | url | state |
|---|---|---|
| `hl39g0tm` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/hl39g0tm | finished |
| `l2xex14v` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/l2xex14v | finished |

## Hypothesis

The 512-token seed prefill leg, which is 8.6-9.4% of the scored candidate leg
and has never been moved by 470 board runs, contains recoverable time that an
isolated-cell roofline can locate; specifically the two GDN/MLP fusion bounds
(G1 in_proj 9->512, G2 gate_up 16->512) should recover about 115 ms of the
~4043 ms seed leg.

## Summary (verbatim, 3626 characters, fits the 4000-character field)

NEGATIVE / CLOSED. The two assigned prefill fusion gates recover +7.1 ms of a 4042.9 ms seed leg (0.176% local, ~0.016% of the ranked candidate leg) against a stop rule of 40.7 ms. The isolated-cell roofline predicted 115 ms, so it over-predicted by 16x. Nothing in E83 should be submitted. Detail: PR comment 5360553157 and research/results/qwen38-r1-e83-prefill-decomposition.md.

R2 REVISION, no GPU work; every measurement below is unchanged. Revert 313fd74 takes the rung-3 instrument out of Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift, and the branch is rebased onto advisor head 07c75a708c2347021d3148d7bc87b246ba2aec73, so `git diff 07c75a70 HEAD -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/` prints nothing. The branch changes no candidate byte. Gates on the rebased tree: rebuild-and-assert-worker.sh PASS (worker db0dcafe; require qwen35DualRMSNormConcat=14, forbid the 3 instrument symbols=0, two-sided against stale worker 70693f8f which reports the exact inverse); swift test 705 tests / 53 suites / 40 issues / the same 9 known names; twin_audit.py exit 0, 29 twins, 1 waiver; check-editable-budget.sh 770a3ff2 exit 0, growth 29980/262144 all organizer and 0 from E83; verify-ranked-score-boundary.sh PASS. Sources/, Vendor/ and Package.swift are byte-identical at f7f356b2, at 07c75a70 and at this head, so the worker and swift test gates carry to the final head. Instrument preserved at 7ef3f15 and e277761.

RUNG 3 (W&B l2xex14v, 8 ABBA reps x 4 arms, 0 expectation failures). Arm medians ms: baseline 4042.9, g1 4041.7, g2 4037.0, g1g2 4036.5. Paired saving: g1 +2.6 ms (5/8 reps faster), g2 +5.5 ms (8/8), g1g2 +7.1 ms (7/8).

G1 IS NOT BIT-EXACT. first_primary = 271 in every arm, but top2_values move from (21, 15.6875) to (21.125, 15.6875): one bf16 ulp, deterministic in 8/8 reps, because the fused a/b pack turns N=48 into N=16480 and straddles the out_vec_size >= 4096 branch at quantized.h:1917. G2 keeps both widths above 4096 and is bit-identical.

WHY THE ROOFLINE OVER-PREDICTED 16x. (1) An isolated cell below saturation measures its own eval() round trip: gdn.in_proj_a/b reach 0.26/0.33 TFLOP/s isolated vs 6.4 at N>=6144, and in situ they hide inside 2233 dispatches; the ladder is 82.7 ms isolated -> 28.0 ms in-situ -> 2.6 ms recovered. (2) Fusion removes launches and activation re-reads, not FLOP. (3) The isolated cells that execute sum to 4090.3 ms, 44 ms MORE than measured begin(). RULE: an isolated-cell roofline over-states recoverable time whenever the cell does not saturate the GPU; a fusion saving is bounded by removed traffic and removed launches, never by the difference of two isolated cell times.

RUNGS 0-2 (W&B hl39g0tm). begin() median 4046.1 ms (n=32); target_forward_build 75.0%, target_forward_eval 24.5%, tail_norm+lm_head 0.5%; positive control passes. H-221, a 512-width cliff, is FALSIFIED: the ladder is linear, -17.79 ms residual at 512 against a +7.7 ms prediction and a 31.5 ms noise band. GEMM share is 99.7% at 6.18 TFLOP/s, so non-GEMM headroom is <= 32 ms (<= 0.8% local). The measured 4.0461 s seed leg supersedes the inherited P = 4.0086 s and 6.15-6.18 TFLOP/s supersedes 6.415, so g = 7.69; the write-up cites neither old value.

LABELS: harness=local, M4 Pro 20-core / 48 GiB (ranked host is M5 / 128 GiB), cool_gate_passed_real_gate=false, gate_qualified_for_timing=false, official_or_ranked_score=false, entry temp spread 52.03-58.53 C.

Follow-ups: (a) attack the 6.18 TFLOP/s quantized GEMM path at M=512, not the fusion boundaries; (b) reuse the rung-3 ABBA gate harness; it caught a 1-ulp break an argmax check would miss.

## The r2 revision, as it would have been posted to the PR

### (a) The candidate-surface instrument is out

Revert commit `313fd74` removes the three `nonisolated(unsafe)` globals, the two
comparisons that read them at `Qwen35.swift:1031` and `:1322`, and the three
`qwen35FusedPackBuildCount` increments. `Qwen35.swift` is byte-identical to the
base, and
`git diff 07c75a70 HEAD -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/`
prints nothing.

Two consequences, both recorded in the write-up:

1. The rung-3 arms in `Tests/MLXFastTests/E83PrefillDecompositionTests.swift`
   drove those globals and cannot compile without them, so that section is
   removed and replaced by a comment that points at the replay commits.
2. `research/e83_prefill.sh gates` now exits 2 with replay instructions.
   `research/e83_report.py` is unchanged and still reads the recorded arms out
   of `research/results/e83/gates.json`. I re-ran it: same table, same
   +2.6 / +5.5 / +7.1 ms paired medians, same one-ulp G1 verdict.

Replay commits: `7ef3f15` is the instrument as measured on the r1 base;
`e277761` is the same instrument replayed onto this base by the rebase.

### (b) Rebase

Two steps, both clean and without conflict. Step one is
`--onto f7f356b2834518ced918f3049ca1b88afb6003f3` from the old merge base
`117e5bb5`. It flattens the assignment-marker merge commit, so the history is
linear. Step two is `--onto 07c75a708c2347021d3148d7bc87b246ba2aec73`, the
advisor head named in the r2 feedback. `07c75a70` descends from `f7f356b2` and
adds only `research/CURRENT_RESEARCH_STATE.md`, `senpai/campaign-ledger.md` and
`senpai/notes/r1-compose-on-8e83c6b3.md`, so no build input moves. The organizer
concat block lands near line 1736, far from every E83 region.

### (b2) Reconciling the rebase with the published head

`submit_experiment_result` requires the local head to fast-forward the
published head, and a rebase breaks that by construction. The first submission
attempt failed with `local head does not fast-forward remote head`.

The fix keeps every published commit. A merge with `-s ours` records the
published head `8fb1a9bf7feddf4acd989a58fe1336d264173efe` as a second parent and
keeps the rebased tree byte for byte:

```
git merge -s ours 8fb1a9bf7feddf4acd989a58fe1336d264173efe
tree(HEAD) == tree(dfde3813112f5c3a9102eeeb930cd96d4dfd1807)   YES
8fb1a9b is an ancestor of HEAD                                 YES
07c75a70 is an ancestor of HEAD                                YES
git diff 07c75a70 HEAD -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/
                                                               prints nothing
```

The published branch therefore fast-forwards again, the merge base with
`senpai/qwen38-mtp-r1` is `07c75a70`, and the pull-request diff is the
research-only file set. Nothing published is discarded and nothing is
force-overwritten.

### (c) The four gate numbers on the rebased tree

```
senpai/rebuild-and-assert-worker.sh: PASS
  worker_mtime   2026-08-20T20:19:11Z
  worker_sha256  db0dcafe6d58d4df3323584ac099be9c64a1e65c50546b70bf00d7dcf0b46606
  ok require-symbol qwen35DualRMSNormConcat  : 14
  ok forbid-symbol  qwen35FusedInProjMaxRows : 0
  ok forbid-symbol  qwen35FusedGateUpMaxRows : 0
  ok forbid-symbol  qwen35FusedPackBuildCount: 0

swift test --force-resolved-versions: 705 tests in 53 suites, 40 issues,
  9 failing functions, 20.9 s, exactly the nine names in
  senpai/known-test-failures.md

python3 research/twin_audit.py: exit 0
  29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)

senpai/check-editable-budget.sh 770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf: exit 0
  source=2484815/3000000 headroom=515185 growth=29980/262144
  exempt=2410/2147483648 files=154

senpai/verify-ranked-score-boundary.sh: PASS
```

All five gates were re-run on the final head after the second rebase, at
2026-08-20T20:56Z. `rebuild-and-assert-worker.sh` reports PASS again with the
same `db0dcafe` digest and 574,507 extracted symbols; the rebuild is a
content-addressed no-op because no build input moved. `twin_audit.py`,
`check-editable-budget.sh` and `verify-ranked-score-boundary.sh` print the same
numbers as above.

The worker witnesses are two-sided. The stale worker built at 19:25:49Z from the
pre-rebase, pre-revert tree, digest
`70693f8f1d175d86fc955dc98bb180dc0df0f9ec84ea8dd1ef64f3f6c69f8809`, reports the
exact inverse: `qwen35DualRMSNormConcat` 0 and each instrument symbol 4. Neither
the require side nor the forbid side is a guard that cannot fail.

`growth=29980` is measured against the contract base `770a3ff2`, so it prices
the organizer commits this base adopted. E83 contributes 0 bytes.

### The known-test-failures gate, corrected

`senpai/known-test-failures.md` now states that the gate is the failing name set
**and** the issue count; that the bare exit code is never the gate, because the
run exits 1 whenever any of the nine fails; and that the test and suite counts
are never the gate, because they move with what a branch adds.

One correction to the advisor's figure, measured: on this tree the branch
reports **705 tests in 53 suites**, with the E83 suite included in that 705. So
the 705-versus-710 difference is not explained by the E83 instrument, and the
document no longer claims it is. It records both totals and rests the gate on
the 9 names and 40 issues, which are identical on every branch. The document
also gained a "re-measured at `f7f356b2`" line: same nine names, same 40 issues,
so the organizer sync introduces no new failure. `07c75a70` carries the same
`Sources/`, `Vendor/`, `Tests/` and `Package.swift` trees as `f7f356b2`, so that
measurement applies to the final head.

### Write-up changes

`research/results/qwen38-r1-e83-prefill-decomposition.md` records the r2
identity, the revert and replay commits, the empty candidate diff, and the four
gates. The two superseded constants are gone: the local-to-ranked seed ratio now
reads `g = 7.69` from this session's measured 4.0461 s against the frontier's
published 0.5264 s, and the transfer table is recomputed on it. The inherited
`P = 4.0086 s` and 6.415 TFLOP/s are cited nowhere in the file.

No number in rungs 0-3 moved.
