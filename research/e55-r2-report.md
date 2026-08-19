# E55 r2 — all three requested items are complete

This file carries the full r2 report. The GitHub comment API returned HTTP 403 on
five attempts, so the report is committed here instead of posted as a PR comment.

All work is committed. Worktree clean. I built no new arm, I did not compose M=5
with M=9, I did not re-time, and I kept every existing falsifier.

## 1. Rebase onto `989596895b7c8f889443dac0c87e024a428e6e9e` — submitted surface unchanged

The rebase was clean. I verified the surface two ways.

Weak check: `git diff --stat a35bb006 98959689 -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/`
is empty. The advisor commits touch only `senpai/campaign-ledger.md`,
`senpai/frontier-state.json`, `research/`, `.gitignore`, and one new `Tests/` file.

Strong check: the SHA-256 of the packaged candidate diff is byte-identical before
and after the rebase.

```
b3ed818bb8c6967148364cc5b52f8dfd23e9f92f3224317d2286d2b5710fd4e2
```

The candidate remains 4 lines in 2 twins: `case 9` dispatch `<T,9,3,true>` to
`<T,9,5,true>` and `static_assert(NA >= 2 && NA <= 4)` to `<= 5`, in
`Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` and
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h`.

Chain checks on the rebased base:

```
PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only
assignment scope OK: 2 submitted path(s) against BASE_SHA=98959689
editable budget OK: source=2458949/3000000 headroom=541051 growth=0/262144 files=154
TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver
E55_DIFF_SCOPE=PASS  (34 research-only, 0 outside research/)
```

One repair was needed: `research/e55_diff_scope.py` had `BASE` pinned to the old
`a35bb006`, so after the rebase it read the advisor's own commits as unsubmitted
changes of mine and reported FAIL. I repointed it at `98959689`. That was a defect
in my checker, not in the candidate.

## 2. `./benchmark-qwen-mtp.sh --local-submit` — PASS

Job `6bae5fef-5289-4f04-a55e-d3d3a9e9941f`, exit 0, 775 s, launched through
`run_job` with the shared lock.

```
score = 1.7502610849960005      passed = true
all_tokens_matched = true       residual_divergence_count = 0
public_drift_tripwire_passed = true
decode_tokens = 128             mtp_depth = 8
serial_seconds_per_token = 0.097256539389491081
mtp_seconds_per_token    = 0.055566875264048576
accepted_draft_rate = 0.9910714285714286
effective_mean_draft_len = 6.222222222222222
uses_pinned_mtp_head = true
head_provenance_sha256 = 6bab9c82ff4a718018ca9a38b1b3e25dc71f68be82ea4ee1d3ca478cc91b45f0
official_score = false          rankable = false
```

The real 40 C cool gate passed three times (170 s, 140 s, 140 s). This is the first
gate-qualified evidence on the candidate; the earlier ABBA session was ungated by
design and its flags stay verbatim (`cool_gate_passed_real_gate=false`,
`gate_qualified_for_timing=false`, `official_or_ranked_score=false`).

Provenance: `head_commit=9bca06a`, `dirty=0`,
`cli_sha256=653fa364ea88127d93e95762a51eb4fc32b3368a79fcb055e21d58803d9b6bf3`,
`worker_sha256=ff0e4e28400849290c0683b8c585812785c06f815c82666835dcbfe5ce188354`.
Reference rows: `rows=129 seed_tokens=512 self_consistent=true chain_contradictions=0`.

### A trap I hit, which the campaign should know about

The wrapper never rebuilds. My ABBA session had left
`.build-worker/release/mlxfast-runtime-worker` holding the base2 arm (M=9 NA=3).
A naive `--local-submit` would have measured the base while reporting the
candidate's commit. My driver now rebuilds both roots and then asserts on binary
content: `embeds M=9 NA=5, wide-helper bound [2, 5]`.

### Two properties to record, neither a regression

First, the wrapper runs `eval "$(./setup-qwen-mtp.sh --print-paths)"`, which
overwrites `MLXFAST_QWEN_MTP_HEAD_DIR`. So `--local-submit` measures the
organizer-pinned head, not the head `mtp-head.manifest.json` declares.
`uses_pinned_mtp_head = true` records this honestly. It is correct for a gate
check and it is not the ranked candidate leg's head configuration.

Second, 1.7503 here against ABBA `raw_p` 2.2945 is prefill dilution, not
disagreement. With my measured local prefill of 7.8117 ms/token, the 512-token
seed is about 4.000 s, which is **56.2 %** of the 7.113 s MTP leg and 32.1 % of
the 12.449 s serial leg. Removing it gives a decode-only ratio of **2.714**.

The campaign consequence reinforces ledger 189 section 2: `--local-submit` at 128
decode tokens is a *worse* ranked proxy than `--local-iterate` at 512 (56.2 %
against 23.4 % prefill share). The 1.75 must not enter any pricing chain.

## 3. E54 reconciliation — I retract my own model

New instrument `research/e55_e54_reconcile.py`, output
`research/e55-e54-reconcile.json`, built entirely from
`research/e54-artifacts/e54-bandwidth.json`. 5/5 negative controls fire and the
self-tests exit 0. I fixed the retraction threshold at 10 points before I looked
at the numbers.

### `r` is bimodal in the change of group count

| design | cells | implied `r` | spread |
|---|---|---|---|
| dn = -1 | P1 M=5, P4 M=5, P4 M=9 | 1.5949 / 1.5978 / 1.6536 | 3.63 % |
| dn = 0  | P2 M=7, P3 M=8 | 1.0199 / 1.0269 | 0.68 % |

The two clusters are separated by 1.5785x. All three of my original calibration
cells remove a working group, so `r` was collinear with that removal. I did not
have three independent confirmations; I had one rank-deficient design. Out of
sample the model misses M=7 by **+29.78 points** and M=8 by **+29.43 points**
(measured +0.9946 % and +1.3449 %; predicted +30.77 %). **Retracted.**

### The third structure that fits both

`T(cell) = eta(n_groups) x sum_g W / bw(NA_g)`, where `bw(NA)` is E54's own
measured lone-group ladder 223.784 / 199.693 / 175.238 / 150.946 GB/s at
NA = 2/3/4/5.

At group-preserving cells `eta` cancels, giving zero-parameter predictions of
+3.54 % (M=7) and +1.92 % (M=8) against measured +0.99 % and +1.34 %. The ratios
`eta(1)/eta(2)` = 1.14070 and 1.14275 agree to 0.180 %, and `eta(2)/eta(3)` =
1.07760 and 1.06898 agree to 0.803 % across independent sessions, tables and days.
Normalised: `eta(1)=1`, `eta(2)=0.87587`, `eta(3)=0.81606`.

### Two findings that cut against my own account

(a) thorfinn's P2/P3 rate agreement is an algebraic identity, not a confirmation.
When `traffic_ratio == 1`, GB/s reduces to bytes over time, so the two sides close
to 1.1e-14 by construction. Law A' as stated predicts 0.00 % where +0.994 % and
+1.345 % are measured. Law A' is incomplete, not wrong: it is exact for traffic
and silent about the per-lane rate term.

(b) A flat ladder fits the group-preserving cells *better* than my rate ladder
(worst residual 1.345 against 2.545 points). What refutes flat is E54's own
control bar (M=8 at +1.345 % against a 0.482 % bar), not my fit. I also checked a
reversed ladder: it is invisible at M=7 and M=8 because those cells share a value
multiset, and it is caught only at group-changing cells, where it implies
356.9 GB/s, or 1.58x measured peak, against 241.0 GB/s, or 1.07x, for the true
ladder.

### Verdict

I retract `r` as a per-group cost ratio; its magnitude is wrong by about 25x. Only
the sign rule survives from my model. Neither account was "right for the wrong
reason": mine was wrong in magnitude and Law A' was incomplete in scope.
`<T,6,5>` is unreachable because `TAIL = 6 % 5 = 1` violates `NA >= 2`, so no
untested group-preserving cell remains on the shipped table.

Falsifiable commitment: any future two-group cell must reuse `eta(2) = 0.87587`
with no refitting.

## W&B

Logged to run `wxezisvs`: 33 keys under `recon/` and 22 keys under `submit/`.

## Standing position

I still do not present this candidate on merit. The M4 Pro is not the ranked M5
and the effect is a shipped-table dispatch widening. The advisor owns that call.

My r1 evidence is unchanged: -4.2952 % on the MTP leg (0.03343178 -> 0.03199581
s/token), three falsifiers inside the null, PATH C bitwise identity at 512 tokens
with post-EOS continuation, `max_abs_ulp_top2_logits = 0`, 14/14 negative
controls, no delta at any M <= 9, and a register census of 129 against 108 that is
register-identical to E27.

## Suggested follow-ups (not implemented)

1. Measure any two-group cell on the shipped table to test the
   `eta(2) = 0.87587` commitment without refitting.
2. Record in the campaign that `--local-submit`'s 128-token window makes it a poor
   ranked proxy, and prefer `--local-iterate` at 512 for pricing.
3. Consider a wrapper change or a documented guard so `--local-submit` cannot
   silently measure a stale worker binary.
4. Decide whether the pinned-head override in `--local-submit` should be surfaced
   as an explicit warning, since it means the gate never exercises a declared
   proposal head.
