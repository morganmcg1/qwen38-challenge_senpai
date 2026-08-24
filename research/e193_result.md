# E193 terminal result — E165 round-start head-chain prefetch, retested alone

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":"2681c3ac-9d9e-4367-aef1-fb3e4c587f35","yukon_status":"rejected","primary_metric":{"name":"ranked_published_score","available":true,"value":3.67457094706895},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- Student / branch: qwen-alphonse / `qwen-alphonse/e193-e165-prefetch-retest-build` (PR #190)
- Hypothesis: the E165 round-start head-chain prefetch, run alone and unconditionally
  on the current tree, still lowers candidate MTP seconds per token.
- Target cost: head-chain step latency at round start, ~35 drafting rounds per
  256-token leg.
- Decision: PROMOTION FAILED. Official run rejected. Local screen was green but
  below the campaign MUE.
- BASE_SHA: `0416ea0bf8a04bf3a65c4eba1b5af438fcc6b837` (`senpai/qwen38-mtp-r1`)
- UPSTREAM_SHA: `80021bc03e4b270f7dfef5b4425107bfc57b8d70`
- Candidate commit (measured + submitted surface): `67662100`
- Frozen tip at submission: `e50b1511`
- Branch tip at result: `74b5f76d` plus this record
  VERIFIED: `git diff 67662100 74b5f76d -- Sources/ mtp-head.manifest.json mtp-head/` is EMPTY.
- Candidate build fingerprint: worker sha256
  `1ece42428e0b87784ce012360b7ceaffca30dfddbc79123cef546b80d56e3251`,
  mtime `2026-08-24T16:42:34Z`. Digest identical before and after the timed run.
- Submitted candidate files: `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` (ONE file, +12/-31, subtractive)
- Supporting research-only files: `research/e193_screen.sh`, `research/e193_gates.py`, `research/e193_wandb.py`
- MTP head: DECLARED, `mtp-head.manifest.json` ->
  `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274`, archive sha256 `559b24eb`,
  427742600 bytes. Run tree digest `dadbfb80...`. Draft policy: depth 8, parent-offered cap.
- Token window / fixture / reference / harness:
  screen 256 tokens `harness=local` `--local-iterate` ungated;
  confirmation 512 tokens `harness=local` `--local-submit` REAL 40 C gate;
  reference rows candidate-generated (rankable=false).
- Official causal path: candidate-editable code lowers candidate MTP seconds per
  token only. d ln(ranked baseline serial time)/dx = 0. `senpai/verify-ranked-score-boundary.sh`
  PASS: ranked numerator is pinned baseline. No psi_serial subtraction anywhere.
- Assignment-scope preflight: `assignment scope OK: 1 submitted path`
- Editable budget: `source=2650169/3000000 growth=-975/262144` (NEGATIVE growth)
- Scored-path reachability: RULE 384/387 rebuild-and-assert PASS via three
  same-file env-var literals (require MLX_QWEN_MTP_TRACE_SYNC_HEAD / MLX_E159_FIXED_DRAFT_DEPTH /
  MLX_QWEN_MTP_TRACE_PATH; forbid MLX_E165_HEAD_PREFETCH; require-symbol
  prefetchHeadStep, undoHeadPrefetch).
- Promotion rule (predeclared): CONTINUE if contrast + 2 sigma < 0.
- Verdict: CONTINUE (-0.1317 % < 0).
- Pre-official evidence budget: 8 ungated screen legs + 2 gated 512-token
  confirmations (one pinned-head, RULE 389 auxiliary; one declared-head, the
  confirmation of record). The second confirmation was NOT budgeted; it was
  required after I found the head-provenance gap.
- Frozen candidate SHA: `67662100`
- Evidence that invalidated the frozen SHA: NONE.

## Official receipt — TERMINAL

Submission `2681c3ac-9d9e-4367-aef1-fb3e4c587f35`, submitted 2026-08-24T17:58Z,
terminal 2026-08-24T23:06:22Z. Wall time 5 h 08 min.

Verbatim board row:

```
2681c3a     morganmcg1            rejected          3.67457094706895   {"mode":"qwen-mtp-paired-decode-only","commit":"504a7bd7325a0c02a5b4da9712c9c...  -0.054539 (-5.49%)    504a7bd  8/24/26, 5:58 PM
```

Crown, checked immediately after the receipt and unchanged during the stall:

```
ec24d59     newjordan             promoted          3.7291100105909    {"mode":"qwen-mtp-paired-decode-only","commit":"0863b06ac16e26e48fc06e9744409...  +0.009513 (+0.96%)    0863b06  8/23/26, 9:44 AM
```

- ranked_published_score: `3.67457094706895`
- crown baseline: `3.7291100105909` (`ec24d59`, newjordan, src `0863b06`)
- delta: `-0.05453906352195`, reported by Yukon as `-0.054539 (-5.49%)`
- outcome: `rejected`, frontier not moved
- The board `commit` column shows `504a7bd`, which is Yukon's snapshot commit for
  the packaged submission. It is not a commit in this tree and it is not the
  candidate SHA `67662100`.

The delay was a board-wide validation stall, not an E193 anomaly: at 21:31-22:00Z
eight receipts from seven solvers were all `validating` and nothing had gone
terminal since about 17:35Z. The stall broke between 23:05:20Z and 23:06:22Z.
Two bounded read-only watchers covered the wait
(`ba95fb61-a6f6-4d36-9813-276150e44112` exhausted its 4 h budget,
`b0bd07df-6727-4349-a865-9ad43f95b7eb` exited 0 on the terminal row).
Exactly one official submission exists for E193. No duplicate was sent.

## Evidence

- Host: `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 51539607552 bytes,
  metallib fingerprint `ce5e223e...`. NOT the ranked M5.
- head_provenance_sha256, every leg:
  - screen legs p1..p8: `dadbfb80...` (DECLARED) -- all eight identical
  - confirmation of record: `dadbfb80...` (DECLARED)
  - auxiliary confirmation: `3a7fed84...` (PINNED) -- see caveat 5
- Cheapest falsification gate: 512 rows compared, 0 mismatched, positive control fires.
- Row-ledger closure RULE 179: `(35, 226, 221)` identical across all 8 legs.
- swift test: 41 issues / 10 distinct tests = recorded pre-existing count, ZERO new.

### Stage 1 screen, harness=local, 256 tokens, ungated, 8-leg palindrome

```
contrast on - off   -0.2165 %   (2 sigma 0.0848 %)
effect + 2 sigma    -0.1317 %   -> CONTINUE
per decode round    -545.3 us over 35 drafting rounds
```
Complete arm separation: on max 0.034383 < off min 0.034421, exact p = 1/70 = 0.0143.
Serial null channel +0.0484 % (2 sigma 0.2894 %), opposite sign.
Reproduces E165 historical -539.0 us/round to 1.2 %.

### Confirmation of record, harness=local, 512 tokens, DECLARED head, real 40 C gate

```
passed true | all_tokens_matched true | residual_divergence_count 0
rows 512/512 serial, 574/574 speculative | rounds 78 | EDL 6.358974358974359
accepted_draft_rate 0.8770161290322581
serial_seconds_per_token 0.07317157625220716
mtp_seconds_per_token    0.028624425642192364
local ratio 2.55626356199624 | rankable false
cool gate 39.9 C / 39.9 C
```

### Auxiliary confirmation, PINNED head (demoted under RULE 389)

```
local ratio 2.338153976942449 | mtp_seconds_per_token 0.031297115376219153
serial_seconds_per_token 0.073177474783733487
EDL 6.3766233766233764 | accepted_draft_rate 0.88594704684317716
```

## Interpretation

Two separate statements, kept apart on purpose.

1. **Promotion: failed.** `3.67457094706895` is below the crown
   `3.7291100105909`, so the run was rejected and the frontier did not move.
2. **Mechanism: unresolved officially.** Our seven official receipts on 8/24 read
   `3.70465`, `3.66785`, `3.65821`, `3.54743`, `3.65474`, `3.52364`, `3.67457`:
   a spread of `0.18`, about 5 %. The prefetch's measured local effect is
   `-0.2165 %` of candidate MTP time, roughly `+0.2 %` of score. The official
   channel is about 25x too noisy to resolve it, and the same-day receipts come
   from several different candidates, so no part of the gap can be attributed to
   this change. The local matched screen remains valid causal evidence on M4 Pro
   for the mechanism itself.

The local screen honoured its own predeclared rule, and the rule was satisfied
only marginally: `-0.2165 %` sits below the 0.39 % campaign MUE. That is the
clearest scientific lesson here. A sub-MUE local effect, even one with complete
arm separation and an exact p of 0.0143, does not survive contact with official
noise of this size.

## W&B

confirmation (declared) mtcsfj3s | confirmation (pinned, auxiliary) cyb9bm5d
stage1 rollup 1yqlaaza
legs: p1off 1m7ve4rp, p2on qpt31r8f, p3on ncyczevy, p4off qq2doc0d,
      p5off 0jpjvqjj, p6on 9zie5ke1, p7on peb45w4p, p8off sfavx6ox

## Reproduction

```bash
git checkout 67662100
./setup.sh && ./setup-qwen-mtp.sh
# screen
research/e193_screen.sh
# confirmation of record
MLXFAST_QWEN_MTP_HEAD_DIR="$PWD/mtp-head-declared-run" \
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit
```

## Honest caveats

1. -0.2165 % is BELOW the 0.39 % campaign MUE and at the bottom of the
   0.38-0.73 % expectation band.
2. Screen legs are ungated. Directional causal evidence within one
   counterbalanced session; NOT gate-qualified, never an official score.
3. M4 Pro only. M5 transfer unverified.
4. Local ratio is not a ranked estimate: both legs use the candidate build and
   the reference rows are candidate-generated.
5. Head-provenance history: the first 512-token confirmation ran the PINNED head
   because run_job starts a clean environment and benchmark-qwen-mtp.sh does not
   read the tracked manifest. Demoted to auxiliary under RULE 389; the
   declared-head rerun is the confirmation of record.
6. The official run cannot separate this change from base-versus-crown gap. We
   never submitted the unchanged base on this same tree, so there is no matched
   official control.

## Follow-ups suggested, NOT implemented

- Exclude `.mlxfast-reference-cache.lock` from `computeQwenMTPHeadProvenance`.
  It contains a timestamp, inode numbers and nanosecond mtimes, so the pinned
  head digest is host-local and time-local. Trusted-harness code, out of scope.
- Make `benchmark-qwen-mtp.sh` resolve the tracked manifest itself (fuller fix
  than the advisor's fail-closed guard `b4c76db1`).
- E65 / PR #68 flagged: merged, on the submitted surface, only 512-token
  confirmation recorded a pinned digest on a manifest-declaring base.
- E190 ship confirmation head provenance is UNRECORDED in local records.
- Audit whether E165-era conclusions in `research/e165_prefetch.py` inherited
  the HARNESS DEFECT 43 position-map collision.
- Raise the bar for official submission: require a local effect at or above the
  campaign MUE, or compose several sub-MUE winners before spending a slot.
  E193 spent a slot and 5 h of queue on a `+0.2 %` predicted effect inside a
  5 % noise channel.
