SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":true,"value":2.051860356452388},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

# e12 / r1 — What does the 512-token seed prefill actually charge the score?

- **Student / branch:** `qwen-alphonse` / `qwen-alphonse/seed-prefill-charge` (PR #14, assignment `qwen38-r1-e12-seed-prefill-charge`, revision `r1`)
- **Hypothesis and target cost:** the 512-token seed prefill is inside every timed leg, so it is an Amdahl floor on the decode ratio. The assignment predicted the charge was small (`p = P/D_mtp ∈ [0.01, 0.04]`, worth `<0.005` ranked score points). **Measured `p = 0.531` at the 300-token window and `p ≈ 0.311` projected at the ranked 512-token window — the prefill lever is worth `≈ 0.605` ranked score points, about 20.5 % of the promoted 2.9466.**
- **Decision:** green locally as a *measurement*. It is not a speed candidate: **zero `Sources/` bytes changed**. Prediction 1 is refuted by roughly 8–13x, prediction 5 is wrong in kind, and predictions 2–3 are confirmed.
- **`BASE_SHA` / `UPSTREAM_SHA` / candidate commit:** `fe38ecc21e4084e4d17dac3aa76264bb5897a614` / `32b94cb67d2f3a102a36382d2beb62eee8d99db5` / see PR head
- **Yukon promoted submission / source ref used as frontier:** `03dedda8-fc70-4e3e-881f-5384a17af405`, source ref `32b94cb67d2f3a102a36382d2beb62eee8d99db5`, score `2.94661597308114`
- **Submitted candidate files:** none. This experiment changed no file in `benchmark.json` `editablePaths`.
- **Supporting test, tooling, or documentation files:** `research/e12-run.sh`, `research/e12-analyze.sh`, `research/prefill_amdahl.py`, this report. All research-only; Yukon does not submit them.
- **MTP head provenance and draft policy:** organizer-pinned head, `uses_pinned_mtp_head: true`, `head_provenance_sha256 05a8613e3d86456f5df9bc8ab8c53daa5d19604c08d1b0bd215ad0d599cb2863`. Draft policy unchanged from base: depth 0 for the serial control arm, depth 8 for the MTP arm.
- **Assignment-scope preflight:** `git diff --stat fe38ecc..HEAD` = 3 files, +295 lines, all under `research/`. `senpai/validate-assignment-scope.sh` takes submitted paths as arguments and there are none, so there was nothing to validate.
- **Editable source bytes / headroom / growth / exempt-head bytes:** `source=2402203/3000000 headroom=597797 growth=0/262144 exempt=2410/2147483648 files=154` (`senpai/check-editable-budget.sh fe38ecc…`). **Growth is exactly 0.**
- **Scored-path reachability evidence:** the numbers come from the trusted parent's own per-phase ranked payloads captured through `MLXFAST_CAPTURE_DIR`, i.e. from the same `Qwen36MTPBlockSession` rounds the scored worker runs — not from a side harness. `seed_prefill_seconds` and `prefill_seconds_per_token` are published by base commit `3352917` ("Publish the seed-prefill rate in ranked payloads (observability only)"), so the charge is read directly out of the scored path rather than inferred.

## Evidence

- **Host, memory profile, toolchain, and thermal policy:** Apple **M4 Pro, 48 GiB** (`hw.memsize=51539607552`), macOS 26.5.2 (25F84), Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`), target `arm64-apple-macosx26.0`. **This is not the ranked M5 host `m5-qwen38-27b-mtp`, so every number here is directional.** All runs went through `benchmark-qwen-mtp.sh`, which holds the single-model lock and the 40 °C cooling gate; `research/await-lock-then-run.sh 420` waited for the lock rather than bypassing it. `macmon` is absent on this host, but `benchmark.sh` has its own probe and the job logs carry real per-phase pre-run temperatures (below).
- **Exact baseline and candidate commands:**
  ```bash
  research/e12-run.sh build
  research/e12-run.sh iterate 64  1 traced-64     # traced pair, MLX_QWEN_MTP_TRACE=1
  research/e12-run.sh iterate 512 0 raw-512       # FAILED, see blocker
  research/e12-run.sh iterate 300 0 clean-300     # headline pair
  research/e12-analyze.sh clean-300 traced-64
  ```
  `e12-run.sh iterate` sets `MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS`, `MLX_QWEN_MTP_TRACE`, `MLXFAST_CAPTURE_DIR`, `MLXFAST_SCORE_PATH` and then runs `./benchmark-qwen-mtp.sh --local-iterate`.
- **Tests and risk-based checks:** no `Sources/` change, so no numerical boundary was touched and no `swift test` regression is possible from this branch. The build was still rebuilt from scratch because the pre-existing binaries predated `3352917` and would have silently reported no `seed_prefill_seconds` field; `prefill_amdahl.py` therefore asserts `has_direct_prefill_fields` as a stale-binary guard, and it passed on both arms. `python3 research/twin_audit.py` is not relevant (no Metal edit).
- **Exact-token and row-ledger verdict:** **both arms clean at 300 tokens.** Serial `all_tokens_matched: true`, `residual_divergence_count: 0`, `declared_rows_total = 300`, reference-checked `300/300`. MTP `all_tokens_matched: true`, `residual_divergence_count: 0`, `declared_rows_total = 309`, reference-checked `309/309`. `public_drift_tripwire_passed: true`, `passed: true`, `rankable: false` (candidate-generated reference rows, as always in `--local-iterate`).
- **Divergent tokens or failure category, if any:** none in the headline pair. One separate hard failure is reported as a blocker below.
- **Generated-twin audit, if relevant:** not relevant.
- **Peak RAM or head/artifact size, if relevant:** not separately instrumented. Both arms ran resident on 48 GiB under the wrapper lock without pressure or OOM.
- **Official status and score, if submitted:** not submitted. Nothing here is a submittable candidate.
- **Binary identity:** all three runs used one build — CLI `.build/release/mlxfast-swift` `sha256 0a904c0df531ff7b1bb5a5d18da4fdd9b8024cfce0eccdc67be37fa7ebee20e5`, worker `.build-worker/release/mlxfast-runtime-worker` `sha256 1a0bc7fadbb08b5d7676bb18058d78c74a6390d55337f3267ebb89f07e86b3c1`.
- **Pre-run GPU temperatures (300-token job):** 39.4 °C before the reference pass, 39.9 °C before the serial control, 39.5 °C before the MTP arm — a 0.5 °C spread across the pair, all under the 40 °C gate.
- **Lock exit codes:** `lock_wrapper_exit=0` for `traced-64` and `clean-300`; `lock_wrapper_exit=1` for the failed `raw-512`.
- **W&B:** run `cl0jpovu` — <https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/cl0jpovu> (group `qwen38-r1-e12-seed-prefill-charge`, 139 scalars plus per-round `block_request_seconds` tables for both arms).

### Headline pair — matched `--local-iterate`, 300 emitted tokens, 512-token seed

| Metric | Baseline (serial, depth 0) | Candidate (MTP, depth 8) | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | 0.07890314022699992 | — | — |
| MTP seconds/token | — | 0.03845443964004516 | **2.051860356452388x** |
| local serial-relative speedup | 1.0 | 2.051860356452388 | +1.051860 |
| effective mean draft length | 0.0 | 6.725 | — |
| accepted draft rate | 0.0 | 0.966542750929368 | — |
| **`seed_prefill_seconds` (P)** | **3.999000072479248** | **4.003351926803589** | +0.11 % |
| `prefill_seconds_per_token` (seed) | 0.007810547016561031 | 0.00781904673203826 | +0.11 % |
| decode leg total (`decode_seconds`) | 23.670942068099976 | 11.53633189201355 | 2.0518x |
| decode work only (Σ `block_request_seconds` = D) | 19.671941995620728 (n=300) | 7.532979965209961 (n=40) | 2.6113x |
| rounds | 300 | 40 | — |
| **prefill share of the timed leg** | **16.894 %** | **34.702 %** | — |

Because the candidate code is byte-identical to `BASE_SHA` on every submitted path, "baseline" and "candidate" above are the two *arms* of one matched pair, not two code versions. The local score is a one-prompt directional measurement; it is not the ranked median over eight hidden prompts.

### The actual answer to the question

At the 300-token window, directly measured:

| Quantity | Value |
| --- | ---: |
| `p = P / D_mtp` | **0.5314433259204887** |
| `r` measured | 2.051860356452388 |
| `r_ideal` if prefill were free | 2.6114422295655775 |
| local leverage `r_ideal − r` | **0.5595818731131894** |
| identity cross-check `p·(r−1)` | 0.5590041662369679 (rel. error `1.03e-3`) |

Projected to the ranked 512-token window using the measured steady per-token rates:

| Quantity | Value |
| --- | ---: |
| `D_serial(512)` | 33.49540542041179 s |
| `D_mtp(512)` | 12.870633430154914 s |
| `p(512)` | **0.3110454468716388** |
| local-host `r(512)` | 2.2220242995190866 |
| prefill share of the ranked serial leg / MTP leg | 10.67 % / 23.74 % |
| **ranked leverage at the promoted `r = 2.94661597308114`** | **0.6054860352344933 score points** |
| implied `r_ideal` on the ranked host if prefill were free | 3.552102008315633 |
| score points per 100 ms of prefill removed (ranked window) | 0.007268313978889516 |

So a **1 %** cut in seed-prefill time is worth `≈0.006` ranked score points — already above the `0.005` threshold the assignment set for the whole of Phase 3. A 20 % cut is worth `≈0.121`; a 30 % cut `≈0.182`.

Two independent checks say the `3.999 s` is real fixed work and not a mis-attributed slice of decode:

1. **The exact asymmetric identity closes.** `local_leverage_measured = local_leverage_exact_asymmetric = 0.5595818731131894`, versus `0.5590041662369679` from `p·(r−1)` — `1.03e-3` relative error. The advisor's algebra is correct as written.
2. **A two-window fixed/variable solve agrees.** Solving `leg(n) = F + c·n` from the 64- and 300-token captures gives `F_serial = 4.011719430907298 s` and `F_mtp = 4.023294719599061 s`, i.e. `1.0032x` and `1.0050x` the directly measured `seed_prefill_seconds`. The solved per-token rates (`0.0655307` serial, `0.0250435` MTP) also agree with the direct steady rates to `0.2 %` and `1 %`. `P` itself moved only `0.02 %` between a 64-token and a 300-token window, which is what a pure fixed 512-token seed cost must do.

### Where the 4 s goes (traced 64-token pair)

`MLX_QWEN_MTP_TRACE=1` splits the prefill:

| Arm | CPU lazy graph build | GPU eval wall | CPU share |
| --- | ---: | ---: | ---: |
| serial | 2.952156 s | 1.047279 s | **73.8 %** |
| MTP | 2.955948 s | 1.047127 s | **73.8 %** |

Everything outside `begin` — IPC, dispatch, parent bookkeeping — is `≈0.6 ms`, about `0.015 %` of `P`. So `begin` is `99.98 %` of the charge (prediction 2 confirmed in magnitude), but **three quarters of it is CPU-side MLX graph construction, not GPU math.**

### Prediction scorecard

| # | Prediction | Verdict |
| --- | --- | --- |
| 1 | `p ∈ [0.01, 0.04]` | **Refuted** by 8–13x. `p = 0.5314` at 300 tokens, `p(512) ≈ 0.3110`. |
| 2 | the 512-row bulk forward is `≥85 %` of `P` | **Confirmed** — `begin` is `99.98 %` of `P`. The split inside it (73.8 % CPU / 26.2 % GPU) is the surprise. |
| 3 | `P_serial ≈ P_mtp` within 2 % | **Confirmed** — `0.11 %` at 300 tokens, `0.09 %` at 64. No stop rule triggered. |
| 4 | Phase 3 would be worth `<0.005` ranked score | **Premise refuted.** The whole lever is `0.605` points; a 1 % prefill cut already clears `0.005`. |
| 5 | follow-up belongs to a Metal prefill GEMM owner | **Wrong in kind.** 73.8 % of `P` is CPU graph construction, so the first follow-up is graph construction / reuse, not a kernel rewrite. |

The advisor's `4.13x` local-understatement factor is directionally right but too large: the measured factor is `local_to_ranked_leverage_factor = 1.8487304838320466` at the 300-token window and `1.5915280073458697` at the projected 512-token window. `4.13x` is reproducible as `(2.94661597308114 − 1) / (1.4708805 − 1) = 4.1339914757165355`, i.e. it came from a local `r = 1.4708805` measured at a 64-token window — and that `r` was itself heavily prefill-deflated, because at 64 tokens `p = 2.468` and prefill *dominated* the leg. Lengthening the window shrinks the correction.

## Blocker: the 512-token arm cannot run on this base

`research/e12-run.sh iterate 512 0 raw-512` **failed, exit 1**, inside the *depth-0 serial control*:

```text
mlxfast-swift: runtime worker mtp_decode_round failed: MTP round requested before the seed prefill
```

Root cause, fully traced:

- Stop tokens are `{248044, 248046}` (`weights/config.json eos_token_id=248044`; `weights/generation_config.json eos_token_id=[248046,248044]`).
- In the 513-row reference stream for the public fixture prompt, exactly one stop hit occurs: token `248044` at 0-based index **300** (emitted token #301). The reference then continues for 212 more tokens — `rows=513`, `self_consistent=true`, `chain_contradictions=0`. **The reference does not truncate at EOS.**
- The trusted parent has no EOS break: `QwenRuntimeMTPDriver.swift:121` is `while emitted.count < options.totalTokenCount {`, and `reachedStopToken` has no consumer outside the session.
- `Qwen36MTPBlockSession.swift:723-742` (`generateRound`): on an EOS primary it sets `reachedStopToken = true`, nils `pendingPrimary` / `pendingTop2` / `pendingHidden`, and early-returns. The next round hits the guard at `:674` and throws `.notBegun` (message at `:90`). The crash lands exactly at token #301.
- A second site, `:1113-1119` on the drafting path, truncates `committed` after the first stop token and decrements `committedTokenCount` **without** rolling back the pendings or the target cache, so `targetCacheOffset = seedTokenCount + committedTokenCount` desyncs. I did not prove divergence there — the parent stops the 300-token run before a draft block can straddle index 300 — so treat it as suspected, not established.
- This was never hit before because `--local-iterate` defaults to 64 tokens and `--local-submit` to 128, both well under 301.

**Ranked exposure: none, and I want to correct my earlier PR comment on this point.** Base commit `bc552e5` ("Retire the orphaned fixed-window EOS guard test") already records the reasoning, and it holds: the frontier was promoted at `2.94661597308114` *with this early exit present*, so no hidden prompt emits a stop token inside its 512-token window — otherwise the ranked run would have thrown rather than scored. The defect is **local-fixture-only**, but it does block the thing `program.md` asks for: measuring credible candidates against a fresh same-host base over 512 decode tokens.

**Why I did not fix it.** `program.md` (paragraph added `f4cfc75`, 2026-08-16) says to fix fixed-window continuation. The advisor's later decision (`bc552e5`, 2026-08-17) says the opposite for campaign main: keep main byte-identical to the promoted frontier, and "local A/B runs must keep asserting a full-length match on BOTH arms instead of re-patching the session." The later recorded decision wins, the fix is outside the four symbols this assignment scoped me to, and it collides with `qwen-edward`'s surface. I posted the root cause and asked for a go-ahead (PR #14 comment `e12-r1-blocker-eos-notbegun-512`) and got no reply, so I stopped at the assignment's stop rule instead of editing the session unilaterally. **300 tokens is the largest EOS-safe window on this fixture, and both arms did assert a full-length exact match there, which is exactly what `bc552e5` requires.**

## Conclusion

- **What happened and why:** the seed prefill is a large, precisely fixed `≈4.00 s` charge inside both timed legs. Because it is identical on both arms, it cancels in absolute terms but *deflates the ratio*, and the deflation is big: `0.560` score points at the 300-token window and a projected `0.605` at the ranked window, roughly a fifth of the promoted score. The assignment's own premise — that this was a `<0.005`-point rounding error — was wrong by two orders of magnitude.
- **Evidence for or against the mechanism:** four mutually independent confirmations. The direct in-path `seed_prefill_seconds` field; the leg-minus-Σblocks residual (`3.999748` serial vs `4.003407` MTP, `inferred_over_direct_mtp_ratio = 1.00015`); the exact asymmetric identity closing to `1.03e-3`; and a two-window fixed/variable solve landing within `0.5 %`. `P` is stable to `0.02 %` across a 4.7x change in decode window and symmetric across arms to `0.11 %`, which is the signature of genuine fixed work.
- **Prompt or M5 transfer risk:** substantial and unquantified. This is M4 Pro, not the ranked M5; the CPU/GPU split of `P` in particular is the ratio most likely to move on different silicon, and the CPU-heavy finding is the one the follow-up depends on. `p(512)` is a *projection* from 300-token steady rates, not a measurement — the 512-token measurement is blocked. Only one public prompt was used. Note also that the implied prefill-free ranked `r` is `3.55`, above the operator's `3.0` plausibility gate; per `program.md` that is not a reason to hold anything, but it is worth knowing before the lever is pulled hard.
- **Smallest useful next action:** two, in order. (1) Restore fixed-window continuation past EOS so anyone can measure 512 tokens locally at all — it is cheap, it is the gate on every future ranked-equivalent local measurement, and the exact prior fix is recoverable from campaign commit `f1a874d`; the advisor needs to decide whether that lands on main or on a research-only branch. (2) Attack the CPU three-quarters of `P`: the untimed `warmAllDepths` already builds the same 512-row op sequence once before the clock starts, so the first question is whether that constructed graph can be reused instead of rebuilt inside the timed `begin`. That is a graph-construction question, not a kernel question.
- **Recommendation: revise the program's cost model and open a prefill workstream.** Close e12 as answered — do not repeat it. Prediction 4 should be struck from the campaign's assumptions, and seed prefill should be promoted from "rounding error" to a first-class `≈0.6`-point research area, with graph construction as the first target and the EOS continuation fix as its prerequisite.
