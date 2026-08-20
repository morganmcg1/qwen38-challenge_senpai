# E82 terminal result — recorded in-repo because `submit_experiment_result` is blocked

`submit_experiment_result` failed twice with:

```
GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/84 returned HTTP 403
```

The failure is on the tool's own read of PR #84, before any mutation. It is not
a lease conflict and not a precondition failure. `post_assignment_comment`
succeeded on the same PR at 20:18:31Z and `get_prs` succeeded immediately
before that, so the credential worked minutes earlier. E80 hit the same blocker
at 19:30Z. Nothing in the result needs to change; only the transport is broken.

This file records the exact payload so the advisor can act on the result and so
the submission can be replayed unchanged once the credential recovers.

## Identity

| field | value |
|---|---|
| repo | `morganmcg1/qwen38-challenge_senpai` |
| pr_number | 84 |
| assignment_id | `qwen38-r1-e82-requantize-the-only-genuinely-retrained-head` |
| revision_id | `r1` |
| student | `qwen-alphonse` |
| branch | `qwen-alphonse/e82-requantize-the-retrained-head` |
| commit_sha / expected_head_sha | the commit that adds this file, that is the branch tip |
| remote_branch_sha_before_push | `1bd04877e072d27057217255556b00fc9bf65b6f` |
| status | `failed` (hypothesis refuted; the experiment itself ran to completion) |

The measured evidence was produced at `30513a5da156c581c3be17ccb1e49f09e34288b0`,
whose parent commits carry the session artifacts. This file is the only later
change, and it touches no code.

## Primary metric

| field | value |
|---|---|
| name | `candidate_mtp_seconds_per_token` (local `--local-iterate`, 512 tokens, ungated, M4 Pro) |
| direction | minimize |
| baseline (`declared`) | 0.03143216494936496 |
| candidate (`noislands`) | 0.031547432648949325 |
| delta | +0.00011526769958436474 (+0.367 %) |

## W&B runs

| run | url | state | contents |
|---|---|---|---|
| `o0rawiol` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/o0rawiol | finished | island acceptance screen and the six-leg timed session |
| `yerghmxz` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/yerghmxz | finished | rungs 0 to 4 head audit, builds and acceptance screen |

## Hypothesis

The pinned MTP head's post-hoc affine-4 g64 quantization leaves recoverable
value, so one of three routes yields a net decode speedup:

- (a) requantizing the only genuinely retrained published head (xkm) to
  affine-4 g64 keeps its trained weights while restoring the 427 MB footprint
  and beats the pinned head;
- (b) a better post-hoc estimator (least squares, HQQ, per-tensor best-of)
  recovers the round-to-nearest acceptance loss;
- (c) deleting the 31,469,568 bytes of precision-island correction tensors buys
  more decode time than the acceptance it costs.

## Summary

All three routes refuted. No candidate code changed:
`git diff --stat 9ec6e087 HEAD -- Sources/ Vendor/` is empty. Nothing to submit
to Yukon; do not reserve a slot. Full tables in PR comments 5360874430 and
5361194836.

**(a) Requantized xkm — no-go at rung 0.** The build met all three hard
constraints but missed the +1.0 pt pooled acceptance gate at depths 3 to 6.

**(b) Better estimator — closed negative.** Least squares and HQQ remove only
8.18 to 11.32 % of reconstruction error against `mx.quantize`; the best-of arm
measured −0.28 pt acceptance against `declared` (McNemar 4/7, chi2 0.36, ns).
Quantization-aware trained weights remove 62 to 69 % of the same error and
measure +0.86 pt, so the axis closes on estimator quality, not effort.

**(c) Precision islands — closed negative, and the decisive measurement.**
Six-leg ungated palindromic timed session, 512 tokens, `--sync-head`, one M4 Pro
session; all legs exit 0, `all_tokens_matched=true`,
`residual_divergence_count=0`.

| arm | leg 1 | leg 2 | mean s/tok | Δ vs `declared` | head bytes |
|---|---:|---:|---:|---:|---:|
| `declared` | 0.031410 | 0.031454 | 0.031432 | ref | 427,746,170 |
| `noislands` | 0.031557 | 0.031538 | 0.031547 | +0.367 % | 396,276,034 |
| `qonly` | 0.032130 | 0.031865 | 0.031998 | +1.799 % | 406,786,962 |

Clean separation: `max(declared) = 0.031454 < min(noislands) = 0.031538`.
Removing all 31,469,568 island bytes makes the candidate slower.

### Why, and the reusable rule

`draft_build` is 10.06 % of round time and `verify_build` is 43.66 %. Head bytes
convert to `draft_build` at 0.810x and to candidate time at 0.0815,
independently confirming the E79 head-to-median factor 0.0843 to 3.3 %.
`noislands` earns −57,720 us of head time, then pays +72,166 us of verify and
+46,988 us of wait for one extra round (78 to 79) and +1.24 % rows.

**Rule: a head-byte reduction converts at 0.0815 only if it is bit-exact.**
Changed head numerics first repay a schedule perturbation, here about 4x the
byte saving.

### Cross-validation against a ranked M5 measurement

The 0.0815 conversion predicts −0.112 % for askeladd's bit-exact mechanism A
(5,898,240 B); he measured −0.172 % ± 0.052 over three ranked submissions. Same
sign and order, model conservative by 1.15 sigma. This refutes byte-dominance:
`noislands` removes 5.34x more bytes and the sign flips. Bytes are not the
currency; bit-exactness is.

### Falsification test passed

Rounds per leg are exactly reproducible across the palindrome's repeat visits —
`declared` [78, 78], `noislands` [79, 79], `qonly` [80, 80], with rows/token
identical to 4 dp — so the work metric is a deterministic head property, not
trajectory noise. Honest limit: one fixed greedy public prompt cannot bound
cross-prompt variance.

### The advisor's arm (a) should not be built

`Qwen35.swift` `qkv()` matches neither the all-quantized branch (`:1725`) nor
the all-dense branch (`:1743`) for a mixed quantized `q_proj` plus unquantized
`k_proj`/`v_proj`, so it falls through to three separate `Linear` calls at
`:1754` — a path that never calls `replaceExactRows` and so silently drops the q
island. `sanitize` also `fatalError`s at `:2877` on a partial island set. No
value anyway: arm (a) and mechanism A are the same mechanism, and arm (a) saves
only 8,192 B more.

### Gates, verbatim

`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`official_or_ranked_score=false`. Host Apple M4 Pro 48 GB, **not** the ranked
M5, so this is directional session evidence and never a score. `declared-1`
entered at 38.6 C and `declared-2` at 61.7 C yet differ by only +0.14 %, so the
effect is 2.6x the thermal drift.

### Head identity asserted per leg

`declared` `dadbfb806d80` 427,746,170 B; `noislands` `92b8419bc3a0`
396,276,034 B; `qonly` `315a03a98a59` 406,786,962 B; correct on all six legs.
`uses_pinned_mtp_head` read `true` everywhere.

### Reproduce (1,296 s)

```bash
research/e82_headcost_session.sh e82-islands declared noislands qonly
python3 research/e82_headcost_report.py e82-islands --out research/e82-headcost-islands.json
python3 research/e82_decompose.py
```

Arms are built by
`research/e82_build_head.py --quantizer raw --islands {none,q-only}`.

### Follow-ups, none implemented

1. The head-byte axis is priced and nearly exhausted; leverage has moved to
   rows/token (1.107) and the verify path.
2. `MLXFAST_QWEN_MTP_EXACT_QKV_ROWS` does not reach the worker
   (`QwenRuntimeWorker.swift:2573-2580` allowlists only `DARKBLOOM_`, `DYLD_`,
   `LC_`, `METAL_`, `MLX_`, `MTL_`).
3. The seed-argmax prefill path differs between
   `QwenRuntimeMTPDriver.swift:692` and `:100`/`:106`.
4. Islands also cost a fixed op boundary beyond their bytes: `d_submit2`
   converts at 0.906x of bytes for `noislands` but only 0.39x for `qonly`.
