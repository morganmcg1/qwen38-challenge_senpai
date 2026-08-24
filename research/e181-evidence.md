# E181 — local Q pair with a prefill/decode split (`harness=local`)

Question: what is Q's LOCAL decode effect at M <= 9, and if it is nonzero
despite zero call-graph consumers, does the shared JIT compilation unit deliver
it?

## Identity tuple

| field | value |
|---|---|
| assignment | `e181-local-q-pair-jit-channel` r0, PR 180 |
| campaign base | `38634ebfbf67f8ccc8a5a404acbc15e749d26b7a` |
| organizer content source | `upstream/main` `0863b06ac16e26e48fc06e97444095b00feb66d4` |
| host | Apple M4 Pro, 48 GiB, macOS 26.5.2 |
| harness | local, `./benchmark-qwen-mtp.sh`, public fixture |
| token window | 512 seed + 512 decode (timing), 512 seed + 128 decode (gate) |
| head | organizer-pinned MTP head, both arms |
| reference source | candidate-generated local reference rows |
| timing source | trusted parent report, no added instrument |
| thermal mode | `MLXFAST_LOCAL_COOL_GATE=0`, ABBA-counterbalanced |

## Arms

Both arms are separate built trees, so every leg times one fixed binary.

| field | arm P | arm PQ |
|---|---|---|
| content | organizer-pure | organizer-pure + the four Q files |
| diff vs organizer over `Sources` + `Vendor` | 0 files | 4 files, +344 -92 |
| needle `mma_op.mma(Xs, Ws + cur * Ws_tile);` | 0 | 1 |
| needle `qmm_t_nax_ws_halves` | 0 | 4 |
| positive control `MLX_QWEN_MTP_TRACE` | 3 | 3 |
| worker sha256 | `93056012d35ea16b…` | `c33c9a8ea4b2c56f…` |
| cli sha256 | `cb1b3ae0bf43b0ab…` | `a6c6734ae6216c96…` |
| metallib sha256 | `5de2569e449410ba…` | `5d6e3f9f1438dc5d…` |
| twin audit, verbatim | `TWIN AUDIT FAILED: 1/29 twin(s)` | `TWIN AUDIT OK: 29 runtime-effective twin(s), 1 allowlisted comment-only waiver(s)` |

The P-arm twin-audit line is the benign comment-only case: `research/twin_audit.py`
is the campaign copy and its waiver pins the campaign comment digests, so an
organizer-pure tree de-pins it. No file was edited to green the audit. No NEW
divergence appeared on either arm.

The PQ metallib digest equals the digest the ledger records for the frozen E175
build, so two independently built trees produced identical kernel bytes.

## Exactness gate — PASSED

`research/e181_session.sh iterate P,PQ 128 gate`

| check | result |
|---|---|
| seed tokens (512) sha256 | `90c07bb15f443d9b` on both arms |
| emitted tokens (129) sha256 | `5be9272dd504e4c6` on both arms |
| reference rows (129) sha256 | `d15cf0f8cc592869` on both arms |
| `all_tokens_matched` | true on all four timed reports |
| rows declared / checked | 128 / 128 on all four |
| `residual_divergence_count` | 0 on all four |

Q is numerically inert on the cells it never serves at M <= 9.

## Timing session — 8 legs, palindromic `P,PQ,PQ,P,P,PQ,PQ,P`, 512 decode tokens

`research/e181_session.sh submit P,PQ,PQ,P,P,PQ,PQ,P 512 abba1`

W&B: https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/3ptd79n1

`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`phase_trace=0`, `added_timing_instrument=none`,
`timing_source=trusted-parent-report`. Worker digest asserted stable before and
after every leg. Entry temperature 35.4 C on leg 1 and 59.8-61.4 C on legs 2-8;
the warm entry spread is 1.6 C.

| leg | arm | T in | T out | serial prefill s | MTP prefill s | serial decode-only s/tok | MTP decode-only s/tok |
|---|---|---|---|---|---|---|---|
| 1 | P | 35.4 | 61.0 | 4.0078 | 4.0156 | 0.065543 | 0.023813 |
| 2 | PQ | 59.8 | 62.9 | 3.9157 | 3.9077 | 0.065588 | 0.023787 |
| 3 | PQ | 59.9 | 64.7 | 3.9083 | 3.9069 | 0.065531 | 0.023778 |
| 4 | P | 61.0 | 63.5 | 3.9940 | 3.9939 | 0.065655 | 0.023824 |
| 5 | P | 60.4 | 63.6 | 3.9938 | 3.9929 | 0.065606 | 0.023798 |
| 6 | PQ | 60.6 | 65.0 | 3.9230 | 3.9078 | 0.065560 | 0.023799 |
| 7 | PQ | 61.1 | 65.2 | 3.9073 | 3.9080 | 0.065640 | 0.023828 |
| 8 | P | 61.4 | 65.1 | 4.0061 | 3.9949 | 0.065583 | 0.023833 |

Every leg matched exact tokens, closed its row ledger (serial 512/512, MTP
568/568 declared and reference-checked), reported
`residual_divergence_count=0`, ran 77 rounds, and reported
`accepted_draft_rate=0.88594705`. Round count and acceptance are identical
across all eight legs, so the decode work itself is unchanged by Q.

## Contrasts, PQ against P, n = 4 per arm

| metric | P mean | PQ mean | delta | 95 % CI | verdict |
|---|---|---|---|---|---|
| MTP decode-only s/token | 0.02381701 | 0.02379807 | **-0.0795 %** | [-0.189 %, +0.030 %] | NULL, inside the +/-0.2 % band, CI spans zero |
| serial decode-only s/token | 0.06559666 | 0.06557969 | -0.0259 % | [-0.124 %, +0.072 %] | NULL |
| MTP prefill seconds | 3.9993185 | 3.9076083 | **-2.293 %** | [-2.560 %, -2.026 %] | resolved speedup |
| serial prefill seconds | 4.0004038 | 3.9135548 | -2.171 % | [-2.429 %, -1.913 %] | resolved speedup |
| MTP charged s/token | 0.03162818 | 0.03143012 | -0.626 % | | prefill-dominated, see below |
| local charged ratio serial/MTP | 2.321031 | 2.329720 | +0.374 % | | prefill-dominated, see below |

Warm-only sensitivity. The cut is the two session-opening legs, leg 1 (the only
cold leg, P) and leg 2 (the first PQ leg), leaving n = 3 per arm: MTP decode-only
-0.0697 %, serial decode-only -0.0576 %, MTP prefill -2.161 %, serial prefill
-2.129 %.

Because "drop one PQ leg for balance" is a choice, here is every variant, so the
cut cannot be doing the work. Dropping leg 1 plus each possible PQ leg in turn
(and dropping leg 1 alone, P n = 3 vs PQ n = 4):

| dropped with leg 1 | MTP decode-only | MTP prefill |
|---|---|---|
| nothing | -0.0846 % | -2.161 % |
| leg 2 | -0.0697 % | -2.161 % |
| leg 3 | -0.0559 % | -2.154 % |
| leg 6 | -0.0859 % | -2.163 % |
| leg 7 | -0.1270 % | -2.164 % |

Decode stays inside the predeclared +/-0.2 % band in all five variants, worst
case -0.127 %, and prefill stays within 0.01 pp of -2.16 %. The branch does not
change under any cut.

This enumeration is not hand arithmetic: `research/e181_analyze.py` emits it as
`warm_only_sensitivity` in `summary.json`, picks the dropped cold leg by lowest
entry temperature rather than by position, and reports
`branch_stable_within_band=true` with `mtp_decode_only_pct_worst_abs=-0.127`.

## Two confounds this session measures directly

1. **The charged headline is a prefill artifact.** The measured MTP charged
   delta is -0.626 %. Moving only the measured prefill delta and holding decode
   fixed predicts -0.566 %. About 90 % of the charged "decode win" is prefill
   spread over 512 tokens.
2. **The local ratio moves the wrong way.** The charged serial/MTP ratio rises
   by +0.374 % for PQ, and moving only prefill predicts +0.337 %. Prefill is a
   larger fraction of the shorter MTP leg, so a real prefill SPEEDUP makes the
   local ratio look like a regression. Read absolute candidate time, never the
   local ratio, for any change that touches prefill.

## Result

- **Branch 1 of the registered decision tree.** P and PQ decode agree within
  the predeclared band, so PQS was not built.
- FINDING 444's derived local Q decode figure of -0.88 % is refuted. The direct
  pair measures -0.0795 % with a 95 % CI of [-0.189 %, +0.030 %], which excludes
  -0.88 % by a wide margin.
- Receipt F's ranked decode change of +1.098 % is not reproduced locally. The
  same CI excludes it.
- **No local decode signature for the compilation-unit channel.** The PQ arm
  changes the `quantized.cpp` JIT source string that also carries every `qmv_*`
  decode kernel, and it changes the metallib bytes, yet decode is null on both
  the depth-0 serial leg and the depth-8 MTP leg. Neither the JIT source-string
  key nor the metallib layout carries a measurable decode tax on this host.
- **Scope limit.** This host executes the non-nax `quantized.cpp` family. The
  ranked M5 executes the `_nax` family from `quantized_nax.cpp`, a different
  compilation unit. The local null constrains transfer claims for the non-nax
  blob only; it cannot settle whether the nax blob carries the ranked effect.
- **Prefill positive control PASSES** under the F3 amendment: a replicated
  |~2 %| effect with either sign proves the instrument detects a real Q prefill
  effect. The SIGN disagrees with FINDING 447, which recorded a derived local
  penalty of +1.936 %. Here Q makes prefill 2.17-2.29 % FASTER, 8 legs of 8 with
  the same sign, and the thermal bias ran against the observation.

## The prefill sign flip, against F3's registered candidate reads

The flip survives the design that was supposed to explain it away.

- **(c) n=1 cold-first-leg artifact: ELIMINATED, and I price its contribution
  rather than waving at it.** The effect is -2.293 % with a 95 % CI of
  [-2.560 %, -2.026 %] over 4 legs per arm, and every one of the 8 legs orders
  the same way (max PQ prefill 3.9080 s < min P prefill 3.9929 s, so the two
  arms do not overlap leg-wise). The cold leg IS a P leg and it is the slowest P
  leg (4.0078 s), so the cold-cache term biases FOR the observed speedup — but
  dropping it costs only 0.13 pp: the warm-only cut (n = 3 per arm) still gives
  -2.161 %. Temperature runs the other way: PQ entered 5.8 C hotter on average
  (60.34 C vs 54.53 C), and in the warm-only cut the arms are balanced to 0.6 C
  (PQ 60.34 C vs P 60.93 C) with the effect intact. Neither the cache term nor
  the thermal term can produce a 2 % flip.
- **(a) instrument difference: LIVE, and now the leading read.** This is the
  first local prefill reading through the trusted `seed_prefill_seconds` field;
  FINDING 447's +1.936 % was derived from `parent_measured_seconds_per_token`
  arithmetic. Section "Two confounds" shows exactly how a charged-time
  instrument mis-signs prefill work on this host: the charged serial/MTP ratio
  RISES 0.374 % for PQ while absolute prefill FALLS 2.293 %, because prefill is
  a bigger fraction of the shorter MTP leg. Any derived split that reads a
  charged quantity can therefore report a penalty where the trusted field
  reports a speedup. Edward's E184 bracket-sum-vs-trusted-field reconciliation
  is the independent test; FINDING 447 should be scoped to its instrument until
  that lands.
- **(b) host difference within the M4 class: LIVE and not excluded.** My host is
  M4 Pro, 48 GiB. Nothing here separates an M4-vs-M4-Pro dispatch difference in
  the non-nax path from an instrument artifact; only a same-instrument reading
  on a plain M4 can.
- **The ranked fact is untouched.** FINDING 457's +1.9705 % ranked prefill price
  (8/8, replicated 0.035 pp across receipts B and F, `harness=ranked`, M5)
  remains the operative number for ship decisions. It prices Q's `_nax` edits;
  this local contrast prices Q's non-nax `quantized.cpp` edits. Different code
  objects, so the disagreement is not a contradiction, and it constrains
  TRANSFER claims only. Q is out of the ship set either way.
