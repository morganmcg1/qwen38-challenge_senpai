# E30 pre-registration — does the M=9 cliff survive NA=5, and is `vbuild` host work?

Student `qwen-alphonse`, PR #35, base `d08feb85bf65959d7eaa1455e36a0173b3edd8d9`.
Committed **before** any E30 measurement was taken. Nothing below is edited after
the first run; adjudication lives in the result report.

Host: Apple M4 Pro `Mac16,11`, 48 GiB, 20 GPU cores, NAX off. Idle GPU 43.0 C at
2026-08-18T23:4x UTC (E29's D0 arm entered at 41.57 C — a +1.4 C hotter entry).

## What changed on the scored path since the E29 measurement

E29's D0 arm was measured at tree `09fd8192df2a188ed483a6f0bb494d792e41ca62`.
`git diff 09fd819..HEAD -- Sources Vendor` is four files, and on **this** host only
one of them is live:

| change | live here? | why |
|---|---|---|
| `quantized.h` / `quantized.cpp` twins: `<T,5,3>`→`<T,5,5>`, `<T,9,3>`→`<T,9,5>`, `static_assert(NA<=5)` | **yes** | E27's treatment |
| `Qwen36MTPBlockSession.warmAllDepths` residency wiring | no | `physicalMemory >= 96 GiB` guard; this host is 48 GiB |
| `RuntimeStartupMemoryPolicy` command-buffer defaults (`MLX_MAX_MB_PER_BUFFER=512`) | no | same 96 GiB guard |
| my own E29 trace + ladder knob | yes, and **byte-identical** to what E29 measured | instrument unchanged |

**M=8's dispatch is `<T,8,4,true>` in both trees.** Only M=5 and M=9 changed, and
M=5 never appears in this fixture's realised widths. That makes M=8 a
same-arm, interleaved-rounds internal control for the M=9 treatment, which is
what lets a cross-session comparison mean anything.

## Baseline numbers this experiment is measured against

E29 D0, steady state (33 rounds, first 2 timed rounds dropped):

| M | rounds | round_mean ms | vbuild ms | eval ms | dbuild ms | host tail ms | full-accept ms/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 70.849 | 35.998 | 34.381 | 0.324 | 36.468 | 35.4245 |
| 6 | 9 | 141.199 | 73.118 | 65.327 | 2.053 | 75.872 | 23.5332 |
| 7 | 3 | 154.183 | 78.183 | 69.168 | 5.795 | 85.015 | 22.0261 |
| **8** | 5 | **167.675** | **83.922** | 75.604 | 6.641 | 92.071 | **20.9593** |
| **9** | 15 | **206.607** | **101.720** | 96.511 | 8.074 | 110.096 | **22.9564** |

Steady round total 5741.673 ms · full-accept tokens 252 · best width 8 ·
`headroom_pct` 8.0102 · `accepted_draft_total` 222.
Both legs, true decode (parent block sums, **not** prefill-inclusive
`decode_seconds`): serial `03` 66.8745 ms/token over 256 rounds; MTP `04`
172.5586 ms/round over 35 rounds = 23.592 ms/token over 256 tokens.

E27's measured QMV width curve (round-weighted kernel time per verify forward):
M=8 149.355 → 150.110 ms (+0.51 %), M=9 186.233 → 164.900 ms (**−21.33 ms**,
−11.45 %).

## Primary metric definition, fixed now so it cannot drift

```
bound_M8 = 100 x (steady_round_total_ms - full_accept_tokens x ms_per_token(M=8))
                 / steady_round_total_ms
```

E29 D0 value 8.0102; the assignment's baseline 8.25 is the 8.0/8.5 D0/L0 midpoint.
`bound_M8` is always computed at M=8 even when M=8 stops being the best width; the
best-width `headroom_pct` is reported separately so the two cannot be conflated.

## Predictions

### P1 — `e30/all_at_m8_upper_bound_pct`

Advisor's threshold: **falls below 2.0**, may go negative.

My own arithmetic, registered now, says that threshold **narrowly fails** while its
substance holds. Holding the realised histogram fixed and transferring E27's full
−21.33 ms to M=9's round only:

```
new M=9 round_mean   206.607 - 21.33          = 185.28 ms  -> 20.587 ms/token
new steady total     70.849 + 9(141.199) + 3(154.183) + 5(167.675) + 15(185.28)
                                              = 5421.8 ms
bound_M8             100 x (5421.8 - 252 x 20.9593) / 5421.8 = 2.58 %
```

- **P1 point estimate 2.6 %**, plausible range **1.8–3.6 %** (a 68–78 % reduction
  from 8.25 either way).
- The residual is **not** M=9 any more: it is the 9 M=6 rounds at 23.53 ms/token
  and 3 M=7 rounds at 22.03 ms/token, which E27 did not touch. A bound below 2.0
  requires those to move too, and nothing in this base moves them.
- **Best width flips 8 → 9** (20.59 < 20.96 ms/token) and the per-row ordering
  inverts, so "put every round at M=8" becomes a *loss* against all-at-M=9.
  Predicted best-width `headroom_pct` ≈ 4.3 %.
- Predicted M=8→M=9 step: 185.28/167.675 = **1.105** for a 1.125x row increase,
  i.e. the 22 % cliff at constant dispatch count is **closed** (E29: 1.232).

### P2 — `vbuild` at M=9

Advisor's threshold: **falls by at least 5 ms** vs E29's 101.720 ms.

- **Point estimate −10.5 ms → 91.2 ms**, range **−4 to −16 ms**.
- Derivation: E29 D0 split M=9's round into vbuild 101.720 / eval 96.511, i.e.
  51.3 % / 48.7 %. If the 21.33 ms of removed QMV time is distributed in
  proportion to where those dispatches sit relative to the eight ladder rungs,
  ~10.9 ms lands in the vbuild window and ~10.4 ms in `eval_wall`.
- Predicted M=9 round_mean **185 ms**, range 178–196.
- Registered in advance, per the advisor's §4: a `vbuild` drop is **not** by
  itself a speedup. The round total is the arbiter. If vbuild falls and the round
  total does not, the honest report is "the backpressure explanation survives and
  it carries no prize."

### P3 — falsification branch, written before the data

Dispersion is estimated from the two identical D0 arms and from per-round spread
inside each width, not assumed. "Flat" below means within 2x the within-config
per-width dispersion.

| round_mean(M=9) | vbuild(M=9) | what I will conclude |
|---|---|---|
| falls >= 10 ms | falls >= 5 ms | Both explanations' shared prediction holds and the GPU-wait reading of `vbuild` survives its only available one-sided test. The cliff shrank; report the new bound. |
| falls >= 10 ms | **flat** | The strong form of my E29 claim — "vbuild size tracks total GPU work in that window" — is **falsified**. Two survivors remain and this experiment cannot separate them: (i) genuine host work, which E29 already prices at an implausible 70–90 µs per op across a *constant* 969-op inventory, or (ii) a **saturation-limited** wait, where the queue is past `MAX_ACTIVE_TASKS=10` either way so the host waits on a fixed buffer count regardless of per-dispatch cost. I will report (ii) as the better-supported survivor, say plainly that I cannot rule out (i) from this arm, and name the ladder-off-at-M=9 arm on this base as the test that would separate them. I will **not** relabel the tail "host work" without it. E29's 4.35 % removable-host-cost ceiling is unaffected either way: it came from the ladder-off arm, not from `vbuild`. |
| **flat** | falls >= 5 ms | Pure relocation between buckets. Backpressure survives, carries no prize, and E27's kernel win did not reach this fixture's round time — which would itself contradict E27's own +7.02 % end-to-end confirmation and would need explaining, not reporting. |
| **flat** | **flat** | Treat as an instrument failure first: stale `.build-worker`, or the JIT twin not reaching the live dispatch. Prove the M=9 `<T,9,5>` arm is actually resident before interpreting anything. |

### Pre-committed control checks

1. **M=8 control.** M=8's dispatch is unchanged, so round_mean(M=8) must land
   within ±3 % of 167.675 ms (162.6–172.7). Outside that, session-level drift is
   confounding and every cross-session delta gets interpreted through that
   offset instead of at face value.
2. **Serial-leg control.** E27 touched M=5 and M=9 only; the serial leg runs at
   M=1, so serial must be **unchanged**: 66.8745 ms/token ± 2 % (65.5–68.2). This
   is also the score-derivative check the advisor's correction requires — E27
   gains on the MTP leg only, which is the direction that raises the score.
3. **Histogram control.** Predicted unchanged (M=2:1, M=6:9, M=7:3, M=8:5,
   M=9:15; `accepted_draft_total` 222) because the token stream is bit-identical
   and width selection uses static cost constants. If it moves, every cost number
   is re-weighted onto the new histogram before any comparison.
4. **Exactness.** `all_tokens_matched: true` and
   `residual_divergence_count: 0` on every arm, or the timing is void.

## Method fixed in advance

- Instrument unchanged: `research/e29-run.sh D0=1:8:default`, 256 decode tokens,
  depth 8, declared head, trace on, same warmup skip of 2, analysed by an
  unmodified `research/e29_analyze.py`. New logic goes in a separate
  `research/e30_adjudicate.py` so the measurement and primary analysis paths stay
  byte-identical to E29's.
- Two identical arms: `D0` (matched entry temperature) and `D0b` (hotter entry)
  for within-config dispersion. `D0` is the primary comparison; `D0b` prices the
  repeat.
- `MLXFAST_LOCAL_COOL_GATE=0` with entry/exit temperatures recorded per arm;
  `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`
  preserved verbatim. Directional causal evidence, not a gate-qualified score.
- `MLX_QWEN_MTP_TRACE_SYNC_HEAD` stays **off** on every timed arm: it destroys
  the head/verify overlap.
- No ladder arm in either direction, no scheduler arm, no `costModelDepth` or
  `headStepCostRatio` edit, no touch of the twin-locked `quantized.h` /
  `quantized.cpp`.
- Every arm reported as **two numbers** — serial leg and MTP leg — on true
  decode, never a ratio alone.
