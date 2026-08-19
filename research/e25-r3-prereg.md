# E25 r3 pre-registration — re-measure the per-row price after E27

Committed **before any r3 run**. Nothing in this file may be edited after the
first measurement; adjudication goes in `research/e25-results.md`.

Base `329d3644dc96972d6843ecfe759141b8b0ab539d` (E27 merged, PR #32, thorfinn).
Credit: E25 is **thorfinn's E22 follow-up #1**; the row-4 cliff this re-measures
is the effect his E27 removed.

## 1. What r3 is

A **measurement**, not a policy arm. Both r2 policy arms are closed by the
advisor: arm D (`DEEP_CAP = 3`) is obsolete because the step it avoided is no
longer mispriced, and a row-5 cap is refused because row-5 steps are 9/1947 =
0.46 % of taped rounds against 167/1947 = 8.58 % for row 4. **No arm D. No
row-5 cap.** The branch source is byte-identical to the base and stays that way.

## 2. Primary metric and the prediction that may fail

| field | value |
|---|---|
| metric | `e25/measured_row_step_ratio_at_depth_3` |
| definition | `(T(4) − T(3)) / T(3)`, mean MTP round wall time at proposed depth d |
| baseline | **0.442442** (E25 r1, shipped-policy tape, 1947 rounds) |
| direction | minimize |
| **advisor pre-registered prediction** | **0.18 ± 0.05** ⇒ accept band **[0.13, 0.23]** |

**My own point prediction, and how I derived it (registered as a separate,
independently falsifiable number): `c_3 = 0.1754`.**

E27 measured the isolated QMV round cost at width M=5 falling 120.683 → 96.423
ms (−24.26 ms) when `<T,5,3,true>` (2 weight streams) became `<T,5,5,true>`
(1 stream). My r2 forced-depth curve on this host with the same declared head
measured `T(4) = 132.2566 ms` pre-E27. Transferring E27's QMV delta:

```
T_post(4) = 132.2566 − 24.26 = 108.00 ms
c_3       = (108.00 − 91.885) / 91.885 = 0.1754
c_4       = (144.103 − 108.00) / 108.00 = 0.3343
```

Full predicted post-E27 curve (mine; pre-E27 measured values are r2's forced-depth
means, unchanged where the dispatch cell did not move):

| d | M=d+1 | streams pre → post | T(d) pre (measured) | T(d) post (predicted) | c_d pre | **c_d predicted** |
|---|---|---|---:|---:|---:|---:|
| 0 | 1 | 1 → 1 | 68.616 | 68.616 | 0.0436 | 0.0436 |
| 1 | 2 | 1 → 1 | 71.605 | 71.605 | 0.1021 | 0.1021 |
| 2 | 3 | 1 → 1 | 78.916 | 78.916 | 0.1643 | 0.1643 |
| 3 | 4 | 1 → 1 | 91.885 | 91.885 | **0.4394** | **0.1754** |
| 4 | 5 | **2 → 1** | 132.257 | **108.00** | 0.0896 | **0.3343** |
| 5 | 6 | 2 → 2 | 144.103 | 144.103 | 0.0880 | 0.0880 |
| 6 | 7 | 2 → 2 | 156.788 | 156.788 | 0.0851 | 0.0851 |
| 7 | 8 | 2 → 2 | 170.134 | 170.134 | — | — |

So I predict the cliff **moves one row deeper rather than disappearing**, and
that the row-4 step lands within the advisor's band.

## 3. Secondary pre-registered predictions

1. **Realised depth histogram, shipped policy, new base.** The policy is
   belief-driven and unchanged, so the histogram must not move materially:
   mean proposed depth **2.30 ± 0.25**, depth ≥ 4 share **5–12 %**, max depth 5.
2. **Residual headroom (deliverable c).** Re-pricing the 1947-round r1 tape with
   the newly measured curve, by the same replay instrument that projected
   **+5.27 % pooled / +4.69 % median-of-8** for arm D pre-E27, yields
   **< +1.0 % pooled** true decode. Point prediction **+0.3 % to +0.8 %**,
   coming almost entirely from the now-expensive row-5 step that fires in
   0.46 % of rounds.
3. **Verdict.** I expect to report **the lever is closed**: no policy arm on the
   depth price is worth a GPU hour on this host after E27.

## 4. Falsifiers

| observation | consequence |
|---|---|
| `c_3 ≥ 0.30` | E27's microbenchmark win does not reach the decode path. Major, report immediately. |
| `c_3 < 0.13` or `> 0.23` | advisor's band fails; my 0.1754 also adjudicated separately. |
| `c_4 < 0.20` | the cliff did not move to row 5; the staircase model of the width curve is wrong. |
| headroom `> +1.5 %` pooled | the lever is **not** closed and r3 must say so. |
| mean depth outside 2.30 ± 0.25 | the tape is not comparable to r1's and the replay is void. |

## 5. Instruments and controls

Two instruments, both already built and used in r1/r2, unchanged in code:

- **BASE** — the base blob verbatim (shipped scalar price `h = 0.18`), traced.
  This is r1's instrument: per-depth mean round time over the depths the
  adaptive rule chooses. It is what makes the comparison against 0.442442
  like-for-like, and it supplies the realised depth histogram.
- **FORCE** — `research/e25r2-force-depth.sh` on the same base blob: the taken
  depth cycles 0..7 round by round, so every depth sees the same positions,
  prompts, cache lengths and temperature. n ≈ 225 per depth per 8-prompt sweep
  versus n = 9 at d = 5 for the adaptive tape. The pre-E27 half of this
  comparison is r2's archived sweep on `d7619a7`, measured on this host with
  this head and this instrument.

Fixed controls, identical to r1/r2:

- 8 prose prompts, pre-registered order `english, narrative, technical,
  dramatic, travel, philosophy, natural_history, medicine`; 512 decode tokens;
  goldens from E17; `--local-iterate` via `research/e11-run.sh`.
- Declared head `mtp-head-declared-q2q4-run`
  (`model.safetensors` = the manifest's 427742600-byte q2-q4-rerank-v1 tree,
  plus the wrapper-required `config.json`), per r3 deliverable (f).
  **Head confound, disclosed:** r1's tape ran the *previous* declaration
  (238934129 bytes). r2 measured `c_3 = 0.439371` with the q2q4 head against
  r1's `0.442442` with the old head — a 0.7 % difference on this metric, so the
  head change cannot account for a move to 0.18.
- Arms are interleaved within prompt, never blocked by arm, so thermal drift
  cannot correlate with arm.
- `MLXFAST_LOCAL_COOL_GATE=0`, with all four disclosures carried verbatim in
  every `meta.txt`: `cool_gate_passed_real_gate=false`,
  `gate_qualified_for_timing=false`, `cool_gate_temp_c=40`,
  `cool_gate_bypass_reason=host idles above the compile-time 40C gate`.
  These are **traced, non-timed** probe passes; the headline is a *ratio between
  depths measured inside the same legs*, not a between-arm score.
- Every leg checks `all_tokens_matched` against the E17 512-step serial
  goldens, and the whole-repo `dirty` flag is recorded per leg. No file in this
  repository is edited while a leg is in flight.

## 6. Scope

Editable surface touched by r3: **none**. `Qwen36MTPBlockSession.swift` is
byte-identical to `329d3644:...`. `quantized.h` and its generated twin are
thorfinn's surface and are not touched. Everything else r3 adds is
research-only (`research/`), never submitted.
