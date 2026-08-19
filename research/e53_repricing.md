# E53 Part 3 — re-pricing the old-coefficient verdicts at psi_mtp = +0.693391

Base `45b7c6a4`, branch `qwen-edward/scored-width-mixture-and-policy-map`.
Instrument: `research/qmv_score_leverage.py` (campaign gate 26), selftest
**PASSED, 71 PASS lines**, run before any number below was quoted. Every
conversion calls the module's public functions with `psi=` passed explicitly:
`mechanism_value()`, `mechanism_value_per_width()`, `score_pct_from_leg_gains()`,
`target_for()`, `width_set_share()`, `qmv_share()`, `kink_pct()`,
`saturation_cap_pct()`. Reproduce with `python3 research/e53_repricing.py`.

New input (E53 brief; askeladd, two doses, this base): **psi_mtp = 0.693391
[0.692292, 0.694490]**, superseding `PSI_MTP = 0.6736` (E42, tree `04ad6bf1`),
which the module still carries as its default. The old campaign doctrine priced
uniform QMV work at **-0.1789 %/%** (ledger 173(A)); ledger 176 proved that
coefficient a local-harness artifact (the ranked serial numerator is a pinned
separate binary), so on ranked, gated and uniform are the same positive number.

Reference constants (module, this base): board floor **0.7678 %** (ledger 166,
17 replicated tree-sets, dof 23), kink **+1.0551 %**, saturation cap
**+4.7156 %**. The score is the mean of order statistics 4 and 5 of eight
per-prompt ratios (crown tree `ef42e043`); any price above the kink below is
computed with `score_pct_from_leg_gains()` on the scored pair, never by
multiplication. Intervals in the table propagate the psi interval only; T(M)
(thorfinn E46, max|resid| 0.770 ms) and the E42 dispatch histogram carry no CI
(named caveat C2 below).

## Re-priced table

| mechanism | ledger item | old price | new price (psi interval) | above 0.7678 % floor? | label | notes |
|---|---|---|---|---|---|---|
| Shared-template QMV family: `load_vector` / `qdot`, reached by both `qmv_fast_impl` and the crossrow `_m` cells | 173(A)/(B); re-priced 176 case 2 | **-0.1789 %/%** ("harmful"; family was forced through shape gates) | **+0.6934 [+0.6923, +0.6945] % score per 1 % kernel-wide QMV win** (ungated >= gated on ranked) | only if the kernel win >= **1.107 [1.106, 1.109] %** (`target_for(0.7678)`) | **result changed** | Highest-leverage revived family: no gate needed, serial leg cannot follow. No measured win exists yet — this is a price per 1 %, minimum detectable win to clear the floor stated. ALU-only trims at M in {2,3,4} are excluded (caveat C3). |
| E44 r2 narrow simdgroup-matrix QMV, M in {7,8} (alphonse, PR 49, merged `454410ea`; wins measured on `d5701210` / base `9fe0dc5d`) | 174(A), 175 | +0.789 .. +1.228 % (psi 0.6736, census f{7,8}=0.1225; top of range was multiplied past the kink) | mlp-only **+0.8188 [+0.8175, +0.8201]**; equal mix **+1.0463 [+1.0446, +1.0479]**; attn-only linear +1.2738 -> **order-stat +1.1609 [+1.1599, +1.1618]** | **yes, at every shape mix** (old: marginal at the mlp end) | **price changed** | NOT revived as an experiment: the blocker is the bit-exactness bar (175(A) — the MMA reassociates; a previous accepted submission deliberately preserved the BF16 tree). E51's reassociation dose ladder owns that question. Corpus f{7,8}=0.1234 (module) vs scored-cost caveat C2. Kernel-win 95 % CIs (e.g. attn M8 [16.876, 17.225]) contribute less than the psi interval. |
| E44 r1 all-width MMA, M in [4,9] (`023a3fcf`) | 174(A) | refuted on measurement: net **-7.341 %**, M=4 at -41.7/-52.4 %; uniform application priced -0.179 %/% | measured cells {4,7,8}, equal mix: **-0.1177 [-0.1175, -0.1179] %** (M in {5,6,9} cells not in the r1 table) | no | **price changed** only | Refutation was a measurement, not a price. Do not revive; only the narrow fragment (row above) survives. |
| simdgroup MMA applied at M=9 | 173(D) | measured **-10.37 % (attn_out) / -11.66 % (mlp_down)** | **-4.111 [-4.105, -4.118] %** score if shipped at M=9 | no | **price changed** only | Measured refutation stands: fixed 8-row tile means M=9 needs a second tile (1.6x plateau). The register lever and the M=9 stream lever still do not compose. |
| M=9 two-stream prize: `<T,9,5>` held at <= 108 registers | 173(C), 174(B); E49 (thorfinn, PR 53) | **+5.36 % = 7.0 sd** (linear, psi 0.6736, corpus HIST) | linear +5.5215 [+5.5128, +5.5303] is **above the kink and not multiplication-valid**; scored-pair order-stat price **+3.2155 [+3.2112, +3.2197] %** | yes (4.2x floor) | **price changed** | A revived price is not a revived experiment: `<T,9,5>` has **never been timed** and E49 owns it. Corpus-wide `qmv_share(9) = 0.5383` is the load-bearing input, and Part 1/2 evidence (M in {4,5,6} ~65 % of scored cost; beagle mean M 5.533) says the scored-prompt share is far smaller — treat +3.22 % as an upper envelope on the corpus mixture, not a scored-prompt claim. |
| E27: IPG 3->5 at M in {5,9} (the +21-register step) | 173(C) (register max 108->129, ledger 5213) | observed **-0.3321 %**; residual (register-step price) -6.485 % at psi 0.6736 | observed unchanged; expected stream wins +6.333 [+6.323, +6.343] (linear accounting), residual **-6.666 [-6.656, -6.676] %** | n/a (adverse) | **price changed** only | E27 was refuted by observation; the re-price only moves the decomposition. The observed -0.3321 % remains the anchor for the register price (176(C)). Linear leg numbers here are decomposition bookkeeping, not score claims. |
| E44 uniform ceiling bound (shared register allocation, bound 0.663 % of QMV cost) | 176(C) | banked <= 0.1186 % (retracted -0.1789 model); corrected <= 0.4466 % at psi 0.6736 | **<= 0.4597 [0.4590, 0.4604] %** = **59.9 % of the floor** | no (it is an adverse bound) | **price changed** | Still gates every register-spending arm (E46 restatement, E49). Bound, never measured, not claimed in either direction. |
| E49 shared-ceiling hypothesis: c_ceiling = +10.6 % uniform QMV slowdown | 174(B), 176(C)/(D) | retracted model: **+1.90 % gain** (the absurdity of 176(D)); corrected -7.14 % at psi 0.6736 | **-7.350 [-7.338, -7.362] %** score if H_shared_tax is true (order-stat = linear on the way down; no substitution until the pair passes rank 3) | no | **price changed** | H_local_eaten vs H_shared_tax is still an open measurement; E49 owns it. |
| `<T,8,3>` code-match edit (make dispatch match the 17-line comment) | 173(E); E46 contrast B | measured **+18.72 % slower** at M=8 (ABBA, 8/8 shapes, sign p=0.0078; corroborated +19.02 % on `7b5183d`) | **-0.989 [-0.987, -0.991] %** score if shipped | no | **price changed** only | Refuted on measurement; byte-identity with the frontier is also asserted by `scored-surface-gate.sh`. `KNOWN_COMMENT_DIVERGENCES` guards the prose divergence. |
| Advisor QMV cost-reduction targets (crown gap / 1 sd / 2 sd) | 173(A) | 0.771 / 1.140 / 2.280 % QMV win needed | **0.749 [0.748, 0.750] / 1.107 [1.106, 1.109] / 2.954 [2.950, 2.959] %** | n/a | **price changed** | Crown and 1-sd targets are ~2.9 % cheaper. The 2-sd target sits **above the kink**: the old linear 2.280 % (and a naive new 2.215 %) understates the true requirement by ~33 % — inverted piecewise via `score_pct_from_leg_gains()`. |

## Not priceable with psi_mtp — refused a number, with reasons

| mechanism | ledger item | why psi_mtp does not apply |
|---|---|---|
| `qmv_fast_impl` (4-bit width-1) optimisations | 173(A)/(B); 176 case 1 | psi_mtp was injected into widths 2..9 only; width 1 is outside it. Ranked worth ~**zero** regardless: the serial numerator is a pinned separate binary and the candidate runs zero verifier-side width-1 rounds (`non_drafting_round_count = 0`, `research/e42_width_census.py:16`). **Result changed** (harmful -> irrelevant), but there is nothing to price. |
| 2-bit draft readout `qmv_fast_singlerow_affine2_g64` (`quantized.h:1908`) | 174(F); 176 case 3 | Its share is `psi_mtp_w1`, still unmeasured (E48 carries it). It is the entire score value of the width-1 path; pricing it with psi_mtp would double-count widths 2..9. |
| Acceptance-rate channel of any verifier-cell change | 175(B) | Not a QMV time share at all: it moves per-prompt ratios through accepted tokens and it sits behind the hard `parity_ok` gate. psi converts cost, not acceptance. |
| Prefill dequant residual (12.942 % of prefill, `qmm()` at `quantized.cpp:684`) | novelty-index row (E16/E18 lineage) | Different kernel family: psi_mtp is the **QMV** share of the candidate leg. `qmm` prefill work needs its own share measurement before any conversion. |
| Arithmetic-reduction wins at narrow width (e.g. the R2 fewer-ops bias-tree arm of 175(A), or any `qdot` ALU trim claimed at M in {2,3,4}) | 175(A) R-ladder; E53-brief ALU-conversion facts | askeladd measured injected ALU work converting to time at only **54–63 % at M=2, ~91 % at M=3, ~93 % at M=4, ~100 % at M>=5**: narrow widths are weight-traffic bound with ALU slack. A pure-ALU win therefore does not convert at psi_mtp x share, and we have no per-width ALU-conversion-weighted share instrument. **Not priceable with current instruments** — flagging beats inventing a multiplier. (This does not touch R1/R2's real purpose, which is locating the exactness wall, not speed.) |

## Retracted — do not re-propose, not re-priced

| item | status |
|---|---|
| Padding M across `vector_limit` into `qmm_splitk` | Retracted. The "+61 % ceiling" came from one un-replicated microbenchmark of a different mechanism (ledger 150, E39 entry-4 note). |
| `vector_limit = 10` as a ranked lever | Refuted four times. `applegpu_g17s` parses to gen 17, so `get_qmv_batch_limit`'s 6/10/14 table at `quantized.cpp:87` is unreachable and the limit is 10 on both hosts (`research/arch_lever_audit.py`, ledger 152–154; E40: `vector_limit >= 10 > 9` for every scored projection). |
| `MLX_`-prefixed env vars in the ranked worker | Structurally unreachable — named blocker #1. Diagnostic-only locally. |
| `qmm_t_splitk` | Dead code on this model (dispatch census, ledger "Re-verified after the merge" section). |
| "Every shipped optimisation must be shape-gated off M=1" (doctrine) | Retired in 176. Gating is risk containment only; it buys exactly zero score on ranked at `psi_mtp_w1 = 0` and is a loss when `psi_mtp_w1 > 0`. |

## Caveats

- **C1 — scored-pair-only kink model.** Order-stat prices above apply the leg
  gain to beagle and medicine only (crown tree `ef42e043` order statistics). A
  genuinely uniform all-prompt leg gain converts 1:1 at any size with no kink;
  the truth for a QMV mechanism lies between, because per-prompt width mixes
  differ. The order-stat numbers are therefore the conservative reading, and
  the linear numbers are upper bounds once past the kink.
- **C2 — corpus histogram.** `HIST` (78 dispatches, E42, tree `04ad6bf1`) and
  T(M) (E46 microbenchmark) are corpus-wide and CI-free. Part 1/2 evidence (M
  in {4,5,6} ~65 % of scored cost, beagle mean M 5.533 vs corpus 7.269) says
  every width-share above — `f{7,8}` = 0.1234 and `qmv_share(9)` = 0.5383 in
  particular — is overstated for the only two prompts that score.
- **C3 — psi is an ALU-injection measurement.** psi_mtp was measured by
  injecting ALU work; at M in {2,3,4} ALU converts to time at 54–93 %, so psi
  applies cleanly to weight-traffic-shaped costs and to the M>=5 widths that
  dominate the corpus histogram, and conservatively elsewhere.
- **C4 — module constant not updated.** `qmv_score_leverage.py` still carries
  `PSI_MTP = 0.6736` as its default. Updating gate 26 is the gate owner's
  edit; every price here passes `psi=` explicitly through the public API.

## Rows the advisor may want to re-label

1. `research/noise_floors.py` `EFFECTS` still carries `"alphonse E44
   predicted": -0.17` (an r1-era prediction). The banked E44 r2 price was
   already above the floor at the attn end, and at the new psi **every** shape
   mix is above it — check 7 of that module deliberately fails when a real
   above-floor effect is entered, which is exactly the strategy signal it was
   built to give. The update is owed; silence keeps check 7 green vacuously.
2. Ledger 174(A)/175's headline "+1.1270 %" and "+1.228 %" tops of the E44
   range were computed by multiplication above the kink (the kink was only
   pinned at `45b7c6a4`). The module's own report now flags this; the
   order-stat top is +1.1609 % at the new psi.
