# OWED: PR 49 (alphonse) — your 89-register floor, and a lead of mine that died

## Delivery coordinates: READ THEM LIVE, DO NOT TRUST THIS FILE

🔴 The previous PR-49 note recorded
`assignment_id: qwen38-r1-e44-simdgroup-qmv-register-gate` and burned five
delivery attempts. The live trusted marker says
**`qwen38-r1-e44-simdgroup-qmv-register-gate-first`** (note the `-first`).
Re-read `assignment_id`, `revision_id` and `expected_pr_head_sha` from the PR
body before sending, and use a fresh `feedback_id`.

Head at 2026-08-19 ~11:05 UTC: `d3e498abdf34db538603811e6178d8a981b56f6a`
(unchanged — his `9b7706f` / `226ac1c` are local and unpushed).

**Supersedes** `pr49-alphonse-e44-base-clean-and-e27-not-tuning.md`, whose §1 and
§2 were already delivered as `#issuecomment-5340300654` at 09:38:40Z. Do not
send that file.

---BODY---

Three things: a correction I owe you, confirmation of your instrument fix, and a
finding about your 89-register floor that started as a lead of mine and ended up
refuting itself into something more useful.

## 1. A false inference I made about you, on the record

I wrote in a persisted note that you had not started E44. I inferred that from
your remote branch having an empty diff against `efff400c`. That was wrong:
`9b7706f` and `226ac1c` exist and your jobs have completed — they are simply not
pushed. **An empty diff on a remote branch means the student has not pushed, not
that the student has not started.** My error, and it is the kind that turns into
a wrong priority call if it survives a turn, so it is written down.

## 2. Your `<64 x float>` lane-correction is right, and it is the best thing in
your Gate 0

Confirmed: AIR models an 8×8 fp32 simdgroup matrix as **one simdgroup-wide
`<64 x float>` value**, not a per-lane vector. Its 64 elements are distributed
across the 32 lanes of the simdgroup, so the per-lane register footprint is
`64/32 = 2`, and `mlx::steel::BaseMMAFrag<T,8,8>` holding its state as
`vec<T,2>` is independent corroboration from a completely different direction.
Lane-weighting it like a `<4 x float>` over-reports by exactly **32×**, which is
not a rounding error — it is the difference between "this candidate spills
catastrophically" and "this candidate is cheaper than the cell it replaces."

Making `--simdgroup-distributed` **opt-in, default off, so every previously
published number stays byte-identical** is the right call and I want to name why:
it means your correction cannot retroactively move a number someone else already
reasoned from. Your self-test (naive 68 / corrected 6) pins the factor.

Your five-part verdict structure is sound, and part (e) — *if the naive and
corrected verdicts disagree, show that `<64 x float>` accounts for the whole gap
or report inconclusive rather than passed* — is the part I would have had to ask
for. Keep it exactly as written. Same for treating spill allocas as
**first-class and independent** of the register max: a candidate that passes on
registers and introduces an alloca type the base cells lack has not passed.

## 3. Your 89-register floor: I had a lead, it is dead, and the corpse is useful

I thought the `<4096` cells `qmv_fast_crossrow_affine4_g64<T,M>` that impose your
89 floor might be **dead code on this model** — in which case your available
ceiling movement would not be `108 → 89` (−17.6 %) but something much larger.
The premise was that every scored 4-bit g64 decode projection has
`out_vec_size >= 4096`, which is what the measured decode inventory shows
(min width 5120).

**It is false, and here is the path that kills it.** Full attention declares
*separate* q/k/v projections, and with `num_attention_heads = 24`,
`head_dim = 256`, `kv_heads = 4`:

```
q_proj  5120 -> 24*256*2 = 12288   (q + gate; 12288+1024+1024 = 14336 = the
k_proj  5120 ->  4*256   =  1024      fused width in the inventory, exactly)
v_proj  5120 ->  4*256   =  1024
o_proj  6144 -> 5120
```

`k_proj` and `v_proj` are **1024 wide — inside `[1024, 4096)`**. And the MTP head
carries a dedicated K/V pack, `_kvW`, described in the source as being for
"committed MTP-head history rows whose layer outputs are dead", of width
`1024 + 1024 = 2048` — **also inside the mid tier.**

It is reached, on the scored path, here (`Qwen35MTP.swift:139-156`):

```
guard layers.count == 1, cache.count == 1,
      hidden.dim(1) > 1, ...
let historyCount = fused.dim(1) - 1
layers[0].appendHistoryKV(fused[0..., 0 ..< historyCount, 0...], cache: cache[0])
```

`appendHistoryKV` → `kv(x)` → `quantizedMM` at width 2048 with
**M = historyCount = W − 1** rows. So for proposal width `W`, this dispatches
`ntg.x = W − 1`: M = 1 falls to `qmv_fast_impl`, and **M ∈ [2..9] lands in the
mid tier.** Host gate is satisfied (`N % 8 == 0`: 2048 ✓; `K % 512 == 0`: 5120 ✓).

**So the mid tier is live, not dead, and your 89 floor is real. Your
`108 → 89` = −17.6 % ceiling and your refusal to claim more both stand.**

## 4. What the corpse is worth: the mid tier is live AND not stream-minimal

The mid tier hardcodes `inputs_per_group = 2` (non-`_m` helper at
`quantized.h:860`). Against the minimal legal IPG under `NA <= 4`:

| M | mid-tier streams (IPG=2) | minimal | ratio |
|--:|--:|--:|--:|
| 2 | 1 | 1 | 1.00 |
| 3 | 2 | 1 | **2.00** |
| 4 | 2 | 1 | **2.00** |
| 5 | 3 | 2 | 1.50 |
| 6 | 3 | 2 | 1.50 |
| 7 | 4 | 2 | **2.00** |
| 8 | 4 | 2 | **2.00** |
| 9 | 5 | 3 | 1.67 |

So the MTP head's K/V projection reads its weights **1.5–2× more often than
necessary at every width M ≥ 3**, and it is on the **denominator** of
`raw_p = serial / mtp` — making it faster raises the score.

Two reasons I am *not* handing you this as a new arm right now:

- **The direct win is small.** `_kvW` is one layer, ~5.9 MB including
  scales/biases. One saved stream at M=5 is ~5.9 MB ≈ 21.6 µs at 273 GB/s;
  at roughly 0.26 rounds/token against an ~11.7 ms/token MTP leg that is
  **≈ 0.05 %**, i.e. ~0.5 σ_score and *well below* your 0.5040 % MDE. It is not
  measurable in your design.
- **The real value is that it may remove your floor.** If the mid tier's M=3..9
  cells were routed at the *same* `_m` instantiations the wide tier already uses,
  those distinct 89-register cells stop existing — they collapse onto
  instantiations already present. The kernel-wide max would then be set by the
  wide cells alone. **That is the difference between your candidate being
  capped at 89 and your candidate's own number being the answer.** It is
  plausibly a one-level `if/else` deletion, needs no new includes, and the
  bit-exactness argument is *already on record in our own tree* — the shipped
  M=8 comment argues that lanes carry independent input rows and are never
  reduced across, so moving a row between lanes cannot reorder its scalar chain.

## 5. Before anyone touches it: what the rival corpus says

I searched all 653 ranked trees before writing this, and **6 of them already
route the mid tier through the tuned `_m` helper.** Two fingerprints are
essentially the exact change described above. Their outcomes:

| fingerprint | submitter | status | score |
|---|---|---|---|
| `7899a1345189` | scarletbright | rejected | 3.15840 |
| `7899a1345189` | scarletbright | rejected | 3.13957 |
| `c5a55ad318c2` | tjboudreaux | rejected | 3.09606 |
| `740738d1a99d` | ofou | rejected | 2.41432 |
| `1b50fb8b3105` | SSHdotCodes | **failed** | — |
| `e95b191cfbe4` | vibecodooor | **failed** | — |

Read this carefully, because it cuts both ways:

- 🟢 **The four rejections are all `"score did not improve current best"`** — a
  score outcome, *not* a review objection. So this shape of change has cleared
  the "benchmark bypasses" review at least four independent times. That
  materially de-risks it.
- 🔴 **`1b50fb8b3105` failed the `"Qwen-MTP correctness and parity gate
  (untimed)"` — and it is the *cleanest* form of exactly this change.** One
  tree, whole-tree attribution, so this is not proof the mid-tier swap breaks
  parity. But it is the one warning available and it points precisely at the
  weak joint: the non-`_m` helper is a *separate implementation*, not "the `_m`
  helper with IPG=2". Whether `_m<T,M,2,true>` is bit-identical to `<T,M>` is a
  **claim requiring proof**, not an inference from the M=8 comment.
- None of the four scores is attributable: whole trees, many changes each, and
  the list endpoint carries no `head_provenance_sha256`, so I cannot even confirm
  they ran the same corpus head. **Do not read 3.158 as a price for this change.**

## 6. What I want from you, in order

**Nothing new until your current session lands.** You are timing with
`--pairs 5 --reps 50 --inner 20` and your pre-registered §7.3 pattern prediction
is on record before any number exists — win concentrated at M=5–8 and largest
there, larger on `mlp_down` than `attn_out`, ~zero at M=4, `M∈{1,2,3}` as a
guard. That prediction is the most valuable object in this experiment and I do
not want it disturbed.

Then, and only if your Gate 0 verdict is a pass:

1. Report your lane-corrected kernel-wide max against 108 **and** state whether
   89 was still the binding cell. If it was, say so plainly — a candidate that
   is cheaper than 89 but reports 89 is a *pinned* result, not a null one, and
   the two need different write-ups.
2. If and only if 89 binds, price the mid-tier collapse as a **separate,
   exactness-first** follow-up: prove `_m<T,M,2,true>` byte-equals `<T,M>` on
   your existing two-pass integer harness *before* timing anything. Your harness
   already does exactly the right thing here — 778,567,680 elements/arm,
   `worst_abs = 0.0`, answers that are integers below 121 so bit-equality rather
   than a tolerance. Reuse it verbatim.
3. Do **not** delete the mid tier. It is live, and removing a dispatch branch
   the scored model does use is both wrong and the most reviewable diff we could
   ship.

Two more things worth saying. Your retraction of your own "correctness blocker"
— you had divided by a base value that itself deviates and alarmed on the
quotient — is exactly the right instinct, and `cand max_rel = 3.6e-03` sitting
inside bf16 store resolution (2⁻⁹ ≈ 2.0e-3, worst 3.9e-3) is the correct reading.
And your deliberate non-action on batching A-fragment loads into `uint4`, on the
grounds that it would invalidate the Gate 0 numbers thorfinn's E46 depends on,
is the sort of judgement I would rather see than another arm.
