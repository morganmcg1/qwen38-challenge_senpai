# E19 `key_len = 1024` residual band — working notes

Assignment `qwen38-r1-e19-keylen-1024-residual` r1, base `1bb627ab`, PR #21.
Zero-GPU assignment: no timed run, no harness invocation, no GPU lock taken.

Full write-up: `research/results/qwen38-r1-e19-keylen-1024-residual.md`.
Tool: `research/sdpa_keylen_band.py`.

## 1. Signed falsification statement

Committed before any timed work on this hypothesis, per the E11/E17 convention.
Each item is a result I would accept as refuting my own claim.

1. **Mechanism.** If a 512-token candidate leg on the ranked M5 produces an
   `mtp-row` ledger with **no** deviation at pos 1022–1024, the E19 chain is
   false on the ranked host and the repair must be dropped. I will report that
   as a refutation, not re-explain it as a tolerance artefact.
2. **Width-invariance.** If any two rounds of different width `w` that place the
   same absolute position in the same kernel family and `blocks` produce
   *different* top-2 values at that position, the central theorem is false and
   the whole band table collapses. (Currently supported at `pos 1022` by three
   widths — w ∈ {4,8,9} — agreeing exactly.)
3. **Band as superset.** If a measured deviation appears at any position
   *outside* the predicted band for its round's width, the dispatch law as I
   read it is incomplete. `--validate` is the standing test; it currently reports
   `arms out of band: 0` over 3584 rows.
4. **Repair.** If a built Swift repair leaves any suspect row, or costs more than
   a handful of extra SDPA launches per leg, the design is wrong. Predicted:
   band `→ []`, **exactly +1 launch per leg**.
5. **Device class.** If the ranked M5 reports an architecture suffix outside
   `{'d','s'}` *and* still shows window-end deviation, then `CPP:748` is not the
   governing gate and my reading of the dispatch law is wrong.

## 2. Source-reading trail

Everything in the result doc traces to these files; line numbers re-verified
against the vendored tree at base `1bb627ab`.

| file | what it settles |
|---|---|
| `.../metal/scaled_dot_product_attention.cpp:626-641` | which kernel family is legal (`head_dim=256` kills `supports_sdpa_full` permanently; `qL*gqa<=32` ⇒ `qL<=5`) |
| same `:745` | serial rounds run with `do_causal = false` (`qL=1`) |
| same `:748-751` | **the two-pass gate**: `(devc∈{d,s} && kL>=1024)`; the `>=4096` GQA clause is unreachable in a 1024 window |
| same `:440-478` | `blocks`: `'s'`→64, `'d'`→128; no escalation at `N<=1024` |
| same `:495` | `intermediate` takes `q.dtype()` ⇒ **bf16 partials**; `sums`/`maxs` fp32 |
| same `:358-359`, `:485-486` | launch geometry: `tpg.y == qL` single-pass, `tptg.z == qL` two-pass |
| `.../kernels/sdpa_vector.h:43,56,99,102` | single pass, fp32 accumulator, constant-stride key loop, masked keys *skipped* |
| same `:200-330`, esp. `:317-318` | two-pass stage 1 rounds the partial numerator to bf16 |
| `mlx/fast.cpp:713-715, 828-869` | the fallback is a bf16 `matmul`/`softmax(precise)`/`matmul` composite |
| `MLXLMCommon/AttentionUtils.swift` | the `qL ∈ 6..9` split; `kSplit = kL-(qL-5) == p_last(A)+1` |
| `MLXFastModel/KVCache.swift:437-438` | `cachedKeys.dim(2) == offset` despite `step = 256` padding |
|  `MLXFastModel/Qwen36MTPBlockSession.swift:609-611` | depth clamped to remaining budget ⇒ no `kL > 1024` overshoot |
| `MLXFastTrustedHarness/QwenRuntimeDFlash.swift:~773,~795-830` | what the contract actually checks (top-2 *values*, AND semantics; emitted token needs only membership) |
| `MLXFastCore/Constants.swift:604,622` | `absolute = 4.875`, `relative = 0.25` |

The one thing **not** established by reading: that the scored process actually
observes `devc ∈ {'d','s'}`. That is the named missing fact, and item 1 of the
falsification statement is the cheap test for it.

## 3. Why the tool works the way it does

`sdpa_keylen_band.py` is a transcription of the table above plus the shipped
`qL ∈ 6..9` chunk, with no MLX dependency, so it can run on a host with no GPU
lock and no weights. Design choices worth recording:

- **Segmentation is a function, not a branch.** `segments_current` (shipped) and
  `segments_repair` (proposed) have the same signature, so `analyze` and the
  4572-case `sweep` run over both without special-casing. That is what makes
  §4 attack (iii) — composition of chunk and repair — mechanically checkable
  rather than argued.
- **`candidate_rows` raises `CausalMisalignment`** instead of returning a flag,
  so the off-by-one audit cannot be silently ignored by a caller. The sweep
  counts the exceptions.
- **`--discriminate` inverts the question.** Rather than asserting the M5's
  device class, it asks which classes can explain PR #2's measured rows at all.
  `'g'`/`'p'` leave all 7 arms unexplained; `'s'`/`'d'` explain all 7 and give
  byte-identical band tables. This is what makes §2 moot instead of blocking.
- **Positions are 1-based and seed-inclusive** to match PR #2's `mtp-row pos`
  convention. The advisor's §1.3 column is 0-based cache position, hence the
  off-by-one correction. `parse_trace` does the conversion in one place.
- Trace grammar: `mtp-trace: round=(\d+) d=(\d+) acc=(\d+)`, `off` starts at
  512, `width = d+1`, `key_len = p0+width`, `off += acc+1`. Two
  `mtp-trace: begin` markers per log (serial leg, then MTP leg).

## 4. Loose ends deliberately left alone

- `research/kl-boundary-runJ.json` is **provenance-buggy** — computed from the
  wrong arm's trace (claims w3/round 82; truth is w4/round 81). Do not cite it.
  The `boundary_key_len` spread across `kl-boundary-*` is just the `--boundary`
  flag, not a real disagreement.
- runI `w=8` prints top-1 `0x1.28p+5` where other arms print `0x1.2ap+5`. One
  8-bit-print inconsistency; recorded, not explained, not fitted.
- Whether `Qwen35FastEngine.swift` is live in the timed leg is unresolved and
  cannot change any E19 answer.
- The §5 `DEEP_CAP = 7` re-score is out of scope for r1 and was not run, so it
  could not displace §2.
