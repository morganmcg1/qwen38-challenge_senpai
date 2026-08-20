# E62 — the warm-up leg reproduces the E60 anchor, and what that costs us

The declared discarded warm-up leg of session `r1ops` is also a free
cross-session control, because it runs the shipped geometry on the stock arm.

| quantity | value |
|---|---|
| leg | `e62-r1ops-01-warmup` |
| decode tokens | 512 |
| `mtp_seconds_per_token` | 0.03400300769135356 |
| E60 arm-B anchor | 0.0340649975114502 |
| **delta** | **-0.182 %** |
| `serial_seconds_per_token` | 0.07248233980499208 |
| `mtp_decode_speedup` | 2.1316449551438716 |
| `all_tokens_matched` | true |
| `residual_divergence_count` | 0 |
| `effective_mean_draft_len` | 6.5131578947368425 |
| `wired_residency_active` | false |
| worker env | `MB=512`, `OPS=50`, `PROFILE=full` |
| wall | 01:36:49 -> 01:41:03, 254 s |
| W&B run | `258zcwrd`, project `wandb-applied-ai-team/qwen38-mlx-challenge-senpai` |

Every leg of session `r1ops` resumes that one run id, so the whole ladder is a
single W&B record rather than fifteen disconnected runs.

## The good part

The stock arm's worker `__TEXT,__text` is byte-identical to the E60 arm-B
binary, and it reproduces the anchor to under two parts in a thousand on a
different day. The harness, the geometry exports and the timing path are all
behaving.

## The uncomfortable part

**-0.182 % is larger than the -0.15 % minimum useful effect this assignment is
hunting.** Identical machine code, identical geometry, identical host,
identical token window, and the number still moved by more than the effect
size that would count as a win.

That is a direct measurement of cross-session drift, and it settles a
methodological question for this PR:

- **No cross-session comparison against the anchor can establish a win here.**
  Any candidate that beats `0.0340649975114502` by 0.2 % on a different day is
  indistinguishable from this warm-up leg, which changed nothing at all.
- Only **within-session** contrasts, position-balanced and fitted with
  `leg_position` in the model, can resolve effects at this scale. That is what
  session `r1ops` is built to do.

So the anchor keeps its role as a sanity check on the harness. It is not a
baseline that a candidate can be scored against, and I will not report any
ladder result as a delta against it.

## Consequence for the leg budget

A leg is 254 s. The 15-leg session is about 64 min, which matches the plan.
