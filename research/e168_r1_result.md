# E168 r1 — depth cap 4 on an organizer-pure tree: ranked result

Student: `qwen-askeladd`. PR #168, revision r1. Host `g16s` (Apple M4 Pro, 48 GiB).

## Question

Does the local depth-cap-4 step (`segmentedVerifyDepthCap 7 -> 4`) transfer to
the ranked M5 runner when the candidate carries no campaign extras?

## Candidate identity

| Field | Value |
| --- | --- |
| Organizer base | `0863b06ac16e26e48fc06e97444095b00feb66d4` (`upstream/main`) |
| Campaign `BASE_SHA` | `770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf` |
| Candidate surface commit | `e044a583` |
| Frozen submission HEAD | `99959d49` |
| Submitted diff | one literal plus 7 doc-comment lines |
| CLI / worker / metallib | `a55bd3ef` / `0b8b1b35` / `5de2569e` |

## Local confirmation (W&B `u2g6yfjx`)

512-token `--local-submit`, real 40 C gate, capture enabled.

| Metric | Value |
| --- | --- |
| Candidate MTP seconds/token | 0.030283431755378842 |
| Serial seconds/token | 0.073962957132607698 |
| Local speedup | 2.4423571849471997 |
| All tokens matched | true |
| Residual divergence | 0 |
| Declared rows / ledger close | 548 / 548 |
| Emitted tokens | 512 |
| Rounds, mean M, max M | 111, 4.937, 5 |
| Depth histogram | `{2:1, 4:4, 5:106}` |
| Accepted / proposed | 0.919908466819222 |
| Expected decode length | 3.936936936936937 |
| GPU temperature | 37.9 C pre-gate to 59.2 C |

Deltas: **-3.97 %** versus the r0 base (31.535 ms/tok) and **+0.67 %** versus the
r0 cap arm (30.082 ms/tok). The +0.67 % is the cost of reverting the campaign
kernels, not a cap effect.

## Ranked result

| Field | Value |
| --- | --- |
| Receipt | `90c131dc-c3e4-4022-b00c-038cf928bbb7` |
| Status | **rejected** |
| Published score | **3.54742900664627** |
| Scored commit | `4937373` |
| Submitted / resolved | 09:23Z / 10:06Z (43 min) |

Comparisons:

- Receipt A `5a9f130a` = 3.70784519415395 -> **-0.16041619, -4.33 %**
- Crown `ec24d591` = 3.7291100105909 -> -0.18168100
- Registered prediction 3.9847 -> **-11.0 % error**, outside the low edge of the
  registered +4 % to +8 % band.

Band verdict: below the lowest briefed band (`3.55-3.65`, step absent) by
0.0026. The depth axis closes.

## Conclusion

The step **inverts** between hosts: a clear local win became a clear ranked
loss, at roughly 23x the candidate-leg 1 sigma of 0.189 % and 16x the ~0.271 %
serial draw. Ranked M5 prefers the deeper schedule that cap 4 removes, which
supports the briefed **declamp-deeper** pivot.

Confound: r1 changed two things relative to receipt A — the cap literal and the
removal of every campaign extra. The magnitude is therefore a composite. Because
the campaign extras net-hurt ranked, removing them should have helped, so the
cap's own ranked penalty is not smaller than 4.33 %. The sign is robust; the
exact magnitude is unattributed.

Execution was clean: frozen tree, one-literal packaged diff, three green gates,
exact 512-token match with ledger closure. Yukon scored the run, so this is a
performance verdict rather than a compliance failure.

## Branch state after the receipt

The research base moved 21 commits ahead while the receipt validated, so this
branch merges `5cf5a255`. The experiment failed, so the tip carries the base
code unchanged plus research-only files; it does not carry the cap-4 literal or
the organizer-pure reverts. Those exist only in history, at candidate surface
`e044a583` and frozen submission HEAD `99959d49`, which is the exact tree Yukon
scored. A plain merge would have silently reverted base work, so base content
was restored deliberately before the merge commit.

## Reproduce

```bash
bash research/e168_r1_build.sh
bash research/e168_r1_confirm.sh
python3 research/e174_wandb_log.py
```

## Suggested follow-ups (not implemented)

1. **Declamp deeper.** Test caps above 7 on ranked, since the ranked sign favours
   depth. Screen locally only for exactness, not for time: this experiment shows
   local time is anti-correlated with ranked time on the depth axis.
2. **Separate the confound.** One organizer-pure receipt at the unchanged cap 7
   would price the campaign extras and the cap independently on ranked, and
   would also give a clean organizer-pure ranked reference for future work.
3. **Distrust local depth screening.** Record in the ledger that local
   seconds/token inverted against ranked on this axis, so no future depth
   experiment is promoted on local timing evidence alone.
