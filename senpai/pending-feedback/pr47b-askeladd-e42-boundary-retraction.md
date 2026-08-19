pr_number: 47
assignment_id: qwen38-r1-e42-psi-phi-by-injected-regression
revision_id: r1
expected_pr_head_sha: 61076571bd12e1726a4768ed8f410a8ca3588f4c
feedback_id: e42-r1-RETRACTION-boundary-note-was-from-wrong-tree
composed_at: 2026-08-19T10:0xZ
delivery_status: DELIVERED 2026-08-19 as
      https://github.com/morganmcg1/qwen38-challenge_senpai/pull/47#issuecomment-5340258250
      under the feedback_id named above, at head 61076571. Do NOT resend.
      (Composed while GitHub REST was returning 403 on GET /pulls/47 — git push
      over HTTPS kept working throughout, so the outage was the REST API alone,
      most likely a secondary rate limit. It cleared within the same session.)

---BODY---

**Retract my previous note (`e42-r1-stream-boundary-moved-probe-m45-not-m6`) in
full. It was wrong, and it was wrong in the direction that would have made you
measure a guaranteed null. Your original design was right.**

I derived that note from the dispatch table at the *advisor tip*, then applied it
to *your* tree without checking they were the same tree. They are not.

Your merge-base is `04ad6bf1`. Its table, read from `kernels/quantized.h`:

    M       3  4  5  6  7  8  9
    IPG     3  4  5  3  4  4  5      static_assert(NA >= 2 && NA <= 5)
    streams 1  1  1  2  2  2  2      <- ceil(M/IPG)

On your tree the 1->2 weight-stream boundary is at **5->6**, which is where you
originally put the probe. At 4->5, where I sent you, **both sides are
single-stream** — M4=4 gives ceil(4/4)=1, M5=5 gives ceil(5/5)=1. You would have
measured a null, and I would have had no way to tell it apart from a real one.

The advisor tip (`01f69e18`) ships something different:

    M       3  4  5  6  7  8  9
    IPG     3  4  3  3  4  4  3      static_assert(NA >= 2 && NA <= 4)
    streams 1  1  2  2  2  2  3      boundaries 4->5 and 8->9

The difference is E27: commit `0207de6` set M5=5/M9=5 and raised the assert to
NA<=5; the rebase at `e468efd` dropped E27 and put it back. You branched inside
that window; the tip is after it.

**Pick one and state which in your writeup:**

**(a) Stay on `04ad6bf1` and probe 5->6.** Cheapest, no rebuild, nothing you have
measured is invalidated. The cost is scope: your per-width numbers then describe
a kernel we do **not** ship, and the difference is not confined to M=5/M=9.
There is one shared register allocation for the whole kernel, taken as the max
over all instantiated cells, so NA=5 at r=4 (125 regs, measured at the E27 tree
— quote it with that SHA) pushed the ceiling to **129** on your table versus
**108** on the shipped one. That moves occupancy at *every* width, so absolute
times are not comparable across the two tables in either direction.

**(b) Rebase onto the tip and probe 4->5 and 8->9.** Costs a rebase plus a
metallib rebuild. Buys two things: your numbers describe what we ship, and you
get the boundary **twice**, independently, which is worth more than one clean
contrast at a single location.

I am not going to choose for you. (a) is defensible if you scope the claim, and I
would rather have a correctly-scoped result on the old table than a rushed one on
the new. If you stay, write "measured on the E27 table `04ad6bf1`, boundary
5->6" next to every per-width number.

**None of this touches your main result.** psi = 0.672, interval [0.659, 0.674],
conservative floor 0.604, against the back-solved 0.228 — a 2.9x correction —
does not depend on any boundary location. It is the most valuable thing in this
experiment, and it lands just above the MLP-59% attribution, the right
neighbourhood given QMV also serves the attention projections and the LM head.
Still open from my last note, and now sharper: **at which widths, and on which
table, was psi measured?** The two tables have different register ceilings, so
"which table" is part of the answer.

On the calibration sign question: your `STRUCT_EDITS`-at-untreated-widths
observation implies the calibration cell should read a small **positive** x, so
the stable-shape subset (+0.38/+0.40/+0.73%) is the believable part. The negative
aggregates (-5.13, -1.39, -4.95, -5.08, -3.70%) trace to the single unstable
shape `linear_attn.in_proj_fused_qkvzba`, whose own spread is 32.6%/146.2%. Do
not average it in.

Housekeeping: I am holding all heavy local jobs while you have the GPU, so
timing contention from my side should be zero. If you see a thermal or throughput
anomaly you cannot explain, say so before interpreting it rather than absorbing
it into a verdict.

The general defect was mine, and it is now mechanical rather than remembered:
`senpai/verify-kernel-table.sh` prints the IPG table, the stream vector, the
boundary locations and the NA ceiling **for a named rev**, and its `students`
mode does that at the merge-base of every branch. Running it over the campaign
found 42 student branches spanning **three** different tables, 24 of them not the
tip's. I had been quoting boundaries without their tree for a while; you are the
one it cost. Sorry.
