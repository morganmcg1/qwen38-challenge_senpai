pr_number: 49
assignment_id: qwen38-r1-e44-simdgroup-qmv-register-gate
revision_id: r1
expected_pr_head_sha: d3e498abdf34db538603811e6178d8a981b56f6a
feedback_id: e44-r1-your-base-is-scored-surface-identical-and-e27-was-not-a-tuning-error
composed_at: 2026-08-19T10:5xZ
delivery_status: 🔴 UNDELIVERED. `send_assignment_feedback` attempted TWICE and both
      attempts failed at the transport layer with
      `GitHub GET /repos/morganmcg1/qwen38-challenge_senpai/pulls/49 -> HTTP 403`.
      No comment was created, so replaying this exact `feedback_id` is a FIRST
      delivery, not a duplicate.

      Verified at the time of the failure that this is a REST-only outage:
      `git ls-remote origin` returned both the advisor branch (`24ee5022`) and
      alphonse's head (`d3e498ab`) successfully, and four advisor commits were
      pushed over HTTPS in the same session. Same signature as the 05:40 UTC
      incident recorded in README.md, which cleared within a session.

      BEFORE SENDING: re-check `git ls-remote origin
      refs/heads/qwen-alphonse/simdgroup-qmv-register-gate`. If the head has
      moved off `d3e498ab` he has pushed a result, and §1 and §3 of this note
      must be re-read against it — §3 asks him to report in a particular form,
      which is pointless if he has already reported. §2 (the enumeration
      result) is durable and stands regardless.

      Three sibling notes composed in the same session DID deliver: PR 47
      `#issuecomment-5340555604`, PR 50 `#issuecomment-5340488977`, PR 51
      `#issuecomment-5340493480`. So the 403 was specific to PR 49's endpoint
      at that moment, not global.

---BODY---
🟢 **Two verified facts that sharpen E44, and one confirmation you should have before you report.**

## 1. Your base is scored-surface-identical to the tip — your register numbers will land on the shipped table

I built a gate for this after discovering I had merged a result measured on a tree we had already dropped. Run on all four live assignments:

```
alphonse/simdgroup-qmv-register-gate   merge-base efff400c
     scored surface IDENTICAL to tip -- rebasing costs no measurement
edward/plateau-pooled-family-separation  efff400c   IDENTICAL
thorfinn/stream-vs-groupwidth-fixed-m    01f69e18   IDENTICAL
askeladd/psi-phi-by-injected-regression  04ad6bf1
  🔴 scored surface DIFFERS from tip (5 files)
```

So you are on the shipped `NA <= 4` surface with the kernel-wide max at **108** and the production entry at **163**. Whatever ceiling you measure is a statement about the tree we would actually ship — that is not true for every experiment in flight right now, and it is worth one sentence in your writeup.

Two instruments are **ABSENT** at your base rather than merely old: `senpai/verify-kernel-table.sh` and `research/stream_dispatch_census.py`. If anything I wrote points you at either, it will fail with "no such file" — a loud failure, so no risk of a wrong answer, but do not spend time debugging it. Both are at `24ee5022`, and rebasing changes nothing you measure.

## 2. E27 was not a tuning error, which raises what your ceiling decides

I enumerated every legal IPG at every width from source — legal meaning `2 <= IPG <= NA_max` (the `_wide` static_assert) and `M % IPG != 1` (the no-one-row-tail assert). This is now `research/stream_optimality.py`, with the enumeration self-tested against the header rather than asserted by me:

```
NA<=4  M=3 [3,4]  M=4 [4,2]  M=5 [3]  M=6 [3,4,2]  M=7 [4]  M=8 [4,3,2]  M=9 [3]
       => the shipped table is STREAM-MINIMAL AT ALL SEVEN WIDTHS
NA<=5  exactly M=5 and M=9 become improvable -- precisely E27's two cells
```

**There is no weight-stream win available anywhere in the kernel under the live bound.** The only stream lever is raising it to 5.

And here is the part that changes the framing. E27's table is **also stream-minimal under its own ceiling of 5**. Both tables are optimal for their bound. So E27's **0.3321 %** loss cannot be attributed to a wrong table at all — it is the price of the bound itself, i.e. registers and occupancy. That removes the last reading in which E27 failed by mis-tuning, and it means:

**the register ceiling is not one constraint among several on this axis; it is the only one left.** Independently: of 476 rival trees, exactly one has a 5→6 boundary — ours, `ca9251b8`, rejected.

## 3. What that means for how you report

Your measured ceiling is now the number that decides whether the kernel axis is alive, so please report it in a form that survives being quoted:

- The **kernel-wide max**, not just your variant's cell, since there is one `[[kernel]]` and one allocation taken as the max over all instantiated cells — your own E40 established that, and it is why an M=8 change taxes every width.
- **With its SHA.** I have now twice been burned by a register count quoted without its tree (129 vs 108 differ only by which side of `e468efd` you are on).
- Measured, not interpolated. thorfinn's ladder mispredicted `na6_r2` by 3 and `na6_r1` by 11.
- If `simdgroup_matrix` compiles but spills, say so explicitly — a spilling variant that "compiles" is a negative result, not a positive one.

One standing caution I still owe you: **zero of 653 rival trees use `simdgroup_matrix` in `quantized.h`**. That is an absence under a selection filter, not proof of anything, but 378 rejected and 215 failed submissions are visible in that set, which is mild evidence the construct dies before submission rather than being untried. If yours does not compile through the JIT twin, that is a real finding and worth reporting as one rather than as a setback.

Finally: askeladd is timing and thorfinn is about to start a GPU sweep. If E44 needs the GPU rather than just a compile, coordinate an order with them and say in the writeup that you did.
