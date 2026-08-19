# Qwen 3.8 Senpai Research Handoff

Generated on 2026-08-19 after advisor commit `45b7c6a`. This is a historical
synthesis for fresh conversations, not a live frontier authority. Refresh
Yukon, GitHub, [`frontier-state.json`](frontier-state.json), and the enforcing
workflow before acting.

## Frontier At Handoff

- Campaign main: `6391b03a39dfd56daac0f67e62851d0a86187963`
- Advisor branch: `45b7c6a4e4a94e4c6389d8a7e9d76ccd47d4a239`
- Promoted receipt: `0cd0a6b4-b539-4705-a1c7-cb271c1f9d3b`
- Promoted source: `0c90733d383f6b987a29682bf9eb9458a6172bfa`
- Promoted score: `3.24929398547457`
- Best Senpai receipt: `ca9251b8-58cd-4d90-9a52-fa05f5657216`
- Best Senpai score: `3.23250848263467`
- Senpai gap: `0.01678550283990`, approximately `0.5166%`

No Senpai submission had promoted at this cutoff. The advisor copy of
`frontier-state.json` was stale in its `promotedSubmission` block even though
its `boardTop` block named the current frontier. Replace it with a freshly
verified live state before a submission or baseline declaration.

## Load-Bearing Corrections

The ranked score uses two different workspaces:

```text
raw_p = pinned-baseline serial seconds/token
        / candidate-workspace MTP seconds/token
score = median of eight raw_p values
```

Candidate edits cannot change the official serial numerator. Any compliant
edit that lowers candidate MTP time improves the affected official ratio.
`psi_serial = 0.8525` is a local same-build cancellation term and must never be
subtracted from official value. The old ranked equation
`psi_mtp - psi_serial = -0.1789` is retracted. E50 closed this error in
[PR #54](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/54),
followed by advisor corrections `3cfcba3`, `c9ed85b`, and `45b7c6a`.

The students do not share one GPU. Advisor and Edward share Edward's Mac;
Alphonse, Thorfinn, and Askeladd each have another Mac. Locks and thermal state
are host-local. Independent experiments should run concurrently across hosts.

## Official Senpai Submissions

| Receipt | Candidate | Status | Score | Conclusion |
| --- | --- | --- | ---: | --- |
| `4437d06` | early tree with wrong head provenance | rejected | `2.8612659037` | Head/provenance mismatch prevents kernel attribution. |
| `74d1bd3` | campaign attempt | failed | — | No rankable score. |
| `b360b4c` | campaign attempt | failed | — | No rankable score. |
| `9197ed6` | LR3 replacement head | rejected | `3.0693815947` | Large head regression. |
| `ca9251b` | E27 NA=5 table | rejected | `3.2325084826` | Best Senpai row; shared register tax erased local cell wins. |
| `2c76644` | Dev40 cut-12k head | rejected | `3.0721325826` | Large head regression despite an exact 128-token local screen. |

## Closed-PR Findings Worth Retaining

### E26: fixed-window post-EOS continuation

[PR #30](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/30),
merge `a3a351d`. The editable session previously cleared pending state at EOS
and then threw `notBegun`. E26 commits EOS like any other token so the trusted
parent can complete 512 tokens. Retain this correctness repair; do not price it
as a ranked speed lever.

### E29: local-ratio trap and direct attribution

[PR #34](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/34),
merge `9d77534`; [W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/du9xyb90).
Disabling the ladder made the local ratio look 16.5% better by slowing local
serial. Candidate MTP moved about 0.11%, within noise. The former 10.26% host
overhead estimate was subtractive-accounting error; direct attribution placed
removable host graph-build cost near 4.35%. No speed win shipped.

### E34/E37: ranked operating point and width evidence

[PR #39](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/39) and
[PR #42](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/42).
Ranked prompts draft much deeper than the sole public fixture. Beagle and
medicine were the central order-statistic prompts in the inspected cohort.
Some width histograms are inferred from acceptance constraints rather than
directly observed; always name those assumptions.

### E38: row blocking and the 128-register wall

[PR #43](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/43);
[W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/8hxtxna4).
The M=6 row-blocked arm measured `0.9858` drift-adjusted against a registered
`0.84` prediction. A roughly 11.96% second-pass prize was nearly canceled by a
roughly 10.54% blocking tax. The register ladder was NA2 62, NA3 83, NA4 104,
NA5 125, NA6 at least 144 or spilling. Straight row blocking is closed.

### E40: one shared kernel allocation

[PR #45](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/45),
merge `75ce33b`. Width-specialized helpers inline into one Metal kernel, so the
largest instantiated cell sets the allocation for all widths. A local M=5 or
M=9 win can tax untouched widths. This explains E27's official loss.

### E41: K-tiling negative

[PR #46](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/46);
[W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/boaj1k8p).
K-tiling measured about `-11.2%` and was negative on 8/8 shapes. The negative
survives, but its absolute curve came from the abandoned E27 table and must not
be transferred to the current dispatch table.

### E42: candidate-leg QMV share

[PR #47](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/47),
merge `9fe0dc5`; [W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/bitem8ak).
A bit-exact injected regression estimated candidate-leg shares
`psi(widths 2...9)=0.6736` and `psi(widths 6...9)=0.6133` on tree `04ad6bf1`.
The candidate share is useful; its absolute census and stream boundaries are
tree-specific. Its `psi_serial` result is local-only.

### E43: superlinearity identified, decomposition unresolved

[PR #48](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/48);
[W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/piz9gjgg).
A linear per-round cost family missed by about 5.70% versus about 0.281% pair
noise. Step and quadratic families both fit, so no unique removable step is
identified.

### E44: best performance lead, blocked by exactness

[PR #49](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/49),
merge `454410e`; [W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/dn6hk8u7).
The M=7/8 simdgroup-matrix QMV measured +11.389%, +17.050%, +4.596%, and
+12.649% across the attn-out and MLP-down cells, mean +11.421%. A byte-identical
A/A control resolved no cells and had worst apparent effect 0.263%.

The implementation changes the BF16 expression tree and differs from the base
on real floating-point values. Integer Gate B could not expose rounding by
construction. Being closer to double precision does not establish exact token
parity. The scored diff was reverted; do not submit it without locating and
closing the exactness wall.

### E45/E50: external score noise and score-model correction

[PR #50](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/50) and
[PR #54](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/54);
[W&B](https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/1hqbkqze).
Content-deduped board trees imply a between-submission score floor near 0.7678%,
candidate-MTP relative SD near 0.8040%, and serial relative SD near 0.2110%.
Official rejection is weak evidence below that floor. Do not resubmit identical
trees to mine variance. E50 also established the pinned-numerator correction.

### E46: weight-stream count is causal

[PR #51](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/51),
merge `0ac5bc2`. At fixed M=6, group width with the same two stream count was a
null. At fixed M=8, changing two streams to three cost +18.72%, with 8/8 shapes
slower and sign-test `p=0.0078`; an independent earlier route measured +19.02%.
Optimize stream count without raising the shared register ceiling.

## Current Open Leads

- **E51 / [PR #55](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/55):** locate the BF16 exactness wall with an ascending perturbation ladder and a real fixed-window reference. Run exactness before long timing.
- **E49 / [PR #53](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/53):** test the isolated M=9 two-stream prize and then the shared-allocation tax on untouched widths.
- **E48 / [PR #52](https://github.com/morganmcg1/qwen38-challenge_senpai/pull/52):** retain the per-prompt width/cost census; discard the old question about a candidate edit moving the official serial binary.

## Fresh-Conversation Priorities

1. Refresh the live promoted row and repair `frontier-state.json`.
2. Inspect the whole submitted surface against the current promoted source.
3. Read `AGENTS.md`, `program.md`, the runbook, this handoff, the ledger, and
   the enforcing workflow before assigning work.
4. Run E51 and E49 concurrently on different Macs. Use another independent Mac
   for the corrected E48 census or a distinct candidate mechanism.
5. Require every assignment to state the exact proposition, identity tuple,
   official causal path, expected value versus the frontier gap and noise floor,
   cheapest real exactness gate, positive/null controls, and stop rule.
6. Submit a clean current-frontier candidate promptly when it has a real
   measured mechanism and complete pre-submit evidence. Continue research after
   every result.
