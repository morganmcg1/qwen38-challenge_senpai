# Autoresearch Result Template

Use this for a terminal experiment result. Never infer an unmeasured score.

## Machine-readable marker

Begin with exactly one single-line marker:

```text
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"local_serial_relative_speedup","available":true,"value":1.0123},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}
```

`status` describes report completeness, not whether the experiment won. Use
`"available":false,"value":null` for an unmeasured metric; numeric zero is a
real measured value, not a missing-value sentinel.

- Student / branch:
- Hypothesis and target cost:
- Decision: green locally, ambiguous, invalid, or dead
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
- Yukon promoted submission / source ref used as frontier:
- Candidate build fingerprint:
- Submitted-surface / generated-twin / metallib digests, if applicable:
- Submitted candidate files:
- Supporting test, tooling, or documentation files:
- MTP head provenance, digest, and draft policy:
- Token window, fixture, reference source, and harness (`local` or `ranked`):
- Exact cell: shape, width/M, dispatch family, source form, and M5 variant:
- Official causal path and score equation:
- Assignment-scope preflight:
- Editable source bytes / headroom / growth / exempt-head bytes:
- Scored-path reachability evidence:

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy:
- `head_provenance_sha256` for every leg, baseline and candidate:
  (mandatory. `score.json.uses_pinned_mtp_head` reads `true` for both
  artifacts and must not be used. See the runbook section "Prove which
  proposal head you measured".)
- Exact baseline and candidate commands:
- Cheapest real falsification gate and positive-control verdict:
- Tests and risk-based checks, in execution order:
- Exact-token and row-ledger verdict:
- Divergent tokens or failure category, if any:
- Generated-twin audit, if relevant:
- Peak RAM or head/artifact size, if relevant:
- Official status and score, if submitted:

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | ... | ... | ... |
| MTP seconds/token | ... | ... | ...x |
| local serial-relative speedup | ... | ... | ... |
| effective mean draft length | ... | ... | ... |
| accepted draft rate | ... | ... | ... |

The local score is a one-prompt directional measurement. It is not the ranked
median across eight hidden prompts.

State whether every compared identity field matched. Label every inference,
interpolation, or extrapolation and identify its source tree and domain. Never
present a local cancellation term as part of the ranked score equation.

## Conclusion

- What happened and why:
- Evidence for or against the mechanism:
- Prompt or M5 transfer risk:
- Smallest useful next action:
- Recommendation: promote, repeat, revise, compose later, or close
