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
- Submitted candidate files:
- Supporting test, tooling, or documentation files:
- MTP head provenance and draft policy:
- Assignment-scope preflight:
- Editable source bytes / headroom / growth / exempt-head bytes:
- Scored-path reachability evidence:

## Evidence

- Host, memory profile, toolchain, and thermal policy:
- Exact baseline and candidate commands:
- Tests and risk-based checks:
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

## Conclusion

- What happened and why:
- Evidence for or against the mechanism:
- Prompt or M5 transfer risk:
- Smallest useful next action:
- Recommendation: promote, repeat, revise, compose later, or close
