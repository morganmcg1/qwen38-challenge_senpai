# Experiment Assignment Template

Complete this before an experiment branch starts. Keep Yukon-submitted paths
separate from research-only tests, scripts, notes, and instrumentation.

- Student / branch:
- Full `BASE_SHA`:
- Full `UPSTREAM_SHA`:
- Yukon promoted submission / source ref:
- Host, chip, memory profile, and toolchain:
- MTP head provenance:
- One causal hypothesis:
- Scored cost and expected direction:
- Scored call-path proof:
- Submitted candidate paths:
- Research-only support paths:
- Draft-policy or head change, if any:
- Runtime-effective JIT/AOT source and `_nax` variant, if relevant:
- Numerical, recurrence, cache, or ledger risk:
- Cheapest decisive test:
- Stop rule:

## Required preflight

Run these from the recorded base before expensive work, replacing the example
with every proposed submitted path:

```bash
senpai/validate-assignment-scope.sh "$BASE_SHA" \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift
senpai/check-editable-budget.sh "$BASE_SHA"
```

The scope check reads `benchmark.json` from `BASE_SHA`, never from the working
tree. The budget check applies the source caps and separately accounts for the
contract's exempt `mtp-head/` weights path.

Before timing a selector or kernel, prove it reaches the native-MTP worker at
the scored shape. If a generated Metal family is involved, run:

```bash
research/twin_audit.py "<generated-stem>"
```

## Authority boundary

- The student may edit, test, commit, push, and report the assigned branch.
- Only `benchmark.json` `editablePaths` enter a Yukon submission.
- The advisor owns integration of `upstream/main` and promotion of campaign
  `main`; students do not merge organizer history into an active arm.
- An authorized advisor, student, or operator may submit a committed,
  preflighted candidate, but must use an exact model name and a reviewed public
  note with `senpai/submit-official.sh "$BASE_SHA" ...`.
- Never print, commit, or copy Yukon credentials into notes or logs.
- Documentation-only changes do not advance the submitted solver snapshot.
