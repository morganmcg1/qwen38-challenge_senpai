---
name: check-senpai-status-health
description: Run the standard read-only operational and scientific audit for the Senpai Qwen 3.8 Native-MTP campaign, with explicit callouts for major scientific progress and major scientific issues. Use when the user asks to check in, get campaign status, assess advisor or student health, inspect experiments, find official submissions, compare W&B evidence, or determine whether the Yukon frontier moved.
---

# Check Senpai Qwen 3.8 Status and Health

Produce a fresh, evidence-ranked snapshot of the live campaign. Compare with
the preceding audit when it is available in conversation; otherwise report a
timestamped current state.

## Preserve the read-only boundary

- Do not launch, restart, terminate, submit, promote, merge, comment, push,
  edit, sync, reset, or otherwise mutate campaign or external state.
- Never run `yukon sync`, `yukon sync --harness-only`, or `yukon reset` in the
  maintained checkout.
- Treat refreshing the local `sandbox-sso` session as an allowed monitoring
  prerequisite. It does not authorize any AWS resource mutation. Follow the
  recovery flow below when the cached session is expired.
- Never print credentials, environment values, API keys, or tokens.
- Treat thermal waits and thermal aborts as legitimate. Never bypass a gate.
- Distinguish pushed Git evidence, live-agent log claims, local/W&B evidence,
  and official Yukon results in every conclusion.

## Use the fixed campaign coordinates

```text
repository       morganmcg1/qwen38-challenge_senpai
base branch      main
advisor branch   senpai/qwen38-mtp-r1
launcher         /Users/mmcguire/.codex/worktrees/docker-launcher/senpai
AWS profile      sandbox-sso
run tag          qwen38-mlx-senpai-r1
W&B              wandb-applied-ai-team/qwen38-mlx-challenge-senpai
Yukon binary     /Users/mmcguire/.local/bin/yukon
```

Use `/opt/homebrew/bin/gh` with host access. Sandboxed GitHub authentication
can falsely fail because it cannot read the macOS keychain.

## 1. Reload authority before interpreting results

Read these files completely from the campaign checkout:

1. `AGENTS.md`
2. `senpai/program.md`
3. `senpai/experiment-runbook.md`
4. `senpai/campaign-ledger.md`
5. `senpai/frontier-state.json`

Consult `benchmark.json`, the Qwen fixture, and the ranked workflow when a
score, exactness, token-window, editable-surface, or runner claim needs
enforcing-source confirmation. Narrative research notes never outrank those
sources.

## 2. Audit GitHub research state

Start with the campaign-standard query:

```bash
/opt/homebrew/bin/gh pr list \
  --repo morganmcg1/qwen38-challenge_senpai \
  --state all --limit 50 \
  --json number,title,state,isDraft,headRefName,baseRefName,updatedAt,mergedAt,url,labels,commits,statusCheckRollup
```

If GitHub rejects that aggregate query for exceeding its node limit, retry
without `commits,statusCheckRollup`, then inspect every open or recently
changed PR separately:

```bash
/opt/homebrew/bin/gh pr view <N> \
  --repo morganmcg1/qwen38-challenge_senpai \
  --json title,state,isDraft,body,comments,commits,labels,headRefName,baseRefName,updatedAt,mergedAt,url,statusCheckRollup
```

Read the advisor branch independently because it carries cross-PR corrections
and synthesis:

```bash
/opt/homebrew/bin/gh api \
  repos/morganmcg1/qwen38-challenge_senpai/commits/senpai%2Fqwen38-mtp-r1
/opt/homebrew/bin/gh api \
  'repos/morganmcg1/qwen38-challenge_senpai/commits?sha=senpai%2Fqwen38-mtp-r1&per_page=20'
```

For each PR, report:

- the causal hypothesis and changed cost center;
- whether it creates a major-progress or major-issue callout under the criteria
  below;
- new measurements and their host/window/provenance;
- exactness and ledger gates actually demonstrated;
- the predeclared stop rule and whether it fired;
- whether a candidate diff is pushed, only present in live logs, research-only,
  merged, closed unmerged, or ready for review;
- W&B run links and whether CI is queued, running, passed, or failed.

Track `student:qwen-edward`, `student:qwen-alphonse`,
`student:qwen-thorfinn`, `student:qwen-askeladd`, `status:wip`, and
`status:review`. Queued CI is not failed CI. Flag any advisor request for human
approval before a legitimate crossing submission as policy drift, but do not
intervene during a monitoring run.

## 3. Audit fleet and live conversations

Run from the launcher checkout:

```bash
uv run python k8s/aws_mac.py status qwen38-mlx-senpai-r1 --profile sandbox-sso
uv run python k8s/aws_mac.py logs qwen38-mlx-senpai-r1 --profile sandbox-sso --role advisor --tail 300
uv run python k8s/aws_mac.py logs qwen38-mlx-senpai-r1 --profile sandbox-sso --role student-qwen-edward --tail 300
uv run python k8s/aws_mac.py logs qwen38-mlx-senpai-r1 --profile sandbox-sso --role student-qwen-alphonse --tail 300
uv run python k8s/aws_mac.py logs qwen38-mlx-senpai-r1 --profile sandbox-sso --role student-qwen-thorfinn --tail 300
uv run python k8s/aws_mac.py logs qwen38-mlx-senpai-r1 --profile sandbox-sso --role student-qwen-askeladd --tail 300
```

Revalidate the node mapping from status rather than assuming it:

```text
qwen-edward   i-0e0722787ff3f1552  (advisor is co-located here)
qwen-alphonse i-0fd02cb39ea09c0e3
qwen-thorfinn i-0ffb1e5df5f691346
qwen-askeladd i-0a134765ff2073ca8
```

Prioritize the most recent agent-authored `MessageEvent`, `FinishAction`,
`OPENHANDS_RESULT`, assignment feedback, deferred job ID, and
`get_job_status` terminal observation. `SENPAI_WORKSPACE_JOB_DEFERRED` normally
means a supervised benchmark is running; follow its job ID to completion.
Ignore repeated cipher-save and `.eventlog.lock` warnings unless accompanied
by a stopped service, crash loop, or real terminal job error.

When AWS reports an expired SSO token or missing cached session:

1. Start `aws sso login --profile sandbox-sso --no-browser` in a persistent
   terminal with host access. Attempt re-authentication once; do not loop.
2. Capture only the public AWS verification URL and, when present, the short
   device code. Never expose credentials or cached-token contents.
3. Use the Chrome browser-control skill to open that exact AWS URL. Inspect the
   visible page and complete or confirm the authorization using the existing
   signed-in Chrome session. Do not inspect cookies, storage, passwords, or
   browser profiles.
4. If Chrome requests a password, MFA secret, security-key touch, or other
   user-only approval that cannot be completed safely, ask the user to finish
   it in Chrome and tell you when it is ready. Keep the terminal login pending.
5. Resume the terminal command, then prove recovery with
   `aws sts get-caller-identity --profile sandbox-sso` without printing any
   credential material. Retry fleet status and every role log after it passes.

If re-authentication fails or is denied, state exactly what remains unverified:
node health, live job IDs, and terminal job states. Do not call the fleet
unhealthy from an authentication failure. Use recent GitHub or W&B activity
only as weaker evidence of agent activity.

## 4. Audit W&B without promoting it to official evidence

Use `wandb.Api()` in the configured launcher environment to query the project
read-only. Report newly observed runs, state, group, job type, URL, and only
relevant exactness, score, token-window, `official_score`, and `rankable`
fields. If project enumeration is unexpectedly empty but PRs cite run IDs,
resolve those IDs directly and say that enumeration was incomplete.

An absent `official_score` or `rankable` field is absent, not `true` or
`false`. A finished W&B run is still local or analytical unless Yukon has a
matching official receipt. Never expose the W&B key.

## 5. Audit official Yukon state

Run only:

```bash
/Users/mmcguire/.local/bin/yukon submissions --all
```

Identify the highest-scoring row whose status is explicitly `promoted`.
Ignore `promotion failed`, `rejected`, `failed`, `superseded`, and merely
`validating` rows when naming the current frontier. Use `yukon
submission-note <id-or-prefix>` read-only when provenance or the exact
`Model: senpai` attribution must be checked.

Resolve a promoted source prefix against the organizer GitHub repository when
needed, then compare its receipt, full source ref, and score with
`senpai/frontier-state.json` and the campaign ledger. Report stale campaign
pins as a blocker for a future submission; do not sync them during this audit.
Do not infer a Senpai submission from a score, solver badge, branch, or
validation commit.

## 6. Deliver the status in this order

Lead with a UTC snapshot time and these four sections:

1. **Operational health:** advisor/student health, live jobs, real failures,
   and monitoring limitations.
2. **Scientific progress:** begin with two plainly labelled callouts:
   **Major progress** and **Major issues**. Always include both, writing
   `none identified` when the fresh audit supports no item. Then give the
   PR-by-PR changes since the last check, measured insights, exactness,
   negative results, stop decisions, and evidence residency.
3. **Submission status:** Senpai W&B/local evidence versus official Yukon,
   current promoted receipt/source/score, frontier movement, and stale pins.
4. **Links:** concise links to the relevant PRs, W&B runs, and promoted source.

Never describe a projection, one-fixture ratio, microbenchmark, or W&B run as
an official score. State uncertainty plainly when exactness, 512-token
coverage, matched-host controls, logs, or official receipts are missing.

### Scientific callout criteria

Use **Major progress** for developments that materially change the campaign's
decision state, such as:

- a credible measured winner or a completed matched 512-token comparison;
- an exactness, token-window, ledger, scope, or budget gate newly closed;
- a causal hypothesis decisively confirmed or refuted;
- a negative result or stop-rule decision that prevents substantial further
  work;
- a pushed candidate that advances from live exploration to review or
  submission readiness.

Use **Major issues** for developments that materially weaken, invalidate, or
block a scientific conclusion, such as:

- a failed benchmark, crash loop, thermal blocker, or terminal job error;
- missing matched controls, 512-token coverage, exactness, ledger closure,
  provenance, or rankable evidence behind a headline claim;
- a post-EOS/notBegun harness failure or any result measured over the wrong
  effective token window;
- a result present only in live logs when the report implies it is pushed;
- a retracted, contradicted, mislabelled, or non-reproducible measurement;
- failed CI or a stale official-frontier pin that blocks advancement.

Do not inflate routine activity into a major callout. Rank callouts by impact,
name the affected PR or role, state the decisive evidence, and say whether the
issue invalidates a result, merely limits confidence, or only blocks the next
step. A development may legitimately appear under both callouts—for example,
a useful causal discovery whose headline score is invalid because exactness
or the token window did not close.
