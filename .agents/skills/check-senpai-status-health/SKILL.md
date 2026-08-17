---
name: check-senpai-status-health
description: Run the standard read-only operational and scientific audit for the Senpai Qwen 3.8 Native-MTP campaign, with explicit callouts for major scientific progress and major scientific issues. Diagnose agents that appear stuck or stopped and recommend clean, simple fixes in the challenge or Senpai harness repository. Use when the user asks to check in, get campaign status, assess advisor or student health, investigate stalled agents, inspect experiments, find official submissions, compare W&B evidence, or determine whether the Yukon frontier moved.
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

### Diagnose apparent stalls and stops

Do not equate quiet logs with a stuck agent. Classify each affected role from
evidence before proposing a fix:

- **Active:** recent agent-authored work, a live supervised job, a legitimate
  thermal gate, or a long reasoning/tool call still inside its configured
  timeout. `SENPAI_WORKSPACE_JOB_DEFERRED` with a live job is active work.
- **Blocked:** the process is healthy but forward progress depends on a real
  external condition such as authentication, a serialized GPU lock, a stale
  PR-head lease, missing exactness evidence, or user-only approval.
- **Stuck:** the service is running but repeated turns, tool failures, or
  recovery attempts make no scientific or operational progress, with no live
  legitimate wait explaining the inactivity.
- **Stopped:** the launch service is stopped/unhealthy, crash-looping, the
  active turn is terminal or quarantined without recovery, or a supervised
  job has a genuine terminal error and no follow-up turn is running.

For every role classified as blocked, stuck, or stopped, trace this evidence
chain as far as read-only access permits:

1. Service state, current PID, restart history, and the live process's actual
   `SENPAI_OPENHANDS_TIMEOUT_SECONDS` value. A saved role descriptor alone
   does not prove the running process loaded it.
2. Current conversation and turn identity; latest `OPENHANDS_RUN`,
   agent-authored `MessageEvent`, `FinishAction`, `OPENHANDS_RESULT`, timeout,
   interrupt, quarantine, recovery, and repeated `AgentErrorEvent` evidence.
   An `InterruptEvent source=user` may be generated by timeout machinery;
   corroborate it before attributing it to a human.
3. Every deferred job ID, its supervisor state, PID liveness, elapsed time,
   deadline, concise progress tail, and terminal result. Do not interrupt a
   live benchmark or bypass a thermal gate during diagnosis.
4. Workspace branch, HEAD, dirty state, current PR head, expected-head lease,
   pending inbox delivery, and whether an unpushed result or remote policy
   commit explains the wait.
5. External prerequisites: AWS authentication, GitHub access, W&B visibility,
   Yukon state, serialized runner availability, and advisor assignment state.

State the most likely root cause, supporting evidence, confidence, and what
changed since the prior audit. If evidence does not distinguish active deep
work from a stall, say so and recommend another bounded observation rather
than an arbitrary restart or short watchdog.

Recommend the smallest clean fix without applying it:

- Put contract, solver, benchmark-script, exactness-test, EOS/token-window,
  frontier, branch-policy, or campaign-instruction fixes in
  `morganmcg1/qwen38-challenge_senpai`.
- Put launcher defaults, OpenHands timeout handling, conversation/inbox
  recovery, quarantine policy, job supervision, orphan cancellation, service
  lifecycle, authentication flow, or logging/observability fixes in
  `wandb/senpai`.
- Name the likely file or subsystem, the minimal behavioral change, and the
  focused regression test or live verification needed. Prefer one direct
  invariant over new flags or broad fallback logic.
- Say when no code change is justified. For a healthy long-running turn or
  benchmark, the clean fix is often to wait and observe. If a proposed fix
  requires a restart, identify the next safe job boundary and the conversation
  state that must be preserved.

Keep all recommendations read-only during the status check. Report them for
operator decision; do not patch either repository or touch live services
unless the user separately authorizes that action.

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
   and monitoring limitations. For every blocked, stuck, or stopped role,
   include a concise **Root cause** and **Clean fix** recommendation, naming
   the challenge or Senpai harness repository. Say `no stuck or stopped roles`
   when the evidence supports it.
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
