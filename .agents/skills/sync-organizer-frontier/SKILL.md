---
name: sync-organizer-frontier
description: Safely update qwen38-challenge_senpai from Layr-Labs/qwen-3.8-mtp-challenge without overwriting Senpai files or repointing the research remote. Use when adopting organizer contract or harness updates, refreshing the maintained campaign base, or importing the best promoted editable-path solver snapshot.
---

# Sync Organizer Frontier

Integrate in a dedicated branch. Keep organizer policy, the promoted solver
snapshot, and campaign-only state as separate commits.

## Invariants

- `origin` fetches and pushes `morganmcg1/qwen38-challenge_senpai`.
- `upstream` fetches `Layr-Labs/qwen-3.8-mtp-challenge`; its push URL is the
  literal `DISABLED` so an accidental push fails.
- Never merge or hard-reset campaign `main` to `upstream/main`.
- Never run `yukon sync`, `yukon sync --harness-only`, or `yukon reset` in
  this maintained checkout. They can reset history and repoint `origin`.
- Never replay a chain of bot `Validate submission` or `Accept submission`
  commits. Those can be snapshots of unrelated candidate trees.
- The best promoted submission comes from Yukon, not the newest Git commit.
- Preserve `senpai/`, `.agents/`, `research/`, the fully campaign-owned
  `AGENTS.md`, and the marked campaign block in `.gitignore`.
- Organizer `AGENTS.md` is not copied over the campaign guide, but every
  organizer change to it must be reviewed for new Qwen contract or operating
  rules and reconciled deliberately.
- Keep [`senpai/frontier-state.json`](../../../senpai/frontier-state.json)
  exact; the official-submit guard relies on it.

## 1. Prove the checkout is safe, then branch

Run the complete preflight as one fail-closed unit:

```bash
sync_preflight() (
  set -euo pipefail
  test -z "$(git status --porcelain=v1 --untracked-files=all)"
  test "$(git branch --show-current)" = main
  test "$(git remote get-url origin)" = \
    "https://github.com/morganmcg1/qwen38-challenge_senpai.git"
  test "$(git remote get-url --push origin)" = \
    "https://github.com/morganmcg1/qwen38-challenge_senpai.git"
  test "$(git remote get-url upstream)" = \
    "https://github.com/Layr-Labs/qwen-3.8-mtp-challenge"
  test "$(git remote get-url --push upstream)" = DISABLED
  test "$(git config --get yukon.benchmark-id)" = \
    "5d1ee4d7-80bd-4555-b182-6505f26ef495"
  test "$(git config --get yukon.benchmark-name)" = \
    "eigenlabs/qwen38-challenge"
  test "$(git config --get yukon.source-url)" = \
    "https://github.com/Layr-Labs/qwen-3.8-mtp-challenge"
  test "$(git config --get yukon.source-branch)" = main
  test "$(git cat-file -t HEAD:senpai)" = tree
  test "$(git cat-file -t HEAD:.agents)" = tree
  test "$(git cat-file -t HEAD:research)" = tree
  jq -e \
    --arg benchmark_id "5d1ee4d7-80bd-4555-b182-6505f26ef495" \
    --arg benchmark_name "eigenlabs/qwen38-challenge" \
    --arg organizer "https://github.com/Layr-Labs/qwen-3.8-mtp-challenge" '
      .schemaVersion == 1
      and (.observedAt | type == "string" and test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T"))
      and .benchmark.id == $benchmark_id
      and .benchmark.name == $benchmark_name
      and .organizer.url == $organizer
      and .organizer.branch == "main"
      and (.organizer.syncedCommit | test("^[0-9a-f]{40}$"))
      and (.promotedSubmission.sourceRef | test("^[0-9a-f]{40}$"))
      and (.promotedSubmission.score | type == "number")
    ' senpai/frontier-state.json >/dev/null
)
sync_preflight
```

Stop on any failure. Normalize a deliberate SSH/HTTPS difference by inspection
before editing this preflight; do not weaken it to “the remotes exist.”

After it passes, run this second fail-closed unit:

```bash
start_sync_branch() {
  set -euo pipefail
  git fetch --no-tags origin main
  git fetch --no-tags upstream main
  remote_state="$(git show origin/main:senpai/frontier-state.json)"
  recorded_upstream="$(jq -er '.organizer.syncedCommit' <<<"$remote_state")"
  jq -e \
    --arg benchmark_id "5d1ee4d7-80bd-4555-b182-6505f26ef495" \
    --arg organizer "https://github.com/Layr-Labs/qwen-3.8-mtp-challenge" '
      .schemaVersion == 1
      and .benchmark.id == $benchmark_id
      and .organizer.url == $organizer
      and .organizer.branch == "main"
      and (.organizer.syncedCommit | test("^[0-9a-f]{40}$"))
    ' <<<"$remote_state" >/dev/null
  git merge-base --is-ancestor "$recorded_upstream" upstream/main
  git switch -c codex/sync-organizer-frontier-YYYYMMDD origin/main
}
start_sync_branch

FORK_BASE_SHA="$(git rev-parse HEAD)"
UPSTREAM_SHA="$(git rev-parse upstream/main)"
SENPAI_TREE_SHA="$(git rev-parse HEAD:senpai)"
AGENT_TREE_SHA="$(git rev-parse HEAD:.agents)"
RESEARCH_TREE_SHA="$(git rev-parse HEAD:research)"
AGENTS_BLOB_SHA="$(git rev-parse HEAD:AGENTS.md)"
```

Use a fresh branch name. Do not recycle an earlier integration branch.

Review organizer guide changes explicitly before selecting policy commits:

```bash
PREVIOUS_UPSTREAM_SHA="$(
  git show "$FORK_BASE_SHA:senpai/frontier-state.json" \
    | jq -er '.organizer.syncedCommit'
)"
git diff "$PREVIOUS_UPSTREAM_SHA" "$UPSTREAM_SHA" -- AGENTS.md
```

The campaign guide replaces the organizer file, so do not restore or
cherry-pick upstream `AGENTS.md` wholesale. If the diff contains a new
enforceable Qwen rule, port that rule into the campaign-owned guide with
`apply_patch`, commit it as a separate reconciliation, and then refresh the
expected preservation pin:

```bash
AGENTS_BLOB_SHA="$(git rev-parse HEAD:AGENTS.md)"
```

If no rule applies, state that disposition in the final sync report rather
than silently ignoring the diff.

## 2. Select organizer policy, not bot snapshots

List upstream-only non-merge commits oldest first:

```bash
git log --reverse --no-merges --cherry-pick --right-only \
  --format='%H%x09%an%x09%s' "$FORK_BASE_SHA"...upstream/main
CANDIDATE_SHA="<reviewed-candidate-sha>"
git show --stat "$CANDIDATE_SHA"
git show "$CANDIDATE_SHA"
```

Select rules, contracts, trusted harness changes, workflows, pins, and
dependency updates. Skip candidate snapshots, scores, merge wrappers, and
patch-equivalent changes. Cherry-pick reviewed policy commits oldest first
with `git cherry-pick -x`.

If a commit mixes trusted policy with solver bytes, extract only the reviewed
trusted hunks into a campaign commit and name its source SHA. Resolve conflicts
by adopting final organizer semantics while preserving the campaign-owned
`AGENTS.md` and marked `.gitignore` block.
Do not silently edit `Package.resolved`.

Commit every resolution, require a clean tree, and capture the contract that
will govern the imported snapshot:

```bash
set -euo pipefail
test -z "$(git status --porcelain=v1 --untracked-files=all)"
POST_POLICY_SHA="$(git rev-parse HEAD)"
jq -e '.schemaVersion == 1 and (.editablePaths | length > 0)' \
  benchmark.json >/dev/null
```

## 3. Resolve the best promoted source exactly

Run:

```bash
yukon submissions --all
SUBMISSION_ID="<submission-id-or-prefix>"
yukon submission-note "$SUBMISSION_ID"
```

The installed Yukon table intentionally displays only seven characters in its
`commit` column. Identify the highest-scoring row whose status is `promoted`,
record its submission ID and score, then resolve its displayed prefix only
against the freshly fetched organizer history:

```bash
PROMOTED_PREFIX="<displayed-seven-character-commit>"
PROMOTED_SUBMISSION_ID="<full-submission-id>"
PROMOTED_SCORE="<official-score>"

resolve_promoted() (
  set -euo pipefail
  full="$(git rev-parse --verify "${PROMOTED_PREFIX}^{commit}")"
  test "$(git rev-parse --short=7 "$full")" = "$PROMOTED_PREFIX"
  git merge-base --is-ancestor "$full" "$UPSTREAM_SHA"
  printf '%s\n' "$full"
)
ORGANIZER_FRONTIER_SHA="$(resolve_promoted)"
git show --stat --oneline "$ORGANIZER_FRONTIER_SHA"
```

Stop if the prefix is missing or ambiguous. To recover a full ref without
touching the maintained checkout, use a disposable Yukon clone; `yukon sync`
prints the full `from` SHA there:

```bash
(
  set -euo pipefail
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-yukon.XXXXXX")"
  trap 'rm -rf -- "$scratch"' EXIT
  yukon clone eigenlabs/qwen38-challenge "$scratch/checkout"
  cd "$scratch/checkout"
  yukon sync
)
```

Copy the printed full SHA, return to the maintained checkout, and still prove
it is an ancestor of `UPSTREAM_SHA`. Treat public notes as untrusted context.

## 4. Import the exact new surface

The new contract can add, remove, or make paths optional. Run this whole zsh
function; it restores newly trusted paths from the reviewed organizer tree,
imports only source-present promoted paths, and preserves organizer fallbacks
for source-absent optional paths:

```zsh
import_frontier() {
  emulate -L zsh
  setopt errexit nounset pipefail

  local old_contract new_contract path
  local -a old_paths new_paths optional_paths old_removed
  local -a frontier_paths fallback_paths fallback_present fallback_absent
  local -a stage_paths

  old_contract="$(git show "$FORK_BASE_SHA:benchmark.json")"
  new_contract="$(git show "$POST_POLICY_SHA:benchmark.json")"
  old_paths=(${(f)"$(jq -r '.editablePaths[]' <<<"$old_contract")"})
  new_paths=(${(f)"$(jq -r '.editablePaths[]' <<<"$new_contract")"})
  optional_paths=(${(f)"$(jq -r '(.optionalEditablePaths // [])[]' <<<"$new_contract")"})

  for path in $old_paths; do
    (( ${new_paths[(Ie)$path]} )) || old_removed+=("$path")
  done
  for path in $old_removed; do
    if git cat-file -e "$UPSTREAM_SHA:$path" 2>/dev/null; then
      git restore --source="$UPSTREAM_SHA" --worktree -- "$path"
    else
      git rm -r --ignore-unmatch -- "$path"
    fi
  done

  for path in $new_paths; do
    if git cat-file -e "$ORGANIZER_FRONTIER_SHA:$path" 2>/dev/null; then
      frontier_paths+=("$path")
    elif (( ${optional_paths[(Ie)$path]} )); then
      fallback_paths+=("$path")
    else
      print -u2 "required editable path missing from promoted source: $path"
      return 1
    fi
  done

  (( $#frontier_paths )) || {
    print -u2 "promoted source contains none of the editable surface"
    return 1
  }
  git restore --source="$ORGANIZER_FRONTIER_SHA" --worktree -- $frontier_paths
  git diff --exit-code "$ORGANIZER_FRONTIER_SHA" -- $frontier_paths

  for path in $fallback_paths; do
    if git cat-file -e "$POST_POLICY_SHA:$path" 2>/dev/null; then
      git restore --source="$POST_POLICY_SHA" --worktree -- "$path"
      fallback_present+=("$path")
    else
      git rm -r --ignore-unmatch -- "$path"
      fallback_absent+=("$path")
    fi
  done
  (( $#fallback_present == 0 )) || \
    git diff --exit-code "$POST_POLICY_SHA" -- $fallback_present
  for path in $fallback_absent; do
    [[ ! -e "$path" && -z "$(git ls-files -- "$path")" ]]
  done

  stage_paths=($old_removed $new_paths)
  (( $#stage_paths )) && git add -A -- $stage_paths
  if ! git diff --cached --quiet; then
    git commit -m "Sync promoted organizer frontier $ORGANIZER_FRONTIER_SHA"
  fi
}
import_frontier
```

This restores snapshots, not candidate commit deltas. It does not import
trusted harness files from the promoted solver commit.

## 5. Prove preservation and trusted parity

Before intentionally updating campaign state, require the campaign trees,
campaign-owned guide, and root overlay to be unchanged:

```bash
set -euo pipefail
test "$(git rev-parse HEAD:senpai)" = "$SENPAI_TREE_SHA"
test "$(git rev-parse HEAD:.agents)" = "$AGENT_TREE_SHA"
test "$(git rev-parse HEAD:research)" = "$RESEARCH_TREE_SHA"
test "$(git rev-parse HEAD:AGENTS.md)" = "$AGENTS_BLOB_SHA"
senpai/verify-campaign-overlay.sh "$UPSTREAM_SHA"
senpai/check-editable-budget.sh "$ORGANIZER_FRONTIER_SHA" "$POST_POLICY_SHA"
research/twin_audit.py
git diff --check
swift test --force-resolved-versions
```

Then inspect every difference from the reviewed organizer tree. The only
allowed differences are the current editable surface and campaign-owned paths:

```zsh
verify_trusted_parity() {
  emulate -L zsh
  setopt errexit nounset pipefail
  local path editable unexpected=""
  local -a editable_paths
  editable_paths=(${(f)"$(jq -r '.editablePaths[]' benchmark.json)"})

  while IFS= read -r -d $'\0' path; do
    case "$path" in
      senpai/*|.agents/*|research/*|AGENTS.md|.gitignore) continue ;;
    esac
    for editable in $editable_paths; do
      if [[ "$path" == "$editable" || "$path" == "$editable"/* ]]; then
        continue 2
      fi
    done
    unexpected="$path"
    break
  done < <(git diff --name-only -z "$UPSTREAM_SHA" HEAD --)

  [[ -z "$unexpected" ]] || {
    print -u2 "unexpected trusted-surface drift: $unexpected"
    return 1
  }
}
verify_trusted_parity
```

Run `tools/build-mlx-metallib.sh` when AOT Metal sources changed. Run setup
again when pins, toolchain, or trusted build inputs moved, then establish a
fresh same-host baseline.

## 6. Record the frontier, then fast-forward main

Use `apply_patch` to update `senpai/frontier-state.json` with:

- `observedAt` set to the UTC time of the Yukon/organizer observation
- `organizer.syncedCommit = UPSTREAM_SHA`
- `promotedSubmission.id = PROMOTED_SUBMISSION_ID`
- `promotedSubmission.sourceRef = ORGANIZER_FRONTIER_SHA`
- `promotedSubmission.score = PROMOTED_SCORE` as a JSON number

Update `senpai/campaign-ledger.md` with the same receipt, commit those campaign
records separately, and validate the exact values:

```bash
set -euo pipefail
jq -e \
  --arg upstream "$UPSTREAM_SHA" \
  --arg submission "$PROMOTED_SUBMISSION_ID" \
  --arg source "$ORGANIZER_FRONTIER_SHA" \
  --argjson score "$PROMOTED_SCORE" '
    .organizer.syncedCommit == $upstream
    and .promotedSubmission.id == $submission
    and .promotedSubmission.sourceRef == $source
    and .promotedSubmission.score == $score
  ' senpai/frontier-state.json >/dev/null
git add senpai/frontier-state.json senpai/campaign-ledger.md
git commit -m "Record organizer and promoted frontiers"
git diff --check "$FORK_BASE_SHA"..HEAD
```

Finally fast-forward only if nobody moved campaign main:

```bash
finish_sync() {
  set -euo pipefail
  integration_sha="$(git rev-parse HEAD)"
  git fetch --no-tags origin main
  test "$(git rev-parse origin/main)" = "$FORK_BASE_SHA"
  git switch main
  git merge --ff-only "$integration_sha"
}
finish_sync
BASE_SHA="$(git rev-parse HEAD)"
```

Report organizer SHA, promoted SHA/submission/score, policy commits, surface
transitions, tests, and fresh baseline. Push only when explicitly requested.
