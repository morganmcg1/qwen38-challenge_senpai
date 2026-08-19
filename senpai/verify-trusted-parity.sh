#!/usr/bin/env bash
# Fail if the campaign differs from the organizer frontier anywhere that is
# neither an editablePaths entry nor a DECLARED overlay.
#
# Why this exists. The generic sync check allows a broad prefix list
# (senpai/ .agents/ research/ AGENTS.md .gitignore). That is fine for campaign
# scaffolding, but this campaign also carries three differences inside the
# organizer's own Sources/ and Tests/ trees. Widening the allow-list to
# `Tests/*` or `Sources/*` would let a future frontier sync hide real trusted
# drift behind a rule that was written for scaffolding. So every such path is
# declared individually in senpai/trusted-overlay-manifest.json, with the
# obligation that keeps it honest, and anything undeclared is a hard failure.
#
# Usage:
#   senpai/verify-trusted-parity.sh <UPSTREAM_SHA> [REV]
#
#   UPSTREAM_SHA  organizer frontier commit to compare against
#   REV           campaign revision to check (default HEAD)
#
# Exit codes:
#   0  parity OK
#   1  undeclared drift, or a declared overlay whose obligation broke
#   2  usage / environment error

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: senpai/verify-trusted-parity.sh <UPSTREAM_SHA> [REV]" >&2
  exit 2
fi

UPSTREAM_SHA="$1"
REV="${2:-HEAD}"

command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 2; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"

MANIFEST="senpai/trusted-overlay-manifest.json"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST" >&2; exit 2; }
[ -f benchmark.json ] || { echo "missing benchmark.json" >&2; exit 2; }

git cat-file -e "${UPSTREAM_SHA}^{commit}" 2>/dev/null || {
  echo "unknown UPSTREAM_SHA: $UPSTREAM_SHA" >&2; exit 2; }
git cat-file -e "${REV}^{commit}" 2>/dev/null || {
  echo "unknown REV: $REV" >&2; exit 2; }

blob_at() {  # blob_at <rev> <path> -> full sha, or ABSENT
  git rev-parse --verify --quiet "$1:$2" || echo ABSENT
}

is_editable() {  # is_editable <path>
  local candidate="$1" entry
  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    entry="${entry%/}"
    if [ "$candidate" = "$entry" ] || [ "${candidate#"$entry"/}" != "$candidate" ]; then
      return 0
    fi
  done <<EOF
$(jq -r '(.editablePaths // [])[], (.optionalEditablePaths // [])[]' benchmark.json)
EOF
  return 1
}

is_campaign_owned() {  # is_campaign_owned <path>
  local candidate="$1" entry
  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    if [ "${candidate#"$entry"}" != "$candidate" ]; then return 0; fi
  done <<EOF
$(jq -r '(.campaignOwnedPrefixes // [])[]' "$MANIFEST")
EOF
  while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    if [ "$candidate" = "$entry" ]; then return 0; fi
  done <<EOF
$(jq -r '(.campaignOwnedPaths // [])[]' "$MANIFEST")
EOF
  return 1
}

failures=0
declared=0

echo "trusted parity: $REV ($(git rev-parse --short "$REV")) vs organizer $(git rev-parse --short "$UPSTREAM_SHA")"

while IFS= read -r path; do
  [ -n "$path" ] || continue
  is_editable "$path" && continue
  is_campaign_owned "$path" && continue

  entry="$(jq -c --arg p "$path" '.overlays[] | select(.path == $p)' "$MANIFEST")"
  if [ -z "$entry" ]; then
    echo "  FAIL  UNDECLARED trusted-surface drift: $path" >&2
    echo "        organizer=$(blob_at "$UPSTREAM_SHA" "$path") campaign=$(blob_at "$REV" "$path")" >&2
    failures=$((failures + 1))
    continue
  fi

  declared=$((declared + 1))
  kind="$(printf '%s' "$entry" | jq -r '.kind')"
  organizer_now="$(blob_at "$UPSTREAM_SHA" "$path")"
  campaign_now="$(blob_at "$REV" "$path")"

  case "$kind" in
    added)
      if [ "$organizer_now" != "ABSENT" ]; then
        echo "  FAIL  [added] declared campaign-added path now EXISTS upstream, reconcile: $path" >&2
        echo "        organizer=$organizer_now" >&2
        failures=$((failures + 1))
      else
        echo "  ok    [added]  absent upstream, campaign=$campaign_now: $path"
      fi
      ;;
    repair|seam)
      pinned="$(printf '%s' "$entry" | jq -r '.organizerBlob // ""')"
      if [ -z "$pinned" ]; then
        echo "  FAIL  [$kind] manifest entry lacks organizerBlob: $path" >&2
        failures=$((failures + 1))
      elif [ "$organizer_now" != "$pinned" ]; then
        echo "  FAIL  [$kind] organizer CHANGED a path the campaign patched, re-review: $path" >&2
        echo "        pinned=$pinned organizer_now=$organizer_now" >&2
        failures=$((failures + 1))
      else
        echo "  ok    [$kind] organizer blob stable $organizer_now, campaign=$campaign_now: $path"
      fi
      must="$(printf '%s' "$entry" | jq -r '.mustContain // ""')"
      if [ -n "$must" ]; then
        # Deliberately NOT `git show ... | grep -q`: grep -q closes the pipe on
        # the first match, git show takes SIGPIPE, and `set -o pipefail` then
        # reports the successful search as a failure. Match in-shell instead.
        content="$(git show "$REV:$path")"
        case "$content" in
          *"$must"*)
            echo "        obligation ok: contains \"$must\""
            ;;
          *)
            echo "  FAIL  [$kind] obligation BROKEN, \"$must\" missing from $path" >&2
            printf '        %s\n' "$(printf '%s' "$entry" | jq -r '.obligation // ""')" >&2
            failures=$((failures + 1))
            ;;
        esac
      fi
      ;;
    *)
      echo "  FAIL  unknown overlay kind \"$kind\" for $path" >&2
      failures=$((failures + 1))
      ;;
  esac
done <<EOF
$(git diff --name-only "$UPSTREAM_SHA" "$REV" --)
EOF

# --- a declaration must still describe something real ------------------------
# The loop above walks the DIFF, so an overlay that has STOPPED differing is
# never visited and never checked. The manifest goes on asserting an overlay
# that does not exist and the gate goes on passing -- the stale-acknowledgement
# failure this campaign has now hit in three separate instruments (ledger 162).
# It already hid a real event: the rebase made Sources/MLXFastCLI/main.swift
# byte-identical to the organizer while the manifest still declared a seam in
# it, and the only visible trace was the declared-overlay COUNT dropping from 3
# to 2 in a line nobody reads. A declaration that cannot fail is not a
# declaration.
#
# Branch scoping is honoured: an overlay marked `branches` is only required to
# be present in revisions that actually CONTAIN that branch tip. The test is
# "branch tip is an ancestor of REV", not the reverse -- the reverse would be
# true for the campaign base and would demand branch-only overlays on main.
diff_paths="$(git diff --name-only "$UPSTREAM_SHA" "$REV" --)"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  case $'\n'"$diff_paths"$'\n' in
    *$'\n'"$path"$'\n'*) continue ;;
  esac

  entry="$(jq -c --arg p "$path" '.overlays[] | select(.path == $p)' "$MANIFEST")"
  in_scope=1
  branch_list="$(printf '%s' "$entry" | jq -r '(.branches // [])[]')"
  if [ -n "$branch_list" ]; then
    in_scope=0
    while IFS= read -r branch; do
      [ -n "$branch" ] || continue
      tip="$(git rev-parse --verify --quiet "refs/heads/$branch" \
        || git rev-parse --verify --quiet "refs/remotes/origin/$branch")" || tip=""
      if [ -z "$tip" ]; then
        echo "  FAIL  declared overlay names an unresolvable branch \"$branch\": $path" >&2
        echo "        Refusing to skip a presence check on a branch that cannot be read." >&2
        failures=$((failures + 1))
        in_scope=0
        break
      fi
      if git merge-base --is-ancestor "$tip" "$REV"; then in_scope=1; break; fi
    done <<INNER
$branch_list
INNER
  fi

  if [ "$in_scope" -eq 1 ]; then
    echo "  FAIL  declared overlay no longer differs from the organizer: $path" >&2
    echo "        The manifest still describes an overlay here. Either the frontier" >&2
    echo "        ADOPTED it or a rebase REVERTED it, and those need opposite" >&2
    echo "        responses. Delete the entry recording which, or restore the" >&2
    echo "        overlay; do not leave a declaration that cannot fail." >&2
    failures=$((failures + 1))
  else
    echo "  ok    [scope] overlay not required in this revision: $path"
  fi
done <<EOF
$(jq -r '.overlays[].path' "$MANIFEST")
EOF

if [ "$failures" -ne 0 ]; then
  echo "trusted parity FAILED: $failures problem(s), $declared declared overlay(s)" >&2
  exit 1
fi

echo "trusted parity OK: $declared declared overlay(s), 0 undeclared drift"
