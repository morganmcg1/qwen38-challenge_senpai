#!/usr/bin/env bash
# E26 content gate (PR #33, item v).
#
# The fixed-window continuation fix has now been silently discarded by a
# content-mirror sync five separate times. Every one of those regressions was
# invisible to branch-level bookkeeping, because the branch name and the merge
# commit both survived while the file content reverted. This gate inspects the
# CONTENT of Qwen36MTPBlockSession.swift at an arbitrary revision, so it cannot
# be fooled by a merge that landed and was later overwritten.
#
#   research/e26-content-gate.sh                       # HEAD
#   research/e26-content-gate.sh HEAD origin/main upstream/main
#   research/e26-content-gate.sh 0f41bbf904d09c28e93736217fd90729ba0636e7
#   research/e26-content-gate.sh /tmp/candidate-session.swift
#
# An argument is a ref (resolved as <ref>:<path>), a raw blob SHA, or a file.
# Blob SHAs work even when the ref that once carried them is not fetched
# locally, which is the normal situation in a student checkout; a file path
# covers an unmerged sync candidate and the working tree itself.
#
# Exit 0 when every inspected revision passes, 1 when any fails, 2 on a usage
# or resolution error.

set -uo pipefail

PATH_IN_TREE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

# Git lookups need the repo root, but a file argument is the caller's, so keep
# their working directory to resolve relative paths against.
ORIGINAL_PWD="$PWD"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

# Delete ALL whitespace and drop `self.`, collapsing the source to one line, so
# that a reformat cannot hide the defect. An earlier revision only squeezed
# whitespace RUNS and was demonstrably fooled by `stopTokens .contains( primary
# )`; deleting outright also matches a marker that a formatter has wrapped
# across lines.
normalize() {
    tr -d '[:space:]' | sed -e 's/self\.//g'
}

# Occurrence count, not line count: the normalized source is a single line.
count() {
    grep -o -F -- "$1" "$2" | wc -l | tr -d ' '
}

# Sets RESOLVED_ID (a blob SHA, always) and RESOLVED_CAT (a command that emits
# the raw source on stdout). Hashing a plain file keeps the BLOB column a real
# content identity, which is the whole point of this gate.
resolve_source() {
    local arg="$1" blob
    if blob="$(git rev-parse --verify --quiet "${arg}:${PATH_IN_TREE}")"; then
        RESOLVED_ID="$blob"
        RESOLVED_CAT=(git cat-file -p "$blob")
        return 0
    fi
    if [ "$(git cat-file -t "$arg" 2>/dev/null)" = "blob" ]; then
        blob="$(git rev-parse --verify "$arg")"
        RESOLVED_ID="$blob"
        RESOLVED_CAT=(git cat-file -p "$blob")
        return 0
    fi
    local candidate
    for candidate in "$arg" "${ORIGINAL_PWD}/${arg}"; do
        if [ -f "$candidate" ]; then
            RESOLVED_ID="$(git hash-object -- "$candidate")"
            RESOLVED_CAT=(cat -- "$candidate")
            return 0
        fi
    done
    return 1
}

refs=("$@")
[ "${#refs[@]}" -eq 0 ] && refs=(HEAD)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

printf '%-24s %-12s %6s %7s %7s %7s %7s %7s  %s\n' \
    REF BLOB LINES 'stop(p)' reached truncate 'draft()' pending VERDICT

failures=0
for ref in "${refs[@]}"; do
    if ! resolve_source "$ref"; then
        printf '%-24s %s\n' "$ref" \
            "UNRESOLVED (no ${PATH_IN_TREE}, not a blob, not a file)"
        exit 2
    fi
    blob="$RESOLVED_ID"

    raw="${work}/${blob}.raw"
    src="${work}/${blob}.norm"
    if [ ! -f "$raw" ]; then
        "${RESOLVED_CAT[@]}" >"$raw"
        normalize <"$raw" >"$src"
    fi
    lines="$(wc -l <"$raw" | tr -d ' ')"

    # Defect markers: the early return that ends the window at a stop token,
    # the flag it set, and the truncation of the committed-token list.
    stop_primary="$(count 'stopTokens.contains(primary)' "$src")"
    reached="$(count 'reachedStopToken' "$src")"
    truncate="$(count 'firstIndex(where:{stopTokens.contains(' "$src")"

    # Fix markers: drafting is still truncated at a stop token (legitimate and
    # required), and the primary is still carried across rounds.
    draft_stop="$(count 'stopTokens.contains(drafts[' "$src")"
    pending="$(count 'pendingPrimary' "$src")"

    verdict=PASS
    [ "$stop_primary" -eq 0 ] || verdict=FAIL
    [ "$reached" -eq 0 ] || verdict=FAIL
    [ "$truncate" -eq 0 ] || verdict=FAIL
    [ "$draft_stop" -ge 1 ] || verdict=FAIL
    [ "$pending" -ge 1 ] || verdict=FAIL
    [ "$verdict" = PASS ] || failures=$((failures + 1))

    printf '%-24s %-12s %6s %7s %7s %7s %7s %7s  %s\n' \
        "$ref" "${blob:0:12}" "$lines" \
        "$stop_primary" "$reached" "$truncate" "$draft_stop" "$pending" "$verdict"
done

if [ "$failures" -ne 0 ]; then
    cat >&2 <<'EOF'

E26 / PR #33: a revision above still ends the fixed decode window at a stop
token. The trusted parent owns the window length and continues the serial
trajectory past EOS; a session that stops early emits a short row ledger and
then throws `notBegun`. Columns:

  stop(p)   `if stopTokens.contains(primary)` -- the early return.  Must be 0.
  reached   the `reachedStopToken` flag it sets and reports. Must be 0.
  truncate  `firstIndex(where: { stopTokens.contains(` on the committed list.
            Must be 0. (Whitespace is deleted before matching.)
  draft()   `stopTokens.contains(drafts[` -- truncating the PROPOSAL at a stop
            token is correct and must stay. Must be >= 1.
  pending   `pendingPrimary`, the carry that spans rounds. Must be >= 1.
EOF
    exit 1
fi
exit 0
