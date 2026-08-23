#!/usr/bin/env bash
# RULE 162 gate. Enumerate Metal kernel names and top-level Swift symbols that
# the organizer's promoted source ships and our candidate does not.
#
# A missing promoted mechanism is free score and it is invisible to a self-diff.
# FINDING 283 (ledger 319): this check would have found +1.075 % of a +1.2578 %
# gap at any point in the campaign, with no GPU time.
#
#   senpai/diff-promoted-surface.sh [OURS_REF] [PROMOTED_REF]
#
# Defaults: OURS_REF=HEAD, PROMOTED_REF=upstream/main.
# Exit 1 when the promoted source ships a kernel name we do not.

set -uo pipefail

OURS="${1:-HEAD}"
PROMOTED="${2:-upstream/main}"

cd "$(git rev-parse --show-toplevel)" || exit 2

if ! git rev-parse --verify --quiet "$PROMOTED^{commit}" >/dev/null; then
    echo "FAIL: cannot resolve promoted ref '$PROMOTED'." >&2
    echo "      Run 'git fetch upstream' first." >&2
    exit 2
fi

OURS_SHA=$(git rev-parse "$OURS")
PROMOTED_SHA=$(git rev-parse "$PROMOTED")

echo "RULE 162 promoted-surface diff"
echo "  ours     $OURS_SHA  ($OURS)"
echo "  promoted $PROMOTED_SHA  ($PROMOTED)"
echo

# ---------------------------------------------------------------- lineage ----
BEHIND=$(git rev-list --count "$OURS..$PROMOTED")
MERGE_BASE=$(git merge-base "$OURS" "$PROMOTED")
echo "== lineage =="
echo "  merge-base            $MERGE_BASE"
echo "  organizer commits we are behind   $BEHIND"
if git merge-base --is-ancestor "$PROMOTED_SHA" "$OURS_SHA"; then
    echo "  promoted is an ancestor of ours   YES"
else
    echo "  promoted is an ancestor of ours   NO   <-- stale-base overlay risk"
fi
echo

# ---------------------------------------------- editable path enumeration ----
# bash 3.2 on macOS has no `mapfile`, so the path list goes through a file.
EDITABLE_FILE=$(mktemp -t rule162-editable)
trap 'rm -f "$EDITABLE_FILE"' EXIT

python3 "$(git rev-parse --show-toplevel)/senpai/list-editable-paths.py" \
    "$PROMOTED_SHA" > "$EDITABLE_FILE"
PY_STATUS=$?

EDITABLE_COUNT=$(grep -c . "$EDITABLE_FILE" 2>/dev/null || echo 0)
if [ "$PY_STATUS" -ne 0 ] || [ "$EDITABLE_COUNT" -eq 0 ]; then
    echo "FAIL: no editablePaths recovered from benchmark.json at $PROMOTED_SHA." >&2
    echo "      Refusing to report PASS on an empty scope." >&2
    exit 2
fi

echo "== scope =="
echo "  editable paths from promoted benchmark.json   $EDITABLE_COUNT"
echo

# --------------------------------------------------------- kernel names ------
kernel_names() {
    ref="$1"; shift
    git grep -h -oE 'name:[[:space:]]*"[a-z0-9_]+"' "$ref" -- "$@" 2>/dev/null \
        | sed -E 's/.*"([a-z0-9_]+)".*/\1/' | sort -u
}

kernel_names "$OURS_SHA"     $(cat "$EDITABLE_FILE") > /tmp/rule162-ours-kernels.txt
kernel_names "$PROMOTED_SHA" $(cat "$EDITABLE_FILE") > /tmp/rule162-promoted-kernels.txt

if [ ! -s /tmp/rule162-promoted-kernels.txt ]; then
    echo "FAIL: no kernel names found in the promoted surface." >&2
    echo "      The scan matched nothing, so a PASS would be meaningless." >&2
    exit 2
fi

MISSING=$(comm -13 /tmp/rule162-ours-kernels.txt /tmp/rule162-promoted-kernels.txt)
EXTRA=$(comm -23 /tmp/rule162-ours-kernels.txt /tmp/rule162-promoted-kernels.txt)

echo "== Metal kernel names =="
echo "  ours     $(wc -l < /tmp/rule162-ours-kernels.txt | tr -d ' ')"
echo "  promoted $(wc -l < /tmp/rule162-promoted-kernels.txt | tr -d ' ')"
echo

STATUS=0
if [ -n "$MISSING" ]; then
    STATUS=1
    echo "  PROMOTED SHIPS, WE DO NOT  <-- classify each as deletion or rename"
    echo "$MISSING" | sed 's/^/      /'
else
    echo "  PROMOTED SHIPS, WE DO NOT  (none)"
fi
echo
if [ -n "$EXTRA" ]; then
    echo "  OURS ONLY  (genuine divergence, must be independently measured)"
    echo "$EXTRA" | sed 's/^/      /'
else
    echo "  OURS ONLY  (none)"
fi
echo

# --------------------------------------------------- top-level Swift decls ---
swift_decls() {
    ref="$1"; shift
    git grep -h -oE \
        '^[[:space:]]*(public |private |internal |fileprivate |static |final )*(let|var|func|enum|struct|class|extension) [A-Za-z_][A-Za-z0-9_]*' \
        "$ref" -- "$@" 2>/dev/null \
        | awk '{print $NF}' | sort -u
}

swift_decls "$OURS_SHA"     $(cat "$EDITABLE_FILE") > /tmp/rule162-ours-decls.txt
swift_decls "$PROMOTED_SHA" $(cat "$EDITABLE_FILE") > /tmp/rule162-promoted-decls.txt
DECL_MISSING=$(comm -13 /tmp/rule162-ours-decls.txt /tmp/rule162-promoted-decls.txt)

echo "== Swift declarations promoted ships and we do not =="
if [ -n "$DECL_MISSING" ]; then
    echo "$DECL_MISSING" | sed 's/^/      /'
else
    echo "      (none)"
fi
echo

# ------------------------------------------------------------- verdict -------
if [ "$STATUS" -ne 0 ]; then
    cat <<'EOF'
VERDICT: FAIL

The promoted source ships Metal kernels our candidate does not. Classify every
name above before pricing any gap to the bar:

  deletion   a stale-base overlay dropped a promoted mechanism. Free score.
             Re-found on the promoted source or re-land the mechanism.
  rename     the same kernel under a different name. Prove it from the source
             body, not from the name, and record the mapping.

Do not assign work to close a gap until this list is empty or fully classified.
EOF
else
    echo "VERDICT: PASS  (no promoted kernel name is missing from our surface)"
fi

exit "$STATUS"
