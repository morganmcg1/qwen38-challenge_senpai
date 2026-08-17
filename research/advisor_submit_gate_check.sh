#!/usr/bin/env bash
# Reproduce senpai/submit-official.sh's protected-path gate without submitting.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

MAIN_SHA="$(git rev-parse origin/main)"
BASE_SHA="${1:-HEAD}"
BASE_SHA="$(git rev-parse "${BASE_SHA}^{commit}")"

protected=(benchmark.json)
while IFS= read -r p; do protected+=("$p"); done < <(jq -r '.editablePaths[]' benchmark.json)

echo "main_sha = ${MAIN_SHA}"
echo "base_sha = ${BASE_SHA}"
echo "protected path count = ${#protected[@]}"
echo
echo "--- gate 1: base is ancestor of origin/main ---"
if git merge-base --is-ancestor "${BASE_SHA}" "${MAIN_SHA}"; then
  echo "PASS"
else
  echo "FAIL: base is NOT an ancestor of origin/main"
fi
echo
echo "--- gate 2: protected-path diff main..base must be EMPTY ---"
if git diff --quiet "${MAIN_SHA}" "${BASE_SHA}" -- "${protected[@]}"; then
  echo "PASS (identical editable surface)"
else
  echo "FAIL: differing protected paths:"
  git diff --name-status "${MAIN_SHA}" "${BASE_SHA}" -- "${protected[@]}"
  echo
  echo "--- per-file line stats ---"
  git diff --stat "${MAIN_SHA}" "${BASE_SHA}" -- "${protected[@]}"
fi
echo
echo "--- gate 3: remotes required by the wrapper ---"
for r in origin upstream; do
  if git remote get-url "$r" >/dev/null 2>&1; then
    echo "remote $r: $(git remote get-url "$r")"
  else
    echo "remote $r: MISSING"
  fi
done
echo
echo "--- gate 4: yukon git config pins ---"
for k in yukon.benchmark-id yukon.benchmark-name yukon.source-url yukon.source-branch; do
  v="$(git config --get "$k" || true)"
  echo "$k = ${v:-<unset>}"
done
