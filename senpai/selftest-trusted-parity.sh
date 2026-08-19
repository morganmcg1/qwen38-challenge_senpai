#!/usr/bin/env bash
# Positive validation for senpai/verify-trusted-parity.sh.
#
# A gate that has only ever passed is not a gate, it is a decoration. This
# perturbs each condition the gate claims to detect and requires the gate to
# fail with the right message. It restores the manifest unconditionally.
#
# Usage: senpai/selftest-trusted-parity.sh <UPSTREAM_SHA> [REV]
# Exit:  0 all perturbations detected, 1 a perturbation went undetected, 2 usage

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: senpai/selftest-trusted-parity.sh <UPSTREAM_SHA> [REV]" >&2
  exit 2
fi

UPSTREAM_SHA="$1"
REV="${2:-HEAD}"

cd "$(git rev-parse --show-toplevel)"
GATE=senpai/verify-trusted-parity.sh
MANIFEST=senpai/trusted-overlay-manifest.json
BACKUP="$(mktemp)"
cp "$MANIFEST" "$BACKUP"

# The manifest must come back even if we die, AND a death must never look like a
# pass.
#
# History, stated accurately because the wrong lesson is worse than none: the
# first run of this script aborted at victim selection and looked like success.
# The EXIT trap was NOT the cause -- a plain `trap ... EXIT` preserves the exit
# status here, verified. The status was invisible because the script was invoked
# as `selftest ... | tail -60`, and a pipeline reports the LAST command's status.
# What actually made the abort undetectable was the output: "baseline passes" and
# then nothing, with no verdict line to distinguish "finished clean" from "died".
#
# So the defence is a mandatory verdict, not a clever trap: capture $? first,
# restore the manifest, and refuse to exit 0 unless we reached the end and said
# so out loud. Keeping the status-preserving form as well costs nothing and makes
# the guarantee independent of how the caller invokes us.
selftest_complete=0
cleanup() {
  status=$?
  cp "$BACKUP" "$MANIFEST"
  rm -f "$BACKUP"
  if [ "$selftest_complete" -ne 1 ]; then
    # Always say something. A non-zero status already means failure, but a reader
    # scanning output must never have to infer death from missing lines.
    echo "SELF-TEST ABORTED before completion (status=$status, manifest restored)" >&2
    if [ "$status" -eq 0 ]; then
      status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT

failures=0

# Substring test with no pipeline. `printf ... | grep -qF` is wrong here: grep -q
# exits at the first match, the producer takes SIGPIPE, and `pipefail` then turns
# a SUCCESSFUL search into a non-zero status. That false negative already bit the
# gate once; do not reintroduce it.
contains() {
  case "$1" in
    *"$2"*) return 0 ;;
    *) return 1 ;;
  esac
}

# expect_fail <label> <expected substring>
expect_fail() {
  local label="$1" expect="$2" out rc
  set +e
  out="$("$GATE" "$UPSTREAM_SHA" "$REV" 2>&1)"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "UNDETECTED  $label: gate still passed (rc=0)" >&2
    failures=$((failures + 1))
  elif ! contains "$out" "$expect"; then
    echo "WRONG CAUSE $label: rc=$rc but message lacks \"$expect\"" >&2
    printf '%s\n' "$out" | sed 's/^/            /' >&2
    failures=$((failures + 1))
  else
    echo "detected    $label (rc=$rc, \"$expect\")"
  fi
  cp "$BACKUP" "$MANIFEST"
}

# Baseline must pass, otherwise the perturbations prove nothing.
if ! "$GATE" "$UPSTREAM_SHA" "$REV" >/dev/null 2>&1; then
  echo "BASELINE FAILS: fix real drift before trusting this self-test" >&2
  "$GATE" "$UPSTREAM_SHA" "$REV" || true
  exit 1
fi
echo "baseline    passes"

# Pick a declared overlay that is actually present at REV, so the perturbation
# exercises a live path rather than a manifest entry for another branch.
#
# Deliberately not `... | while read; done | head -1`: `head -1` closes the pipe
# as soon as it has its line, the loop's slow `git` calls take SIGPIPE, and
# `pipefail` aborts the whole script. Iterate in this shell so `break` works and
# nothing can be killed by a downstream reader.
victim=""
overlay_paths="$(jq -r '.overlays[].path' "$MANIFEST")"
saved_ifs="$IFS"
IFS='
'
for p in $overlay_paths; do
  IFS="$saved_ifs"
  if git rev-parse --verify --quiet "$REV:$p" >/dev/null 2>&1 \
    && ! git diff --quiet "$UPSTREAM_SHA" "$REV" -- "$p"; then
    victim="$p"
    break
  fi
  IFS='
'
done
IFS="$saved_ifs"
[ -n "$victim" ] || { echo "no live declared overlay at $REV to perturb" >&2; exit 1; }
echo "victim      $victim"

# 1. Undeclared drift: drop the entry entirely.
jq --arg p "$victim" '.overlays |= map(select(.path != $p))' "$BACKUP" > "$MANIFEST"
expect_fail "undeclared drift" "UNDECLARED trusted-surface drift: $victim"

# 2. Organizer moved a path we patched: corrupt the pinned blob. Only
#    meaningful for kinds that pin one.
kind="$(jq -r --arg p "$victim" '.overlays[] | select(.path == $p) | .kind' "$BACKUP")"
if [ "$kind" = "repair" ] || [ "$kind" = "seam" ]; then
  jq --arg p "$victim" \
    '(.overlays[] | select(.path == $p) | .organizerBlob) = "0000000000000000000000000000000000000000"' \
    "$BACKUP" > "$MANIFEST"
  expect_fail "organizer blob moved" "organizer CHANGED a path the campaign patched"
fi

# 3. Obligation broken. This is tested against whichever overlay actually
#    declares mustContain, NOT against $victim: the victim is whatever sorted
#    first, and when it had no obligation this check silently skipped. The
#    behavioural guard on the trace seam is the single most important thing in
#    the manifest and the one that already produced a false negative, so it does
#    not get to be untested because of selection order.
must_path=""
saved_ifs="$IFS"
IFS='
'
for p in $overlay_paths; do
  IFS="$saved_ifs"
  if [ "$(jq -r --arg p "$p" '.overlays[] | select(.path == $p) | .mustContain // ""' "$BACKUP")" != "" ] \
    && git rev-parse --verify --quiet "$REV:$p" >/dev/null 2>&1; then
    must_path="$p"
    break
  fi
  IFS='
'
done
IFS="$saved_ifs"

if [ -n "$must_path" ]; then
  echo "obligation  $must_path"
  jq --arg p "$must_path" \
    '(.overlays[] | select(.path == $p) | .mustContain) = "SENTINEL_THAT_MUST_NOT_EXIST_9f3a"' \
    "$BACKUP" > "$MANIFEST"
  expect_fail "obligation broken" "obligation BROKEN"
else
  echo "NOTE        no live overlay declares mustContain; obligation check untested" >&2
  failures=$((failures + 1))
fi

# 4. Unknown kind must not be silently tolerated.
jq --arg p "$victim" '(.overlays[] | select(.path == $p) | .kind) = "bogus"' \
  "$BACKUP" > "$MANIFEST"
expect_fail "unknown overlay kind" "unknown overlay kind"

# 5. A campaign-owned allow-list entry must not launder a trusted path.
# The victim is DERIVED. This case used to name Sources/MLXFastCLI/main.swift
# literally; the rebase made that file organizer-identical, so the perturbation
# silently stopped perturbing anything and printed a NOTE that read like an
# insight. If no trusted (non-editable) path under Sources/ differs today, say
# so plainly instead of naming a file that proves nothing.
# The victim is taken from the GATE'S OWN baseline report, not recomputed from
# benchmark.json. editablePaths entries are directory PREFIXES and the gate
# matches them as such; a first attempt here used exact membership and nominated
# an editable file, so the perturbation "failed" for a reason that was in the
# self-test rather than in the gate. Whatever the gate reports is, by definition,
# a path it considers trusted and differing.
set +e
baseline_report="$("$GATE" "$UPSTREAM_SHA" "$REV" 2>&1)"
set -e
launder_victim="$(printf '%s\n' "$baseline_report" \
  | sed -n 's|.*[ :]\(Sources/[^ ][^ ]*\)$|\1|p' | head -1)"
if [ -z "$launder_victim" ]; then
  echo "NOTE        no trusted Sources/ path differs at $REV, so allow-list"
  echo "            laundering is NOT EXERCISABLE in this tree. Recorded as"
  echo "            untested rather than reported as passing."
else
  jq '.campaignOwnedPrefixes += ["Sources/"]' "$BACKUP" > "$MANIFEST"
  set +e
  laundered="$("$GATE" "$UPSTREAM_SHA" "$REV" 2>&1)"
  set -e
  cp "$BACKUP" "$MANIFEST"
  if contains "$laundered" "$launder_victim"; then
    echo "detected    allow-list laundering is visible ($launder_victim still reported)"
  else
    echo "UNDETECTED  widening campaignOwnedPrefixes to Sources/ hid $launder_victim" >&2
    failures=$((failures + 1))
  fi
fi

# 6. A declaration for a path that NO LONGER DIFFERS must be refused.
# This is the case that was missing entirely, and its absence hid a real event:
# the rebase made Sources/MLXFastCLI/main.swift organizer-identical while the
# manifest still declared a seam in it, and the gate's only symptom was the
# declared-overlay count falling from 3 to 2. Adopted-by-frontier and
# reverted-by-rebase are indistinguishable in a diff and demand opposite
# responses, so the gate must refuse to stay quiet.
# The victim is any path that is IDENTICAL in both trees -- derived, so it cannot
# rot into a file that has started to differ. Editability is deliberately NOT
# filtered: the presence check is editability-independent by design, because a
# declaration has to be true about whatever it names.
same_victim=""
alldiff="$(git diff --name-only "$UPSTREAM_SHA" "$REV" --)"
while IFS= read -r p; do
  [ -n "$p" ] || continue
  case $'\n'"$alldiff"$'\n' in *$'\n'"$p"$'\n'*) continue ;; esac
  same_victim="$p"
  break
done <<EOF
$(git ls-tree -r --name-only "$UPSTREAM_SHA" -- Sources Tests)
EOF
if [ -z "$same_victim" ]; then
  echo "UNDETECTED  could not derive an identical trusted path to declare" >&2
  failures=$((failures + 1))
else
  echo "stale-decl  $same_victim (identical in both trees)"
  jq --arg p "$same_victim" \
    '.overlays += [{"path": $p, "kind": "seam", "organizerBlob": "deadbeef",
                    "why": "self-test perturbation: declares an overlay that does not exist"}]' \
    "$BACKUP" > "$MANIFEST"
  expect_fail "declaration for a path that no longer differs" \
    "declared overlay no longer differs from the organizer: $same_victim"
fi

selftest_complete=1
if [ "$failures" -ne 0 ]; then
  echo "SELF-TEST FAILED: $failures perturbation(s) undetected" >&2
  exit 1
fi
echo "SELF-TEST PASS: every perturbation was detected with the right cause"
