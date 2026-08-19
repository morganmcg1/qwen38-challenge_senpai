#!/usr/bin/env bash
# Cheap validation after reverting the E58 instrument. The submitted surface is
# byte-identical to the base, so this only has to prove the tree still builds and
# that the research-only storm test compiles without the deleted census source.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== swift build -c release ==="
swift build -c release --force-resolved-versions || exit 1

echo "=== swift test ==="
swift test --force-resolved-versions || exit 1

echo "=== validation complete ==="
