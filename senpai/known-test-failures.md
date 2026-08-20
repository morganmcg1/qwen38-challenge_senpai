# Known `swift test` failures on the campaign base

This file records the `swift test --force-resolved-versions` failures that the
campaign base carries on purpose. It exists so a student does not stop a
submission for a failure that is already understood, and so a genuinely new
failure stands out immediately.

This is a research-only record. It changes no candidate byte and no submitted
path.

## How to use this file

Run the suite and compare:

```bash
swift test --force-resolved-versions
```

The gate is the **failing name set** and the **issue count**, and nothing else:

1. every failing test function name is a member of the nine names listed below;
   and
2. the total issue count is **40 or fewer**.

Treat the result as a **stop** when a test outside this list fails, or when the
issue count exceeds 40. Either condition means the change under test broke
something new.

**The bare exit code is never the gate.** The run exits 1 whenever any of the
nine fails, so exit 1 carries no information on its own. Read the name set and
the issue count out of the run.

**The test count and the suite count are never the gate either.** They move with
whatever tests a branch adds, so two students on the same base report different
totals. On `f7f356b2` this branch reports 705 tests in 53 suites and other
branches report 710 tests. Every one of them reports the same 9 names and the
same 40 issues, and that is the gate.

Do **not** edit the organizer's tests to make them green. These failures are
evidence about the base, not defects to hide.

## Measured at

- commit: `222f23325d0645375dc731eae4ecf1119fbef0fe` (advisor branch tip,
  `senpai/qwen38-mtp-r1`)
- host: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 26.5.2, Swift 6.3.3
- totals: 703 tests across 52 suites, **9 failing test functions**, **40 issues**
- independent corroboration: the 06:18 UTC job log for the same base also
  records `swift_test exit=1 failing=9` with 40 issues.

Re-measured at `f7f356b2834518ced918f3049ca1b88afb6003f3`, the base that adopts
organizer commit `8b54ff11`, on the same host at 2026-08-20T20:26Z: 705 tests
across 53 suites, the **same 9 names** and the **same 40 issues**. The organizer
sync introduced no new failure.

That measurement also covers advisor head
`07c75a708c2347021d3148d7bc87b246ba2aec73`. `07c75a70` adds only record-only
Markdown over `f7f356b2`, so the `Sources/`, `Vendor/`, `Tests/` and
`Package.swift` trees are byte-identical at both commits.

## The nine failing tests

| test function | issues | failing input | cause |
|---|---:|---|---|
| `theCheckedInDeclarationSelectsThePinnedHead` | 6 | `mtp-head.manifest.json` | asserts the organizer-pinned head; this tree declares the remote q2/q4 head |
| `startupMemoryPolicyKeepsRanked128GiBProfile` | 2 | `RuntimeStartupMemoryPolicy.swift:149-150` | source holds `512`/`50`; the test expects `320`/`128` |
| `qwen36ConfigContractDigestMatchesTheReferenceManifest` | 2 | reference manifest | organizer placeholder that asserts `entry == nil` |
| `theEvenMedianRuleIsTheMeanOfTheTwoCentralValues` | 3 | track fixture | expects the noop median `0.9438`; the armed fixture gives `1.0` |
| `theSeededCalibrationExpectationMatchesItsRecordedProvenance` | 2 | track fixture | expects `seeded`/`pending`; the fixture says `measured_qwen38_cutover_2026_08_14` |
| `theQwenMTPTrackIsArmedOnQwen38` | 11 | track fixture | expects `QWEN38-PENDING-RELEASE`; the fixture holds real digests |
| `contestantDocsCommandBlocksKeepTheDependencyGraphFrozen` | 1 | `AGENTS.md` | the campaign overlay dropped `git checkout -- Package.resolved` |
| `participantDocsExposeDefaultCLIInstallDirectory` | 2 | `AGENTS.md`, `CLAUDE.md` | the campaign overlay dropped ``Yukon CLI (`yukon`)`` |
| `submissionStaticReviewPromptCoversMeasurementStructureExploitation` | 11 | `AGENTS.md`, `CLAUDE.md` | the campaign overlay dropped the serial-track rule text |
| **total** | **40** | | |

## The seven failing input files

1. `mtp-head.manifest.json`
2. `Sources/MLXFastHarness/RuntimeStartupMemoryPolicy.swift`
3. the Qwen 3.8 MTP reference manifest
4. `fixtures/qwen3_8_27b_mtp_track.json`
5. `AGENTS.md`
6. `CLAUDE.md`
7. the submission static-review prompt text

`CLAUDE.md` is a symlink to `AGENTS.md`, so a single overlay edit is counted
twice by the documentation tests.

## The two classes

**Class A - organizer test staleness (26 issues).** The organizer's tests still
encode the pre-release placeholder state of the track: an unarmed fixture, a
pending calibration provenance, the pinned head, and the earlier startup memory
constants. The live fixture and manifest have since been armed with real
digests. The tests were not updated with them. Nothing in this class reads
campaign code.

**Class B - campaign `AGENTS.md` overlay (14 issues).** Three documentation
tests assert that specific sentences appear in `AGENTS.md`. The campaign overlay
rewrote that file and dropped three of those sentences. The tests are correct
about the text being absent; the omission is in the campaign overlay, not in the
organizer's contract.

**Neither class reads either of the two submitted Swift files.** Zero of the 40
issues touch `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` or
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. The suite therefore
gives no evidence against the candidate.

## Two specific cautions

**`qwen36ConfigContractDigestMatchesTheReferenceManifest` is not coverage.** The
test asserts `entry == nil`. It is a self-declared organizer placeholder. Do not
read a pass from it as proof that the config contract digest matches anything,
and do not cite it as a correctness gate.

**`theCheckedInDeclarationSelectsThePinnedHead` fails on `upstream/main` too.**
The failure is not introduced by the campaign. I confirmed it against the
organizer commit itself. Record this and take no action: changing the
declaration to satisfy the test would replace the declared remote q2/q4 head
with the pinned head and change the candidate.
