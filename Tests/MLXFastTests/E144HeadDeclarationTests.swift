import Foundation
import Testing

import MLXFastCore
@testable import MLXFastHarness

// E144 R-D — the `in_branch` head declaration rehearsal.
//
// Two parties compute the head tree digest and they must agree, because the
// candidate declares one number and two different implementations check it:
//
//   1. the trusted CLI, through `computeQwenMTPHeadProvenance`, which seals
//      `head_provenance.sha256` into the report;
//   2. the ranked workflow's own shell recipe, at
//      `.github/workflows/qwen-mtp-ranked-benchmark.yml`, step
//      "Resolve the declared Qwen-MTP head", which refuses the run when the
//      declared digest does not match what it hashes.
//
// A candidate that declares the CLI's number and is checked by the workflow's
// number fails the run. These tests hash real files with both implementations
// and compare, in both polarities per Rule 101.
//
// Nothing here loads a model, reaches the network, or needs a GPU.

@Suite
struct E144HeadDeclarationTests {
    /// The workflow's own recipe, run as the workflow runs it.
    ///
    /// Copied from the "Resolve the declared Qwen-MTP head" step so the
    /// comparison is against the enforcing text and not a paraphrase of it.
    private func workflowDigest(_ root: URL) throws -> (sha: String, bytes: Int) {
        let script = """
        set -euo pipefail
        cd "$1"
        got_bytes="$(find . -type f ! -name README.md -print0 \
          | xargs -0 -n1 wc -c | awk '{ total += $1 } END { print total + 0 }')"
        got_sha="$( find . -type f ! -name README.md \
          | sed 's|^\\./||' \
          | LC_ALL=C sort \
          | while read -r f; do
              printf '%s  %s\\n' "$(shasum -a 256 "${f}" | awk '{print $1}')" "${f}"
            done | shasum -a 256 | awk '{print $1}')"
        printf '%s %s\\n' "${got_sha}" "${got_bytes}"
        """
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", script, "bash", root.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        let fields = (String(data: data, encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: " ")
        return (String(fields[0]), Int(fields[1]) ?? -1)
    }

    private func scratchTree() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("e144-" + UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    private func parse(_ json: String) throws -> QwenMTPHeadDeclaration {
        try QwenMTPHeadDeclaration.parse(data: Data(json.utf8), origin: "e144")
    }

    /// POSITIVE POLARITY. On the tree shape a real head ships -- weight files
    /// beside a top-level README -- the sealed digest and the workflow digest
    /// are the same number, and so are the byte counts.
    @Test
    func theSealedDigestMatchesTheWorkflowDigestOnAHeadShapedTree() throws {
        let root = try scratchTree()
        defer { try? FileManager.default.removeItem(at: root) }
        try "weights".write(
            to: root.appendingPathComponent("model.safetensors"),
            atomically: true, encoding: .utf8)
        try "{\"n\":1}".write(
            to: root.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
        try "docs that must not count".write(
            to: root.appendingPathComponent("README.md"),
            atomically: true, encoding: .utf8)

        let sealed = computeQwenMTPHeadProvenance(
            headDirectory: root.path,
            declaration: QwenMTPHeadDeclaration.pinnedDefault)
        let workflow = try workflowDigest(root)

        #expect(sealed.sha256 == workflow.sha)
        #expect(sealed.bytes == workflow.bytes)
        #expect(sealed.fileCount == 2)
    }

    /// NEGATIVE POLARITY. One byte of one weight file moves BOTH digests, so a
    /// declaration written against the old tree is refused rather than
    /// silently accepted. A positive control that cannot fail proves nothing.
    @Test
    func aSingleByteEditMovesBothDigestsAndBreaksTheDeclaration() throws {
        let root = try scratchTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let weights = root.appendingPathComponent("model.safetensors")
        try "weights".write(to: weights, atomically: true, encoding: .utf8)

        let before = computeQwenMTPHeadProvenance(
            headDirectory: root.path,
            declaration: QwenMTPHeadDeclaration.pinnedDefault)
        let workflowBefore = try workflowDigest(root)
        #expect(before.sha256 == workflowBefore.sha)

        try "weightt".write(to: weights, atomically: true, encoding: .utf8)
        let after = computeQwenMTPHeadProvenance(
            headDirectory: root.path,
            declaration: QwenMTPHeadDeclaration.pinnedDefault)
        let workflowAfter = try workflowDigest(root)

        #expect(after.sha256 != before.sha256)
        #expect(workflowAfter.sha != workflowBefore.sha)
        #expect(after.sha256 == workflowAfter.sha)
        // This is exactly the comparison the workflow refuses on.
        #expect(before.sha256 != workflowAfter.sha)
    }

    /// THE DIVERGENCE. The two implementations do not read "except a top-level
    /// README.md" the same way.
    ///
    /// `computeQwenMTPHeadProvenance` skips `relative == "README.md"`, so it
    /// excludes the TOP-LEVEL file only. The workflow, and the equivalent shell
    /// printed in `mtp-head/README.md`, use `find . -type f ! -name README.md`,
    /// which excludes a file called `README.md` at ANY depth.
    ///
    /// A head tree with a nested `README.md` therefore hashes to two different
    /// numbers, and whichever one the candidate declares, the other party
    /// refuses. This test pins the divergence. If a later change makes the two
    /// agree, this test fails and says so.
    ///
    /// The divergence opens no benchmark escape. Before any timed work the
    /// ranked workflow re-scrubs the resolved head and refuses a `README.md` at
    /// ANY depth (`qwen-mtp-ranked-benchmark.yml`, "declared MTP head tree grew
    /// a digest-exempt README.md before timing"), then digests the survivor
    /// with no exclusion at all. So a nested README always fails the job; it can
    /// never reach a timed round as digest-exempt payload. What remains is a
    /// consistency and diagnosability defect: three implementations, two rules,
    /// and prose in `mtp-head/README.md` that says "top-level" beside its own
    /// equivalent shell that matches any depth.
    @Test
    func aNestedReadmeSplitsTheTwoImplementationsOfTheDigestRule() throws {
        let root = try scratchTree()
        defer { try? FileManager.default.removeItem(at: root) }
        try "weights".write(
            to: root.appendingPathComponent("model.safetensors"),
            atomically: true, encoding: .utf8)
        try "top level docs".write(
            to: root.appendingPathComponent("README.md"),
            atomically: true, encoding: .utf8)

        let flatSealed = computeQwenMTPHeadProvenance(
            headDirectory: root.path,
            declaration: QwenMTPHeadDeclaration.pinnedDefault)
        let flatWorkflow = try workflowDigest(root)
        #expect(flatSealed.sha256 == flatWorkflow.sha)

        let nested = root.appendingPathComponent("shard")
        try FileManager.default.createDirectory(
            at: nested, withIntermediateDirectories: true)
        try "nested docs".write(
            to: nested.appendingPathComponent("README.md"),
            atomically: true, encoding: .utf8)

        let sealed = computeQwenMTPHeadProvenance(
            headDirectory: root.path,
            declaration: QwenMTPHeadDeclaration.pinnedDefault)
        let workflow = try workflowDigest(root)

        // The sealer counts the nested README; the workflow does not.
        #expect(sealed.fileCount == 2)
        #expect(sealed.sha256 != flatSealed.sha256)
        #expect(workflow.sha == flatWorkflow.sha)
        #expect(sealed.sha256 != workflow.sha)
        #expect(sealed.bytes != workflow.bytes)
    }

    /// The parser accepts a well-formed `in_branch` declaration and keeps every
    /// field the workflow re-reads.
    @Test
    func aWellFormedInBranchDeclarationParses() throws {
        let sha = String(repeating: "a", count: 64)
        let declaration = try parse(
            #"{"source":"in_branch","path":"mtp-head/head","sha256":"\#(sha)","bytes":427742600}"#)
        #expect(declaration.source == .inBranch)
        #expect(declaration.path == "mtp-head/head")
        #expect(declaration.sha256 == sha)
        #expect(declaration.bytes == 427_742_600)
        #expect(declaration.maxBytes == QwenMTPHeadDeclaration.defaultMaxBytes)

        let provenanceRoot = try scratchTree()
        defer { try? FileManager.default.removeItem(at: provenanceRoot) }
        try "w".write(
            to: provenanceRoot.appendingPathComponent("model.safetensors"),
            atomically: true, encoding: .utf8)
        let provenance = computeQwenMTPHeadProvenance(
            headDirectory: provenanceRoot.path, declaration: declaration)
        #expect(provenance.source == "in_branch")
        #expect(provenance.origin == "mtp-head/head")
    }

    /// Every unsafe or unverifiable `in_branch` declaration is a REFUSAL, never
    /// a fall back to the pinned head. These are the same rejections the
    /// workflow's `case "${rel}" in ""|/*|*..*|*'\'*)` arm makes.
    @Test
    func everyUnsafeOrUnverifiableInBranchDeclarationIsRefused() throws {
        let sha = String(repeating: "a", count: 64)
        let refusals = [
            #"{"source":"in_branch","path":"","sha256":"\#(sha)","bytes":1}"#,
            #"{"source":"in_branch","path":"/etc/passwd","sha256":"\#(sha)","bytes":1}"#,
            #"{"source":"in_branch","path":"../secrets","sha256":"\#(sha)","bytes":1}"#,
            #"{"source":"in_branch","path":"mtp-head\\head","sha256":"\#(sha)","bytes":1}"#,
            #"{"source":"in_branch","path":"mtp-head","bytes":1}"#,
            #"{"source":"in_branch","path":"mtp-head","sha256":"abc","bytes":1}"#,
            #"{"source":"in_branch","path":"mtp-head","sha256":"\#(sha)","bytes":0}"#,
            #"{"source":"in_branch","path":"mtp-head","sha256":"\#(sha)","bytes":2147483649}"#,
            #"{"source":"in_branch","path":"mtp-head","sha256":"\#(sha)","bytes":9,"max_bytes":8}"#,
        ]
        for json in refusals {
            #expect(throws: (any Error).self) { try parse(json) }
        }
    }

    /// The checked-in declaration is untouched by E144: still `remote`, still
    /// the promoted head. A research branch must not leave a live in-branch
    /// declaration behind.
    @Test
    func e144LeavesTheShippedDeclarationOnTheRemotePromotedHead() throws {
        let declaration = try QwenMTPHeadDeclaration.parse(
            contentsOf: URL(fileURLWithPath: QwenMTPHeadDeclaration.relativePath))
        #expect(declaration.source == .remote)
        #expect(declaration.bytes == 427_742_600)
        #expect(
            declaration.sha256
                == "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71")
    }
}
