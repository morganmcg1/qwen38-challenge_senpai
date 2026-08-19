import Foundation
import Testing

@testable import MLXFastModel

/// The depth schedule prices an extra draft row from a model of what the
/// machine charges for a wider verify. Two live sources decide that charge:
/// the QMV dispatch switch, which says when a wider verify buys another pass
/// over the 4-bit backbone, and our own wide-decode attention chunk, which
/// says when a wider verify buys a second SDPA call.
///
/// These tests re-parse both sources. They exist so that a change to either
/// one fails loudly instead of leaving a stale staircase inside the schedule,
/// which the schedule itself could never detect.
@Suite
struct QwenMTPDepthCostModelTests {
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXFastTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repository root
    }

    private static func source(_ relativePath: String) throws -> String {
        try String(contentsOf: repositoryRoot.appending(path: relativePath),
                   encoding: .utf8)
    }

    private static func matches(
        _ pattern: String, in text: String, groups: Int
    ) throws -> [[String]] {
        let regex = try NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        return regex.matches(in: text, range: range).map { match in
            (1 ... groups).map { index in
                guard let captured = Range(match.range(at: index), in: text)
                else { return "" }
                return String(text[captured])
            }
        }
    }

    /// The staircase the schedule believes in must be the one the machine runs.
    @Test
    func inputsPerGroupTableMatchesTheShippedDispatchSwitch() throws {
        let header = try Self.source(
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h")
        // Only the wide-output branch dispatches the `_m` cells whose IPG sets
        // the number of weight streams; the narrow branch keeps every row
        // resident and reads the weights once.
        let branch = try #require(header.range(of: "if (out_vec_size >= 4096)"))
        let narrow = try #require(header.range(of: "} else {", range: branch.upperBound ..< header.endIndex))
        let wideBranch = String(header[branch.lowerBound ..< narrow.lowerBound])

        var live: [Int: Int] = [:]
        for row in try Self.matches(
            #"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>"#,
            in: wideBranch, groups: 2)
        {
            live[Int(row[0])!] = Int(row[1])!
        }
        #expect(!live.isEmpty, "no `_m` dispatch cells parsed from quantized.h")
        let schedule = Qwen36MTPBlockSession.verifyInputsPerGroup
        let movedWidths = Set(live.keys).union(schedule.keys)
            .filter { live[$0] != schedule[$0] }
            .sorted()
        let movedDetail = movedWidths.map { width in
            let before = schedule[width].map(String.init) ?? "absent"
            let after = live[width].map(String.init) ?? "absent"
            let streamsBefore = Qwen36MTPBlockSession.verifyWeightStreams(width: width)
            let streamsAfter = live[width].map { (width + $0 - 1) / $0 }
                .map(String.init) ?? "1"
            return "width \(width): IPG \(before) -> \(after), "
                + "weight streams \(streamsBefore) -> \(streamsAfter)"
        }
        #expect(
            movedWidths.isEmpty,
            """
            The QMV dispatch table moved at \(movedWidths). \
            \(movedDetail.joined(separator: "; ")). \
            The depth schedule's stream staircase is now stale: re-derive \
            `verifyInputsPerGroup`, recheck which widths `pricedBoundaryWidths` \
            should name, and re-measure the schedule before shipping.
            """)
    }

    /// The surcharge must sit where the live table puts a boundary, and the
    /// mean price must stay at the bracketed `headStepCostRatio`.
    @Test
    func marginalPriceStepsOnlyWhereAWeightStreamIsAdded() throws {
        let marginals = Qwen36MTPBlockSession.marginalCostRatio
        #expect(marginals.count == Qwen36MTPLimits.maxDepth)

        let mean = marginals.reduce(0.0, +) / Double(marginals.count)
        #expect(abs(mean - 0.18) < 1e-12, "mean price drifted from the live h")

        let priced = { (depth: Int) -> Bool in
            Qwen36MTPBlockSession.pricedBoundaryWidths.contains(depth + 2)
                && Qwen36MTPBlockSession.verifyWeightStreams(width: depth + 2)
                    > Qwen36MTPBlockSession.verifyWeightStreams(width: depth + 1)
        }
        let withinTier = marginals.enumerated().filter { !priced($0.offset) }
        let crossings = marginals.enumerated().filter { priced($0.offset) }
        // Every width this schedule prices must still be a live crossing. If
        // the dispatch table stops adding a weight stream there, the surcharge
        // is charging for work the machine no longer does.
        let pricedButFlat = Qwen36MTPBlockSession.pricedBoundaryWidths.filter {
            Qwen36MTPBlockSession.verifyWeightStreams(width: $0)
                <= Qwen36MTPBlockSession.verifyWeightStreams(width: $0 - 1)
        }.sorted()
        #expect(pricedButFlat.isEmpty,
                """
                `pricedBoundaryWidths` names \(pricedButFlat) where the live \
                dispatch table adds no weight stream, so the schedule charges \
                for a pass the machine no longer makes.
                """)
        #expect(crossings.count == Qwen36MTPBlockSession.pricedBoundaryWidths.count)
        for (_, value) in withinTier {
            #expect(abs(value - withinTier[0].element) < 1e-12)
        }
        for (_, value) in crossings {
            let ratio = value / withinTier[0].element
            #expect(abs(ratio - Qwen36MTPBlockSession.verifyStreamCostRatio)
                < 1e-12,
                """
                A boundary row is priced at \(ratio)x an ordinary row, but the \
                measured round-level surcharge is \
                \(Qwen36MTPBlockSession.verifyStreamCostRatio)x.
                """)
        }

        var cumulative = 1.0
        for (depth, step) in marginals.enumerated() {
            #expect(abs(Qwen36MTPBlockSession.cumulativeCostRatio[depth] - cumulative)
                < 1e-12)
            cumulative += step
        }
    }

    /// A price can close a depth step silently. The walk extends while
    /// `reach > marginalCostRatio[d] * (1 + expected) / cumulativeCostRatio[d]`,
    /// `reach` is a product of probabilities so it never exceeds 1, and
    /// `expected` is a sum of partial products each at least `reach`, so
    /// `expected >= (d + 1) * reach` once the walk has reached depth `d`.
    /// Substituting gives a necessary condition that depends on the price
    /// alone: if `marginalCostRatio[d] * (d + 1) / cumulativeCostRatio[d] >= 1`
    /// the step is unreachable at EVERY acceptance rate, on every prompt.
    ///
    /// The first version of this cost model closed depth 3 that way, and the
    /// schedule then behaved as an unconditional width-4 cap while still
    /// looking like a walk. Nothing in the other tests could see it. This test
    /// pins which steps are closed so that the next change to the price has to
    /// say so out loud.
    @Test
    func onlyTheDeclaredDepthStepsAreClosedAtEveryAcceptanceRate() throws {
        let closed = (0 ..< Qwen36MTPLimits.maxDepth).filter { depth in
            let bestCase = Qwen36MTPBlockSession.marginalCostRatio[depth]
                * Double(depth + 1)
                / Qwen36MTPBlockSession.cumulativeCostRatio[depth]
            return bestCase >= 1.0
        }
        // A closed step that prices a crossing is the session-1 failure: the
        // walk pays for a boundary it can never cross, so the schedule is a
        // width cap wearing a walk.
        let closedBoundaries = closed.filter {
            Qwen36MTPBlockSession.pricedBoundaryWidths.contains($0 + 2)
        }
        #expect(closed == Qwen36MTPBlockSession.declaredClosedDepthSteps,
                """
                Closed depth steps are \(closed), declared \
                \(Qwen36MTPBlockSession.declaredClosedDepthSteps); of those, \
                \(closedBoundaries) price a weight-stream crossing the walk can \
                never take. A step listed here is unreachable whatever the \
                prompt or the acceptance rate, so the schedule is a fixed cap \
                rather than a walk at that depth. Re-derive the price or declare \
                the new closure in `declaredClosedDepthSteps`.
                """)
    }

    /// The wide-decode chunk is the only reason a verify above width 5 issues
    /// two SDPA calls, and the schedule deliberately does NOT price that: the
    /// measured cost of the chunk at width 6 is between -0.1 ms and +1.2 ms
    /// per round, against a ~19 ms verify row. This test fails if the live
    /// predicate changes, because that measurement, and the decision not to
    /// price it, would both need to be redone.
    @Test
    func attentionChunkPredicateStillMatchesTheMeasuredRoute() throws {
        let attention = try Self.source(
            "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift")
        let predicate = try Self.matches(
            #"queries\.dim\(0\) == (\d+), qL >= (\d+), qL <= (\d+), kL >= qL"#,
            in: attention, groups: 3)
        #expect(predicate.count == 1, "wide-decode chunk predicate not found")
        let split = try Self.matches(#"let split = (\d+)"#, in: attention, groups: 1)
        #expect(split.count == 1, "wide-decode chunk split not found")

        let minQL = Int(predicate[0][1])!
        let maxQL = Int(predicate[0][2])!
        let splitRow = Int(split[0][0])!
        #expect(minQL == 6 && maxQL == 9 && splitRow == 5,
                """
                The wide-decode chunk predicate changed to \(minQL)...\(maxQL) \
                split \(splitRow). E56 measured the cost of the OLD predicate at \
                width 6; re-measure it before the schedule assumes the step is free.
                """)

        // The chunk exists to keep both halves on the fused vector route. The
        // trusted host selects that route on the query length alone, so the
        // invariant is that neither half exceeds it.
        let dispatch = try Self.source(
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp")
        let vectorMode = try Self.matches(
            #"if \(q_pre\.shape\(2\) <= (\d+)\)"#, in: dispatch, groups: 1)
        #expect(vectorMode.count >= 1, "host vector-mode condition not found")
        let vectorModeMaxQL = Int(vectorMode[0][0])!
        #expect(splitRow <= vectorModeMaxQL)
        #expect(maxQL - splitRow <= vectorModeMaxQL)
    }
}
