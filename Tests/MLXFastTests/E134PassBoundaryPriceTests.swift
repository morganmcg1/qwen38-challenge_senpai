import Foundation
@testable import MLXFastModel
import Testing

// E134: the depth price at verify width 6.
//
// The first version of this file justified `passBoundaryVerifyWidth = 6` from
// the QMV dispatch table: width 6 was the first width whose rows needed more
// than one read of the projection weights, so the cost step there was
// structural. That justification is now WRONG. The promoted one-pass table
// `{6: 6, 7: 7}` gives widths 6 and 7 one pass each, so the structural pass
// boundary moved to width 8.
//
// The priced boundary did not move with it. Ranked receipt `623e77af`, the
// promotion of that same table, kept 89 to 96 percent of the measured cost
// step at width 6. The constant therefore rests on a MEASUREMENT, not on the
// table, and these tests pin the measurement instead of the table.
//
// Every predicate here is stated in both polarities, as Rule 101 requires: for
// each assertion there is a positive control that the same predicate rejects.
// A test that cannot fail is not evidence.

private let h = 0.18
private let maxDepth = Qwen36MTPLimits.maxDepth

/// The ranked receipt whose inversion selects the priced boundary.
private let receipt = "623e77af"

private func repoRoot(file: StaticString = #filePath) -> URL {
    var root = URL(fileURLWithPath: "\(file)")
    for _ in 0 ..< 3 { root.deleteLastPathComponent() }
    return root
}

private enum PassBoundaryFailure: Error {
    case routeNotFound
    case planNotFound(String)
    case artifactMissing(String)
}

// MARK: - the compiled QMV route

private struct WidthPlanEntry {
    let m: Int
    let ipg: Int
    var passes: Int { (m + ipg - 1) / ipg }
}

private func quotedLiteral(on line: Substring) -> String? {
    guard let open = line.firstIndex(of: "\""),
          let close = line.lastIndex(of: "\""), open < close
    else { return nil }
    return String(line[line.index(after: open) ..< close])
}

/// `(table name) -> width plan`, parsed from the witness literals.
///
/// `Qwen35.swift` belongs to another agent, so this stack must not copy its
/// tables into constants that can drift. That agent's own suite fails the
/// build if a witness ever disagrees with the plan it describes, which makes
/// the literals a safe proxy for the plans themselves.
private func widthPlans() throws
    -> (compiled: String, byName: [String: [WidthPlanEntry]]) {
    let source = repoRoot()
        .appendingPathComponent("Vendor/mlx-swift-lm/Libraries/MLXLLM")
        .appendingPathComponent("Models/Qwen35.swift")
    let lines = try String(contentsOf: source, encoding: .utf8)
        .split(separator: "\n", omittingEmptySubsequences: false)

    guard let routeLine = lines.first(where: {
        $0.contains("defaultRouteWitness") && $0.contains("e120_default_route/")
    }), let route = quotedLiteral(on: routeLine),
        let compiled = route.split(separator: "/").last
    else { throw PassBoundaryFailure.routeNotFound }

    var byName: [String: [WidthPlanEntry]] = [:]
    var pending: String?
    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("case ."), trimmed.hasSuffix(":") {
            pending = String(trimmed.dropFirst("case .".count).dropLast())
        }
        guard trimmed.contains("e120_width_plan/"),
              let name = pending,
              let literal = quotedLiteral(on: line)
        else { continue }
        let body = literal.replacingOccurrences(
            of: "e120_width_plan/", with: "")
        byName[name.lowercased()] = body.split(separator: ",").compactMap {
            let parts = $0.split(separator: ":").compactMap { Int($0) }
            guard parts.count == 3 else { return nil }
            return WidthPlanEntry(m: parts[0], ipg: parts[1])
        }
    }
    return (String(compiled).lowercased(), byName)
}

/// The narrowest width that reads the projection weights more than once.
private func structuralPassBoundary(_ plan: [WidthPlanEntry]) -> Int? {
    plan.sorted { $0.m < $1.m }.first { $0.passes > 1 }?.m
}

// MARK: - the measured ranked curve

private struct MeasuredCurve: Decodable {
    struct Shape: Decodable {
        let rows: [Int]
        let roundUs: [Double]
        let steps: [Double]
        let argmaxBoundary: Int

        enum CodingKeys: String, CodingKey {
            case rows
            case roundUs = "round_us"
            case steps
            case argmaxBoundary = "argmax_boundary"
        }
    }

    struct Entry: Decodable { let shape: Shape }

    let receipt: String
    let harness: String
    let bestForm: String
    let curves: [String: Entry]

    enum CodingKeys: String, CodingKey {
        case receipt
        case harness
        case bestForm = "best_form"
        case curves
    }
}

private func measuredCurve() throws -> MeasuredCurve {
    let path = repoRoot()
        .appendingPathComponent("research/e134-artifacts")
        .appendingPathComponent("item2-measured-curve.json")
    guard let data = try? Data(contentsOf: path) else {
        throw PassBoundaryFailure.artifactMissing(path.path)
    }
    return try JSONDecoder().decode(MeasuredCurve.self, from: data)
}

/// Which boundary carries the largest step in a round-cost vector.
///
/// This is the whole selection rule the arm rests on, written once so the
/// measured curve and its positive control run through identical code.
private func argmaxBoundary(_ roundUs: [Double]) -> Int {
    let steps = (0 ..< roundUs.count - 1).map { roundUs[$0 + 1] - roundUs[$0] }
    return steps.indices.max { steps[$0] < steps[$1] } ?? 0
}

// MARK: - the walk

/// Replay of the shipped walk under a flat acceptance rate, matching
/// `QwenMTPDepthPriceTests`.
private func walkDepth(
    _ price: Qwen36MTPBlockSession.DepthPrice, cap: Int, p: Double
) -> Int {
    var reach = 1.0
    var expected = 0.0
    var depth = 0
    while depth < cap {
        reach *= p
        let threshold = price.marginal[depth] * (1.0 + expected) /
            price.cumulative[depth]
        guard reach > threshold else { break }
        expected += reach
        depth += 1
    }
    return depth
}

@Suite("E134 pass boundary depth price")
struct E134PassBoundaryPriceTests {
    // MARK: the table no longer justifies the constant

    @Test("the compiled QMV route moved the structural pass boundary off 6")
    func compiledRouteMovedTheStructuralBoundary() throws {
        let (compiled, plans) = try widthPlans()
        #expect(compiled == "onepass67")
        guard let live = plans[compiled] else {
            throw PassBoundaryFailure.planNotFound(compiled)
        }
        #expect(live.first { $0.m == 6 }?.passes == 1)
        #expect(structuralPassBoundary(live) == 8)
        #expect(structuralPassBoundary(live)
            != Qwen36MTPBlockSession.passBoundaryVerifyWidth)

        // Positive control, same parser and same predicate. The table this
        // stack was first fitted on does put the structural boundary at the
        // priced width, so the 8 above is a real move rather than a parse
        // failure that silently returns nothing.
        guard let preArm = plans["shipped"] else {
            throw PassBoundaryFailure.planNotFound("shipped")
        }
        #expect(preArm.first { $0.m == 6 }?.passes == 2)
        #expect(structuralPassBoundary(preArm)
            == Qwen36MTPBlockSession.passBoundaryVerifyWidth)
    }

    // MARK: the measurement that does justify it

    @Test("the measured ranked curve still puts its largest step at width 6")
    func measuredCurveKeepsTheStepAtWidthSix() throws {
        let blob = try measuredCurve()
        #expect(blob.receipt == receipt)
        #expect(blob.harness == "ranked")
        guard let best = blob.curves[blob.bestForm]?.shape else {
            throw PassBoundaryFailure.planNotFound(blob.bestForm)
        }
        #expect(best.rows.first == 1)
        // Boundary `b` is the step from `b + 1` rows to `b + 2` rows, so the
        // priced width is `b + 2`.
        #expect(best.argmaxBoundary == argmaxBoundary(best.roundUs))
        #expect(best.argmaxBoundary + 2
            == Qwen36MTPBlockSession.passBoundaryVerifyWidth)

        // Positive control. Cut the priced step down to the shallow step and
        // shift the rest of the curve with it, which removes the cliff and
        // leaves every other step untouched. The same selection rule must
        // then choose another boundary. Without this the argmax assertion
        // would also pass on a curve that has no cliff at all.
        var flattened = best.roundUs
        let priced = Qwen36MTPBlockSession.passBoundaryVerifyWidth - 1
        let excess = (flattened[priced] - flattened[priced - 1]) - best.steps[0]
        for row in priced ..< flattened.count { flattened[row] -= excess }
        #expect(argmaxBoundary(flattened) != best.argmaxBoundary)
    }

    @Test("the shipped tier is inside the plateau the measured curve supports")
    func shippedTierIsInsideTheMeasuredPlateau() throws {
        let blob = try measuredCurve()
        guard let best = blob.curves[blob.bestForm]?.shape else {
            throw PassBoundaryFailure.planNotFound(blob.bestForm)
        }
        let ratio = best.steps[best.argmaxBoundary] / best.steps[0]
        // The measured step at the priced width is still a multiple of the
        // shallow step, which is what makes any tier above 1.0 admissible.
        let admissible = { (tier: Double) in tier > 1.0 && tier <= ratio }
        #expect(ratio > 1.0)
        #expect(admissible(Qwen36MTPBlockSession.passBoundaryTierFactor))

        // Positive control: the shipped flat price is tier 1.0 by definition
        // and a tier above the measured ratio overcharges the boundary, so the
        // same predicate must reject both ends.
        #expect(!admissible(1.0))
        #expect(!admissible(ratio + 1.0))
    }

    // MARK: the price construction, unchanged by the receipt

    @Test("the pb6 table is exactly the table the replayer priced")
    func pb6TableMatchesTheReplayer() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let within = Double(maxDepth) * h / (Double(maxDepth - 1) + tier)
        let price = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for depth in 0 ..< maxDepth {
            let expected = depth == width - 2 ? within * tier : within
            #expect(abs(price.marginal[depth] - expected) <= expected.ulp)
        }
        let total = price.marginal.reduce(0.0, +)
        #expect(abs(total - Double(maxDepth) * h) < 1e-12)
        var running = 1.0
        #expect(price.cumulative[0] == 1.0)
        for depth in 0 ..< maxDepth {
            running += price.marginal[depth]
            #expect(price.cumulative[depth + 1] == running)
        }
    }

    // The tier parameter carries E56's constant as its default, so the arms
    // that experiment published must not move by one ulp.
    @Test("the tier parameter leaves the E56 arms bit-identical")
    func tierParameterLeavesE56ArmsUnchanged() {
        let e56 = Qwen36MTPBlockSession.boundaryTierFactor
        let within = Double(maxDepth) * h / (Double(maxDepth - 1) + e56)
        for width in [5, 7] {
            let price = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
                enteringVerifyWidth: width)
            for depth in 0 ..< maxDepth {
                let expected = depth == width - 2 ? within * e56 : within
                #expect(price.marginal[depth] == expected)
            }
        }
    }

    @Test("the shipped arm is pb6 at the measured width and tier")
    func shippedArmIsPB6() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .pb6)
        #expect(Qwen36MTPBlockSession.passBoundaryVerifyWidth == 6)
        #expect(Qwen36MTPBlockSession.passBoundaryTierFactor == 1.45)
    }

    // The mechanism, in two halves. Holding the total means every step below
    // the boundary gets CHEAPER as the one priced step rises, so pb6 is not a
    // level change. A level change is the `flat` control, which the replayer
    // scored at `-0.30 %` against pb6's `+2.34 %`.
    @Test("pb6 charges less below the pass boundary and more across it")
    func pb6ChargesLessBelowAndMoreAcross() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for depth in 0 ..< maxDepth {
            if depth == width - 2 {
                #expect(pb6.marginal[depth] > ship.marginal[depth])
            } else {
                #expect(pb6.marginal[depth] < ship.marginal[depth])
            }
        }
    }

    // The second half: at both ranked acceptance rates the flat-rate walk
    // stops with `width - 2` drafts taken, while the shipped price crosses.
    // The flat-rate walk cannot separate one tier from another, so it pins the
    // direction only; the tier itself comes from the replayer.
    @Test("pb6 stops the flat-rate walk at the priced width")
    func pb6StopsTheWalkAtThePassBoundary() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for p in [0.8351, 0.8750] {
            #expect(walkDepth(pb6, cap: 7, p: p) == width - 2)
            #expect(walkDepth(ship, cap: 7, p: p) > width - 2)
        }
    }
}
