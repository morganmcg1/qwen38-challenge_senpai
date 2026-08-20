#!/usr/bin/env python3
"""Install the E68 per-position depth price into the MTP scheduler.

Anchored, fail-closed, idempotent. Every anchor must appear exactly once.

The scored change is two lines inside `costModelDepth`. Everything else is
the price table those two lines read. The `ship` arm is arithmetically
identical to the tip: its `cumulative[d]` repeats the tip's closed form
`1.0 + Double(d) * h` rather than accumulating, because repeated addition of
0.18 differs from the closed form in the last bits at d = 3...6.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
TARGET = REPO / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

ANCHOR_RATIO = "    private static let headStepCostRatio = 0.18\n"

PRICE_TABLE = '''
    /// E68: the depth price as a per-position vector.
    ///
    /// `headStepCostRatio` prices every extra draft the same, so the shipped
    /// cost model is `T(d) = V + d * h * V` with the verify forward `V` flat
    /// in width. The measured verify curve is not flat in width: the QMV
    /// dispatch table changes group shape at several widths, so the step into
    /// one width can cost a multiple of the step into its neighbour.
    ///
    /// Every arm holds the total at `maxDepth * headStepCostRatio`, so an arm
    /// changes the SHAPE of the price and never its level. The level is
    /// already measured: `h = 0.32` scored 2.84585, a clean -3%, because it
    /// shortened every draft. This pool rewards depth, so E68 asks only
    /// whether the price is distributed correctly across positions.
    internal struct DepthPrice {
        /// `marginal[d]` prices the step into verify width `d + 2`.
        let marginal: [Double]
        /// `cumulative[d]` is the running cost BEFORE step `d` is taken, so
        /// `cumulative[0]` is 1.0: the verify forward on its own.
        let cumulative: [Double]
    }

    /// The one-boundary tier factor E56 fitted, retained so `pb5` and `pb7`
    /// reproduce that experiment's published arithmetic exactly.
    internal static let boundaryTierFactor = 2.0301

    /// The shipped flat price. `cumulative` repeats the tip's closed form
    /// instead of accumulating: `1.0 + 0.18 + 0.18 + 0.18` and
    /// `1.0 + 3.0 * 0.18` differ by one ulp, and a control arm that is not
    /// bit-identical to the tip is not a control.
    internal static func makeUniformDepthPrice() -> DepthPrice {
        DepthPrice(
            marginal: [Double](repeating: headStepCostRatio,
                               count: Qwen36MTPLimits.maxDepth),
            cumulative: (0 ... Qwen36MTPLimits.maxDepth).map {
                1.0 + Double($0) * headStepCostRatio
            })
    }

    /// One priced boundary, holding the total. `width` is the verify width
    /// the priced step ENTERS, so it selects index `width - 2`.
    internal static func makeBoundaryDepthPrice(
        enteringVerifyWidth width: Int
    ) -> DepthPrice {
        let count = Qwen36MTPLimits.maxDepth
        let within = Double(count) * headStepCostRatio
            / (Double(count - 1) + boundaryTierFactor)
        var marginal = [Double](repeating: within, count: count)
        marginal[width - 2] = within * boundaryTierFactor
        return DepthPrice(marginal: marginal,
                          cumulative: prefixCosts(marginal))
    }

    /// `headStepCostRatio + (C(d + 2) - C(d + 1)) / V` from the E68 rung-1
    /// session, before rescaling. Empty until that session fills it, and
    /// `makeMeasuredDepthPrice` traps rather than silently running as `ship`.
    internal static let measuredRawDepthPrice: [Double] = []

    internal static func makeMeasuredDepthPrice() -> DepthPrice {
        precondition(
            measuredRawDepthPrice.count == Qwen36MTPLimits.maxDepth,
            "E68 pbfit: measuredRawDepthPrice is not filled from rung 1")
        let total = Double(Qwen36MTPLimits.maxDepth) * headStepCostRatio
        let scale = total / measuredRawDepthPrice.reduce(0.0, +)
        let marginal = measuredRawDepthPrice.map { $0 * scale }
        return DepthPrice(marginal: marginal,
                          cumulative: prefixCosts(marginal))
    }

    internal static func prefixCosts(_ marginal: [Double]) -> [Double] {
        var out = [1.0]
        var running = 1.0
        for value in marginal {
            running += value
            out.append(running)
        }
        return out
    }

    internal enum DepthPriceArm: String {
        case ship, pb5, pb7, pbfit
    }

    /// THE ONE LINE THE E68 LEG RUNNER PATCHES.
    internal static let depthPriceArm: DepthPriceArm = .ship

    /// Built once. A computed property here would allocate two arrays on
    /// every round, inside the timed path.
    internal static let depthPrice: DepthPrice = {
        switch depthPriceArm {
        case .ship: return makeUniformDepthPrice()
        case .pb5: return makeBoundaryDepthPrice(enteringVerifyWidth: 5)
        case .pb7: return makeBoundaryDepthPrice(enteringVerifyWidth: 7)
        case .pbfit: return makeMeasuredDepthPrice()
        }
    }()
'''

ANCHOR_WALK_HEAD = """        let h = Self.headStepCostRatio
        var reach = 1.0
"""
REPLACE_WALK_HEAD = """        let price = Self.depthPrice
        var reach = 1.0
"""

ANCHOR_THRESHOLD = (
    "            let threshold = h * (1.0 + expected)"
    " / (1.0 + Double(depth) * h)\n")
# The operator stays at the end of the line. A continuation line that starts
# with `/` can be lexed as the opening delimiter of a regex literal.
REPLACE_THRESHOLD = (
    "            let threshold = price.marginal[depth] * (1.0 + expected) /\n"
    "                price.cumulative[depth]\n")

ANCHOR_TRACE = """        scheduleTrace = String(
            format: "m=%.6f streak=%d cap=%d ema=",
            margin, fullAcceptStreak, widthCap) + emas + " sched="
"""
REPLACE_TRACE = """        scheduleTrace = "arm=" + Self.depthPriceArm.rawValue + " " + String(
            format: "m=%.6f streak=%d cap=%d ema=",
            margin, fullAcceptStreak, widthCap) + emas + " sched="
"""

EDITS = [
    ("price table", ANCHOR_RATIO, ANCHOR_RATIO + PRICE_TABLE),
    ("walk head", ANCHOR_WALK_HEAD, REPLACE_WALK_HEAD),
    ("threshold", ANCHOR_THRESHOLD, REPLACE_THRESHOLD),
    ("trace tag", ANCHOR_TRACE, REPLACE_TRACE),
]


def main() -> int:
    text = TARGET.read_text()
    if "internal static let depthPriceArm" in text:
        print("apply_swift_patch: already installed, nothing to do")
        return 0
    for name, anchor, replacement in EDITS:
        found = text.count(anchor)
        if found != 1:
            raise SystemExit(
                "apply_swift_patch: anchor %r matched %d times, expected 1"
                % (name, found))
        text = text.replace(anchor, replacement)
    TARGET.write_text(text)
    print("apply_swift_patch: installed %d edits into %s"
          % (len(EDITS), TARGET.relative_to(REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
