#!/usr/bin/env python3
"""E40 deliverable 1 -- width-dependence classification of the whole shipped delta.

Enumerates every hunk of the 5-file / +229/-74 shipped surface against the
campaign baseline, attaches a hand-assigned scaling class, and proves the
per-hunk line counts sum to the gate's totals.  The classification itself is a
source-reading judgement; this tool exists so the ARITHMETIC is checked and the
table cannot silently omit a hunk.

Scaling classes
  none        cost cannot enter a scored round at all (comment, doc, dead field,
              declaration, or code on a path the scored run never takes)
  once        one-time cost, paid once per process, outside the timed window
  per-token   cost proportional to emitted tokens (512, identical every prompt)
  per-round   cost proportional to round count R (varies 84..487 by prompt)
  per-row     cost proportional to M within a round  <-- the only class that can
  per-width   one-time cost per DISTINCT width first touched in a run
              produce a width-confined deficit

Zero GPU.  `git diff` only.
"""

import re
import subprocess
import sys

BASE = '527306761f70e2c4024f347915328894db80c181'

FILES = [
    'Sources/MLXFastModel/Qwen36MTPBlockSession.swift',
    'Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift',
    'Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift',
    'Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp',
    'Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h',
]

# (file, hunk index within file) -> (label, class, hypothesis, why)
LABEL = {
    ('Qwen36MTPBlockSession.swift', 0): (
        'drop reachedStopToken from RoundResult', 'none', 'H3',
        'struct field removal; part of the EOS-latch fix.'),
    ('Qwen36MTPBlockSession.swift', 1): (
        'drop session reachedStopToken var', 'none', 'H3',
        'stored-property removal, same fix.'),
    ('Qwen36MTPBlockSession.swift', 2): (
        'wireResidentWeightsIfEnabled (Memory.clearCache)', 'per-width', 'H3',
        'RETIRED FAMILY. Memory.clearCache() after the shape warm returns every '
        'warmed buffer to the OS, so the scored run pays first-touch again at '
        'each width it visits. The baseline has no such call, so its warm '
        'buffers stay in the allocator cache. Cost is outside the timed window '
        'but its EFFECT is inside it. Only per-width mechanism in the delta.'),
    ('Qwen36MTPBlockSession.swift', 3): (
        'warm split + wireResidentWeightsIfEnabled call', 'once', 'H3',
        'warm path, outside every scored window.'),
    ('Qwen36MTPBlockSession.swift', 4): (
        'traceSyncHeadChain + traceSink statics', 'none', 'H3',
        'two static lets read from env once; false / stderr in a scored run.'),
    ('Qwen36MTPBlockSession.swift', 5): (
        'traceWrite -> traceSink.write', 'none', 'H3',
        'body only reached when the phase trace is on.'),
    ('Qwen36MTPBlockSession.swift', 6): (
        'if traceRounds { snapshotScheduleSignal }', 'per-round', 'H3',
        'one static-Bool test per round; the call never fires when scored.'),
    ('Qwen36MTPBlockSession.swift', 7): (
        'if traceRounds { scheduleTrace += } in walk', 'per-row', 'H3',
        'inside the depth-extension walk, so <=depth predicted-not-taken tests '
        'per round, i.e. O(M). Genuinely per-row; see the magnitude budget.'),
    ('Qwen36MTPBlockSession.swift', 8): (
        'scheduleTrace var + snapshotScheduleSignal body', 'none', 'H3',
        'declaration plus a method body the scored run never calls.'),
    ('Qwen36MTPBlockSession.swift', 9): (
        'tailPending binding -> pendingTop2 != nil', 'none', 'H3',
        'removes one unused let binding; no work either way.'),
    ('Qwen36MTPBlockSession.swift', 10): (
        'remove pre-draft EOS early-return', 'none', 'H3',
        'EOS-latch fix. Reachable only on a round whose primary is a stop '
        'token, so it cannot scale with M.'),
    ('Qwen36MTPBlockSession.swift', 11): (
        'drop reachedStopToken from depth-0 result', 'none', 'H3',
        'initialiser argument removal.'),
    ('Qwen36MTPBlockSession.swift', 12): (
        'if traceSyncHeadChain { eval(draft chain) }', 'per-round', 'H3',
        'one static-Bool test per round; the eval() that would destroy '
        'head/verify overlap never executes in a scored run.'),
    ('Qwen36MTPBlockSession.swift', 13): (
        'scheduleTrace appended to trace line', 'none', 'H3',
        'inside the `if Self.traceRounds` block already counted at hunk 6.'),
    ('Qwen36MTPBlockSession.swift', 14): (
        'remove post-commit EOS truncation', 'none', 'H3',
        'EOS-latch fix. The removed firstIndex(where:) was itself O(M), so the '
        'candidate does strictly LESS per-row work here than the baseline.'),
    ('Qwen36MTPBlockSession.swift', 15): (
        'drop reachedStopToken from final result', 'none', 'H3',
        'initialiser argument removal.'),
    ('RuntimeStartupMemoryPolicy.swift', 0): (
        'import Foundation', 'none', 'H4',
        'import only.'),
    ('RuntimeStartupMemoryPolicy.swift', 1): (
        'installQwenMTPFullProfileCommandBufferDefaults', 'once', 'H4',
        'RETIRED FAMILY. setenv MLX_MAX_MB_PER_BUFFER=512 / OPS=50 with '
        'overwrite=0, before first MLX device access. One-time. NOTE: the crown '
        'row is the same setenv with overwrite=1, so if the harness already '
        'exports either variable our call is a NO-OP and theirs is not -- a '
        'readable source difference bearing on H4.'),
    ('RuntimeStartupMemoryPolicy.swift', 2): (
        'resolve() calls the installer', 'once', 'H4',
        'one call at startup.'),
    ('Qwen35.swift', 0): (
        'qwen35DecodeLadderRungs global Set<Int>', 'once', 'H2',
        'lazily-initialised global; shipped default [0,1,9,19,29,39,49,57] is '
        'EXACTLY the baseline switch cases, so the asyncEval schedule is '
        'unchanged. Initialiser (ProcessInfo environment build) runs once and '
        'is triggered by warmup.'),
    ('Qwen35.swift', 1): (
        'ladder comment S<=2 -> S<=9', 'none', 'H2',
        'comment only. `ladderActive = inputs.dim(1) <= 9 || prefillLadder` is '
        'CONTEXT in the diff -- the baseline already gated at 9; only the stale '
        'comment moved. The shipped rung set is also unchanged, so the asyncEval '
        'SCHEDULE is byte-identical between the two arms.'),
    ('Qwen35.swift', 2): (
        'comment: rung set overridable', 'none', 'H2',
        'comment only.'),
    ('Qwen35.swift', 3): (
        'fused loop: switch -> Set.contains', 'per-round', 'H2',
        'ONE Set<Int>.contains per layer = 64 per forward pass = 64 per round. '
        'Independent of M: swift_once guard + SipHash-1-3 + L1 bucket probe. '
        'This is the ONLY functional change in the file and it is the answer to '
        'the brief\'s question -- nothing here allocates, reshapes, concatenates '
        'or broadcasts per row.'),
    ('Qwen35.swift', 4): (
        'unfused loop: switch -> Set.contains', 'none', 'H2',
        'same rewrite on the `dtype != bfloat16 || dim != 5120` branch, which '
        'the scored Qwen 3.8 path never takes (bf16, hidden 5120).'),
    ('quantized.cpp', 0): (
        'E27 twin: static_assert NA<=4 -> NA<=5', 'none', 'H1',
        'compile-time only -- but it is what ADMITS the NA=5 instantiation.'),
    ('quantized.cpp', 1): (
        'E27 twin: IPG comment ceil(M/4) -> ceil(M/5)', 'none', 'H1',
        'comment only.'),
    ('quantized.cpp', 2): (
        'E27 twin: case 5 <T,5,3> -> <T,5,5>', 'per-row', 'H1',
        'runtime-effective generated twin of the kernel source.'),
    ('quantized.cpp', 3): (
        'E27 twin: case 9 <T,9,3> -> <T,9,5>', 'per-row', 'H1',
        'runtime-effective generated twin of the kernel source.'),
    ('quantized.h', 0): (
        'static_assert NA<=4 -> NA<=5', 'none', 'H1',
        'compile-time only -- but it is what ADMITS the NA=5 instantiation that '
        'raises the shared kernel register ceiling. Zero cost by itself, yet it '
        'is a necessary link in the confirmed H1 mechanism.'),
    ('quantized.h', 1): (
        'IPG comment ceil(M/4) -> ceil(M/5)', 'none', 'H1',
        'comment only.'),
    ('quantized.h', 2): (
        'case 5: <T,5,3> -> <T,5,5>', 'per-row', 'H1',
        'switch is on ntg.x, a RUNTIME value, inside the single '
        '[[kernel]] affine_qmv_fast (quantized.h:1869). Every width cell '
        'compiles into that ONE kernel, so its register allocation is the max '
        'over all cells: 108 -> 129 (+19.4 %). Every M pays.'),
    ('quantized.h', 3): (
        'case 9: <T,9,3> -> <T,9,5>', 'per-row', 'H1',
        'same shared-kernel argument; this is the cell that BECOMES the '
        'kernel-wide maximum at 129 registers.'),
}

CLASSES = ['none', 'once', 'per-token', 'per-round', 'per-row', 'per-width']
CAN_BE_WIDTH_CONFINED = {'per-row', 'per-width'}
MAGNITUDE_KILLED = {'per-round'}
# Qwen36MTPBlockSession hunk 7: the only per-row survivor the magnitude budget
# removes, so its line count is subtracted explicitly in the ladder.
SESSION_TRACE_WALK = (4, 0)


def hunks(path):
    out = subprocess.run(
        ['git', '--no-pager', 'diff', '--no-color', '-U0', BASE, 'HEAD', '--', path],
        capture_output=True, text=True, check=True).stdout
    res, cur = [], None
    for line in out.splitlines():
        if line.startswith('@@'):
            if cur:
                res.append(cur)
            cur = {'hdr': line, 'add': 0, 'del': 0}
        elif cur is not None:
            if line.startswith('+') and not line.startswith('+++'):
                cur['add'] += 1
            elif line.startswith('-') and not line.startswith('---'):
                cur['del'] += 1
    if cur:
        res.append(cur)
    return res


def main():
    print('=' * 100)
    print('E40 DELIVERABLE 1 -- WIDTH-DEPENDENCE CLASSIFICATION OF THE SHIPPED DELTA')
    print('base %s -> HEAD' % BASE[:12])
    print('=' * 100)
    tot_a = tot_d = 0
    by_class = {c: [0, 0, 0] for c in CLASSES}      # add, del, hunks
    missing = []
    for path in FILES:
        short = path.rsplit('/', 1)[1]
        hs = hunks(path)
        fa = sum(h['add'] for h in hs)
        fd = sum(h['del'] for h in hs)
        tot_a += fa
        tot_d += fd
        print()
        print('%s   +%d/-%d   %d hunks' % (path, fa, fd, len(hs)))
        print('  %-3s %6s %-42s %-10s %-4s' % ('#', '+/-', 'what', 'class', 'hyp'))
        for i, h in enumerate(hs):
            key = (short, i)
            if key not in LABEL:
                missing.append(key)
                lbl, cls, hyp, why = h['hdr'][:40], '?', '?', ''
            else:
                lbl, cls, hyp, why = LABEL[key]
            print('  %-3d %6s %-42s %-10s %-4s'
                  % (i, '+%d/-%d' % (h['add'], h['del']), lbl[:42], cls, hyp))
            if cls in by_class:
                by_class[cls][0] += h['add']
                by_class[cls][1] += h['del']
                by_class[cls][2] += 1

    print()
    print('=' * 100)
    print('ARITHMETIC CHECK')
    print('=' * 100)
    print('  hunk sums:   +%d/-%d' % (tot_a, tot_d))
    print('  gate totals: +229/-74')
    print('  match: %s' % ((tot_a, tot_d) == (229, 74)))
    print('  every hunk labelled: %s%s'
          % (not missing, '' if not missing else '  MISSING %s' % missing))

    print()
    print('=' * 100)
    print('ELIMINATION LADDER')
    print('=' * 100)
    print('  %-10s %8s %8s %7s   %s' % ('class', 'added', 'removed', 'hunks', 'verdict'))
    for c in CLASSES:
        a, d, n = by_class[c]
        verdict = ('CANDIDATE for a width-confined deficit'
                   if c in CAN_BE_WIDTH_CONFINED else 'ELIMINATED by scaling class')
        print('  %-10s %8d %8d %7d   %s' % (c, a, d, n, verdict))
    elim_a = sum(by_class[c][0] for c in CLASSES if c not in CAN_BE_WIDTH_CONFINED)
    elim_d = sum(by_class[c][1] for c in CLASSES if c not in CAN_BE_WIDTH_CONFINED)
    keep_a = sum(by_class[c][0] for c in CAN_BE_WIDTH_CONFINED)
    keep_d = sum(by_class[c][1] for c in CAN_BE_WIDTH_CONFINED)
    print()
    print('  step 1  scaling class eliminates      %3d added / %2d removed lines'
          % (elim_a, elim_d))
    print('          leaving                       %3d added / %2d removed lines'
          % (keep_a, keep_d))
    print('          i.e. %.1f %% of the +%d added lines cannot produce the effect'
          % (100.0 * elim_a / tot_a, tot_a))
    m_a = sum(by_class[c][0] for c in MAGNITUDE_KILLED)
    m_d = sum(by_class[c][1] for c in MAGNITUDE_KILLED)
    print()
    print('  step 2  magnitude budget (research/e40_overhead_budget.py) kills the')
    print('          per-row trace guard: it would need 3305 ns per branch on')
    print('          beagle against a ~3 ns predicted-not-taken test, short by')
    print('          ~10^3.  That removes a further %d added / %d removed lines.'
          % (SESSION_TRACE_WALK[0], SESSION_TRACE_WALK[1]))
    fin_a = keep_a - SESSION_TRACE_WALK[0]
    fin_d = keep_d - SESSION_TRACE_WALK[1]
    print('          leaving                       %3d added / %2d removed lines'
          % (fin_a, fin_d))
    print()
    print('  step 3  TWO mechanisms survive both filters:')
    print('            +4/-4   quantized.h + .cpp  case 5 / case 9  IPG 3 -> 5')
    print('                    (H1, per-row, cost INSIDE the timed window)')
    print('            +76/-0  Qwen36MTPBlockSession wireResidentWeightsIfEnabled,')
    print('                    specifically its Memory.clearCache()')
    print('                    (H3, per-width, cost OUTSIDE the timed window but')
    print('                     its EFFECT inside it -- RETIRED FAMILY)')
    print()
    print('  So of +229/-74 shipped lines, exactly %d added / %d removed can'
          % (fin_a, fin_d))
    print('  produce a width-confined deficit, and only +4/-4 of those act')
    print('  inside a scored round.  %.1f %% of the added lines are eliminated.'
          % (100.0 * (tot_a - fin_a) / tot_a))


if __name__ == '__main__':
    main()
