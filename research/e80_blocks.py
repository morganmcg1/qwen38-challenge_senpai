#!/usr/bin/env python3
"""Rung 3 without a fit: `F(M) - F(1)` decomposed by command-buffer block.

Every buffer records an exact GPU interval and the exact multiset of dispatches
inside it, and every buffer belongs to exactly one width and phase. Summing
buffer intervals therefore reproduces the measured phase total exactly, with no
solver, no null space and no identifiability caveat.

The only thing needed to compare a serial round with a verify round is a key
that survives the width change. A dispatch grid carries the row count M in one
axis, so the key replaces that literal with `M`. `6x640x1` and `1x640x1` then
become the same block, `17408x6x1` and `17408x1x1` become the same block, and
the difference of the two sides is an exact, exhaustive decomposition of the
width tax.

Blocks that exist on one side only are reported as such. That is a real result:
it is where MLX changes its plan with M, not a gap in the measurement.
"""
import argparse
import json
import sys
import pathlib
import collections
import re

sys.path.insert(0, "/tmp")
import e80_census_report as R

SHORT = (
    ("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0", "qmv"),
    ("custom_kernel_qwen35_fused_residual_rms_norm_bfloat16_t_bfloat16_t_"
     "bfloat16_t_floats_bfloat16_t_bfloat16_t", "residual_rms_norm"),
    ("custom_kernel_qwen35_attention_qk_rms_rope_bf16_v1", "qk_rms_rope"),
    ("custom_kernel_qwen35_packed_gdn_prework", "gdn_prework"),
    ("custom_kernel_gated_delta_step", "gated_delta_step"),
    ("custom_kernel_qwen_mtp_linear_top2_partial", "top2_partial"),
    ("custom_kernel_qwen_mtp_linear_top2_finalize", "top2_finalize"),
    ("CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB", "swiglu_fusion"),
    ("CV2ISigmoidBDV2IBroadcastACEV2IBroadcastCAFV2OMultiplyDE", "attn_gate"),
    ("Cf4IAsTypeADf4ISigmoidCEf4IBroadcastCDFf4IBroadcastDCG", "gdn_gate"),
    ("Bf4ISigmoidACf4IBroadcastABDf4IBroadcastBAEf4OMultip", "gdn_gate_m1"),
    ("BV2ISigmoidACV2IBroadcastABDV2IBroadcastBAEV2OMultip", "gdn_gate_m1b"),
    ("affine_dequantize_bfloat16_t_gs_64_b_4", "dequant"),
    ("gemv_al_bfloat16_bm8", "gemv_bm8"),
    ("gemv_al_bfloat16_bm4", "gemv_bm4"),
    ("steel_gemm_splitk_accum", "steel_splitk_accum"),
    ("steel_gemm_splitk_nt", "steel_splitk_nt"),
    ("steel_gemm_fused_nt", "steel_fused_nt"),
    ("custom_kernel_qwen_mtp_draft_select", "draft_select"),
    ("rope_single_bfloat16", "rope_single"),
    ("rope_bfloat16", "rope"),
    ("sdpa_vector_2pass_1", "sdpa_2pass_1"),
    ("sdpa_vector_2pass_2", "sdpa_2pass_2"),
    ("sdpa_vector_bfloat16_t_256_256_nomask_qnt_c_nosinks", "sdpa_c"),
    ("sdpa_vector_bfloat16_t_256_256_nomask_qnt_nc_nosinks", "sdpa_nc"),
    ("depthwise_conv_1d_bfloat16", "depthwise_conv1d"),
)

QMV_OUT = {640: "5120", 4352: "34816", 2060: "16480", 1792: "14336",
           31040: "248320"}


def shorten(kernel):
    for long, short in SHORT:
        if kernel.startswith(long):
            return short
    return kernel[:34]


def learn_axis_rules(leg):
    """Which grid axes scale with M, learned from the widths in the data.

    At M = 1 a row axis is indistinguishable from a size-1 axis, so the rule
    cannot be read off a serial round. It can be read off the verify rounds,
    where two or more distinct widths pin every axis: an axis is `c*M` only if
    the same `c` explains every observed width, and `24` cannot be `c*M` for
    both M = 5 and M = 6. The learned rule is then applied to M = 1.
    """
    seen = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (width, phase), counts in leg.shape_dispatches.items():
        if phase != "target_verify" or width <= 1:
            continue
        for shape in counts:
            parsed = R.parse_shape(shape)
            if not parsed or not parsed["grid"]:
                continue
            for axis, value in enumerate(parsed["grid"]):
                seen[parsed["kernel"]][axis][width] = value
    rules = collections.defaultdict(dict)
    for kernel, axes in seen.items():
        for axis, obs in axes.items():
            if len(obs) < 2:
                continue
            for label, f in (("M", lambda m: m), ("D", lambda m: m - 1)):
                cs = {v / f(m) for m, v in obs.items() if f(m)}
                if len(cs) == 1:
                    c = cs.pop()
                    if c == int(c) and c >= 1:
                        rules[kernel][axis] = (label, int(c))
                        break
    return rules


def norm_shape(shape, m, rules):
    parsed = R.parse_shape(shape)
    if not parsed:
        return shorten(shape)
    kernel = parsed["kernel"]
    axes = []
    for axis, value in enumerate(parsed["grid"] or ()):
        rule = rules.get(kernel, {}).get(axis)
        if rule is None:
            axes.append(str(value))
            continue
        label, c = rule
        span = m if label == "M" else m - 1
        if value == c * span:
            axes.append(label if c == 1 else f"{c}{label}")
        else:
            axes.append(str(value))
    return f"{shorten(kernel)} {'x'.join(axes)}"


def block_key(counts, m, rules):
    parts = sorted(f"{c}x {norm_shape(s, m, rules)}" for s, c in counts.items())
    return " + ".join(parts)


def collect(leg, phase, width, rules):
    rounds = leg.round_count(width, phase) or 1
    acc = collections.defaultdict(lambda: [0, 0])
    total = 0
    for counts, gpu_ns, n in leg.signature_rows(phase, {width}):
        key = block_key(counts, width, rules)
        slot = acc[key]
        slot[0] += n
        slot[1] += gpu_ns
        total += gpu_ns
    return {k: (v[0] / rounds, v[1] / rounds / 1e6) for k, v in acc.items()}, \
        total / rounds / 1e6, rounds


# Kernels that can dominate a command buffer, most expensive first. The list
# only decides which dispatch owns a buffer's interval.
PRIORITY = ("qmv", "steel_splitk_nt", "steel_fused_nt", "steel_splitk_accum",
            "gemv_bm8", "gemv_bm4", "gdn_prework", "gated_delta_step",
            "sdpa_c", "sdpa_nc", "sdpa_2pass_1", "sdpa_2pass_2",
            "draft_select", "top2_partial", "swiglu_fusion", "qk_rms_rope",
            "rope", "rope_single", "depthwise_conv1d", "attn_gate",
            "gdn_gate", "residual_rms_norm", "dequant")


def owner_of(counts, m, rules):
    """The one dispatch a buffer's GPU interval is charged to.

    Charging the whole interval to one dispatch is only honest if the interval
    does not move when the other dispatches in the buffer change. The
    validation table reports exactly that spread, so the assumption is checked
    rather than assumed.
    """
    named = [(norm_shape(s, m, rules), c) for s, c in counts.items()]
    for want in PRIORITY:
        hits = [n for n, _ in named if n.split(" ")[0] == want]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return "AMBIGUOUS: " + " + ".join(sorted(hits))
    return " + ".join(sorted(n for n, _ in named))


def attribute(leg, phase, width, rules):
    rounds = leg.round_count(width, phase) or 1
    acc = collections.defaultdict(lambda: {"buffers": 0, "gpu_ns": 0,
                                           "spans": [], "disp": 0,
                                           "kdisp": collections.Counter()})
    for counts, gpu_ns, n in leg.signature_rows(phase, {width}):
        slot = acc[owner_of(counts, width, rules)]
        slot["buffers"] += n
        slot["gpu_ns"] += gpu_ns
        slot["spans"].append((gpu_ns / n, n))
        for shape, c in counts.items():
            slot["disp"] += c * n
            slot["kdisp"][norm_shape(shape, width, rules)] += c * n
    return acc, rounds


# Ranked draft-width histogram from the assignment. Local legs force one width
# each, so the ranked mix is applied as weights over the measured widths and
# the covered mass is disclosed instead of interpolating the missing widths.
RANKED = {3: 3.25, 4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}

# family of an owning dispatch, for the rider verdicts
FAMILY = {
    "qmv": "qmv", "gdn_prework": "gdn_recurrence",
    "gated_delta_step": "gdn_recurrence", "sdpa_c": "sdpa", "sdpa_nc": "sdpa",
    "sdpa_2pass_1": "sdpa", "sdpa_2pass_2": "sdpa",
    "top2_partial": "top2_readout", "top2_finalize": "top2_readout",
    "swiglu_fusion": "compiled_fusion", "attn_gate": "compiled_fusion",
    "gdn_gate": "compiled_fusion", "gdn_gate_m1": "compiled_fusion",
    "gdn_gate_m1b": "compiled_fusion", "qk_rms_rope": "norm",
    "residual_rms_norm": "norm", "rmsbfloat16": "norm",
    "rms_loopedbfloat16": "norm", "depthwise_conv1d": "depthwise_conv",
    "dequant": "quant_dequant",
    # proposal-head path
    "gemv_bm8": "gemv", "gemv_bm4": "gemv",
    "steel_splitk_nt": "steel_gemm", "steel_splitk_accum": "steel_gemm",
    "steel_fused_nt": "steel_gemm", "draft_select": "draft_select",
    "rope": "rope", "rope_single": "rope",
}


# MLX names a compiled elementwise kernel after its operand kinds: `v` vector,
# `s` scalar, so `vv_Addbfloat16`, `sv_Multiplybfloat16` and `v_Expfloat32` are
# all elementwise. Anything that matches none of the rules below is genuinely
# unclassified and must be reported as such.
ELEMENTWISE = re.compile(r"^(v|s|vv|vs|sv|ss)_[A-Z]")


def family_of_kernel(head):
    """Family of one short kernel name, or `UNCLASSIFIED`."""
    name = head.split(" ")[0]
    if name in FAMILY:
        return FAMILY[name]
    if "copy" in name:
        return "copy"
    if name.startswith("gather"):
        return "gather"
    if ELEMENTWISE.match(name):
        return "elementwise"
    return "UNCLASSIFIED"


def family_of_owner(key):
    """A buffer owned by several small kernels is charged to the heaviest."""
    fams = {family_of_kernel(part.strip()) for part in key.split("+")}
    for pref in ("qmv", "steel_gemm", "gemv", "sdpa", "gdn_recurrence",
                 "draft_select", "rope", "compiled_fusion", "norm",
                 "top2_readout", "depthwise_conv", "quant_dequant", "gather",
                 "copy", "elementwise"):
        if pref in fams:
            return pref
    return "UNCLASSIFIED"


def riders(acc, rounds, total):
    fam = collections.Counter()
    for key, v in acc.items():
        fam[family_of_owner(key)] += v["gpu_ns"] / rounds / 1e6
    checks = [
        ("copy", "<= 1 % or ledger 218 reopens; ~0.02 % expected", 1.0),
        ("elementwise", "unary/binary/ternary_ops < 3 %", 3.0),
        ("norm", "rms_norm < 3 %", 3.0),
        ("sdpa", "sdpa_vector < 3 %", 3.0),
        ("gemv", "gemv < 2 %", 2.0),
    ]
    print("\n### riders\n")
    print("| family | rider | ms/round | share | verdict |")
    print("|---|---|---:|---:|---|")
    for name, text, limit in checks:
        ms = fam.get(name, 0.0)
        share = 100 * ms / total
        print(f"| {name} | {text} | {ms:.3f} | {share:.2f}% | "
              f"{'PASS' if share <= limit else 'FAIL'} |")
    qmv = fam.get("qmv", 0.0)
    print(f"| qmv | five linear families dominate | {qmv:.3f} | "
          f"{100*qmv/total:.2f}% | "
          f"{'PASS' if 100*qmv/total > 50 else 'FAIL'} |")
    rest = sorted(((v, k) for k, v in fam.items() if k != "qmv"), reverse=True)
    rest_txt = ", ".join(f"{k} {v:.3f}" for v, k in rest if v >= 0.02)
    print(f"\nremainder outside qmv, {total-qmv:.3f} ms: {rest_txt}. No "
          f"further matrix-multiply family appears, so the remainder is not a "
          f"sixth linear family.")
    return dict(fam)


def kernel_census(leg, rules, acc, rounds, total, width, phase):
    """Per-kernel and per-family census with exact dispatch counts.

    A dispatch count is recorded at encode time and is exact. The GPU time is
    the attributed buffer interval, so a kernel that never owns a buffer shows
    its true dispatch count against no attributed time, and the owner that
    hides it is named. That is the honest form: the census must not invent a
    number for a kernel whose cost the hardware never exposed on its own.
    """
    disp = collections.Counter()
    for shape, n in leg.shape_dispatches[(width, phase)].items():
        disp[norm_shape(shape, width, rules)] += n
    owned = {p.strip() for key in acc for p in key.split("+")}

    print(f"\n### per-kernel census, w{width} {phase}, {rounds} rounds, "
          f"{total:.3f} ms/round\n")
    print("| kernel x grid | disp/round | co-dispatches in the same buffers | "
          "GPU ms/round | share of round | note |")
    print("|---|---:|---:|---:|---:|---|")
    rows = []
    for key, v in acc.items():
        own = sum(v["kdisp"].get(p.strip(), 0) for p in key.split("+"))
        rows.append((v["gpu_ns"] / rounds / 1e6, key, own / rounds,
                     (v["disp"] - own) / rounds, "owns its command buffer"))
    for key, n in disp.items():
        if key in owned:
            continue
        rows.append((0.0, key, n / rounds, 0.0,
                     "never owns a buffer; its time is inside an owner above"))
    for ms, key, n, co, note in sorted(rows, reverse=True):
        if ms < 0.02 and n < 0.5:
            continue
        print(f"| {key} | {n:6.1f} | {co:6.1f} | {ms:7.3f} | "
              f"{100*ms/total:5.2f}% | {note} |")

    fam_ms = collections.Counter()
    fam_disp = collections.Counter()
    for key, v in acc.items():
        fam_ms[family_of_owner(key)] += v["gpu_ns"] / rounds / 1e6
        for head, n in v["kdisp"].items():
            fam_disp[family_of_kernel(head)] += n
    hidden = sum(n for key, n in disp.items() if key not in owned)
    print("\n| family | disp/round | GPU ms/round | share of round |")
    print("|---|---:|---:|---:|")
    for f in sorted(set(fam_ms) | set(fam_disp),
                    key=lambda f: -fam_ms.get(f, 0.0)):
        print(f"| {f} | {fam_disp[f]/rounds:6.1f} | {fam_ms[f]:7.3f} | "
              f"{100*fam_ms[f]/total:5.2f}% |")
    print(f"| **total** | {sum(fam_disp.values())/rounds:6.1f} | "
          f"{total:7.3f} | 100.00% |")
    unclassified = {h: n for h, n in
                    ((h, sum(v["kdisp"].get(h, 0) for v in acc.values()))
                     for h in {p for v in acc.values() for p in v["kdisp"]})
                    if family_of_kernel(h) == "UNCLASSIFIED"}
    if unclassified:
        print("\n`unclassified_kernels`: "
              + ", ".join(f"`{k}` ({n/rounds:.1f}/round)"
                          for k, n in sorted(unclassified.items(),
                                             key=lambda kv: -kv[1])))
    else:
        print("\n`unclassified_kernels`: 0. Every dispatched kernel above "
              "belongs to a named family.")
    print(f"\n{hidden/rounds:.1f} dispatches per round never own a command "
          f"buffer, so their GPU time is reported inside the owner that "
          f"shares the buffer with them instead of being invented. The "
          f"dispatch counts come from the encoder and are exact; the GPU time "
          f"is the measured interval of the command buffer the kernel owns.")
    return dict(fam_ms)


def report_attribution(leg, rules, width, phase):
    acc, rounds = attribute(leg, phase, width, rules)
    total = sum(v["gpu_ns"] for v in acc.values()) / rounds / 1e6
    print(f"\n### dominant-dispatch attribution, w{width} {phase}, "
          f"{rounds} rounds, {total:.3f} ms/round\n")
    print("| owning dispatch | buffers/round | mean ns | ms/round | share | "
          "span spread over partners |")
    print("|---|---:|---:|---:|---:|---|")
    for name, v in sorted(acc.items(), key=lambda kv: -kv[1]["gpu_ns"]):
        ms = v["gpu_ns"] / rounds / 1e6
        if ms < 0.02:
            continue
        spans = sorted(v["spans"], key=lambda s: -s[1])[:4]
        spread = ", ".join(f"{int(s):,}({n})" for s, n in spans)
        print(f"| {name} | {v['buffers']/rounds:5.1f} | "
              f"{v['gpu_ns']/v['buffers']:,.0f} | {ms:7.3f} | "
              f"{100*ms/total:5.2f}% | {spread} |")
    return acc, rounds, total


def merge_acc(a, b):
    """Sums two attribution tables that cover the same rounds.

    A drafting round runs `draft_head` and then `target_verify`, so the round's
    cost is the sum of the two phases. The riders ask for a share of the round,
    not a share of one phase: `gemv` lives only on the head path and would read
    as exactly zero against the verify phase alone.
    """
    out = collections.defaultdict(lambda: {"buffers": 0, "gpu_ns": 0,
                                           "spans": [], "disp": 0,
                                           "kdisp": collections.Counter()})
    for src in (a, b):
        for key, v in src.items():
            slot = out[key]
            slot["buffers"] += v["buffers"]
            slot["gpu_ns"] += v["gpu_ns"]
            slot["spans"].extend(v["spans"])
            slot["disp"] += v["disp"]
            slot["kdisp"].update(v["kdisp"])
    return out


def verify_widths(leg, min_rounds):
    return sorted(w for (w, ph), n in leg.rounds.items()
                  if ph == "target_verify" and n >= min_rounds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isolated", action="append", default=[])
    ap.add_argument("--default", action="append", default=[])
    # A forced-width leg still emits a few rounds at a smaller width near the
    # end of generation. Those samples are too few to mean anything, so a
    # width needs a real sample before it gets a table.
    ap.add_argument("--min-rounds", type=int, default=20)
    ap.add_argument("--json", help="write the full census to this path")
    args = ap.parse_args()
    iso_leg = R.Leg([pathlib.Path(p) for p in args.isolated]) \
        if args.isolated else None
    default_leg = R.Leg([pathlib.Path(p) for p in args.default]) \
        if args.default else None
    # The attribution needs small buffers to be meaningful, so it runs on the
    # isolated packing. The default packing supplies the absolute anchor and
    # the concurrency discount.
    leg = iso_leg or default_leg
    per_width = {}
    per_round = {}
    in_situ = None
    if iso_leg is not None and default_leg is not None:
        d_rules = learn_axis_rules(default_leg)
        in_situ = {}
        for w in verify_widths(default_leg, args.min_rounds):
            a, r = attribute(default_leg, "target_verify", w, d_rules)
            in_situ[w] = (a, r, sum(v["gpu_ns"] for v in a.values()) / r / 1e6)
    rules = learn_axis_rules(leg)
    scaled = sum(len(v) for v in rules.values())
    print(f"axis rules learned from the verify widths: {scaled} axes across "
          f"{len(rules)} kernels scale with M or with the draft count M-1.")
    lo, lo_ms, lo_n = collect(leg, "target_forward", 1, rules)
    for width in verify_widths(leg, args.min_rounds):
        hi, hi_ms, hi_n = collect(leg, "target_verify", width, rules)
        tax = hi_ms - lo_ms
        print(f"\n## fit-free block decomposition of F({width}) - F(1)\n")
        print(f"F(1) = {lo_ms:.3f} ms/round over {lo_n} rounds; "
              f"F({width}) = {hi_ms:.3f} ms/round over {hi_n} rounds; "
              f"tax = **{tax:.3f} ms/round**. Every number below is a sum of "
              f"measured command-buffer intervals.\n")
        print(f"| block | buf/round M=1 | buf/round M={width} | F(1) ms | "
              f"F({width}) ms | tax ms | share of tax |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        keys = sorted(set(lo) | set(hi),
                      key=lambda k: -(hi.get(k, (0, 0))[1] - lo.get(k, (0, 0))[1]))
        acc = 0.0
        for k in keys:
            a_n, a_ms = lo.get(k, (0.0, 0.0))
            b_n, b_ms = hi.get(k, (0.0, 0.0))
            d = b_ms - a_ms
            if abs(d) < 0.02 and a_ms < 0.05 and b_ms < 0.05:
                continue
            acc += d
            print(f"| {k} | {a_n:.1f} | {b_n:.1f} | {a_ms:7.3f} | "
                  f"{b_ms:7.3f} | {d:7.3f} | {100*d/tax:6.2f}% |")
        print(f"\nrows shown sum to {acc:.3f} ms, {100*acc/tax:.2f} % of the "
              f"tax; the omitted rows are all below 0.05 ms on both sides.")

    lo_att, lo_rounds, lo_tot = report_attribution(leg, rules, 1,
                                                   "target_forward")
    kernel_census(leg, rules, lo_att, lo_rounds, lo_tot, 1, "target_forward")
    for width in verify_widths(leg, args.min_rounds):
        hi_att, hi_rounds, hi_tot = report_attribution(leg, rules, width,
                                                       "target_verify")
        tax = hi_tot - lo_tot
        print(f"\n### rung 3 by owning dispatch: F({width}) - F(1) = "
              f"{tax:.3f} ms/round\n")
        print(f"| owning dispatch | disp/round M=1 | disp/round M={width} | "
              f"ns M=1 | ns M={width} | F(1) ms | F({width}) ms | tax ms | "
              f"share |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        named = 0.0
        for key in sorted(set(lo_att) | set(hi_att),
                          key=lambda k: -(hi_att.get(k, {}).get("gpu_ns", 0)
                                          / hi_rounds
                                          - lo_att.get(k, {}).get("gpu_ns", 0)
                                          / lo_rounds)):
            a = lo_att.get(key)
            b = hi_att.get(key)
            a_ms = a["gpu_ns"] / lo_rounds / 1e6 if a else 0.0
            b_ms = b["gpu_ns"] / hi_rounds / 1e6 if b else 0.0
            d = b_ms - a_ms
            if abs(d) < 0.02:
                continue
            named += d
            a_n = f"{a['buffers']/lo_rounds:.1f}" if a else "0.0"
            b_n = f"{b['buffers']/hi_rounds:.1f}" if b else "0.0"
            a_ns = f"{a['gpu_ns']/a['buffers']:,.0f}" if a else "-"
            b_ns = f"{b['gpu_ns']/b['buffers']:,.0f}" if b else "-"
            print(f"| {key} | {a_n} | {b_n} | {a_ns} | {b_ns} | {a_ms:7.3f} | "
                  f"{b_ms:7.3f} | {d:7.3f} | {100*d/tax:6.2f}% |")
        print(f"\nnamed rows sum to {named:.3f} ms, {100*named/tax:.2f} % of "
              f"the tax.")
        kernel_census(leg, rules, hi_att, hi_rounds, hi_tot, width,
                      "target_verify")
        head_att, head_rounds, head_tot = report_attribution(
            leg, rules, width, "draft_head")
        kernel_census(leg, rules, head_att, head_rounds, head_tot, width,
                      "draft_head")
        round_att = merge_acc(hi_att, head_att)
        round_tot = hi_tot + head_tot
        print(f"\n### whole drafting round, w{width} = draft_head + "
              f"target_verify = {head_tot:.3f} + {hi_tot:.3f} = "
              f"{round_tot:.3f} ms/round\n")
        print("The riders below are shares of this whole round. A rider "
              "evaluated against `target_verify` alone would read `gemv` as "
              "exactly zero, because the proposal head is the only phase that "
              "dispatches it.\n")
        riders(round_att, hi_rounds, round_tot)
        per_width[width] = (hi_att, hi_rounds, hi_tot)
        per_round[width] = (round_att, hi_rounds, round_tot)

    if in_situ:
        print("\n## concurrency discount per owning dispatch\n")
        print("The discount is the in-situ default packing divided by the "
              "one-dispatch-per-MLX-op isolated packing. Both legs run the "
              "same dispatches on the same host, so a value below one means "
              "the default packing lets that dispatch overlap other work, and "
              "a value at one means the dispatch is already serialised.\n")
        print("| width | owning dispatch | family | default ms/round | "
              "isolated ms/round | discount |")
        print("|---|---|---|---:|---:|---:|")
        for width in sorted(per_width):
            if width not in in_situ:
                continue
            i_att, i_rounds, i_tot = per_width[width]
            d_att, d_rounds, d_tot = in_situ[width]
            print(f"| {width} | ALL (phase level, no attribution) | - | "
                  f"{d_tot:.3f} | {i_tot:.3f} | {d_tot/i_tot:.3f} |")
            fam_d = collections.Counter()
            fam_i = collections.Counter()
            for key, v in d_att.items():
                fam_d[family_of_owner(key)] += v["gpu_ns"] / d_rounds / 1e6
            for key, v in i_att.items():
                fam_i[family_of_owner(key)] += v["gpu_ns"] / i_rounds / 1e6
            for fam, i_ms in fam_i.most_common():
                if i_ms < 0.05:
                    continue
                d_ms = fam_d.get(fam, 0.0)
                ratio = f"{d_ms/i_ms:.3f}" if i_ms else "-"
                print(f"| {width} | (family) | {fam} | {d_ms:.3f} | "
                      f"{i_ms:.3f} | {ratio} |")
            for key in sorted(i_att, key=lambda k: -i_att[k]["gpu_ns"]):
                a = d_att.get(key)
                b_ms = i_att[key]["gpu_ns"] / i_rounds / 1e6
                if b_ms < 0.2 or a is None:
                    continue
                a_ms = a["gpu_ns"] / d_rounds / 1e6
                print(f"| {width} | {key} | {family_of_owner(key)} | "
                      f"{a_ms:.3f} | {b_ms:.3f} | {a_ms/b_ms:.3f} |")

    if len(per_width) > 1:
        print("\n## ranked-weighted verify cost\n")
        covered = sum(RANKED[w] for w in per_width if w in RANKED)
        print(f"The ranked histogram puts {covered:.2f} % of its mass on the "
              f"widths measured here. No missing width is interpolated. The "
              f"weighted mean below is renormalised over the covered mass and "
              f"is only valid for that mass.\n")
        print("| width | ranked weight % | verify ms/round | qmv share | "
              "draft_head ms/round | whole round ms/round |")
        print("|---:|---:|---:|---:|---:|---:|")
        acc_ms = 0.0
        acc_round = 0.0
        for width in sorted(per_width):
            att, rnds, tot = per_width[width]
            w = RANKED.get(width, 0.0)
            fam = collections.Counter()
            for key, v in att.items():
                fam[family_of_owner(key)] += v["gpu_ns"] / rnds / 1e6
            acc_ms += w * tot
            r_tot = per_round[width][2] if width in per_round else tot
            acc_round += w * r_tot
            print(f"| {width} | {w:.2f} | {tot:.3f} | "
                  f"{100*fam['qmv']/tot:.2f}% | {r_tot - tot:.3f} | "
                  f"{r_tot:.3f} |")
        print(f"\nranked-weighted verify cost over the covered "
              f"{covered:.2f} % of the histogram: "
              f"**{acc_ms/covered:.3f} ms/round**; whole drafting round "
              f"**{acc_round/covered:.3f} ms/round**.")

    if args.json:
        out = {
            "method": "dominant-dispatch attribution over measured "
                      "command-buffer intervals; no fit",
            "isolated_legs": args.isolated,
            "default_legs": args.default,
            "min_rounds": args.min_rounds,
            "serial_reference": {
                "width": 1, "phase": "target_forward", "rounds": lo_rounds,
                "gpu_ms_per_round": lo_tot,
                "kernels": dump(lo_att, lo_rounds),
            },
            "verify": {},
            "whole_round": {},
            "default_leg_verify": {},
        }
        for width, (att, rnds, tot) in per_width.items():
            out["verify"][str(width)] = {
                "rounds": rnds, "gpu_ms_per_round": tot,
                "tax_vs_serial_ms": tot - lo_tot,
                "kernels": dump(att, rnds),
            }
        for width, (att, rnds, tot) in per_round.items():
            out["whole_round"][str(width)] = {
                "phases": ["draft_head", "target_verify"],
                "rounds": rnds, "gpu_ms_per_round": tot,
                "kernels": dump(att, rnds),
            }
        for width, (att, rnds, tot) in (in_situ or {}).items():
            out["default_leg_verify"][str(width)] = {
                "rounds": rnds, "gpu_ms_per_round": tot,
                "kernels": dump(att, rnds),
            }
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\ncensus written to `{args.json}`")


def dump(att, rounds):
    return [
        {
            "owner": key,
            "family": family_of_owner(key),
            "buffers_per_round": v["buffers"] / rounds,
            "dispatches_per_round": v["disp"] / rounds,
            "gpu_ms_per_round": v["gpu_ns"] / rounds / 1e6,
            "mean_buffer_ns": v["gpu_ns"] / v["buffers"],
            "buffer_ns_by_partner": sorted(
                ({"ns": s, "buffers": n} for s, n in v["spans"]),
                key=lambda d: -d["buffers"])[:8],
            "kernel_dispatches_per_round": {k: n / rounds
                                            for k, n in v["kdisp"].items()},
        }
        for key, v in sorted(att.items(), key=lambda kv: -kv[1]["gpu_ns"])
    ]


if __name__ == "__main__":
    main()
