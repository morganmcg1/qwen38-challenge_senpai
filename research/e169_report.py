"""Print the E169 decomposition tables that go into the PR report."""

import json
from pathlib import Path

REPORT = json.loads(Path("research/out/e169/decomposition.json").read_text())


def ranked():
    rr = REPORT["ranked_roofline"]
    print("== ranked roofline (harness=ranked) ==")
    print(f"B_w = {rr['weight_gb_per_forward']:.5f} GB per target forward")
    print(f"roofs: {rr['bandwidth_roof_gb_per_s']} GB/s, "
          f"{rr['compute_roof_tflop_per_s']} TFLOP/s")
    print(f"{'round':<20}{'ms':>9}{'GB/s':>9}{'%BW':>8}{'TFLOP/s':>9}{'%CMP':>7}")
    for name, row in rr["rounds"].items():
        print(f"{name:<20}{row['seconds']*1e3:>9.2f}"
              f"{row['weight_stream_gb_per_s']:>9.1f}"
              f"{100*row['frac_of_bandwidth_roof']:>8.1f}"
              f"{row['tflop_per_s']:>9.2f}"
              f"{100*row['frac_of_compute_roof']:>7.1f}")
    print(f"stream floor at roof     : "
          f"{rr['stream_floor_seconds_at_roof']*1e3:.2f} ms")
    print(f"fixed term of ranked law : {rr['fixed_term_seconds']*1e3:.4f} ms")
    print(f"GB/s needed if the whole stream sat in the fixed term: "
          f"{rr['implied_gb_per_s_if_stream_inside_fixed_term']:.1f}")
    print(f"  fits under the roof? {rr['stream_fits_inside_fixed_term']}")


def stream_arith():
    sa = REPORT["stream_vs_arithmetic"]
    print("\n== stream against arithmetic (harness=local, g16s) ==")
    print(sa["model"])
    print(f"alpha = {sa['alpha_us_per_mb_per_weight_pass']:.4f} us/MB/pass"
          f"  -> {sa['weight_stream_gb_per_s']:.1f} GB/s")
    print(f"beta  = {sa['beta_us_per_mb_per_row']:.4f} us/MB/row"
          f"  -> {sa['arithmetic_tflop_per_s']:.3f} TFLOP/s")
    print(f"max residual {100*sa['max_abs_residual_frac']:.2f} %")
    print(f"{'M':>3}{'P':>3}{'meas':>9}{'pred':>9}{'resid%':>8}")
    for f in sa["fit"]:
        print(f"{f['m']:>3}{f['active_input_groups']:>3}"
              f"{f['measured_us_per_mb']:>9.4f}{f['predicted_us_per_mb']:>9.4f}"
              f"{100*f['residual_frac']:>8.2f}")


def arithmetic():
    ae = REPORT["arithmetic_efficiency"]
    print("\n== routed-kernel arithmetic rate, fixed call cost removed ==")
    print(f"{'M':>3}{'NA':>4}{'TFLOP/s':>10}{'spread%':>9}{'%localroof':>12}")
    for width, row in ae.items():
        print(f"{width:>3}{str(row['inputs_per_group']):>4}"
              f"{row['mean_tflop_per_s']:>10.3f}"
              f"{100*row['spread_frac']:>9.2f}"
              f"{100*row['frac_of_local_compute_roof']:>12.1f}")


def census_marginals():
    curve = REPORT["in_situ"]["arms"]["baseline"]["seconds_by_width"]
    print("\n== census baseline target forward, per-width marginal ==")
    prev = None
    for width in sorted(curve, key=int):
        ms = curve[width] * 1e3
        step = "" if prev is None else f"{ms - prev:+8.3f}"
        print(f"  M={width}  {ms:9.3f} ms  {step}")
        prev = ms


def passes():
    pm = REPORT["in_situ"]["pass_cost_model"]
    print("\n== per-weight-pass cost model (harness=local, g16s) ==")
    print(f"{pm['model']},  fixed = {pm['fixed_ms']:.3f} ms,"
          f"  max residual {100*pm['max_abs_residual_frac']:.2f} %")
    print(f"{'NA':>4}{'pass ms':>10}{'GB/s':>9}{'%226roof':>10}"
          f"{'%273spec':>10}{'ms/row':>9}")
    for na, row in pm["passes"].items():
        print(f"{na:>4}{row['pass_ms']:>10.3f}{row['gb_per_s']:>9.1f}"
              f"{100*row['frac_of_local_measured_roof']:>10.1f}"
              f"{100*row['frac_of_local_spec_roof']:>10.1f}"
              f"{row['ms_per_row_in_pass']:>9.3f}")
    print("  marginal row inside one pass:",
          {k: round(v, 3) for k, v in
           pm["marginal_row_inside_a_pass_ms"].items()})
    print(f"{'M':>3}{'partition':>14}{'meas':>10}{'pred':>10}{'resid%':>8}")
    for f in pm["fit"]:
        print(f"{f['m']:>3}{str(f['partition']):>14}{f['measured_ms']:>10.3f}"
              f"{f['predicted_ms']:>10.3f}{100*f['residual_frac']:>8.2f}")


ranked()
stream_arith()
arithmetic()
census_marginals()
passes()
