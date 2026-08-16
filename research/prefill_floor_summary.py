#!/usr/bin/env python3
"""Summarize research/prefill_floor.py output into the irreducibility bounds.

Answers the stop-rule question directly: what fraction of the measured seed
prefill wall P is quantized-GEMM throughput, and what is the largest gain any
implementation could extract without changing the arithmetic?
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--tag", default="part-a-base-e20268e9")
    args = ap.parse_args()

    d = json.load(open(args.floor))
    P = d["measured_prefill_seconds"]
    modelled = d["modelled_prefill_seconds"]
    comps = d["components"]
    ceiling = next(c for c in comps if c["component"].startswith("ceiling:"))["tflops_achieved"]

    live = [c for c in comps if c["calls_per_prefill"] > 0]
    gemm = [c for c in live if c["tflops_achieved"] > 5.0]
    gemm_seconds = sum(c["total_seconds"] for c in gemm)
    gemm_macs = sum(c["macs_per_call"] * c["calls_per_prefill"] for c in gemm)
    gemm_tflops = 2.0 * gemm_macs / gemm_seconds / 1e12
    gemm_at_ceiling = 2.0 * gemm_macs / (ceiling * 1e12)

    def pick(frag):
        return next(c for c in live if frag in c["component"])

    out = {
        "measured_prefill_seconds": P,
        "modelled_prefill_seconds": modelled,
        "model_over_prediction_fraction": modelled / P - 1.0,
        "dense_bf16_ceiling_tflops": ceiling,
        "quantized_gemm_seconds": gemm_seconds,
        "quantized_gemm_fraction_of_P": gemm_seconds / P,
        "quantized_gemm_tflops": gemm_tflops,
        "quantized_gemm_efficiency_vs_dense_ceiling": gemm_tflops / ceiling,
        "quantized_gemm_seconds_at_dense_ceiling": gemm_at_ceiling,
        "max_extractable_seconds_free_dequant": gemm_seconds - gemm_at_ceiling,
        "max_extractable_fraction_of_P": (gemm_seconds - gemm_at_ceiling) / P,
        "gated_delta_recurrence_fraction_of_P": pick("gated_delta")["total_seconds"] / P,
        "final_norm_512_rows_seconds": pick("final_norm")["total_seconds"],
        "final_norm_512_rows_fraction_of_P": pick("final_norm")["total_seconds"] / P,
        "lm_head_single_row_fraction_of_P": pick("lm_head")["total_seconds"] / P,
        "component_share_of_P": {
            c["component"]: c["total_seconds"] / P for c in live
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"measured P                       = {P:.4f} s")
    print(f"modelled (sum of isolated medians) = {modelled:.4f} s  ({100*(modelled/P-1):+.1f}% vs measured)")
    print(f"dense bf16 GEMM ceiling          = {ceiling:.3f} TFLOP/s\n")
    for c in sorted(live, key=lambda c: -c["total_seconds"]):
        print(
            f"  {c['component']:42s} {c['total_seconds']:7.4f} s "
            f"{100*c['total_seconds']/P:6.2f}% of P  {c['tflops_achieved']:6.3f} TFLOP/s"
        )
    print()
    print(f"quantized GEMM total             = {gemm_seconds:.4f} s = {100*gemm_seconds/P:.1f}% of P")
    print(f"  achieved                       = {gemm_tflops:.3f} TFLOP/s "
          f"= {100*gemm_tflops/ceiling:.1f}% of dense bf16 ceiling")
    print(f"  time if dequant were free      = {gemm_at_ceiling:.4f} s")
    print(f"  MAX extractable from prefill   = {gemm_seconds-gemm_at_ceiling:.4f} s "
          f"= {100*(gemm_seconds-gemm_at_ceiling)/P:.2f}% of P")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=f"prefill-floor-{args.tag}",
            job_type="analysis",
            group="qwen38-r1-e3-seed-prefill-amdahl",
            config={"host": d["host"], "seed_tokens": d["seed_tokens"]},
        )
        flat = {f"floor/{k}": v for k, v in out.items() if isinstance(v, (int, float))}
        flat |= {f"floor/share/{k}": v for k, v in out["component_share_of_P"].items()}
        run.log(flat)
        run.summary.update(flat)
        table = wandb.Table(
            columns=["component", "per_call_seconds", "calls", "total_seconds", "tflops"]
        )
        for c in comps:
            table.add_data(
                c["component"],
                c["per_call_seconds"],
                c["calls_per_prefill"],
                c["total_seconds"],
                c["tflops_achieved"],
            )
        run.log({"floor/components": table})
        print(f"WANDB_RUN_URL {run.url}", file=sys.stderr)
        print(f"WANDB_RUN_ID {run.id}", file=sys.stderr)
        run.finish()


if __name__ == "__main__":
    main()
