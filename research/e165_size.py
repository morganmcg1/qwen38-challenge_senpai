#!/usr/bin/env python3
"""E165: size the head-chain prefetch against the advisor's LAW 377 channels.

LAW 377 says a local measurement has no ranked value until its channel is
named. This script names the channel for every term it prices and refuses to
sum terms from different channels.
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
census = json.loads((REPO / "research" / "e165-census.json").read_text())
legs = {leg["tag"]: leg for leg in census["legs"]}

# LAW 377 transfer factors, ranked ms per 1 ms local, from advisor F2.
RANKED_PER_LOCAL = {
    "weight_stream_pass": 0.00,
    "marginal_verified_row": 0.69,   # midpoint of 0.62 - 0.76
    "per_round_fixed": 0.34,
}
RANKED_DEFICIT_MS = 0.085          # per-round deficit vs crown, ranked


def mean(tags, key):
    return sum(legs[t]["median"][key] for t in tags) / len(tags)


D4 = ["e165c4", "e165c4b"]
rp4 = mean(D4, "round_period")

print("=== replicate noise, d=4 shipped, harness=local ===")
a = legs["e165c4"]["mtp_seconds_per_token"]
b = legs["e165c4b"]["mtp_seconds_per_token"]
print(f"  s/tok        {a:.9f} vs {b:.9f}   spread {abs(a - b) / ((a + b) / 2) * 100:.4f} %")
ra = legs["e165c4"]["median"]["round_period"]
rb = legs["e165c4b"]["median"]["round_period"]
print(f"  round_period {ra:.1f} vs {rb:.1f} us  spread {abs(ra - rb) / rp4 * 100:.4f} %")

print("\n=== gpu_idle_window vs pinned width, harness=local ===")
for tag, d in (("e165c1", 1), ("e165c4", 4), ("e165c4b", 4), ("e165c7", 7)):
    w = legs[tag]["median"]["gpu_idle_window"]
    r = legs[tag]["median"]["round_period"]
    print(f"  d={d}  {tag:8s} idle_window {w:7.1f} us   round {r:9.1f} us   {w / r * 100:.3f} %")
fit = census.get("fit", {})
if "gpu_idle_window" in fit:
    f = fit["gpu_idle_window"]
    print(f"  fit: intercept {f['intercept_us']:.1f} us, slope {f['per_draft_us']:.1f} us/draft")

print("\n=== the idle tail at d=4, by component (channel: per_round_fixed) ===")
tail = ["readout", "commit", "upkeep_net", "protocol_gap",
        "host_draft_pre", "host_head1_build"]
total = 0.0
for k in tail:
    v = mean(D4, k)
    total += v
    print(f"  {k:20s} {v:8.1f} us")
print(f"  {'SUM':20s} {total:8.1f} us   = {total / rp4 * 100:.3f} % of round")

head_build = mean(D4, "host_head1_build")
recoverable = total - head_build
print("\n=== what the prefetch can move ===")
print(f"  host_head1_build stays in the tail   {head_build:8.1f} us")
print(f"  recoverable idle                     {recoverable:8.1f} us"
      f"   = {recoverable / rp4 * 100:.3f} % of round")
print(f"  minimum useful effect, 0.35 % of leg {rp4 * 0.0035:8.1f} us")
print(f"  measured replicate noise             {abs(ra - rb):8.1f} us")

print("\n=== ranked price, channel = per_round_fixed (LAW 377) ===")
factor = RANKED_PER_LOCAL["per_round_fixed"]
ranked = recoverable / 1000.0 * factor
print(f"  transfer factor                      {factor:8.2f} ranked ms per local ms")
print(f"  recoverable, ranked                  {ranked:8.4f} ms/round")
print(f"  per-round deficit vs crown, ranked   {RANKED_DEFICIT_MS:8.4f} ms/round")
print(f"  ratio                                {ranked / RANKED_DEFICIT_MS:8.2f} x the deficit")

print("\n=== NOT priced: channel unnamed ===")
print("  The hardware residency counter puts total GPU idle near 4.33 % at")
print("  d=4, against a host-visible 1.24 %. The difference is an intra-queue")
print("  bubble. LAW 377 forbids pricing it until its channel is named, so it")
print("  is reported and left out of every total above.")
