"""E145 R6-0: price the decode state against the wired residency slack.

Zero GPU. Every byte count is read from the pinned checkpoint config on disk
and from the allocation shapes in the source that the scored session executes.
Nothing here is assumed from the advisor's brief; where a source literal and a
config-derived value both exist, both are computed and cross-checked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_GLOB = os.path.join(
    os.path.expanduser("~"),
    ".cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit"
    "/snapshots/*/config.json",
)

SESSION = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
GDN = "Sources/MLXFastModel/Qwen35GatedDelta.swift"
KVCACHE = "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift"
RESIDENT = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/resident.cpp"
ALLOCATOR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp"

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

SEED_TOKENS = 512
DECODE_TOKENS = 512
MAX_ROUND_WIDTH = 8


def read(path: str) -> str:
    with open(os.path.join(REPO, path), encoding="utf-8") as handle:
        return handle.read()


def source_int(path: str, pattern: str, label: str) -> int:
    text = read(path)
    found = re.search(pattern, text)
    if not found:
        raise SystemExit(f"{label}: pattern not found in {path}")
    return int(found.group(1))


def load_config(path: str | None) -> tuple[str, dict]:
    if path is None:
        matches = sorted(glob.glob(CONFIG_GLOB))
        if not matches:
            raise SystemExit(f"no pinned target config under {CONFIG_GLOB}")
        path = matches[0]
    with open(path, encoding="utf-8") as handle:
        return path, json.load(handle)["text_config"]


def full_attention_kv(text: dict) -> dict:
    """Bytes the 16 full-attention caches add for one token."""
    layer_types = text["layer_types"]
    full_layers = layer_types.count("full_attention")
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    elem = DTYPE_BYTES[text["dtype"]]
    per_layer = 2 * kv_heads * head_dim * elem
    return {
        "full_attention_layers": full_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "activation_dtype": text["dtype"],
        "activation_elem_bytes": elem,
        "bytes_per_token_per_layer": per_layer,
        "bytes_per_token": per_layer * full_layers,
    }


def gated_delta_state(text: dict) -> dict:
    """Bytes the 48 MambaCache entries hold once the seed has run.

    Shapes come from `Qwen35GatedDeltaSpec.convolutionStateShape` and
    `recurrentStateShape` evaluated on the pinned config. The recurrent state
    is float32 unconditionally (`Qwen35GatedDelta.swift` and `Qwen35.swift`
    both allocate it that way); the convolution state carries the activation
    dtype.
    """
    layer_types = text["layer_types"]
    linear_layers = layer_types.count("linear_attention")
    key_heads = text["linear_num_key_heads"]
    key_dim = text["linear_key_head_dim"]
    value_heads = text["linear_num_value_heads"]
    value_dim = text["linear_value_head_dim"]
    kernel = text["linear_conv_kernel_dim"]
    elem = DTYPE_BYTES[text["dtype"]]

    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    conv_shape = [1, kernel - 1, conv_dim]
    recurrent_shape = [1, value_heads, value_dim, key_dim]

    conv_per_layer = (kernel - 1) * conv_dim * elem
    recurrent_per_layer = value_heads * value_dim * key_dim * 4

    return {
        "linear_attention_layers": linear_layers,
        "convolution_dimension": conv_dim,
        "convolution_state_shape": conv_shape,
        "recurrent_state_shape": recurrent_shape,
        "convolution_dtype": text["dtype"],
        "recurrent_dtype": "float32",
        "conv_bytes_per_layer": conv_per_layer,
        "recurrent_bytes_per_layer": recurrent_per_layer,
        "conv_bytes": conv_per_layer * linear_layers,
        "recurrent_bytes": recurrent_per_layer * linear_layers,
        "bytes": (conv_per_layer + recurrent_per_layer) * linear_layers,
    }


def source_shape_crosscheck(gdn: dict) -> dict:
    """Confirm the config-derived shapes against the literals in the source."""
    qwen35 = read("Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift")
    literal = re.search(
        r"MLXArray\.zeros\(\[1, (\d+), (\d+), (\d+)\], dtype: \.float32\)",
        qwen35,
    )
    doc = re.search(
        r"\[B, (\d+), (\d+)\]` and\n///\s*`\[B, (\d+), (\d+), (\d+)\]",
        read(GDN),
    )
    result = {
        "qwen35_recurrent_literal": (
            [1] + [int(g) for g in literal.groups()] if literal else None
        ),
        "gdn_doc_conv_shape": (
            [1, int(doc.group(1)), int(doc.group(2))] if doc else None
        ),
        "gdn_doc_recurrent_shape": (
            [1, int(doc.group(3)), int(doc.group(4)), int(doc.group(5))]
            if doc
            else None
        ),
    }
    result["recurrent_matches_source"] = (
        result["qwen35_recurrent_literal"] == gdn["recurrent_state_shape"]
        and result["gdn_doc_recurrent_shape"] == gdn["recurrent_state_shape"]
    )
    result["conv_matches_source"] = (
        result["gdn_doc_conv_shape"] == gdn["convolution_state_shape"]
    )
    return result


def kv_capacity_walk(bytes_per_token: int, step: int) -> dict:
    """Replay `KVCacheSimple.update` capacity arithmetic for one leg.

    The seed is one forward of `SEED_TOKENS` rows. Each decode round writes
    `M = draftCount + 1` rows and the session rolls the offset back over the
    rejected suffix, so the committed offset walks from `SEED_TOKENS` to
    `SEED_TOKENS + DECODE_TOKENS` while every round transiently needs
    `offset + M` rows of capacity.

    The per-round accepted counts are not in any artifact this experiment
    owns, so the event count is reported as a proven interval rather than a
    point estimate. Both ends are derived, not guessed.
    """
    n_steps = (step + SEED_TOKENS - 1) // step
    seed_capacity = n_steps * step
    final_offset = SEED_TOKENS + DECODE_TOKENS

    # Lower bound: each reset raises capacity by exactly one `step`, so the
    # walk from `seed_capacity` to `final_offset` cannot cost fewer resets.
    lower = max(0, -(-(final_offset - seed_capacity) // step))

    # Upper bound: a reset at offset P leaves capacity P + step (the buffer is
    # trimmed to P first whenever P is not step-aligned), and the next reset
    # needs offset > P + step - MAX_ROUND_WIDTH. Walk that worst case.
    upper = 0
    offset = SEED_TOKENS
    capacity = seed_capacity
    while offset < final_offset:
        if offset + MAX_ROUND_WIDTH > capacity:
            upper += 1
            capacity = (
                capacity + step if offset % step == 0 else offset + step
            )
        offset = max(offset + 1, capacity - MAX_ROUND_WIDTH + 1)

    events = []
    offset = SEED_TOKENS
    capacity = seed_capacity
    while capacity < final_offset:
        old_capacity = capacity
        new_zeros = step
        capacity = capacity + step if offset % step == 0 else offset + step
        events.append(
            {
                "at_offset": offset,
                "old_capacity_tokens": old_capacity,
                "new_zeros_tokens": new_zeros,
                "new_capacity_tokens": capacity,
                "old_bytes": old_capacity * bytes_per_token,
                "new_zeros_bytes": new_zeros * bytes_per_token,
                "result_bytes": capacity * bytes_per_token,
                "concatenate_live_bytes": (
                    old_capacity + new_zeros + capacity
                )
                * bytes_per_token,
                "result_plus_old_bytes": (old_capacity + capacity)
                * bytes_per_token,
            }
        )
        offset = capacity
    return {
        "step": step,
        "seed_capacity_tokens": seed_capacity,
        "final_offset_tokens": final_offset,
        "growth_events_lower": lower,
        "growth_events_upper": upper,
        "aligned_walk_events": events,
    }


def residency_facts() -> dict:
    """Read the residency and allocator contract that decides who is wired."""
    resident = read(RESIDENT)
    allocator = read(ALLOCATOR)
    insert = re.search(
        r"void ResidencySet::insert.*?\n}\n", resident, re.S
    ).group(0)
    free_fn = re.search(
        r"void MetalAllocator::free\(Buffer buffer\).*?\n}\n",
        allocator,
        re.S,
    ).group(0)
    return {
        "insert_is_greedy_first_fit": "allocatedSize() + buf->allocatedSize() "
        "<= capacity_" in insert
        and "unwired_set_.insert(buf)" in insert,
        "insert_has_no_eviction": "removeAllocation" not in insert,
        "repromotion_only_on_resize": (
            resident.count("unwired_set_.erase") == 2
            and "void ResidencySet::resize" in resident
        ),
        "free_recycles_without_erase": (
            "buffer_cache_.recycle_to_cache(buf);" in free_fn
            and free_fn.index("recycle_to_cache")
            < free_fn.index("residency_set_.erase")
        ),
        "malloc_inserts_only_fresh_device_buffers": bool(
            re.search(
                r"num_resources_\+\+;\s*\n\s*if \(!buf->heap\(\)\) \{\s*\n"
                r"\s*residency_set_\.insert\(buf\);",
                allocator,
            )
        ),
        "session_never_ends_the_ticket": bool(
            re.search(
                r"The ticket is never\s*(?:///)?\s*ended", read(SESSION)
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--out", default="research/e145-artifacts/r6-0.json"
    )
    args = parser.parse_args()

    config_path, text = load_config(args.config)
    kv = full_attention_kv(text)
    gdn = gated_delta_state(text)
    crosscheck = source_shape_crosscheck(gdn)

    slack_mb = source_int(
        SESSION,
        r"wiredZHDefaultSlackMB = (\d+)",
        "wired slack",
    )
    guard_gib = source_int(
        SESSION,
        r"physicalMemory >= \(UInt64\((\d+)\) << 30\)",
        "wired residency guard",
    )
    step = source_int(
        KVCACHE, r"public var step = (\d+)", "KVCacheSimple step"
    )

    slack_bytes = slack_mb << 20
    kv_at_1024 = kv["bytes_per_token"] * (SEED_TOKENS + DECODE_TOKENS)
    walk = kv_capacity_walk(kv["bytes_per_token"], step)
    peak_transient = max(
        e["concatenate_live_bytes"] for e in walk["aligned_walk_events"]
    )
    peak_transient_excl_zeros = max(
        e["result_plus_old_bytes"] for e in walk["aligned_walk_events"]
    )

    total_decode_state = kv_at_1024 + gdn["bytes"]
    snapshot_bytes = gdn["bytes"]
    round_peak = total_decode_state + snapshot_bytes

    survives = total_decode_state > slack_bytes

    out = {
        "harness": "local",
        "gpu_used": False,
        "rung": "R6-0",
        "config_path": config_path,
        "e145_r6_kv_bytes_per_token": kv["bytes_per_token"],
        "e145_r6_kv_bytes_at_1024": kv_at_1024,
        "e145_r6_wired_slack_bytes": slack_bytes,
        "e145_r6_slack_equals_kv": slack_bytes == kv_at_1024,
        "e145_r6_kv_growth_events_per_leg": walk["growth_events_lower"],
        "e145_r6_kv_growth_events_lower": walk["growth_events_lower"],
        "e145_r6_kv_growth_events_upper": walk["growth_events_upper"],
        "e145_r6_kv_peak_transient_bytes": peak_transient,
        "e145_r6_kv_peak_transient_bytes_excluding_zeros_operand": (
            peak_transient_excl_zeros
        ),
        "e145_r6_gdn_state_bytes": gdn["recurrent_bytes"],
        "e145_r6_conv_state_bytes": gdn["conv_bytes"],
        "e145_r6_total_decode_state_bytes": total_decode_state,
        "e145_r6_gdn_snapshot_bytes": snapshot_bytes,
        "e145_r6_round_peak_decode_state_bytes": round_peak,
        "e145_r6_total_over_slack": total_decode_state / slack_bytes,
        "e145_r6_round_peak_over_slack": round_peak / slack_bytes,
        "e145_r6_slack_share_of_total_decode_state": (
            slack_bytes / total_decode_state
        ),
        "e145_r6_hypothesis_survives_r6_0": survives,
        "e145_r6_wired_guard_gib": guard_gib,
        "e145_r6_kvcache_step_tokens": step,
        "kv_detail": kv,
        "gdn_detail": gdn,
        "source_shape_crosscheck": crosscheck,
        "kv_capacity_walk": walk,
        "residency_contract": residency_facts(),
        "seed_tokens": SEED_TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "max_round_width": MAX_ROUND_WIDTH,
    }

    mib = 1 << 20
    print("## R6-0  decode state against the wired residency slack")
    print(f"  pinned config: {config_path}")
    print(
        "  full-attention layers %d, kv heads %d, head dim %d, dtype %s"
        % (
            kv["full_attention_layers"],
            kv["kv_heads"],
            kv["head_dim"],
            kv["activation_dtype"],
        )
    )
    print(
        "  KV bytes per token           %12d" % kv["bytes_per_token"]
    )
    print(
        "  KV bytes at 1024 tokens      %12d  (%.4f MiB)"
        % (kv_at_1024, kv_at_1024 / mib)
    )
    print(
        "  wired slack                  %12d  (%d MiB)"
        % (slack_bytes, slack_mb)
    )
    print(
        "  slack equals KV at 1024:     %s"
        % out["e145_r6_slack_equals_kv"]
    )
    print()
    print(
        "  GDN layers %d, conv state %s %s, recurrent state %s float32"
        % (
            gdn["linear_attention_layers"],
            gdn["convolution_state_shape"],
            gdn["convolution_dtype"],
            gdn["recurrent_state_shape"],
        )
    )
    print(
        "  source shape cross-check: recurrent %s, conv %s"
        % (
            crosscheck["recurrent_matches_source"],
            crosscheck["conv_matches_source"],
        )
    )
    print(
        "  GDN recurrent state          %12d  (%.4f MiB)"
        % (gdn["recurrent_bytes"], gdn["recurrent_bytes"] / mib)
    )
    print(
        "  GDN convolution state        %12d  (%.4f MiB)"
        % (gdn["conv_bytes"], gdn["conv_bytes"] / mib)
    )
    print(
        "  total persistent decode state%12d  (%.4f MiB)"
        % (total_decode_state, total_decode_state / mib)
    )
    print(
        "  per-round GDN snapshot       %12d  (%.4f MiB)"
        % (snapshot_bytes, snapshot_bytes / mib)
    )
    print(
        "  round-peak decode state      %12d  (%.4f MiB)"
        % (round_peak, round_peak / mib)
    )
    print()
    print(
        "  total / slack     %.4f      round peak / slack %.4f"
        % (out["e145_r6_total_over_slack"], out["e145_r6_round_peak_over_slack"])
    )
    print(
        "  the slack covers %.2f %% of the persistent decode state"
        % (100.0 * out["e145_r6_slack_share_of_total_decode_state"])
    )
    print(
        "  hypothesis survives R6-0: %s"
        % out["e145_r6_hypothesis_survives_r6_0"]
    )
    print()
    print(
        "  KVCacheSimple.step %d, seed capacity %d tokens, final offset %d"
        % (step, walk["seed_capacity_tokens"], walk["final_offset_tokens"])
    )
    print(
        "  growth events per leg: %d proven minimum, %d worst case"
        % (walk["growth_events_lower"], walk["growth_events_upper"])
    )
    for event in walk["aligned_walk_events"]:
        print(
            "    at offset %4d  %4d -> %4d tokens   live during concatenate "
            "%12d B (%.1f MiB)"
            % (
                event["at_offset"],
                event["old_capacity_tokens"],
                event["new_capacity_tokens"],
                event["concatenate_live_bytes"],
                event["concatenate_live_bytes"] / mib,
            )
        )
    print(
        "  peak transient %d B (%.1f MiB); %d B (%.1f MiB) if the zeros "
        "operand is elided"
        % (
            peak_transient,
            peak_transient / mib,
            peak_transient_excl_zeros,
            peak_transient_excl_zeros / mib,
        )
    )
    print()
    print("  residency contract read from the vendored sources")
    for key, value in out["residency_contract"].items():
        print("    %-42s %s" % (key, value))

    path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
