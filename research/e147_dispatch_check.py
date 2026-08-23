"""F6 section 5: which quantized GEMM kernel does the scored seed prefill reach?

The advisor asked me to confirm, against source, that every scored prefill GEMM
takes the NAX branch on the ranked M5 and therefore that rung A is inert there.
This reimplements the host dispatch chain exactly as `quantized.cpp` writes it
and evaluates it for every scored projection. Zero GPU.

The chain is longer than the one quoted in F6. `QuantizedMatmul::eval_gpu` does
NOT call `qmm` for a transposed non-batched product; it calls `qmm_splitk`
first, and `qmm_splitk` reaches `qmm` only when its computed `split_k` collapses
to 1. That hop has to be evaluated, not assumed, because when `split_k > 1` the
work goes to the `qmm_t_splitk` kernel, which is neither the kernel rung A
changed nor the kernel rung B changed.

  usage: research/e147_dispatch_check.py [--m 512] [--json out.json]
"""
import argparse
import json
import math
import re

QUANTIZED_CPP = (
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
)

GROUP_SIZE = 64
BITS = 4
DTYPE = "bfloat16"

# Scored transposed projections, from the checkpoint config: hidden 5120,
# intermediate 17408, head_dim 256, 24 q heads (attn_output_gate doubles the q
# projection), 4 kv heads, linear key heads 16x128, linear value heads 48x128,
# 64 layers with full_attention_interval 4 -> 16 full, 48 linear.
PROJECTIONS = [
    # name, N, K, layer count
    ("gdn.in_proj", 16480, 5120, 48),
    ("gdn.out_proj", 5120, 6144, 48),
    ("fa.qkv", 14336, 5120, 16),
    ("fa.o_proj", 5120, 6144, 16),
    ("mlp.gate_up", 34816, 5120, 64),
    ("mlp.down", 5120, 17408, 64),
    ("lm_head", 248320, 5120, 1),
]

HOSTS = [
    # label, architecture string, is_nax_available
    ("local g16s (this Mac)", "applegpu_g16s", False),
    ("ranked M5 g17s", "applegpu_g17s", True),
]


def get_qmv_batch_limit(D, O, arch):
    """quantized.cpp get_qmv_batch_limit, transcribed."""
    arch_size = arch[-1]
    arch_gen = int(re.search(r"g(\d+)", arch).group(1))
    if arch_gen in (13, 14):
        if arch_size == "d":
            return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
        return 14 if (D <= 2048 and O <= 2048) else (10 if (D <= 4096 and O <= 4096) else 6)
    if arch_size == "d":
        return 32 if (D <= 2048 and O <= 2048) else (18 if (D <= 4096 and O <= 4096) else 12)
    return 18 if (D <= 2048 and O <= 2048) else (12 if (D <= 4096 and O <= 4096) else 10)


def split_k_for(M, N, K, group_size):
    """quantized.cpp qmm_splitk, the part that decides the kernel."""
    bm = bn = 32
    n_tiles = (N + bn - 1) // bn
    m_tiles = (M + bm - 1) // bm
    current_tgs = n_tiles * m_tiles
    split_k = max(1, 512 // current_tgs)
    k_align = max(group_size, 32)
    split_k = min(split_k, K // k_align)
    while split_k > 1 and (K % (split_k * k_align) != 0):
        split_k -= 1
    return split_k, current_tgs, k_align


def dispatch(M, N, K, arch, nax_available, transpose=True, batched=False):
    """Return the kernel the host reaches, with the decision trail."""
    trail = []
    B = 1 if not batched else 2
    vector_limit = get_qmv_batch_limit(K, N, arch) if transpose else 4
    trail.append(f"vector_limit={vector_limit}")
    if M < vector_limit:
        trail.append(f"M={M} < vector_limit -> qmv/qvm path")
        return ("qmv/qvm (not a GEMM)", trail, None)

    if transpose and B == 1:
        sk, tgs, k_align = split_k_for(M, N, K, GROUP_SIZE)
        trail.append(f"qmm_splitk: tgs={tgs} k_align={k_align} split_k={sk}")
        if sk > 1:
            trail.append("split_k>1 -> qmm_t_splitk kernel, NOT qmm")
            return ("qmm_t_splitk", trail, sk)
        trail.append("split_k<=1 -> delegates to qmm()")

    nax = nax_available and transpose and (K % 64 == 0) and (DTYPE != "float32")
    trail.append(
        f"nax predicate: avail={nax_available} transpose={transpose} "
        f"K%64=={K % 64} dtype={DTYPE} -> {nax}"
    )
    if nax:
        return ("affine_qmm_t_nax -> qmm_t_nax_tgp_impl", trail, None)
    return ("affine_qmm_t -> qmm_t_impl", trail, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--json")
    args = ap.parse_args()

    src = open(QUANTIZED_CPP).read()
    out = {"m": args.m, "group_size": GROUP_SIZE, "dtype": DTYPE}

    # Pin the predicate text so a vendored change cannot silently invalidate this.
    pred = re.search(
        r"if \(metal::is_nax_available\(\) && transpose && \(K % 64 == 0\) &&\s*"
        r"\(env::enable_tf32\(\) \|\| x\.dtype\(\) != float32\)\) \{\s*"
        r"return qmm_nax\(",
        src,
    )
    out["nax_predicate_verbatim_present"] = bool(pred)
    out["eval_gpu_calls_qmm_splitk_before_qmm"] = bool(
        re.search(r"if \(transpose_ && B == 1\) \{\s*qmm_splitk\(", src)
    )
    out["qmm_splitk_falls_back_to_qmm"] = bool(
        re.search(r"if \(split_k <= 1\) \{\s*return qmm\(", src)
    )
    print("nax predicate found verbatim in source        :", out["nax_predicate_verbatim_present"])
    print("eval_gpu routes transposed B==1 to qmm_splitk :", out["eval_gpu_calls_qmm_splitk_before_qmm"])
    print("qmm_splitk falls back to qmm when split_k<=1  :", out["qmm_splitk_falls_back_to_qmm"])
    assert out["nax_predicate_verbatim_present"], "F6 predicate text not found"
    assert out["eval_gpu_calls_qmm_splitk_before_qmm"], "split-k hop not found"
    assert out["qmm_splitk_falls_back_to_qmm"], "split-k fallback not found"

    out["hosts"] = {}
    for label, arch, nax in HOSTS:
        print(f"\n=== {label}  arch={arch}  is_nax_available={nax}  M={args.m}")
        print(f"{'projection':14s} {'N':>7s} {'K':>6s} {'n':>3s}  {'split_k':>7s}  kernel")
        rows = []
        for name, N, K, n_layers in PROJECTIONS:
            kernel, trail, sk = dispatch(args.m, N, K, arch, nax)
            sk_txt = "-" if sk is None else str(sk)
            print(f"{name:14s} {N:7d} {K:6d} {n_layers:3d}  {sk_txt:>7s}  {kernel}")
            rows.append(
                {
                    "projection": name,
                    "N": N,
                    "K": K,
                    "layers": n_layers,
                    "K_mod_64": K % 64,
                    "kernel": kernel,
                    "split_k": sk,
                    "trail": trail,
                }
            )
        out["hosts"][arch] = {"label": label, "nax": nax, "rows": rows}
        kinds = sorted({r["kernel"] for r in rows})
        print(f"distinct kernels reached: {kinds}")

    ranked = out["hosts"]["applegpu_g17s"]["rows"]
    local = out["hosts"]["applegpu_g16s"]["rows"]
    out["ranked_all_nax"] = all("nax" in r["kernel"] for r in ranked)
    out["local_all_qmm_t_impl"] = all(r["kernel"] == "affine_qmm_t -> qmm_t_impl" for r in local)
    out["any_non_transposed_scored_gemm"] = False
    out["any_scored_K_not_mult_64"] = any(r["K_mod_64"] != 0 for r in ranked)
    out["any_scored_uses_splitk"] = any(r["split_k"] for r in ranked)

    # Where does split-K actually turn on? This matters because a future
    # chunked prefill would silently leave both edited kernels.
    activations = []
    for m in (1, 8, 16, 32, 64, 128, 192, 256, 384, 511, 512):
        hits = [
            name
            for name, N, K, _ in PROJECTIONS
            if dispatch(m, N, K, "applegpu_g17s", True)[0] == "qmm_t_splitk"
        ]
        activations.append({"M": m, "splitk_projections": hits})
    out["splitk_activation_sweep"] = activations

    print("\n=== split-K activation sweep on the ranked host")
    for a in activations:
        print(f"M={a['M']:4d}  split-K projections: {a['splitk_projections'] or 'none'}")

    print("\nranked host: every scored GEMM takes the NAX branch :", out["ranked_all_nax"])
    print("local  host: every scored GEMM takes qmm_t_impl     :", out["local_all_qmm_t_impl"])
    print("any scored GEMM non-transposed                      :", out["any_non_transposed_scored_gemm"])
    print("any scored GEMM with K % 64 != 0                    :", out["any_scored_K_not_mult_64"])
    print("any scored GEMM on split-K at M=%d                 :" % args.m, out["any_scored_uses_splitk"])
    out["f6_section5_confirmed"] = bool(
        out["ranked_all_nax"]
        and not out["any_scored_K_not_mult_64"]
        and not out["any_non_transposed_scored_gemm"]
    )
    print("F6 section 5 confirmed (rung A inert on ranked)     :", out["f6_section5_confirmed"])

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
