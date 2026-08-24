// E176: read the Metal architecture string this host reports and replay
// `get_qmv_batch_limit` (backend/metal/quantized.cpp:84) over the Qwen decode
// cell shapes. Device metadata query only: no compute, no allocation, no model.
import Metal

let device = MTLCreateSystemDefaultDevice()!
let name = device.architecture.name
let chars = Array(name)
let gen = (Int(String(chars[chars.count - 3])) ?? 0) * 10
    + (Int(String(chars[chars.count - 2])) ?? 0)
let size = chars[chars.count - 1]
print("architecture: \(name)")
print("arch_gen: \(gen)  arch_size: \(size)")

func qmvBatchLimit(_ D: Int, _ O: Int) -> Int {
    if gen == 13 || gen == 14 {
        if size == "d" {
            return (D <= 2048 && O <= 2048) ? 32 : ((D <= 4096 && O <= 4096) ? 18 : 12)
        }
        return (D <= 2048 && O <= 2048) ? 14 : ((D <= 4096 && O <= 4096) ? 10 : 6)
    }
    if size == "d" {
        return (D <= 2048 && O <= 2048) ? 32 : ((D <= 4096 && O <= 4096) ? 18 : 12)
    }
    return (D <= 2048 && O <= 2048) ? 18 : ((D <= 4096 && O <= 4096) ? 12 : 10)
}

// Exact scored-tree shapes (revision r1). The r0 list carried approximate N
// values taken from the non-scored Sources/MLXFastModel tree; they are kept
// as extra rows because they show the limit does not depend on N here.
let cells: [(String, Int, Int)] = [
    ("mlp.gate_up (fused)", 5120, 34816),
    ("mlp.down", 17408, 5120),
    ("gdn.in_proj (fused)", 5120, 16480),
    ("gdn.out_proj", 6144, 5120),
    ("fa.qkv (fused)", 5120, 14336),
    ("fa.o_proj", 6144, 5120),
    ("lm_head", 5120, 248320),
    ("mtp.fc", 10240, 5120),
    ("fa.q_gate (island path)", 5120, 12288),
    ("unfused fa.kv_proj, N=1024", 5120, 1024),
    ("unfused gdn.in_b, N=48", 5120, 48),
    ("control: small 2048x2048", 2048, 2048),
    ("control: 4096x4096", 4096, 4096),
]
for (label, k, n) in cells {
    print("get_qmv_batch_limit(K=\(k), N=\(n)) = \(qmvBatchLimit(k, n))  [\(label)]")
}
