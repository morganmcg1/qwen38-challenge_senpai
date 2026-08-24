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

let cells: [(String, Int, Int)] = [
    ("fa.qkv (fused)", 5120, 8192),
    ("fa.q_proj", 5120, 6144),
    ("fa.kv_proj", 5120, 1024),
    ("fa.o_proj", 6144, 5120),
    ("gdn.in_proj", 5120, 12288),
    ("gdn.out_proj", 8192, 5120),
    ("mlp.gate_up", 5120, 34816),
    ("mlp.down", 17408, 5120),
    ("lm_head", 5120, 248320),
    ("small 2048x2048", 2048, 2048),
]
for (label, k, n) in cells {
    print("get_qmv_batch_limit(K=\(k), N=\(n)) = \(qmvBatchLimit(k, n))  [\(label)]")
}
