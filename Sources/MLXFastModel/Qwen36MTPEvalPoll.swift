import Cmlx
import Foundation
import MLX

/// Non-blocking completion test for arrays submitted through `asyncEval`.
///
/// `eval` parks the calling thread inside `mlx_eval`. Darwin then sees a very
/// low recent-utilisation estimate for that thread and places it on the
/// efficiency cluster, which is the E89 stuck state. Polling lets the drafting
/// thread stay runnable across the GPU wait instead.
enum Qwen36MTPEvalPoll {
    static func isAvailable(_ array: MLXArray) -> Bool {
        var available = false
        guard _mlx_array_is_available(&available, array.ctx) == 0 else { return true }
        return available
    }
}
