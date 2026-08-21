# E90 GPU interval ledger

A light instrument that answers **"was the device executing at host time T"**.
It records the GPU execution interval of every command buffer the worker
submits, and it emits absolute host anchors for every drafting round on the
same clock. An offline reader intersects the two, so it can attribute GPU busy
and GPU idle time to each inter-anchor window of a round.

Measured cost on a 512-token local leg: **-0.046 %** on
`mtp_seconds_per_token`, which is inside the run-to-run noise of this host.

## Why not the E58/E80 census

`research/e80-artifacts/gputime-census.patch` answers a different question,
"which kernel spent the time", and pays for it with a lock on every dispatch,
every pipeline bind and every barrier. Its own header calls it unfit for
timing. Inflated host phases manufacture GPU idle that the uninstrumented round
does not have. E90 rung 0b measured 840.4 us of idle per round where the census
reported 4,749 us.

Use the census when you need kernel names. Use this ledger when you need idle.

## What it contains

`gpu-interval-ledger.patch` is the exact diff against
`59b67f50bcf3d02e064e624e2a08acfa67dc3fa9`:

| file | change |
| --- | --- |
| `Sources/MLXFastModel/E90GPUIntervalLedger.swift` | new. Swizzles one selector, command buffer `commit`, and records `gpuStartTime` and `gpuEndTime` per completed buffer. |
| `Sources/MLXFastModel/Qwen36MTPBlockSession.swift` | emits the `mtp-anchor:` line with absolute anchors, adds the `tSnapshotDone` anchor, adds `host_thread_cpu_ns`, and adds the `MLX_QWEN_MTP_HEAD_SUBMIT` arm selector. |
| `Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift` | installs the hook at worker startup. |

Apply it with:

```bash
git apply research/e90-artifacts/gpu-interval-ledger.patch
```

## How to run it

```bash
research/e90_leg.sh TAG 512 --intervals
python3 research/e90_intervals.py TAG --skip-rounds 8
```

`research/e90_leg.sh` writes `meta.txt`, `score.json`, `trace.txt` and
`gpu-intervals.jsonl` under `research/out/TAG/`.

Environment gates, all default off:

| variable | effect |
| --- | --- |
| `MLX_E90_GPU_INTERVALS=1` | install the commit hook |
| `MLX_E90_GPU_INTERVALS_PATH` | write the ledger here instead of stderr |
| `MLX_QWEN_MTP_TRACE_SYNC_HEAD=1` | block on the head chain instead of `asyncEval` |
| `MLX_QWEN_MTP_HEAD_SUBMIT` | head-chain submission arm |

The `MLX_` prefix is load-bearing. `sanitizedRuntimeWorkerEnvironment` admits
`MLX_` and drops `MLXFAST_`, so an `MLXFAST_`-spelled gate never reaches the
worker process that owns the decode path.

## Self-check

The twelve inter-anchor windows tile `round_us` exactly, so
`gpu_busy_us + gpu_idle_us` equals each window by construction. The reader
reports `tiling_error_us` per round. It was **0.000 us** on every analysed
round of both E90 rung 0b legs. A non-zero value means the anchors and the
ledger disagree, and the instrument is wrong.

`research/e90_intervals.py` also stratifies by host state, using the E89 gate
of a post-warmup host-phase sum above 1,500 us. Publish the stratum before any
pooled number: an arm-unbalanced comparison is void.

## Two limits, both load-bearing

1. **"GPU busy" means a command buffer was executing, not that the device was
   saturated.** A 1-element `arange` keeps the union busy exactly like a 315 MB
   weight read. The ledger closes scheduling and overlap questions. It cannot
   close dispatch-efficiency questions.
2. **It attributes command buffers, not dispatches.** There is no pipeline
   binding recorded, so no kernel-family split is available from this data.
   Use the census for that.
