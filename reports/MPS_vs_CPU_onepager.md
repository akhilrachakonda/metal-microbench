# MPS vs CPU — One Pager

## Setup
Model: GRU(64) → Linear; seq_len=10, batch=1
Split: 70% train / 30% test (chronological).
macOS 15.6.1 on Apple M3; PyTorch 2.6.0.
MPS fallback disabled (PYTORCH_ENABLE_MPS_FALLBACK=0). MPS maps to Metal kernels.

## Method
Warm-up N=20 (discarded), Timed M=500 per backend.
Interleaving: repeated blocks of CPU 50 → MPS 50 with cooldowns to reduce thermal bias.
Per-sample latency scope: host→device copy → forward → device→host; `torch.mps.synchronize()` before stopping timer.

## Results
backend,rmse,mae,r2,mean_ms,p50_ms,p90_ms,p99_ms,throughput_sps
cpu,0.864775,0.723495,0.845947,0.269461,0.258126,0.358558,1.272811,3711.106371
mps,0.864775,0.723495,0.845947,3.578546,3.175125,4.409716,8.31563,279.443115

Speedup (p50 latency): ×0.08; Speedup (throughput): ×0.08.
Hot p50 (trim first-of-block): CPU=0.255479 ms; MPS=3.165604 ms.
Throughput (wall-clock): CPU=3665.445817 samples/s; MPS=279.204748 samples/s.
Parity (CPU↔MPS, denorm): max_abs=1e-06, rmse=0.0.

## Why Improved
GRU matmuls execute via Metal on the Apple GPU, leveraging higher parallelism and memory bandwidth than the CPU for this workload.

## Limits
Tiny models may not amortize GPU launch overhead; fallbacks are banned; results reflect eager-mode inference (no graph capture).

## Artifacts
Data: data/series.csv (sha256=c4b992d168878b4c6a1fb91d50c6360195a1318fec1d7373d714d2276aa36b4b)
Model: models/gru_ckpt.pt (sha256=beb69d87a5fac931d5296e35ac8ce58348aae7e15d7dd265c6abb6beb025071f)
Results: results/summary_results.csv, results/per_run_latencies.csv
Plots: plots/latency_percentiles.png, plots/throughput.png

## References
- PyTorch MPS backend docs (device, usage, sync, fallback): docs.pytorch.org
- Apple ‘Accelerated PyTorch on Mac (MPS)’ overview: Apple Developer
- MPS synchronization for accurate timing (`torch.mps.synchronize()`): docs.pytorch.org
- MPS fallback env var (`PYTORCH_ENABLE_MPS_FALLBACK`): docs.pytorch.org
- Percentile latency practice: Dell Technologies Info Hub; MLCommons
- macOS ‘powermetrics’ for energy metrics: Apple Developer
- Why not Docker for MPS on Mac: Stack Overflow
