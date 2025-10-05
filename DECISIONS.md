Decisions and Rationale

Why native macOS (not Docker) for MPS
- Apple’s MPS backend requires native access to the Metal driver. macOS Docker VMs do not provide GPU passthrough for Apple GPUs, so MPS acceleration is unavailable inside Docker containers. (Stack Overflow)

Why PyTorch + MPS
- PyTorch provides a first‑party MPS device (`mps`) for Apple Silicon with documented device semantics, synchronization (`torch.mps.synchronize()`), and an env var to control CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK`). (docs.pytorch.org)

Why percentile latency and throughput
- P50/P90/P99 are widely used for SLO-aware latency analysis and in MLPerf reporting. Throughput (samples/s) complements latency for system capacity. (Dell Technologies Info Hub, MLCommons)

Why end-to-end latency scope
- Measuring host↔device transfer plus forward pass reflects realistic inference costs for small models; excluding transfer or sync under-reports GPU time.

Interleaving choice
- To achieve exactly M=500 timed samples per backend while following the “CPU 50 → MPS 50” pattern, we repeat that pattern for 10 blocks. This resolves the apparent 5-block vs 500-sample mismatch while preserving the thermal-bias mitigation intent.

References
- PyTorch MPS backend docs, device usage, synchronization, fallback env var: docs.pytorch.org
- Apple “Accelerated PyTorch on Mac (MPS)”: Apple Developer
- Percentile latency practice: Dell Technologies Info Hub; MLCommons
- macOS power metrics: powermetrics (Apple Developer)
- Docker on Mac lacks GPU passthrough for MPS: Stack Overflow

