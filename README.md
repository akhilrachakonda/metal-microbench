Metal MPS vs CPU Micro-Benchmark (PyTorch)

This project benchmarks inference latency and throughput for a simple GRU(64) → Linear regression model on macOS Apple Silicon, comparing CPU vs Apple Metal (MPS). It follows a strict protocol with deterministic data, one checkpoint for both backends, interleaved timing, and synchronized GPU timing.

Quick start
- Python 3.10+ and PyTorch 2.6+ with MPS support installed.
- On Apple Silicon, ensure MPS is available (torch.backends.mps.is_available()).

Commands
- `python bench.py prepare-data` → Generate deterministic series at `data/series.csv`, compute hash.
- `python bench.py train` → Train GRU(64) once, save `models/gru_ckpt.pt`, compute hash.
- `python bench.py run --backend both [--debug] [--trim-cold-first] [--device-only]` → Run warmups+timed M. `--device-only` keeps test tensors on device and times compute only (no H2D/D2H), useful to estimate best‑case MPS compute.
- `python bench.py sweep --batch-sizes 1,8,32,64 --seq-lens 10,50,100 --layers 1,3 --M 200 --warmup 10 --block-size 50 --cooldown 5 [--device-only]` → Train missing variants and sweep configs. `--device-only` runs compute‑only timing on MPS (preloaded tensors). Saves per‑config CSVs and a consolidated `sweep_summary.csv`.
- `python bench.py enrich-sweep --dir results/sweeps/<stamp>` → Backfill environment columns into an existing sweep (`sweep_summary.enriched.csv` and per‑config `summary_results.enriched.csv`).
- `python bench.py report` → Generate `plots/*.png` and `reports/MPS_vs_CPU_onepager.md` from CSVs.

Requirements
- macOS on Apple Silicon; native run (no Docker) so MPS works.
- PyTorch 2.6+ (MPS backend), numpy, matplotlib, pandas recommended.

Notes
- Timing uses end-to-end per-sample scope (host→device copy → forward → device→host copy) and calls `torch.mps.synchronize()` to avoid under-reporting GPU time.
- Interleaving pattern: CPU 50 → MPS 50 with cooldowns to reduce thermal bias; repeated to reach M=500 per backend.
- Fallbacks disabled during timing: `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Summary now includes optional “hot” metrics (trimmed first-of-block), dual throughput (latency-derived and iteration wall-clock), and CPU↔MPS parity deltas.
- Sweep runs truncate M and cooldown by default to keep total runtime reasonable; adjust via flags.

Environment & Verification
- Environment snapshot: `env/manifest.txt` and `results/run_meta.json` record macOS version, chip, Python/PyTorch versions, MPS built/available, BLAS threads, and fallback flag.
- Sweep metadata: each sweep writes `results/sweeps/<stamp>/sweep_meta.json` and per‑config `config_meta.json` with device placements, timing scope, and the same environment fields.
- Device enforcement: models are explicitly moved via `.to('cpu')` / `.to('mps')`. For MPS, `torch.mps.synchronize()` is called before stopping timers.
- Timing scope: end‑to‑end includes H2D → forward → D2H. CPU timing includes CPU forward only; MPS timing includes H2D/D2H/driver overhead.
