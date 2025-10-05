Environment Snapshot

This folder stores a simple manifest of the environment used during runs.

Files
- `manifest.txt`: Key hardware/software info and run settings duplicated from `results/run_meta.json`.

Recorded details
- Mac chip, RAM size, macOS version
- Python and PyTorch versions
- BLAS/threading settings (PyTorch intra/interop threads)
- Power mode (AC vs Battery) from `pmset`
- MPS built/available flags
- Data and model hashes

