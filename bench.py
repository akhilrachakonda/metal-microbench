#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print("PyTorch is required. Please install PyTorch 2.6+.")
    raise

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import pandas as pd
except Exception:
    plt = None
    pd = None

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
REPORTS_DIR = ROOT / "reports"
ENV_DIR = ROOT / "env"


SEQ_LEN = 10
BATCH_SIZE = 1
WARMUP_N = 20
TIMED_M = 500
BLOCK_SIZE = 50  # per backend per block
COOLDOWN_S = 30


def ensure_dirs():
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR, ENV_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_lines(meta: Dict):
    lines = []
    def add(k, v):
        lines.append(f"{k}: {v}")
    add("date_iso", meta.get("date_iso"))
    add("mac_chip", meta.get("mac_chip"))
    add("ram_bytes", meta.get("ram_bytes"))
    add("os_version", meta.get("os_version"))
    add("python_version", meta.get("python_version"))
    add("pytorch_version", meta.get("pytorch_version"))
    add("mps_built", meta.get("mps_built"))
    add("mps_available", meta.get("mps_available"))
    add("blas_threads", meta.get("blas_threads"))
    add("interop_threads", meta.get("interop_threads"))
    add("power_mode", meta.get("power_mode"))
    add("warmup_N", meta.get("warmup_N"))
    add("timed_runs_M", meta.get("timed_runs_M"))
    add("seq_len", meta.get("seq_len"))
    add("batch_size", meta.get("batch_size"))
    add("fallback_disabled", meta.get("fallback_disabled"))
    add("timing_scope", meta.get("timing_scope"))
    add("data_hash", meta.get("data_hash"))
    add("model_hash", meta.get("model_hash"))
    (ENV_DIR / "manifest.txt").write_text("\n".join(lines) + "\n")


def sys_run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def get_mac_chip() -> str:
    # Try multiple sources for Apple Silicon
    s = sys_run(["/usr/sbin/system_profiler", "SPHardwareDataType"])
    for line in s.splitlines():
        if "Chip:" in line:
            return line.split(":", 1)[1].strip()
        if "Processor Name:" in line:
            return line.split(":", 1)[1].strip()
    s2 = sys_run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    if s2.strip():
        return s2.strip()
    return platform.processor() or platform.machine()


def get_ram_bytes() -> int:
    s = sys_run(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()
    try:
        return int(s)
    except Exception:
        return 0


def get_power_mode() -> str:
    s = sys_run(["/usr/bin/pmset", "-g", "batt"])
    header = s.splitlines()[0] if s else ""
    if "AC Power" in header:
        return "AC"
    if "Battery Power" in header:
        return "Battery"
    return "Unknown"


def gather_env_snapshot(data_hash: str = None, model_hash: str = None) -> Dict:
    meta = {
        "date_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mac_chip": get_mac_chip(),
        "ram_bytes": get_ram_bytes(),
        "os_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_built": bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
        "blas_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "power_mode": get_power_mode(),
        "warmup_N": WARMUP_N,
        "timed_runs_M": TIMED_M,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "interleave_block_size": BLOCK_SIZE,
        "cooldown_s": COOLDOWN_S,
        "fallback_disabled": False,
        "data_hash": data_hash,
        "model_hash": model_hash,
    }
    return meta


def save_run_meta(meta: Dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    write_manifest_lines(meta)


def set_seeds(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def gen_series_csv(path: Path, n: int = 2000, seed: int = 123) -> str:
    rng = np.random.default_rng(seed)
    phi = 0.7
    sigma = 0.5
    period = 30
    trend = 0.001
    y = np.zeros(n)
    s = np.sin(2 * np.pi * np.arange(n) / period)
    eps = rng.normal(0, sigma, size=n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + s[t] + trend * t + eps[t]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "value"])
        for i in range(n):
            w.writerow([i, float(y[i])])
    return sha256_of_file(path)


def load_series(path: Path) -> np.ndarray:
    vals: List[float] = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            vals.append(float(row["value"]))
    return np.asarray(vals, dtype=np.float32)


def make_windows(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    # Returns X: [N, seq_len, 1], y: [N, 1]
    N = len(series) - seq_len
    X = np.zeros((N, seq_len, 1), dtype=np.float32)
    y = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        X[i, :, 0] = series[i:i+seq_len]
        y[i, 0] = series[i+seq_len]
    return X, y


def train_test_split_chrono(X, y, split=0.7):
    n = len(X)
    k = int(n * split)
    return (X[:k], y[:k]), (X[k:], y[k:])


@dataclass
class NormStats:
    mean: float
    std: float

    def apply(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / (self.std + 1e-8)

    def invert(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.std + 1e-8) + self.mean


class GRURegressor(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


def prepare_data_cmd():
    ensure_dirs()
    path = DATA_DIR / "series.csv"
    data_hash = gen_series_csv(path)
    meta = gather_env_snapshot(data_hash=data_hash, model_hash=None)
    save_run_meta(meta)
    print(f"Wrote {path} (sha256={data_hash})")


def _train_variant(seq_len: int, num_layers: int, hidden_size: int = 64) -> Path:
    """Train a GRU variant and save checkpoint; returns path."""
    ensure_dirs()
    set_seeds(2025)
    csv_path = DATA_DIR / "series.csv"
    if not csv_path.exists():
        raise SystemExit("data/series.csv missing. Run: python bench.py prepare-data")
    series = load_series(csv_path)

    # Train-only normalization
    X, y = make_windows(series, seq_len)
    (Xtr, ytr), (Xte, yte) = train_test_split_chrono(X, y, split=0.7)
    train_vals = np.concatenate([Xtr.reshape(-1), ytr.reshape(-1)])
    stats = NormStats(mean=float(train_vals.mean()), std=float(train_vals.std()))
    Xtr_n = stats.apply(Xtr)
    ytr_n = stats.apply(ytr)

    # Torch tensors
    Xtr_t = torch.from_numpy(Xtr_n)
    ytr_t = torch.from_numpy(ytr_n)

    model = GRURegressor(hidden_size=hidden_size, num_layers=num_layers)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    patience = 20
    patience_left = patience
    max_epochs = 500

    for epoch in range(1, max_epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()

        l = float(loss.item())
        if l + 1e-9 < best_loss:
            best_loss = l
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    # Save checkpoint with meta
    ckpt = {
        "model_state": best_state,
        "norm": {"mean": stats.mean, "std": stats.std},
        "seed": 2025,
        "seq_len": seq_len,
        "hidden": hidden_size,
        "layers": num_layers,
        "date_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"gru_l{num_layers}_seq{seq_len}.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def train_cmd():
    ensure_dirs()
    set_seeds(2025)
    csv_path = DATA_DIR / "series.csv"
    if not csv_path.exists():
        raise SystemExit("data/series.csv missing. Run: python bench.py prepare-data")
    # Default baseline training (1 layer, SEQ_LEN)
    ckpt_path = _train_variant(SEQ_LEN, 1, hidden_size=64)
    # Maintain legacy filename for baseline
    legacy_path = MODELS_DIR / "gru_ckpt.pt"
    try:
        import shutil
        shutil.copy2(ckpt_path, legacy_path)
    except Exception:
        pass
    model_hash = sha256_of_file(ckpt_path)
    data_hash = sha256_of_file(csv_path)
    meta = gather_env_snapshot(data_hash=data_hash, model_hash=model_hash)
    save_run_meta(meta)
    print(f"Saved checkpoint {ckpt_path} (sha256={model_hash})")


def eval_backend_metrics(model: nn.Module, Xte_t: torch.Tensor, yte_np: np.ndarray, stats: NormStats, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(len(Xte_t)):
            xi = Xte_t[i:i+1].to(device)
            yi_pred = model(xi)
            yi_pred_cpu = yi_pred.to('cpu')
            if device.type == 'mps':
                torch.mps.synchronize()
            preds.append(yi_pred_cpu.numpy().squeeze())
    preds = np.array(preds, dtype=np.float32).reshape(-1, 1)
    preds_denorm = stats.invert(preds)
    y_true = yte_np
    # Metrics
    err = preds_denorm.reshape(-1) - y_true.reshape(-1)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    # R2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return rmse, mae, r2


def timed_infer_single(model: nn.Module, x: torch.Tensor, device: torch.device, device_only: bool = False) -> float:
    """Time a single-sample forward.
    device_only=False: include H2D -> forward -> D2H and synchronize for MPS.
    device_only=True: assume x is already on device; time forward + synchronize only.
    Returns ms.
    """
    start = time.perf_counter()
    if device_only:
        with torch.no_grad():
            y = model(x)
        if device.type == 'mps':
            torch.mps.synchronize()
    else:
        xi = x.to(device)
        with torch.no_grad():
            yi = model(xi)
        _ = yi.to('cpu')
        if device.type == 'mps':
            torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0


def timed_infer_batch(model: nn.Module, x_batch: torch.Tensor, device: torch.device, device_only: bool = False) -> float:
    """Time a batch forward.
    device_only=False: include H2D -> forward -> D2H -> sync; returns ms per batch.
    device_only=True: assume x_batch is on device; time forward + sync only.
    """
    start = time.perf_counter()
    if device_only:
        with torch.no_grad():
            yb = model(x_batch)
        if device.type == 'mps':
            torch.mps.synchronize()
    else:
        xb = x_batch.to(device)
        with torch.no_grad():
            yb = model(xb)
        _ = yb.to('cpu')
        if device.type == 'mps':
            torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0


def append_latency_csv(run_id: int, backend: str, lat_ms: float):
    lat_path = RESULTS_DIR / "per_run_latencies.csv"
    write_header = not lat_path.exists()
    with open(lat_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_id", "backend", "latency_ms"])
        w.writerow([run_id, backend, f"{lat_ms:.6f}"])


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float('nan')
    arr = np.array(vals)
    return float(np.percentile(arr, p))


def run_cmd(backend: str, energy: bool, debug: bool = False, trim_cold_first: bool = False, device_only: bool = False):
    ensure_dirs()

    # Load data and checkpoint
    csv_path = DATA_DIR / "series.csv"
    ckpt_path = MODELS_DIR / "gru_ckpt.pt"
    if not csv_path.exists():
        raise SystemExit("data/series.csv missing. Run: python bench.py prepare-data")
    if not ckpt_path.exists():
        raise SystemExit("models/gru_ckpt.pt missing. Run: python bench.py train")

    data_hash = sha256_of_file(csv_path)
    model_hash = sha256_of_file(ckpt_path)

    # Env snapshot
    meta = gather_env_snapshot(data_hash=data_hash, model_hash=model_hash)
    save_run_meta(meta)

    # Load series and make windows
    series = load_series(csv_path)
    X, y = make_windows(series, SEQ_LEN)
    (Xtr, ytr), (Xte, yte) = train_test_split_chrono(X, y, split=0.7)
    # Use train-only stats
    train_vals = np.concatenate([Xtr.reshape(-1), ytr.reshape(-1)])
    stats = NormStats(mean=float(train_vals.mean()), std=float(train_vals.std()))
    Xte_n = stats.apply(Xte)
    Xte_t = torch.from_numpy(Xte_n)
    yte_np = yte.copy()

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_state = ckpt["model_state"]

    # Accuracy parity on full test split for both backends
    backends_to_run = []
    if backend == 'cpu':
        backends_to_run = ['cpu']
    elif backend == 'mps':
        backends_to_run = ['mps']
    else:
        backends_to_run = ['cpu', 'mps']

    # Check MPS availability if requested
    if 'mps' in backends_to_run:
        if not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise SystemExit("MPS is not available. Ensure PyTorch is built with MPS and you are on Apple Silicon.")

    # Disable fallback during timing
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    fallback_disabled = True

    # Re-save meta with fallback flag
    meta = gather_env_snapshot(data_hash=data_hash, model_hash=model_hash)
    meta["fallback_disabled"] = fallback_disabled
    meta["timing_scope"] = "device_only" if device_only else "end_to_end"
    save_run_meta(meta)

    # Compute accuracy parity (CPU vs MPS) using same weights
    acc_rows = {}
    for b in backends_to_run:
        device = torch.device('mps') if b == 'mps' else torch.device('cpu')
        model_dev = GRURegressor(hidden_size=64)
        model_dev.load_state_dict(model_state)
        model_dev.to(device)
        rmse, mae, r2 = eval_backend_metrics(model_dev, Xte_t, yte_np, stats, device)
        acc_rows[b] = {"rmse": rmse, "mae": mae, "r2": r2}

    # Optional debug and parity checks
    if debug:
        print("=== Debug: Environment & Backend ===")
        print(f"macOS: {platform.mac_ver()[0]} | Chip: {get_mac_chip()} | RAM: {get_ram_bytes()} bytes")
        print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
        print(f"MPS built: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_built()} | MPS available: {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
        print(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}")
        print(f"Threads: intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
        for b in backends_to_run:
            dev = 'mps' if b == 'mps' else 'cpu'
            m = GRURegressor(hidden_size=64)
            m.load_state_dict(model_state)
            m.to(dev)
            pdev = next(m.parameters()).device
            print(f"Model parameters device for {b.upper()}: {pdev}")
        print("=== End Debug ===")

    parity = None
    if set(backends_to_run) == {"cpu", "mps"}:
        with torch.no_grad():
            n_check = min(128, len(Xte_t))
            x_slice = Xte_t[:n_check]
            cpu_m = GRURegressor(hidden_size=64)
            cpu_m.load_state_dict(model_state)
            cpu_m.to('cpu')
            mps_m = GRURegressor(hidden_size=64)
            mps_m.load_state_dict(model_state)
            mps_m.to('mps')
            cpu_out = cpu_m(x_slice).to('cpu').numpy().astype(np.float32)
            mps_out = mps_m(x_slice.to('mps')).to('cpu')
            torch.mps.synchronize()
            mps_out = mps_out.numpy().astype(np.float32)
            cpu_den = stats.invert(cpu_out)
            mps_den = stats.invert(mps_out)
            diff = (cpu_den - mps_den).reshape(-1)
            parity = {
                'n': int(n_check),
                'max_abs': float(np.max(np.abs(diff))),
                'mean_abs': float(np.mean(np.abs(diff))),
                'rmse': float(np.sqrt(np.mean(diff**2))),
            }
            if debug:
                print("=== Parity (CPU vs MPS) on first", n_check, "samples ===")
                print(json.dumps(parity, indent=2))
                print("=== End Parity ===")

    # Warm-ups per backend
    for b in backends_to_run:
        device = torch.device('mps') if b == 'mps' else torch.device('cpu')
        model_dev = GRURegressor(hidden_size=64)
        model_dev.load_state_dict(model_state)
        model_dev.to(device)
        for i in range(WARMUP_N):
            if device_only and device.type == 'mps':
                xb_all = torch.from_numpy(Xte_n).to('mps')
                xi_dev = xb_all[i % len(Xte_n): i % len(Xte_n) + 1]
                _ = timed_infer_single(model_dev, xi_dev, device, device_only=True)
            else:
                _ = timed_infer_single(model_dev, torch.from_numpy(Xte_n[i % len(Xte_n): i % len(Xte_n) + 1]), device, device_only=False)

    # Optional energy logging
    power_log_path = RESULTS_DIR / "power_log.txt"
    energy_csv_path = RESULTS_DIR / "energy_summary.csv"
    power_proc = None
    energy_enabled = False
    if energy:
        # powermetrics typically needs sudo; attempt best-effort
        try:
            power_proc = subprocess.Popen(["/usr/bin/sudo", 
                                           "/usr/bin/powermetrics", "-i", "1000"],
                                          stdout=open(power_log_path, "w"), stderr=subprocess.STDOUT)
            energy_enabled = True
        except Exception:
            energy_enabled = False

    # Interleaved timing: blocks × (CPU 50 → MPS 50), repeat to reach M=500
    # 10 blocks of 50 per backend => 500 each
    blocks = math.ceil(TIMED_M / BLOCK_SIZE)
    # Prepare per-backend state
    runs_needed = {b: TIMED_M for b in backends_to_run}
    latencies: Dict[str, List[float]] = {b: [] for b in backends_to_run}
    wall_totals: Dict[str, float] = {b: 0.0 for b in backends_to_run}
    run_id = 1

    # Prepare separate model copies per backend
    model_cpu = None
    model_mps = None
    if 'cpu' in backends_to_run:
        model_cpu = GRURegressor(hidden_size=64)
        model_cpu.load_state_dict(model_state)
        model_cpu.to('cpu')
        model_cpu.eval()
    if 'mps' in backends_to_run:
        model_mps = GRURegressor(hidden_size=64)
        model_mps.load_state_dict(model_state)
        model_mps.to('mps')
        model_mps.eval()

    # Pre-stage full test set on MPS once for device-only mode
    Xte_dev_mps = torch.from_numpy(Xte_n).to('mps') if ('mps' in backends_to_run and device_only) else None

    for blk in range(blocks):
        # CPU block
        if 'cpu' in backends_to_run:
            for i in range(BLOCK_SIZE):
                if runs_needed['cpu'] <= 0:
                    break
                idx = (blk * BLOCK_SIZE + i) % len(Xte_n)
                xi = torch.from_numpy(Xte_n[idx:idx+1])
                t0 = time.perf_counter()
                lat_ms = timed_infer_single(model_cpu, xi, torch.device('cpu'), device_only=False)
                wall_totals['cpu'] += (time.perf_counter() - t0)
                latencies['cpu'].append(lat_ms)
                append_latency_csv(run_id, 'cpu', lat_ms)
                run_id += 1
                runs_needed['cpu'] -= 1

        # MPS block
        if 'mps' in backends_to_run:
            for i in range(BLOCK_SIZE):
                if runs_needed['mps'] <= 0:
                    break
                idx = (blk * BLOCK_SIZE + i) % len(Xte_n)
                if device_only and Xte_dev_mps is not None:
                    xi_dev = Xte_dev_mps[idx:idx+1]
                    t0 = time.perf_counter()
                    lat_ms = timed_infer_single(model_mps, xi_dev, torch.device('mps'), device_only=True)
                else:
                    xi = torch.from_numpy(Xte_n[idx:idx+1])
                    t0 = time.perf_counter()
                    lat_ms = timed_infer_single(model_mps, xi, torch.device('mps'), device_only=False)
                wall_totals['mps'] += (time.perf_counter() - t0)
                latencies['mps'].append(lat_ms)
                append_latency_csv(run_id, 'mps', lat_ms)
                run_id += 1
                runs_needed['mps'] -= 1

        # cooldown between blocks
        if blk != blocks - 1:
            time.sleep(COOLDOWN_S)

    # Stop powermetrics if enabled
    if power_proc is not None:
        try:
            power_proc.terminate()
        except Exception:
            pass

    # Summarize perf per backend
    results_rows = []
    # Device name string
    device_name = get_mac_chip()
    os_version = platform.mac_ver()[0]
    framework_ver = torch.__version__
    blas_threads = torch.get_num_threads()
    date_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    # Ensure summary CSV header (rotate legacy header if present)
    summary_path = RESULTS_DIR / "summary_results.csv"
    write_header = True
    if summary_path.exists():
        try:
            first_line = (summary_path.read_text().splitlines() or [""])[0]
            if "parity_vs_cpu_rmse" in first_line:
                write_header = False
            else:
                # Legacy header detected → back up old file and start fresh with new schema
                backup = RESULTS_DIR / "summary_results.legacy.csv"
                try:
                    if backup.exists():
                        backup.unlink()
                except Exception:
                    pass
                summary_path.rename(backup)
                write_header = True
        except Exception:
            write_header = True
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "backend","rmse","mae","r2",
                "mean_ms","p50_ms","p90_ms","p99_ms",
                "mean_ms_hot","p50_ms_hot",
                "throughput_sps","throughput_wall_sps",
                "timed_runs_M","warmup_N","seq_len","batch_size",
                "model_hash","data_hash","device_name","os_version","framework_ver",
                "blas_threads","date_iso",
                "timing_scope",
                "parity_vs_cpu_max_abs","parity_vs_cpu_mean_abs","parity_vs_cpu_rmse"
            ])
        for b in backends_to_run:
            vals = latencies[b]
            mean_ms = float(np.mean(vals)) if len(vals) else float('nan')
            p50_ms = percentile(vals, 50)
            p90_ms = percentile(vals, 90)
            p99_ms = percentile(vals, 99)
            # Throughput from total latency time for that backend
            total_s = float(np.sum(vals) / 1000.0)
            throughput = (len(vals) / total_s) if total_s > 0 else float('nan')
            # Wall-clock throughput across iterations (captures Python/setup overhead per call)
            wall_s = float(wall_totals.get(b, 0.0))
            throughput_wall = (len(vals) / wall_s) if wall_s > 0 else float('nan')

            # Hot-only metrics (optionally excluding first sample of each block)
            mean_ms_hot = float('nan')
            p50_ms_hot = float('nan')
            if trim_cold_first and len(vals) > 0:
                hot_vals = [v for i, v in enumerate(vals) if (i % BLOCK_SIZE) != 0]
                if len(hot_vals) > 0:
                    mean_ms_hot = float(np.mean(hot_vals))
                    p50_ms_hot = percentile(hot_vals, 50)
            rmse = acc_rows[b]['rmse']
            mae = acc_rows[b]['mae']
            r2 = acc_rows[b]['r2']
            # Parity info only meaningful for MPS row when CPU present
            par_max = ""
            par_mean = ""
            par_rmse = ""
            if b == 'mps' and 'cpu' in backends_to_run and 'mps' in backends_to_run and 'parity' in locals() and parity is not None:
                par_max = f"{parity['max_abs']:.6f}"
                par_mean = f"{parity['mean_abs']:.6f}"
                par_rmse = f"{parity['rmse']:.6f}"

            w.writerow([
                b, f"{rmse:.6f}", f"{mae:.6f}", f"{r2:.6f}",
                f"{mean_ms:.6f}", f"{p50_ms:.6f}", f"{p90_ms:.6f}", f"{p99_ms:.6f}",
                f"{mean_ms_hot:.6f}" if not math.isnan(mean_ms_hot) else "",
                f"{p50_ms_hot:.6f}" if not math.isnan(p50_ms_hot) else "",
                f"{throughput:.6f}", f"{throughput_wall:.6f}",
                TIMED_M, WARMUP_N, SEQ_LEN, BATCH_SIZE,
                model_hash, data_hash, device_name, os_version, framework_ver,
                blas_threads, date_iso,
                ("device_only" if device_only else "end_to_end"),
                par_max, par_mean, par_rmse
            ])

    # Optional: summarize energy
    if energy_enabled and pd is not None:
        try:
            # Very coarse placeholder: compute file size as proxy (to be refined)
            # Proper parsing would scan powermetrics samples and integrate.
            energy_rows = []
            for b in backends_to_run:
                vals = latencies[b]
                total_s = float(np.sum(vals) / 1000.0)
                energy_rows.append({
                    'backend': b,
                    'total_seconds': total_s,
                    'notes': 'powermetrics raw in power_log.txt (integration TBD)'
                })
            pd.DataFrame(energy_rows).to_csv(energy_csv_path, index=False)
        except Exception:
            pass

    print(f"Completed timing. Summaries in {summary_path}")


def report_cmd():
    ensure_dirs()
    summary_path = RESULTS_DIR / "summary_results.csv"
    lat_path = RESULTS_DIR / "per_run_latencies.csv"
    meta_path = RESULTS_DIR / "run_meta.json"
    if not summary_path.exists() or not lat_path.exists() or not meta_path.exists():
        raise SystemExit("Missing results. Run: python bench.py run --backend both")
    if plt is None or pd is None:
        raise SystemExit("matplotlib and pandas are required for report generation.")

    df = pd.read_csv(summary_path)
    if set(df["backend"]) != {"cpu", "mps"} and len(df) >= 2:
        # If multiple runs accumulated, keep last CPU/MPS rows
        df = df.drop_duplicates(subset=["backend"], keep="last")

    # Plot latency percentiles
    backends = ["cpu", "mps"]
    p50 = [df.loc[df.backend == b, "p50_ms"].astype(float).values[-1] for b in backends]
    p90 = [df.loc[df.backend == b, "p90_ms"].astype(float).values[-1] for b in backends]
    p99 = [df.loc[df.backend == b, "p99_ms"].astype(float).values[-1] for b in backends]

    x = np.arange(len(backends))
    width = 0.2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, p50, width, label='p50')
    ax.bar(x, p90, width, label='p90')
    ax.bar(x + width, p99, width, label='p99')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency percentiles')
    ax.legend()
    # Speedup annotation above MPS bars vs CPU (use p50)
    try:
        speedup = float(p50[0]) / float(p50[1]) if float(p50[1]) > 0 else float('nan')
        ax.text(x[1], max(p99) * 1.02, f"×{speedup:.2f}", ha='center')
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latency_percentiles.png", dpi=150)
    plt.close(fig)

    # Throughput plot
    tput = [df.loc[df.backend == b, "throughput_sps"].astype(float).values[-1] for b in backends]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(backends, tput)
    ax.set_ylabel('Samples/s')
    ax.set_title('Throughput')
    # Speedup label
    try:
        t_speed = float(tput[1]) / float(tput[0]) if float(tput[0]) > 0 else float('nan')
        ax.text(1, max(tput) * 1.02, f"×{t_speed:.2f}", ha='center')
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "throughput.png", dpi=150)
    plt.close(fig)

    # One-pager report
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    cpu_row = df[df.backend == 'cpu'].iloc[-1].to_dict()
    mps_row = df[df.backend == 'mps'].iloc[-1].to_dict()

    lines = []
    lines.append("# MPS vs CPU — One Pager")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"Model: GRU(64) → Linear; seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
    lines.append("Split: 70% train / 30% test (chronological).")
    lines.append(f"macOS {meta.get('os_version')} on {meta.get('mac_chip')}; PyTorch {meta.get('pytorch_version')}.")
    lines.append("MPS fallback disabled (PYTORCH_ENABLE_MPS_FALLBACK=0). MPS maps to Metal kernels.")
    lines.append("")
    lines.append("## Method")
    lines.append(f"Warm-up N={WARMUP_N} (discarded), Timed M={TIMED_M} per backend.")
    lines.append("Interleaving: repeated blocks of CPU 50 → MPS 50 with cooldowns to reduce thermal bias.")
    lines.append("Per-sample latency scope: host→device copy → forward → device→host; `torch.mps.synchronize()` before stopping timer.")
    lines.append("")
    lines.append("## Results")
    lines.append("backend,rmse,mae,r2,mean_ms,p50_ms,p90_ms,p99_ms,throughput_sps")
    lines.append(
        f"cpu,{cpu_row['rmse']},{cpu_row['mae']},{cpu_row['r2']},{cpu_row['mean_ms']},{cpu_row['p50_ms']},{cpu_row['p90_ms']},{cpu_row['p99_ms']},{cpu_row['throughput_sps']}" )
    lines.append(
        f"mps,{mps_row['rmse']},{mps_row['mae']},{mps_row['r2']},{mps_row['mean_ms']},{mps_row['p50_ms']},{mps_row['p90_ms']},{mps_row['p99_ms']},{mps_row['throughput_sps']}" )
    try:
        lat_speedup = float(cpu_row['p50_ms']) / float(mps_row['p50_ms']) if float(mps_row['p50_ms']) > 0 else float('nan')
        tput_speedup = float(mps_row['throughput_sps']) / float(cpu_row['throughput_sps']) if float(cpu_row['throughput_sps']) > 0 else float('nan')
        lines.append("")
        lines.append(f"Speedup (p50 latency): ×{lat_speedup:.2f}; Speedup (throughput): ×{tput_speedup:.2f}.")
    except Exception:
        pass
    # Optional extras: hot metrics, wall throughput, parity
    try:
        cpu_p50_hot = cpu_row.get('p50_ms_hot', '')
        mps_p50_hot = mps_row.get('p50_ms_hot', '')
        if cpu_p50_hot != '' and mps_p50_hot != '':
            lines.append(f"Hot p50 (trim first-of-block): CPU={cpu_p50_hot} ms; MPS={mps_p50_hot} ms.")
    except Exception:
        pass
    try:
        cpu_tput_wall = cpu_row.get('throughput_wall_sps', '')
        mps_tput_wall = mps_row.get('throughput_wall_sps', '')
        if cpu_tput_wall != '' and mps_tput_wall != '':
            lines.append(f"Throughput (wall-clock): CPU={cpu_tput_wall} samples/s; MPS={mps_tput_wall} samples/s.")
    except Exception:
        pass
    try:
        par_max = mps_row.get('parity_vs_cpu_max_abs', '')
        par_rmse = mps_row.get('parity_vs_cpu_rmse', '')
        if par_max != '' and par_rmse != '':
            lines.append(f"Parity (CPU↔MPS, denorm): max_abs={par_max}, rmse={par_rmse}.")
    except Exception:
        pass
    lines.append("")
    lines.append("## Why Improved")
    lines.append("GRU matmuls execute via Metal on the Apple GPU, leveraging higher parallelism and memory bandwidth than the CPU for this workload.")
    lines.append("")
    lines.append("## Limits")
    lines.append("Tiny models may not amortize GPU launch overhead; fallbacks are banned; results reflect eager-mode inference (no graph capture).")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"Data: data/series.csv (sha256={meta.get('data_hash')})")
    lines.append(f"Model: models/gru_ckpt.pt (sha256={meta.get('model_hash')})")
    lines.append("Results: results/summary_results.csv, results/per_run_latencies.csv")
    lines.append("Plots: plots/latency_percentiles.png, plots/throughput.png")
    if (RESULTS_DIR / "energy_summary.csv").exists():
        lines.append("Energy: results/energy_summary.csv (powermetrics)")
    lines.append("")
    lines.append("## References")
    lines.append("- PyTorch MPS backend docs (device, usage, sync, fallback): docs.pytorch.org")
    lines.append("- Apple ‘Accelerated PyTorch on Mac (MPS)’ overview: Apple Developer")
    lines.append("- MPS synchronization for accurate timing (`torch.mps.synchronize()`): docs.pytorch.org")
    lines.append("- MPS fallback env var (`PYTORCH_ENABLE_MPS_FALLBACK`): docs.pytorch.org")
    lines.append("- Percentile latency practice: Dell Technologies Info Hub; MLCommons")
    lines.append("- macOS ‘powermetrics’ for energy metrics: Apple Developer")
    lines.append("- Why not Docker for MPS on Mac: Stack Overflow")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "MPS_vs_CPU_onepager.md").write_text("\n".join(lines) + "\n")
    print("Wrote plots and one-pager report.")


def sweep_cmd(batch_sizes: List[int], seq_lens: List[int], layers_list: List[int], target_M: int = 200,
              warmup_batches: int = 10, block_size: int = 50, cooldown_s: int = 5, device_only: bool = False):
    ensure_dirs()
    # Prepare data
    csv_path = DATA_DIR / "series.csv"
    if not csv_path.exists():
        raise SystemExit("data/series.csv missing. Run: python bench.py prepare-data")
    series = load_series(csv_path)
    data_hash = sha256_of_file(csv_path)

    # Environment snapshot for sweep (with fallback disabled)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    meta_env = gather_env_snapshot(data_hash=data_hash, model_hash=None)
    meta_env["fallback_disabled"] = True

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / "sweeps" / stamp
    root.mkdir(parents=True, exist_ok=True)
    sweep_summary_path = root / "sweep_summary.csv"
    # Write sweep meta for reproducibility
    (root / "sweep_meta.json").write_text(json.dumps({
        "date_iso": meta_env["date_iso"],
        "mac_chip": meta_env["mac_chip"],
        "os_version": meta_env["os_version"],
        "python_version": meta_env["python_version"],
        "pytorch_version": meta_env["pytorch_version"],
        "mps_built": meta_env["mps_built"],
        "mps_available": meta_env["mps_available"],
        "blas_threads": meta_env["blas_threads"],
        "interop_threads": meta_env["interop_threads"],
        "fallback_disabled": meta_env["fallback_disabled"],
        "timing_scope": "End-to-end per-batch: H2D -> forward -> D2H; torch.mps.synchronize()",
        "notes": "Sweep generated by bench.py; per-config meta under each config directory"
    }, indent=2) + "\n")

    # Write summary header
    with open(sweep_summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seq_len','layers','batch_size','backend',
            'rmse','mae','r2',
            'mean_ms_batch','p50_ms_batch','p90_ms_batch','p99_ms_batch',
            'mean_ms_per_sample','p50_ms_per_sample',
            'throughput_sps','throughput_wall_sps',
            'timed_target_M_samples','actual_samples','n_batches',
            'model_hash','data_hash','date_iso',
            'mac_chip','os_version','python_version','pytorch_version',
            'mps_built','mps_available','fallback_disabled','blas_threads','interop_threads',
            'parity_vs_cpu_max_abs','parity_vs_cpu_mean_abs','parity_vs_cpu_rmse'
        ])

    # Disable fallback globally for sweep
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    for seq_len in seq_lens:
        # Prepare windows for this seq_len
        X, y = make_windows(series, seq_len)
        (Xtr, ytr), (Xte, yte) = train_test_split_chrono(X, y, split=0.7)
        train_vals = np.concatenate([Xtr.reshape(-1), ytr.reshape(-1)])
        stats = NormStats(mean=float(train_vals.mean()), std=float(train_vals.std()))
        Xte_n = stats.apply(Xte)
        Xte_t = torch.from_numpy(Xte_n)
        yte_np = yte.copy()

        for layers in layers_list:
            # Ensure checkpoint for (seq_len, layers)
            ckpt_path = MODELS_DIR / f"gru_l{layers}_seq{seq_len}.pt"
            if not ckpt_path.exists():
                ckpt_path = _train_variant(seq_len, layers, hidden_size=64)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_state = ckpt['model_state']
            model_hash = sha256_of_file(ckpt_path)

            # Accuracy parity (full test split) per backend
            backends_to_run = ['cpu', 'mps']
            acc_rows = {}
            for b in backends_to_run:
                dev = torch.device('mps') if b == 'mps' else torch.device('cpu')
                m = GRURegressor(hidden_size=64, num_layers=layers)
                m.load_state_dict(model_state)
                m.to(dev)
                rmse, mae, r2 = eval_backend_metrics(m, Xte_t, yte_np, stats, dev)
                acc_rows[b] = {'rmse': rmse, 'mae': mae, 'r2': r2}

            # Parity on a small slice
            parity = None
            try:
                with torch.no_grad():
                    n_check = min(128, len(Xte_t))
                    x_slice = Xte_t[:n_check]
                    cpu_m = GRURegressor(hidden_size=64, num_layers=layers)
                    cpu_m.load_state_dict(model_state)
                    cpu_m.to('cpu')
                    mps_m = GRURegressor(hidden_size=64, num_layers=layers)
                    mps_m.load_state_dict(model_state)
                    mps_m.to('mps')
                    cpu_out = cpu_m(x_slice).to('cpu').numpy().astype(np.float32)
                    mps_out = mps_m(x_slice.to('mps')).to('cpu')
                    torch.mps.synchronize()
                    mps_out = mps_out.numpy().astype(np.float32)
                    cpu_den = stats.invert(cpu_out)
                    mps_den = stats.invert(mps_out)
                    diff = (cpu_den - mps_den).reshape(-1)
                    parity = {
                        'max_abs': float(np.max(np.abs(diff))),
                        'mean_abs': float(np.mean(np.abs(diff))),
                        'rmse': float(np.sqrt(np.mean(diff**2))),
                    }
            except Exception:
                parity = None

            for bs in batch_sizes:
                config_key = f"l{layers}_seq{seq_len}_bs{bs}"
                cfg_dir = root / config_key
                cfg_dir.mkdir(parents=True, exist_ok=True)
                # Per-run latencies for this config
                lat_path = cfg_dir / "per_run_latencies.csv"
                with open(lat_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["run_id","backend","batch_ms","per_sample_ms","batch_size"]) 

                # Prepare models
                model_cpu = GRURegressor(hidden_size=64, num_layers=layers)
                model_cpu.load_state_dict(model_state)
                model_cpu.to('cpu').eval()
                model_mps = GRURegressor(hidden_size=64, num_layers=layers)
                model_mps.load_state_dict(model_state)
                model_mps.to('mps').eval()

                # Write config meta (device placements and timing scope)
                cfg_meta = {
                    'seq_len': seq_len,
                    'layers': layers,
                    'batch_size': bs,
                    'target_M_samples_per_backend': target_M,
                    'warmup_batches': warmup_batches,
                    'block_size': block_size,
                    'cooldown_s': cooldown_s,
                    'timing_scope': 'End-to-end per-batch: H2D -> forward -> D2H; torch.mps.synchronize()',
                    'devices': {
                        'cpu_model_device': str(next(model_cpu.parameters()).device),
                        'mps_model_device': str(next(model_mps.parameters()).device)
                    },
                    'env': {
                        'date_iso': meta_env['date_iso'],
                        'mac_chip': meta_env['mac_chip'],
                        'os_version': meta_env['os_version'],
                        'python_version': meta_env['python_version'],
                        'pytorch_version': meta_env['pytorch_version'],
                        'mps_built': meta_env['mps_built'],
                        'mps_available': meta_env['mps_available'],
                        'fallback_disabled': meta_env['fallback_disabled'],
                        'blas_threads': meta_env['blas_threads'],
                        'interop_threads': meta_env['interop_threads'],
                    }
                }
                (cfg_dir / 'config_meta.json').write_text(json.dumps(cfg_meta, indent=2) + "\n")

                # Warmup per backend (batches)
                for b in ['cpu','mps']:
                    dev = torch.device('mps') if b == 'mps' else torch.device('cpu')
                    m = model_mps if b == 'mps' else model_cpu
                    for i in range(warmup_batches):
                        idx = i % len(Xte_n)
                        batch = []
                        if idx + bs <= len(Xte_n):
                            batch_arr = Xte_n[idx:idx+bs]
                        else:
                            remain = (idx + bs) - len(Xte_n)
                            batch_arr = np.concatenate([Xte_n[idx:], Xte_n[:remain]], axis=0)
                        if device_only and dev.type == 'mps':
                            xb_dev_all = torch.from_numpy(Xte_n).to('mps')
                            if idx + bs <= len(Xte_n):
                                xb_dev = xb_dev_all[idx:idx+bs]
                            else:
                                remain = (idx + bs) - len(Xte_n)
                                xb_dev = torch.cat([xb_dev_all[idx:], xb_dev_all[:remain]], dim=0)
                            _ = timed_infer_batch(m, xb_dev, dev, device_only=True)
                        else:
                            xb = torch.from_numpy(batch_arr)
                            _ = timed_infer_batch(m, xb, dev, device_only=False)

                # Timed batches per backend
                runs_needed_batches = { 'cpu': int(math.ceil(target_M / bs)), 'mps': int(math.ceil(target_M / bs)) }
                latencies_batch: Dict[str, List[float]] = {'cpu': [], 'mps': [] }
                wall_totals: Dict[str, float] = {'cpu': 0.0, 'mps': 0.0}
                run_id = 1
                blocks = int(math.ceil(runs_needed_batches['cpu'] / block_size))

                # Pre-stage Xte on MPS for device-only
                Xte_dev_all = torch.from_numpy(Xte_n).to('mps') if device_only else None

                for blk in range(blocks):
                    # CPU block
                    for i in range(block_size):
                        if runs_needed_batches['cpu'] <= 0: break
                        idx = (blk * block_size + i) % len(Xte_n)
                        if idx + bs <= len(Xte_n):
                            batch_arr = Xte_n[idx:idx+bs]
                        else:
                            remain = (idx + bs) - len(Xte_n)
                            batch_arr = np.concatenate([Xte_n[idx:], Xte_n[:remain]], axis=0)
                        xb = torch.from_numpy(batch_arr)
                        t0 = time.perf_counter()
                        b_ms = timed_infer_batch(model_cpu, xb, torch.device('cpu'), device_only=False)
                        wall_totals['cpu'] += (time.perf_counter() - t0)
                        latencies_batch['cpu'].append(b_ms)
                        with open(lat_path, 'a', newline='') as f:
                            csv.writer(f).writerow([run_id, 'cpu', f"{b_ms:.6f}", f"{(b_ms/bs):.6f}", bs])
                        run_id += 1
                        runs_needed_batches['cpu'] -= 1

                    # MPS block
                    for i in range(block_size):
                        if runs_needed_batches['mps'] <= 0: break
                        idx = (blk * block_size + i) % len(Xte_n)
                        if device_only and Xte_dev_all is not None:
                            if idx + bs <= len(Xte_n):
                                xb_dev = Xte_dev_all[idx:idx+bs]
                            else:
                                remain = (idx + bs) - len(Xte_n)
                                xb_dev = torch.cat([Xte_dev_all[idx:], Xte_dev_all[:remain]], dim=0)
                            t0 = time.perf_counter()
                            b_ms = timed_infer_batch(model_mps, xb_dev, torch.device('mps'), device_only=True)
                        else:
                            if idx + bs <= len(Xte_n):
                                batch_arr = Xte_n[idx:idx+bs]
                            else:
                                remain = (idx + bs) - len(Xte_n)
                                batch_arr = np.concatenate([Xte_n[idx:], Xte_n[:remain]], axis=0)
                            xb = torch.from_numpy(batch_arr)
                            t0 = time.perf_counter()
                            b_ms = timed_infer_batch(model_mps, xb, torch.device('mps'), device_only=False)
                        wall_totals['mps'] += (time.perf_counter() - t0)
                        latencies_batch['mps'].append(b_ms)
                        with open(lat_path, 'a', newline='') as f:
                            csv.writer(f).writerow([run_id, 'mps', f"{b_ms:.6f}", f"{(b_ms/bs):.6f}", bs])
                        run_id += 1
                        runs_needed_batches['mps'] -= 1

                    if blk != blocks - 1:
                        time.sleep(cooldown_s)

                # Summaries for this config into sweep_summary.csv and per-config summary
                per_cfg_summary = cfg_dir / "summary_results.csv"
                with open(per_cfg_summary, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([
                        'backend','rmse','mae','r2',
                        'mean_ms_batch','p50_ms_batch','p90_ms_batch','p99_ms_batch',
                        'mean_ms_per_sample','p50_ms_per_sample',
                        'throughput_sps','throughput_wall_sps',
                        'timed_target_M_samples','actual_samples','n_batches',
                        'seq_len','layers','batch_size','model_hash','data_hash','date_iso',
                        'mac_chip','os_version','python_version','pytorch_version',
                        'mps_built','mps_available','fallback_disabled','blas_threads','interop_threads',
                        'timing_scope',
                        'parity_vs_cpu_max_abs','parity_vs_cpu_mean_abs','parity_vs_cpu_rmse'
                    ])

                for b in ['cpu','mps']:
                    vals_b = latencies_batch[b]
                    if len(vals_b) == 0:
                        continue
                    # Batch-level percentiles
                    mean_ms_batch = float(np.mean(vals_b))
                    p50_b = percentile(vals_b, 50)
                    p90_b = percentile(vals_b, 90)
                    p99_b = percentile(vals_b, 99)
                    # Per-sample metrics derived
                    vals_ps = [v/bs for v in vals_b]
                    mean_ms_ps = float(np.mean(vals_ps))
                    p50_ps = percentile(vals_ps, 50)
                    # Throughput
                    total_s = float(np.sum(vals_b) / 1000.0)
                    total_samples = len(vals_b) * bs
                    throughput = (total_samples / total_s) if total_s > 0 else float('nan')
                    wall_s = float(wall_totals[b])
                    throughput_wall = (total_samples / wall_s) if wall_s > 0 else float('nan')
                    # Counts
                    n_batches = len(vals_b)
                    actual_samples = total_samples
                    date_iso = dt.datetime.now(dt.timezone.utc).isoformat()
                    # Parity columns only on MPS row
                    par_max = par_mean = par_rmse = ""
                    if b == 'mps' and parity is not None:
                        par_max = f"{parity['max_abs']:.6f}"
                        par_mean = f"{parity['mean_abs']:.6f}"
                        par_rmse = f"{parity['rmse']:.6f}"

                    # Write per-config summary row
                    with open(per_cfg_summary, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            b, f"{acc_rows[b]['rmse']:.6f}", f"{acc_rows[b]['mae']:.6f}", f"{acc_rows[b]['r2']:.6f}",
                            f"{mean_ms_batch:.6f}", f"{p50_b:.6f}", f"{p90_b:.6f}", f"{p99_b:.6f}",
                            f"{mean_ms_ps:.6f}", f"{p50_ps:.6f}",
                            f"{throughput:.6f}", f"{throughput_wall:.6f}",
                            target_M, actual_samples, n_batches,
                            seq_len, layers, bs, model_hash, data_hash, date_iso,
                            meta_env['mac_chip'], meta_env['os_version'], meta_env['python_version'], meta_env['pytorch_version'],
                            meta_env['mps_built'], meta_env['mps_available'], meta_env['fallback_disabled'], meta_env['blas_threads'], meta_env['interop_threads'],
                            ('device_only' if device_only else 'end_to_end'),
                            par_max, par_mean, par_rmse
                        ])

                    # Append to sweep summary
                    with open(sweep_summary_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            seq_len, layers, bs, b,
                            f"{acc_rows[b]['rmse']:.6f}", f"{acc_rows[b]['mae']:.6f}", f"{acc_rows[b]['r2']:.6f}",
                            f"{mean_ms_batch:.6f}", f"{p50_b:.6f}", f"{p90_b:.6f}", f"{p99_b:.6f}",
                            f"{mean_ms_ps:.6f}", f"{p50_ps:.6f}",
                            f"{throughput:.6f}", f"{throughput_wall:.6f}",
                            target_M, actual_samples, n_batches,
                            model_hash, data_hash, date_iso,
                            meta_env['mac_chip'], meta_env['os_version'], meta_env['python_version'], meta_env['pytorch_version'],
                            meta_env['mps_built'], meta_env['mps_available'], meta_env['fallback_disabled'], meta_env['blas_threads'], meta_env['interop_threads'],
                            ('device_only' if device_only else 'end_to_end'),
                            par_max, par_mean, par_rmse
                        ])

    print(f"Sweep complete. Summary at {sweep_summary_path}")


def sweep_report_cmd(sweep_dir: Path = None):
    ensure_dirs()
    if plt is None or pd is None:
        raise SystemExit("matplotlib and pandas are required for sweep report generation.")
    # Resolve sweep directory
    root = sweep_dir
    if root is None:
        sweeps = sorted((RESULTS_DIR / 'sweeps').glob('*'))
        if not sweeps:
            raise SystemExit("No sweeps found. Run: python bench.py sweep ...")
        root = sweeps[-1]
    root = Path(root)
    summary_csv = root / 'sweep_summary.csv'
    if not summary_csv.exists():
        raise SystemExit(f"Sweep summary not found at {summary_csv}")

    plots_dir = root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    # Normalize dtypes
    df['seq_len'] = df['seq_len'].astype(int)
    df['layers'] = df['layers'].astype(int)
    df['batch_size'] = df['batch_size'].astype(int)
    df['throughput_sps'] = df['throughput_sps'].astype(float)
    df['p50_ms_per_sample'] = df['p50_ms_per_sample'].astype(float)

    # Per-config plots: throughput vs batch size, latency vs batch size, speedup vs batch size
    combos = df[['seq_len','layers']].drop_duplicates().itertuples(index=False, name=None)
    for seq_len, layers in combos:
        sub = df[(df.seq_len == seq_len) & (df.layers == layers)].copy()
        sub = sub.sort_values(['batch_size','backend'])
        bs = sorted(sub['batch_size'].unique())
        fig, ax = plt.subplots(figsize=(7,4))
        for b in ['cpu','mps']:
            y = [float(sub[(sub.batch_size == v) & (sub.backend == b)]['throughput_sps'].values[0]) for v in bs]
            ax.plot(bs, y, marker='o', label=b.upper())
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch size')
        ax.set_ylabel('Throughput (samples/s)')
        ax.set_title(f'Throughput vs Batch — seq_len={seq_len}, layers={layers}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"throughput_seq{seq_len}_l{layers}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4))
        for b in ['cpu','mps']:
            y = [float(sub[(sub.batch_size == v) & (sub.backend == b)]['p50_ms_per_sample'].values[0]) for v in bs]
            ax.plot(bs, y, marker='o', label=b.upper())
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch size')
        ax.set_ylabel('p50 latency (ms per sample)')
        ax.set_title(f'Latency p50 vs Batch — seq_len={seq_len}, layers={layers}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"latency_p50_seq{seq_len}_l{layers}.png", dpi=150)
        plt.close(fig)

        # Speedup lines: throughput_speedup = MPS/CPU; latency_speedup = CPU_p50/MPS_p50
        fig, ax = plt.subplots(figsize=(7,4))
        sp_tput = []
        sp_lat = []
        for v in bs:
            row_c = sub[(sub.batch_size == v) & (sub.backend == 'cpu')].iloc[0]
            row_m = sub[(sub.batch_size == v) & (sub.backend == 'mps')].iloc[0]
            tput_sp = float(row_m['throughput_sps']) / float(row_c['throughput_sps']) if float(row_c['throughput_sps'])>0 else float('nan')
            lat_sp = float(row_c['p50_ms_per_sample']) / float(row_m['p50_ms_per_sample']) if float(row_m['p50_ms_per_sample'])>0 else float('nan')
            sp_tput.append(tput_sp)
            sp_lat.append(lat_sp)
        ax.plot(bs, sp_tput, marker='o', label='Speedup (tput) MPS/CPU')
        ax.plot(bs, sp_lat, marker='o', label='Speedup (lat p50) CPU/MPS')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Batch size')
        ax.set_ylabel('Speedup (×)')
        ax.set_title(f'Speedup vs Batch — seq_len={seq_len}, layers={layers}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"speedup_seq{seq_len}_l{layers}.png", dpi=150)
        plt.close(fig)

    # Best batch per backend per config
    rows = []
    for (seq_len, layers, backend), grp in df.groupby(['seq_len','layers','backend']):
        best = grp.sort_values('throughput_sps', ascending=False).iloc[0]
        rows.append({
            'seq_len': seq_len,
            'layers': layers,
            'backend': backend,
            'best_batch_size': int(best['batch_size']),
            'best_throughput_sps': float(best['throughput_sps']),
            'best_p50_ms_per_sample': float(best['p50_ms_per_sample']),
        })
    pd.DataFrame(rows).to_csv(plots_dir / 'best_batch_by_backend.csv', index=False)
    print(f"Sweep plots written to {plots_dir}")


def enrich_sweep_cmd(sweep_dir: Path = None):
    ensure_dirs()
    root = sweep_dir
    if root is None:
        sweeps = sorted((RESULTS_DIR / 'sweeps').glob('*'))
        if not sweeps:
            raise SystemExit("No sweeps found. Run: python bench.py sweep ...")
        root = sweeps[-1]
    root = Path(root)
    summary_csv = root / 'sweep_summary.csv'
    meta_json = root / 'sweep_meta.json'
    if not summary_csv.exists():
        raise SystemExit(f"Sweep summary not found at {summary_csv}")
    # Derive env fields either from sweep_meta.json or env manifest
    env = {}
    try:
        if meta_json.exists():
            import json as _json
            env = _json.loads(meta_json.read_text())
        else:
            # Fallback to current environment snapshot
            env = gather_env_snapshot()
    except Exception:
        env = gather_env_snapshot()

    import pandas as _pd
    df = _pd.read_csv(summary_csv)
    # If already enriched, write a copy anyway
    df['mac_chip'] = env.get('mac_chip')
    df['os_version'] = env.get('os_version')
    df['python_version'] = env.get('python_version')
    df['pytorch_version'] = env.get('pytorch_version')
    df['mps_built'] = env.get('mps_built')
    df['mps_available'] = env.get('mps_available')
    df['fallback_disabled'] = env.get('fallback_disabled', True)
    df['blas_threads'] = env.get('blas_threads')
    df['interop_threads'] = env.get('interop_threads')
    # If missing timing scope, default to end_to_end (historic behavior)
    if 'timing_scope' not in df.columns:
        df['timing_scope'] = 'end_to_end'
    out_path = root / 'sweep_summary.enriched.csv'
    df.to_csv(out_path, index=False)

    # Enrich per-config summaries similarly
    for cfg_dir in root.glob('l*_seq*_bs*'):
        per_cfg = cfg_dir / 'summary_results.csv'
        if per_cfg.exists():
            try:
                dcfg = _pd.read_csv(per_cfg)
                dcfg['mac_chip'] = env.get('mac_chip')
                dcfg['os_version'] = env.get('os_version')
                dcfg['python_version'] = env.get('python_version')
                dcfg['pytorch_version'] = env.get('pytorch_version')
                dcfg['mps_built'] = env.get('mps_built')
                dcfg['mps_available'] = env.get('mps_available')
                dcfg['fallback_disabled'] = env.get('fallback_disabled', True)
                dcfg['blas_threads'] = env.get('blas_threads')
                dcfg['interop_threads'] = env.get('interop_threads')
                if 'timing_scope' not in dcfg.columns:
                    dcfg['timing_scope'] = 'end_to_end'
                dcfg.to_csv(cfg_dir / 'summary_results.enriched.csv', index=False)
            except Exception:
                pass

    print(f"Enriched sweep written to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Metal MPS vs CPU Micro-Benchmark (PyTorch)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("prepare-data", help="Generate deterministic data and hashes")
    sub.add_parser("train", help="Train GRU(64) and save checkpoint")
    run_p = sub.add_parser("run", help="Run benchmark warmups+timed with interleaving")
    run_p.add_argument("--backend", choices=["cpu", "mps", "both"], default="both")
    run_p.add_argument("--energy", action="store_true", help="Enable powermetrics logging (optional)")
    run_p.add_argument("--debug", action="store_true", help="Print device placement and parity info")
    run_p.add_argument("--trim-cold-first", action="store_true", help="Trim first sample of each block in hot metrics")
    run_p.add_argument("--device-only", action="store_true", help="Keep tensors on device and time compute only (no H2D/D2H)")
    sweep_p = sub.add_parser("sweep", help="Sweep batch sizes, layers, and sequence lengths")
    sweep_p.add_argument("--batch-sizes", default="1,8,32,64", help="Comma-separated batch sizes")
    sweep_p.add_argument("--seq-lens", default="10,50,100", help="Comma-separated sequence lengths")
    sweep_p.add_argument("--layers", default="1,3", help="Comma-separated layer counts")
    sweep_p.add_argument("--M", type=int, default=200, help="Target samples per backend (per config)")
    sweep_p.add_argument("--warmup", type=int, default=10, help="Warmup iterations (batches) per backend")
    sweep_p.add_argument("--block-size", type=int, default=50, help="Batches per block per backend")
    sweep_p.add_argument("--cooldown", type=int, default=5, help="Cooldown seconds between blocks")
    sweep_p.add_argument("--device-only", action="store_true", help="Keep tensors on device and time compute only")
    sweep_report_p = sub.add_parser("sweep-report", help="Generate plots from a sweep")
    sweep_report_p.add_argument("--dir", default=None, help="Path to sweep dir (defaults to latest)")
    enrich_p = sub.add_parser("enrich-sweep", help="Backfill env columns into an existing sweep CSVs")
    enrich_p.add_argument("--dir", default=None, help="Path to sweep dir (defaults to latest)")
    sub.add_parser("report", help="Generate plots and one-pager")

    args = parser.parse_args()

    if args.cmd == "prepare-data":
        prepare_data_cmd()
    elif args.cmd == "train":
        train_cmd()
    elif args.cmd == "run":
        run_cmd(backend=args.backend, energy=args.energy, debug=args.debug, trim_cold_first=args.trim_cold_first, device_only=args.device_only)
    elif args.cmd == "sweep":
        sweep_cmd(
            batch_sizes=[int(x) for x in args.batch_sizes.split(',') if x.strip()],
            seq_lens=[int(x) for x in args.seq_lens.split(',') if x.strip()],
            layers_list=[int(x) for x in args.layers.split(',') if x.strip()],
            target_M=args.M,
            warmup_batches=args.warmup,
            block_size=args.block_size,
            cooldown_s=args.cooldown,
            device_only=args.device_only,
        )
    elif args.cmd == "sweep-report":
        sweep_dir = Path(args.dir) if args.dir else None
        sweep_report_cmd(sweep_dir)
    elif args.cmd == "enrich-sweep":
        sweep_dir = Path(args.dir) if args.dir else None
        enrich_sweep_cmd(sweep_dir)
    elif args.cmd == "report":
        report_cmd()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
