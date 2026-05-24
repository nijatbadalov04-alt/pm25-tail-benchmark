"""
AQ-FM-Bench :: scripts/00_environment_check.py
================================================
Environment & hardware sanity check. Run this FIRST and re-run after each
install stage. It is non-fatal: missing libraries are reported, not crashed on.

What it verifies (the GPU gate):
  1. Python version & platform.
  2. PyTorch import + CUDA availability on the Blackwell RTX 5070 (sm_120).
  3. A real CUDA matmul (compute actually works, not just "is_available").
  4. bf16 autocast on GPU (the inference precision the FMs use).
  5. Presence of the project dependency stack.

Usage:
    .venv\\Scripts\\python.exe scripts\\00_environment_check.py
"""
from __future__ import annotations

import importlib
import platform
import sys

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; DIM = "\033[2m"; END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{END}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{END} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{END} {msg}")


def header(msg: str) -> None:
    print(f"\n{'=' * 68}\n{msg}\n{'=' * 68}")


# ---------------------------------------------------------------------------
# 1. Python & platform
# ---------------------------------------------------------------------------
def check_python() -> None:
    header("1. PYTHON & PLATFORM")
    print(f"  Python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  Platform : {platform.platform()}")
    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 11):
        ok("Python 3.11 -- target version for the FM libraries.")
    else:
        warn(f"Python {major}.{minor} -- target is 3.11; some FM libs may lack wheels.")


# ---------------------------------------------------------------------------
# 2-4. PyTorch / CUDA / bf16  (THE critical gate)
# ---------------------------------------------------------------------------
def check_torch() -> bool:
    header("2. PYTORCH + CUDA + bf16  (Blackwell RTX 5070 gate)")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        fail(f"torch import failed: {e}")
        return False

    print(f"  torch version      : {torch.__version__}")
    print(f"  compiled CUDA      : {torch.version.cuda}")
    if not torch.cuda.is_available():
        fail("CUDA NOT available to torch. FM inference would fall back to CPU (too slow).")
        warn("If this build is CPU-only, reinstall from the cu128 index.")
        return False

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  device             : {name}")
    print(f"  compute capability : sm_{cap[0]}{cap[1]}")
    print(f"  total VRAM         : {total_gb:.1f} GiB")
    ok("CUDA is available to torch.")
    if cap[0] >= 12:
        ok(f"Blackwell (sm_{cap[0]}{cap[1]}) detected and recognised by this torch build.")
    else:
        warn(f"Compute capability sm_{cap[0]}{cap[1]} -- expected sm_120 for RTX 5070.")

    # Real compute test (fp32 matmul on GPU)
    try:
        import torch
        a = torch.randn(2048, 2048, device=dev)
        b = torch.randn(2048, 2048, device=dev)
        c = a @ b
        torch.cuda.synchronize()
        assert c.shape == (2048, 2048)
        ok("fp32 CUDA matmul (2048x2048) succeeded -- compute works, not just detection.")
    except Exception as e:  # noqa: BLE001
        fail(f"CUDA matmul failed: {e}")
        return False

    # bf16 autocast test (FM inference precision)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = torch.randn(1024, 1024, device=dev)
            y = (x @ x).float()
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        ok("bf16 autocast matmul succeeded -- FM inference precision supported.")
    except Exception as e:  # noqa: BLE001
        fail(f"bf16 autocast failed: {e}")
        return False

    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  free VRAM now      : {free_gb:.1f} GiB")
    torch.cuda.empty_cache()
    return True


# ---------------------------------------------------------------------------
# 5. Dependency stack
# ---------------------------------------------------------------------------
STACK = {
    "Core data": ["numpy", "pandas", "polars", "pyarrow", "scipy", "yaml"],
    "Scientific / geo": ["xarray", "netCDF4", "sklearn", "statsmodels"],
    "AQ + weather data": ["openaq", "requests", "gcsfs", "zarr"],
    "Classical / generic models": ["xgboost", "darts"],
    "Foundation models": ["transformers", "chronos", "uni2ts", "timesfm"],
    "Training / acceleration": ["accelerate", "peft"],
    "Conformal / metrics": ["mapie"],
    "Viz": ["matplotlib", "seaborn", "cartopy"],
    "Config / tracking / test": ["hydra", "wandb", "pytest"],
}
# import-name remaps where pip name != import name
REMAP = {"yaml": "yaml", "sklearn": "sklearn", "hydra": "hydra", "netCDF4": "netCDF4"}


def check_stack() -> None:
    header("5. DEPENDENCY STACK")
    n_ok = n_total = 0
    for group, mods in STACK.items():
        print(f"\n  {DIM}{group}{END}")
        for m in mods:
            n_total += 1
            try:
                mod = importlib.import_module(REMAP.get(m, m))
                ver = getattr(mod, "__version__", "")
                ok(f"{m:<14}{ver}")
                n_ok += 1
            except Exception:  # noqa: BLE001
                warn(f"{m:<14}not installed")
    print(f"\n  Stack: {n_ok}/{n_total} importable.")


def main() -> int:
    print(f"\n{'#' * 68}\n#  AQ-FM-Bench environment check\n{'#' * 68}")
    check_python()
    gpu_ok = check_torch()
    check_stack()
    header("VERDICT")
    if gpu_ok:
        ok("GPU GATE PASSED -- Blackwell GPU usable for foundation-model inference.")
    else:
        fail("GPU GATE NOT PASSED -- resolve torch/CUDA before running foundation models.")
    print()
    return 0 if gpu_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
