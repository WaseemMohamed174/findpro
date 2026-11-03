from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field


APP_OVERHEAD_GB = 0.8
RAM_USABLE_FRAC = 0.75
AVAIL_FRAC = 0.90
VRAM_USABLE_FRAC = 0.90


@dataclass
class Hardware:
    ram_total: float = 0.0
    ram_avail: float = 0.0
    cores: int = 1
    gpu_name: str | None = None
    vram_total: float = 0.0
    vram_free: float = 0.0
    gpu_available: bool = False
    cuda_tag: str | None = None
    cuda_version: str = ""
    gpu_note: str = ""

    @property
    def safe_ram(self) -> float:
        usable = min(self.ram_avail * AVAIL_FRAC, self.ram_total * RAM_USABLE_FRAC)
        return max(usable - APP_OVERHEAD_GB, 0.0)

    @property
    def safe_vram(self) -> float:
        return max(self.vram_free * VRAM_USABLE_FRAC - 0.2, 0.0)

    def to_dict(self) -> dict:
        return {
            "ram_total": round(self.ram_total, 2),
            "ram_avail": round(self.ram_avail, 2),
            "cores": self.cores,
            "gpu_name": self.gpu_name,
            "vram_total": round(self.vram_total, 2),
            "vram_free": round(self.vram_free, 2),
            "gpu_available": self.gpu_available,
            "cuda_tag": self.cuda_tag,
            "cuda_version": self.cuda_version,
            "gpu_note": self.gpu_note,
            "safe_ram": round(self.safe_ram, 2),
            "safe_vram": round(self.safe_vram, 2),
        }


def _psutil():
    try:
        import psutil
    except ImportError:
        subprocess.check_call(["python3", "-m", "pip", "install", "-q", "psutil"])
        import psutil
    return psutil


def _nvidia():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8, text=True,
        )
    except Exception:
        return None
    best = None
    for line in out.strip().splitlines():
        try:
            name, tot, free = [p.strip() for p in line.split(",")]
            row = (name, float(tot) / 1024, float(free) / 1024)
        except ValueError:
            continue
        if best is None or row[2] > best[2]:
            best = row
    return best


def _cuda_tag():
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL,
                                      timeout=8, text=True)
    except Exception:
        return None, ""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None, ""
    ver = (int(m.group(1)), int(m.group(2)))
    tags = ["cu125", "cu124", "cu123", "cu122", "cu121"]
    for tag in tags:
        if ver >= (int(tag[2:4]), int(tag[4:])):
            return tag, f"{ver[0]}.{ver[1]}"
    return None, f"{ver[0]}.{ver[1]}"


def detect_hardware() -> Hardware:
    ps = _psutil()
    vm = ps.virtual_memory()
    gpu = _nvidia()
    if gpu:
        name, vt, vf = gpu
        tag, drv = _cuda_tag()
        if tag:
            return Hardware(
                ram_total=vm.total / 1024**3,
                ram_avail=vm.available / 1024**3,
                cores=os.cpu_count() or 1,
                gpu_name=name, vram_total=vt, vram_free=vf,
                gpu_available=True, cuda_tag=tag, cuda_version=drv,
            )
        return Hardware(
            ram_total=vm.total / 1024**3,
            ram_avail=vm.available / 1024**3,
            cores=os.cpu_count() or 1,
            gpu_name=name, vram_total=vt, vram_free=vf,
            gpu_available=False, gpu_note=f"driver supports CUDA {drv} (need 12.1+)",
        )
    return Hardware(
        ram_total=vm.total / 1024**3,
        ram_avail=vm.available / 1024**3,
        cores=os.cpu_count() or 1,
        gpu_note="no NVIDIA GPU detected",
    )
