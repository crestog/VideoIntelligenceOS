"""
vios.process.resources — what this machine actually has, measured now.

The rotation loop's first question is "what fits?", and the honest answer
changes between sessions. Kaggle hands out 2×T4 most of the time and a P100
sometimes; a notebook that has already loaded something has less free VRAM than
one that just started; the 57 GB scratch disk is 57 GB until someone caches a
model into it. So nothing here is read from a config file. Everything is probed.

Three sources, in order of preference:

  torch          authoritative, but only present in the GPU session
  nvidia-smi     present whenever a driver is, and works without torch
  nothing        CPU-only, which is a legitimate mode: the signal passes,
                 ffmpeg artifacts and shot detection all run fine without a GPU

The web UI imports this module, and the web UI often runs where torch does not
exist. Every probe therefore degrades to a blank rather than an exception.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

# Leave this much VRAM unallocated per card. CUDA's own context is ~300 MB, the
# allocator fragments, and a batch that fits in theory OOMs in practice on the
# frame that happens to be 1080×1920 instead of 720×1280. Reserving a gigabyte
# costs one smaller batch; not reserving it costs the whole sweep.
VRAM_HEADROOM_MB = 1024

# Same idea for system RAM: decoding a video, holding frames, and the Python
# process itself all live outside VRAM.
RAM_HEADROOM_MB = 2048


def _nvidia_smi() -> list:
    try:
        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    out = []
    for line in (res.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total, used = int(float(parts[2])), int(float(parts[3]))
        except ValueError:
            continue
        cap = parts[4] if len(parts) > 4 else ""
        out.append({"index": int(parts[0]), "name": parts[1],
                    "total_mb": total, "free_mb": max(total - used, 0),
                    "capability": cap})
    return out


def _torch_gpus() -> list:
    try:
        import torch  # noqa: PLC0415 — optional, and absent on the web host
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            major, minor = torch.cuda.get_device_capability(i)
            out.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_mb": total // (1024 ** 2),
                "free_mb": free // (1024 ** 2),
                "capability": f"{major}.{minor}",
            })
        except Exception:
            continue
    return out


def _system_ram_mb() -> int:
    try:
        import psutil  # noqa: PLC0415
        return int(psutil.virtual_memory().available // (1024 ** 2))
    except Exception:
        pass
    # Linux without psutil — Kaggle has psutil, but a stripped container might
    # not, and MemAvailable is the number that matters (free + reclaimable),
    # not MemFree.
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(int(re.sub(r"\D", "", line)) / 1024)
    except OSError:
        pass
    return 0


def capabilities(gpus: list) -> dict:
    """What the *architecture* allows, which is not the same as what fits.

    This is the check that keeps a model card's benchmark from becoming a
    wasted session. Turing (SM 75 — the T4) has INT8 and INT4 tensor cores but
    no BF16 and no FP8, and FlashAttention-2 needs Ampere. A pipeline that
    assumes bf16 because "everything uses bf16 now" fails on Kaggle's most
    common accelerator, and it fails slowly and confusingly rather than at
    import.
    """
    caps = []
    for g in gpus:
        try:
            caps.append(float(g.get("capability") or 0))
        except ValueError:
            caps.append(0.0)
    lowest = min(caps) if caps else 0.0
    return {
        "compute_capability": lowest,
        "fp16": lowest >= 5.3,
        "bf16": lowest >= 8.0,            # Ampere
        "fp8": lowest >= 8.9,             # Ada / Hopper
        "int4_tensor": 7.5 <= lowest < 8.0 or lowest >= 8.0,
        "flash_attention_2": lowest >= 8.0,
        # The dtype every runner should actually use, so no runner has to
        # rediscover this. On a T4 it is float16 and nothing else.
        "dtype": ("bfloat16" if lowest >= 8.0 else
                  "float16" if lowest >= 5.3 else "float32"),
        "attention": ("flash_attention_2" if lowest >= 8.0 else "sdpa"),
    }


def host() -> str:
    """Which kind of machine this is: "kaggle" or "local".

    The only consumer is the `kaggle_ok` flag in the registry, and it needs this
    rather than a GPU name because the constraint is the whole environment — two
    T4s, a nine-hour ceiling, no persistent disk — not the card alone. A T4 in a
    workstation with no session limit is not the same host.

    Kaggle sets KAGGLE_KERNEL_RUN_TYPE in every kernel, interactive or batch;
    /kaggle/working is checked too so a shell that lost the environment still
    identifies itself correctly.
    """
    if os.environ.get("VIOS_HOST"):
        return os.environ["VIOS_HOST"].strip().lower()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    return "kaggle" if os.path.isdir("/kaggle/working") else "local"


def probe(scratch: str = ".") -> dict:
    """Everything the planner needs, in one dict.

    `usable_vram_mb` is the number the cohort packer bin-packs against: free
    VRAM on the *smallest* card minus headroom, because a component is assigned
    to one card and the plan has to hold for whichever card it lands on.
    """
    gpus = _torch_gpus() or _nvidia_smi()
    caps = capabilities(gpus)
    per_card = [max(g["free_mb"] - VRAM_HEADROOM_MB, 0) for g in gpus]
    ram = _system_ram_mb()
    try:
        disk = shutil.disk_usage(scratch).free // (1024 ** 2)
    except OSError:
        disk = 0
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "host": host(),
        "vram_total_mb": sum(g["total_mb"] for g in gpus),
        "vram_free_mb": sum(g["free_mb"] for g in gpus),
        "usable_vram_mb": min(per_card) if per_card else 0,
        "usable_vram_total_mb": sum(per_card),
        "ram_available_mb": ram,
        "ram_known": bool(ram),   # 0 means "could not measure", not "none free"
        "usable_ram_mb": max(ram - RAM_HEADROOM_MB, 0),
        "disk_free_mb": disk,
        "cpus": os.cpu_count() or 1,
        "scratch": os.path.abspath(scratch),
        **caps,
    }


def describe(res: dict) -> str:
    """One line for the log and the tab's header."""
    ram = (f"{res['ram_available_mb'] / 1024:.1f} GB RAM"
           if res.get("ram_known") else "RAM unknown")
    if not res["gpu_count"]:
        return (f"CPU only · {res['cpus']} cores · {ram} · "
                f"{res['disk_free_mb'] / 1024:.0f} GB disk")
    names = {}
    for g in res["gpus"]:
        names[g["name"]] = names.get(g["name"], 0) + 1
    label = ", ".join(f"{n}× {k}" for k, n in names.items())
    return (f"{label} · {res['vram_free_mb'] / 1024:.1f}/"
            f"{res['vram_total_mb'] / 1024:.0f} GB VRAM free · "
            f"sm_{res['compute_capability']:g} ({res['dtype']}) · "
            f"{ram} · {res['disk_free_mb'] / 1024:.0f} GB disk")
