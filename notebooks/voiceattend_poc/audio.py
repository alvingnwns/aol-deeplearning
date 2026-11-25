import subprocess
from pathlib import Path

import torch

from .pengaturan import SAMPLE_RATE, Pengaturan


def muat_audio(path: Path) -> torch.Tensor:
    hasil = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    return torch.frombuffer(bytearray(hasil.stdout), dtype=torch.float32)


def durasi_detik(suara: torch.Tensor) -> float:
    return suara.numel() / SAMPLE_RATE


def volume_db(suara: torch.Tensor) -> float:
    return float(20 * torch.log10(suara.pow(2).mean().sqrt() + 1e-12))


def masalah_kualitas(suara: torch.Tensor, pengaturan: Pengaturan) -> str | None:
    if durasi_detik(suara) < pengaturan.durasi_minimal_detik:
        return f"rekaman terlalu pendek ({durasi_detik(suara):.1f} detik)"
    if volume_db(suara) < pengaturan.volume_minimal_db:
        return f"rekaman terlalu pelan ({volume_db(suara):.0f} dB)"
    return None
