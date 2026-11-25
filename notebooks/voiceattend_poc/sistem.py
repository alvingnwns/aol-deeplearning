from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .audio import masalah_kualitas, muat_audio
from .model import PengenalSuara
from .pengaturan import Pengaturan

HADIR = "HADIR"
ULANGI = "ULANGI"
TOLAK = "TOLAK"


@dataclass(frozen=True)
class HasilDaftar:
    nim: str
    status: str
    jumlah_rekaman: int
    catatan: str = ""

    def __str__(self) -> str:
        catatan = f", {self.catatan}" if self.catatan else ""
        return f"{self.nim}: {self.status} ({self.jumlah_rekaman} rekaman{catatan})"


@dataclass(frozen=True)
class HasilAbsen:
    keputusan: str
    alasan: str
    skor: float | None = None
    pesaing_terdekat: str | None = None
    skor_pesaing: float | None = None
    selisih: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        skor = "" if self.skor is None else f", skor {self.skor:.3f}"
        return f"{self.keputusan} ({self.alasan}{skor})"


def kemiripan(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def buat_voiceprint(embeddings: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)


def putuskan(embedding: torch.Tensor, klaim_nim: str, voiceprints: dict[str, torch.Tensor], pengaturan: Pengaturan) -> HasilAbsen:
    skor = {nim: kemiripan(embedding, voiceprint) for nim, voiceprint in voiceprints.items()}
    skor_klaim = skor.pop(klaim_nim)
    pesaing = max(skor, key=skor.get, default=None)
    skor_pesaing = skor[pesaing] if pesaing is not None else None
    selisih = skor_klaim - skor_pesaing if skor_pesaing is not None else 1.0
    if skor_klaim < pengaturan.ambang_ulangi:
        keputusan, alasan = TOLAK, "suara tidak cocok dengan NIM"
    elif skor_klaim < pengaturan.ambang_hadir:
        keputusan, alasan = ULANGI, "skor di zona ragu"
    elif selisih < pengaturan.selisih_minimal:
        keputusan, alasan = ULANGI, f"terlalu mirip dengan {pesaing}"
    else:
        keputusan, alasan = HADIR, "suara cocok"
    return HasilAbsen(keputusan, alasan, round(skor_klaim, 3), pesaing, None if skor_pesaing is None else round(skor_pesaing, 3), round(selisih, 3))


class SistemAbsensi:
    def __init__(self, pengenal: PengenalSuara, pengaturan: Pengaturan = Pengaturan()):
        self.pengenal = pengenal
        self.pengaturan = pengaturan
        self.voiceprints: dict[str, torch.Tensor] = {}

    def embedding_dari_file(self, path: Path) -> torch.Tensor | None:
        suara = muat_audio(path)
        if masalah_kualitas(suara, self.pengaturan):
            return None
        return self.pengenal.embedding(suara)

    def daftar(self, nim: str, paths: list[Path]) -> HasilDaftar:
        semua_suara = [muat_audio(path) for path in paths]
        masalah = [f"{Path(path).name}: {pesan}" for path, suara in zip(paths, semua_suara) if (pesan := masalah_kualitas(suara, self.pengaturan))]
        if len(paths) < self.pengaturan.min_rekaman_daftar:
            masalah.append(f"butuh minimal {self.pengaturan.min_rekaman_daftar} rekaman")
        if masalah:
            return HasilDaftar(nim, "gagal", len(paths), "; ".join(masalah))
        self.voiceprints[nim] = buat_voiceprint([self.pengenal.embedding(suara) for suara in semua_suara])
        return HasilDaftar(nim, "terdaftar", len(paths))

    def absen(self, path: Path, klaim_nim: str) -> HasilAbsen:
        if klaim_nim not in self.voiceprints:
            return HasilAbsen(TOLAK, "NIM belum mendaftar")
        suara = muat_audio(path)
        masalah = masalah_kualitas(suara, self.pengaturan)
        if masalah:
            return HasilAbsen(ULANGI, masalah)
        return putuskan(self.pengenal.embedding(suara), klaim_nim, self.voiceprints, self.pengaturan)

    def simpan(self, path: Path) -> None:
        torch.save(self.voiceprints, path)

    def muat(self, path: Path) -> None:
        self.voiceprints = torch.load(path)
