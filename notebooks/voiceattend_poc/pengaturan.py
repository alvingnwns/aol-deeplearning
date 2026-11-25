from dataclasses import dataclass

SAMPLE_RATE = 16_000
MODEL_PRETRAINED = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass(frozen=True)
class Pengaturan:
    ambang_hadir: float = 0.30
    ambang_ulangi: float = 0.20
    selisih_minimal: float = 0.05
    durasi_minimal_detik: float = 2.0
    volume_minimal_db: float = -40.0
    min_rekaman_daftar: int = 3
