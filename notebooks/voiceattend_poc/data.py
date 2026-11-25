from pathlib import Path

import pandas as pd

JENIS_SAH = {"daftar", "uji"}
HARAPAN_SAH = {"hadir", "tidak_hadir"}


def baca_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def periksa(folder: Path, manifest: pd.DataFrame, naskah: pd.DataFrame) -> list[str]:
    masalah = []
    for nomor, baris in manifest.iterrows():
        lokasi = f"manifest.csv baris {nomor + 2}"
        if not (folder / baris.file).is_file():
            masalah.append(f"{lokasi}: file tidak ditemukan: {baris.file}")
        if baris.jenis not in JENIS_SAH:
            masalah.append(f"{lokasi}: jenis harus daftar atau uji")
        if baris.jenis == "uji" and baris.harapan not in HARAPAN_SAH:
            masalah.append(f"{lokasi}: harapan harus hadir atau tidak_hadir")
        if baris.id_kalimat and baris.id_kalimat not in set(naskah.id_kalimat):
            masalah.append(f"{lokasi}: id_kalimat {baris.id_kalimat} tidak ada di naskah.csv")
    return masalah


def baca_data(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    naskah = baca_csv(folder / "naskah.csv")
    mahasiswa = baca_csv(folder / "mahasiswa.csv")
    manifest = baca_csv(folder / "manifest.csv")
    masalah = periksa(folder, manifest, naskah)
    if masalah:
        raise ValueError("Data bermasalah:\n" + "\n".join(masalah))
    return naskah, mahasiswa, manifest


def ringkasan_per_nim(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.groupby(["klaim_nim", "jenis"]).size().unstack(fill_value=0).rename_axis(index="NIM", columns=None)
