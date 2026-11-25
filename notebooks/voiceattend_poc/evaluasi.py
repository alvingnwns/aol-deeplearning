from itertools import combinations
from pathlib import Path

import pandas as pd

from .sistem import HADIR, SistemAbsensi, kemiripan


def jalankan_absensi(sistem: SistemAbsensi, folder: Path, uji: pd.DataFrame) -> pd.DataFrame:
    baris_hasil = []
    for baris in uji.itertuples(index=False):
        hasil = sistem.absen(folder / baris.file, baris.klaim_nim)
        prediksi = "hadir" if hasil.keputusan == HADIR else "tidak_hadir"
        baris_hasil.append({
            "file": baris.file,
            "pemilik_suara": baris.pemilik_suara,
            "klaim_nim": baris.klaim_nim,
            "harapan": baris.harapan,
            **hasil.as_dict(),
            "prediksi": prediksi,
            "benar": prediksi == baris.harapan,
        })
    return pd.DataFrame(baris_hasil)


def ringkasan(hasil: pd.DataFrame) -> pd.DataFrame:
    asli = hasil[hasil.harapan == "hadir"]
    penyusup = hasil[hasil.pemilik_suara != hasil.klaim_nim]
    baris = [
        ("Akurasi", "prediksi sesuai kunci jawaban", int(hasil.benar.sum()), len(hasil)),
        ("FRR", "mahasiswa asli gagal absen", int((asli.prediksi != "hadir").sum()), len(asli)),
        ("FAR", "penyusup berhasil absen", int((penyusup.prediksi == "hadir").sum()), len(penyusup)),
    ]
    return pd.DataFrame(
        [{"metrik": nama, "arti": arti, "nilai": f"{jumlah / total:.1%}" if total else "-", "jumlah": f"{jumlah}/{total}"} for nama, arti, jumlah, total in baris]
    )


def skor_semua_pasangan(sistem: SistemAbsensi, folder: Path, uji: pd.DataFrame) -> pd.DataFrame:
    pemilik = dict(zip(uji.file, uji.pemilik_suara))
    pasangan = []
    for file in pemilik:
        embedding = sistem.embedding_dari_file(folder / file)
        if embedding is None:
            continue
        for nim, voiceprint in sistem.voiceprints.items():
            pasangan.append({"file": file, "nim": nim, "skor": kemiripan(embedding, voiceprint), "orang_sama": pemilik[file] == nim})
    return pd.DataFrame(pasangan)


def tabel_ambang(pasangan: pd.DataFrame, daftar_ambang: list[float]) -> pd.DataFrame:
    skor_sama = pasangan[pasangan.orang_sama].skor
    skor_beda = pasangan[~pasangan.orang_sama].skor
    return pd.DataFrame({
        "ambang": daftar_ambang,
        "FRR: asli ditolak": [(skor_sama < ambang).mean() for ambang in daftar_ambang],
        "FAR: penyusup diterima": [(skor_beda >= ambang).mean() for ambang in daftar_ambang],
    })


def pasangan_termirip(sistem: SistemAbsensi, jumlah: int = 5) -> pd.DataFrame:
    semua = [
        {"NIM 1": a, "NIM 2": b, "skor": kemiripan(sistem.voiceprints[a], sistem.voiceprints[b])}
        for a, b in combinations(sorted(sistem.voiceprints), 2)
    ]
    return pd.DataFrame(semua).nlargest(jumlah, "skor")
