import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import PengenalSuara
from .pengaturan import Pengaturan

WARNA_KEPUTUSAN = {
    "HADIR": "background-color: #d4edda",
    "ULANGI": "background-color: #fff3cd",
    "TOLAK": "background-color: #f8d7da",
}
KOLOM_HASIL = ["file", "pemilik_suara", "klaim_nim", "harapan", "keputusan", "alasan", "skor", "selisih", "benar"]


def tabel(df: pd.DataFrame):
    return df.style.hide(axis="index")


def tabel_hasil(hasil: pd.DataFrame):
    return (
        hasil[KOLOM_HASIL]
        .style.map(lambda keputusan: WARNA_KEPUTUSAN.get(keputusan, ""), subset=["keputusan"])
        .map(lambda benar: f"color: {'#2e7d32' if benar else '#c62828'}; font-weight: bold", subset=["benar"])
        .format({"skor": "{:.3f}", "selisih": "{:.3f}"}, na_rep="–")
        .hide(axis="index")
    )


def tabel_ambang(tabel_df: pd.DataFrame, ambang_sekarang: float):
    sorot = (tabel_df.ambang - ambang_sekarang).abs() < 1e-9
    return (
        tabel_df.style.format({"ambang": "{:.2f}", "FRR: asli ditolak": "{:.1%}", "FAR: penyusup diterima": "{:.1%}"})
        .apply(lambda kolom: ["font-weight: bold; background-color: #e8eaf6" if aktif else "" for aktif in sorot], axis=0)
        .hide(axis="index")
    )


def gambar_fbank_dan_voiceprint(pengenal: PengenalSuara, suara: torch.Tensor, voiceprints: dict[str, torch.Tensor]) -> None:
    fbank = pengenal.fbank(suara).numpy()
    nim_urut = sorted(voiceprints)
    fig, (kiri, kanan) = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    kiri.imshow(fbank.T, origin="lower", aspect="auto", cmap="magma")
    kiri.set_title(f"Langkah 1 · 'gambar' suara (Fbank {fbank.shape[1]} × {fbank.shape[0]})")
    kiri.set_xlabel("waktu (per 10 ms)")
    kiri.set_ylabel("pita frekuensi")
    kanan.imshow(torch.stack([voiceprints[nim] for nim in nim_urut]).numpy(), aspect="auto", cmap="coolwarm")
    kanan.set_yticks(range(len(nim_urut)), nim_urut)
    kanan.set_title("Langkah 2 · voiceprint tiap mahasiswa (192 angka)")
    kanan.set_xlabel("dimensi ke-")
    plt.tight_layout()
    plt.show()


def gambar_sebaran(pasangan: pd.DataFrame, pengaturan: Pengaturan) -> None:
    skor_sama = pasangan[pasangan.orang_sama].skor
    skor_beda = pasangan[~pasangan.orang_sama].skor
    fig, ax = plt.subplots(figsize=(10, 3.6))
    zona = [
        (-0.3, pengaturan.ambang_ulangi, "#f8d7da", "TOLAK"),
        (pengaturan.ambang_ulangi, pengaturan.ambang_hadir, "#fff3cd", "ULANGI"),
        (pengaturan.ambang_hadir, 1.0, "#d4edda", "HADIR"),
    ]
    for awal, akhir, warna, _ in zona:
        ax.axvspan(awal, akhir, color=warna, alpha=0.6, zorder=0)
    bins = np.linspace(-0.3, 1.0, 40)
    ax.hist(skor_beda, bins=bins, color="#d9534f", alpha=0.85, label=f"orang berbeda ({len(skor_beda)} pasangan)")
    ax.hist(skor_sama, bins=bins, color="#2e7d32", alpha=0.85, label=f"orang yang sama ({len(skor_sama)} pasangan)")
    puncak = ax.get_ylim()[1]
    for awal, akhir, _, label in zona:
        ax.text((awal + akhir) / 2, puncak * 0.9, label, ha="center", fontsize=10, fontweight="bold", color="#555")
    ax.set_xlim(-0.3, 1.0)
    ax.set_xlabel("skor kemiripan suara (cosine similarity)")
    ax.set_ylabel("jumlah pasangan")
    ax.legend(fontsize=8, loc="center right")
    plt.tight_layout()
    plt.show()
