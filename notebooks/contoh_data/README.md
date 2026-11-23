# Data contoh VoiceAttend

Folder ini adalah data yang dibaca `notebooks/contoh_absensi_suara.ipynb`. Strukturnya sama dengan yang nanti dipakai backend: rekaman pendaftaran per NIM dan rekaman absensi.

```
contoh_data/
├── naskah.csv          kalimat yang harus diucapkan (spesifikasi rekaman)
├── mahasiswa.csv       daftar NIM dan nama
├── manifest.csv        menghubungkan setiap file audio dengan NIM, jenis, dan label harapan
├── sumber_audio.csv    asal setiap file audio contoh (atribusi lisensi)
├── audio/
│   ├── daftar/<NIM>/K01.wav   minimal 3 klip pendaftaran per mahasiswa
│   └── uji/<NIM>_1.wav        klip absensi yang akan diuji
└── hasil/              dibuat notebook: hasil_inferensi.csv, pendaftaran.csv, voiceprints.db
```

## Naskah rekaman (`naskah.csv`)

| id | jenis | yang diucapkan |
|---|---|---|
| K01 | daftar | "Nama saya {nama}, nomor induk mahasiswa saya {nim}." |
| K02 | daftar | "Saya mengikuti mata kuliah Deep Learning dan hadir di kelas hari ini." |
| K03 | daftar | "Satu, dua, tiga, empat, lima, enam, tujuh, delapan, sembilan, sepuluh." |
| K04 | daftar (opsional) | "Pagi ini saya berangkat ke kampus dengan semangat untuk belajar." |
| K05 | daftar (opsional) | "Kecerdasan buatan dapat mengenali suara setiap orang secara unik." |
| U01 | uji / absen | "Saya hadir. Kode saya {kode}." dengan 6 angka acak dari layar |

Panduan merekam:

- Durasi 3–6 detik per klip. Klip di bawah 2 detik ditolak (`too_short`).
- Ruangan tenang, HP atau mikrofon 20–30 cm dari mulut, suara bicara biasa (tidak berbisik, tidak berteriak).
- Rekam setiap kalimat sebagai file terpisah. Format apa pun bisa (wav, webm, m4a, mp3, ogg) asalkan ada ffmpeg; WAV 16-bit selalu bisa.
- Mesin ini **tidak bergantung pada isi kalimat** (text-independent). Naskah dipakai supaya durasi dan variasi bunyi cukup, dan kode acak di U01 mencegah orang memutar ulang rekaman lama.

## Kolom `manifest.csv`

| kolom | isi |
|---|---|
| `file` | path relatif dari folder ini, misalnya `audio/uji/2702000001_1.wav` |
| `pemilik_suara` | siapa yang sebenarnya berbicara (NIM, atau `tamu` untuk orang yang tidak terdaftar). Ini *ground truth* untuk validasi |
| `jenis` | `daftar` (dipakai membuat voiceprint) atau `uji` (dipakai untuk inferensi absensi) |
| `klaim_nim` | NIM yang diakui saat merekam. Untuk `daftar` sama dengan pemilik suara. Untuk `uji` boleh berbeda: itulah skenario penyusup |
| `harapan` | khusus `uji`: `hadir` jika sistem seharusnya menerima, `tidak_hadir` jika seharusnya menolak |
| `id_kalimat` | id dari `naskah.csv` |

Satu file boleh muncul di beberapa baris `uji` dengan `klaim_nim` berbeda. Contohnya `audio/uji/2702000001_1.wav` diuji sebagai pemiliknya sendiri (harus `hadir`) dan sebagai penyusup yang mengaku `2702000002` (harus `tidak_hadir`).

## Menambah rekaman sendiri

1. Tambahkan NIM dan nama ke `mahasiswa.csv`.
2. Simpan minimal 3 klip pendaftaran di `audio/daftar/<NIM>/K01.wav`, `K02.wav`, `K03.wav`.
3. Simpan klip absensi di `audio/uji/<NIM>_<n>.wav`.
4. Tambahkan satu baris per file ke `manifest.csv`.
5. Jalankan ulang notebook. Hasilnya ada di `hasil/hasil_inferensi.csv`.

Notebook juga punya sel untuk merekam langsung dari mikrofon di Colab dan menambahkan barisnya ke manifest secara otomatis.

## Asal audio contoh

Audio contoh di `audio/` adalah potongan maksimal 5 detik dari **LibriSpeech dev-clean** (Panayotov dkk., 2015), lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), sumber <https://www.openslr.org/12>. Isinya pembaca buku berbahasa Inggris, bukan naskah di atas. Audio ini hanya pengganti sampai tim punya rekaman sendiri, dan cocok untuk demo karena model tidak bergantung pada bahasa atau isi kalimat. NIM dan nama "Mahasiswa A–E" adalah data rekaan. Daftar file sumbernya ada di `sumber_audio.csv`.
