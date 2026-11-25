import base64
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from IPython.display import Javascript, display

REKAM_JS = """
async function rekamSuara(durasiMs) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const potongan = [];
  recorder.ondataavailable = (event) => potongan.push(event.data);
  const selesai = new Promise((resolve) => (recorder.onstop = resolve));
  recorder.start();
  await new Promise((resolve) => setTimeout(resolve, durasiMs));
  recorder.stop();
  await selesai;
  stream.getTracks().forEach((track) => track.stop());
  const isi = new Uint8Array(await new Blob(potongan, { type: recorder.mimeType }).arrayBuffer());
  let biner = "";
  for (let i = 0; i < isi.length; i += 1) biner += String.fromCharCode(isi[i]);
  return btoa(biner);
}
"""


def kalimat_untuk(naskah: pd.DataFrame, mahasiswa: pd.DataFrame, id_kalimat: str, nim: str) -> str:
    template = naskah.set_index("id_kalimat").kalimat[id_kalimat]
    nama = mahasiswa.set_index("nim").nama.get(nim, "<nama>")
    return template.format(nama=nama, nim=nim, kode=f"{random.randint(0, 999_999):06d}")


def rekam_mikrofon(detik: float) -> bytes:
    from google.colab.output import eval_js

    display(Javascript(REKAM_JS))
    return base64.b64decode(eval_js(f"rekamSuara({int(detik * 1000)})"))


def simpan_rekaman(folder: Path, isi_audio: bytes, nim: str, jenis: str, id_kalimat: str, ekstensi: str = "webm") -> Path:
    if jenis == "daftar":
        tujuan = folder / "audio" / "daftar" / nim / f"{id_kalimat}.{ekstensi}"
    else:
        tujuan = folder / "audio" / "uji" / f"{nim}_{datetime.now():%Y%m%d-%H%M%S}.{ekstensi}"
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    tujuan.write_bytes(isi_audio)
    baris = {
        "file": tujuan.relative_to(folder).as_posix(),
        "pemilik_suara": nim,
        "jenis": jenis,
        "klaim_nim": nim,
        "harapan": "hadir" if jenis == "uji" else "",
        "id_kalimat": id_kalimat,
    }
    pd.DataFrame([baris]).to_csv(folder / "manifest.csv", mode="a", header=False, index=False)
    print(f"Tersimpan: {baris['file']} (baris baru ditambahkan ke manifest.csv)")
    return tujuan
