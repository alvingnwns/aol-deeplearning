from pathlib import Path

import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

from .pengaturan import MODEL_PRETRAINED


class PengenalSuara:
    def __init__(self, folder_model: Path, perangkat: str = "cpu"):
        self.perangkat = perangkat
        self.model = EncoderClassifier.from_hparams(
            source=MODEL_PRETRAINED,
            savedir=str(folder_model),
            run_opts={"device": perangkat},
        )
        self.model.eval()

    @property
    def jumlah_parameter(self) -> int:
        return sum(parameter.numel() for parameter in self.model.mods.embedding_model.parameters())

    def embedding(self, suara: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            hasil = self.model.encode_batch(suara.unsqueeze(0).to(self.perangkat)).reshape(-1)
        return F.normalize(hasil, dim=0).cpu()

    def fbank(self, suara: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.mods.compute_features(suara.unsqueeze(0).to(self.perangkat))[0].cpu()
