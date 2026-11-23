import threading
from typing import Protocol

import numpy as np

from .settings import VoiceSettings

EMBEDDING_DIM = 192


class SpeakerEncoder(Protocol):
    version: str

    def embed(self, samples: np.ndarray) -> np.ndarray: ...


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (vector / norm).astype(np.float32)


class EcapaEncoder:
    def __init__(self, settings: VoiceSettings):
        self.version = settings.encoder_source
        self._settings = settings
        self._model = None
        self._load_lock = threading.Lock()

    def _loaded_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from speechbrain.inference.speaker import EncoderClassifier

                    savedir = self._settings.model_dir / self.version.replace("/", "--")
                    self._model = EncoderClassifier.from_hparams(
                        source=self._settings.encoder_source,
                        savedir=str(savedir),
                        run_opts={"device": self._settings.device},
                    )
                    self._model.eval()
        return self._model

    def warm_up(self) -> None:
        self._loaded_model()

    def embed(self, samples: np.ndarray) -> np.ndarray:
        import torch

        model = self._loaded_model()
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            embedding = model.encode_batch(batch).reshape(-1).cpu().numpy()
        assert embedding.shape == (EMBEDDING_DIM,), embedding.shape
        return unit_vector(embedding)
