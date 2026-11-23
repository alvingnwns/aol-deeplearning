from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import audio, matcher
from .archive import ClipArchive, ClipKind, safe_student_id
from .encoder import EcapaEncoder, SpeakerEncoder, unit_vector
from .errors import AudioRejected, RejectionReason
from .matcher import Decision, Verdict
from .settings import VoiceSettings


@dataclass(frozen=True)
class Enrollment:
    student_id: str
    voiceprint: bytes
    encoder_version: str
    clip_count: int

    def as_dict(self) -> dict:
        return {"student_id": self.student_id, "encoder_version": self.encoder_version, "clip_count": self.clip_count}


class VoiceAI:
    def __init__(
        self,
        settings: VoiceSettings | None = None,
        encoder: SpeakerEncoder | None = None,
        archive_root: Path | None = None,
    ):
        self.settings = settings or VoiceSettings.from_env()
        self.encoder = encoder or EcapaEncoder(self.settings)
        root = archive_root if archive_root is not None else self.settings.archive_dir
        self.archive = ClipArchive(root) if root is not None else None

    @property
    def encoder_version(self) -> str:
        return self.encoder.version

    def warm_up(self) -> None:
        warm = getattr(self.encoder, "warm_up", None)
        if warm is not None:
            warm()

    def prepare(self, raw: bytes, clip_index: int | None = None) -> np.ndarray:
        try:
            samples = audio.decode(raw)
        except AudioRejected as rejection:
            rejection.clip_index = clip_index
            raise
        audio.check_quality(samples, self.settings, clip_index)
        return samples

    def enroll(self, student_id: str, clips: Sequence[bytes]) -> Enrollment:
        safe_student_id(student_id)
        prepared = [self.prepare(raw, index) for index, raw in enumerate(clips)]
        embeddings = [self.encoder.embed(samples) for samples in prepared]
        voiceprint = matcher.build_voiceprint(embeddings, self.settings)
        if self.archive is not None:
            for samples in prepared:
                self.archive.save(student_id, samples, ClipKind.ENROLL)
        return Enrollment(student_id, matcher.voiceprint_to_bytes(voiceprint), self.encoder_version, len(prepared))

    def verify(self, student_id: str, clip: bytes, voiceprints: Mapping[str, bytes]) -> Verdict:
        safe_student_id(student_id)
        samples = self.prepare(clip)
        probe = self.encoder.embed(samples)
        decoded = {sid: matcher.voiceprint_from_bytes(blob) for sid, blob in voiceprints.items()}
        verdict = matcher.verify(probe, student_id, decoded, self.settings)
        if self.archive is not None and verdict.decision is Decision.PRESENT:
            self.archive.save(student_id, samples, ClipKind.ATTEND)
        return verdict

    def check_encoder_version(self, stored_version: str) -> None:
        if stored_version != self.encoder_version:
            raise AudioRejected(
                RejectionReason.ENCODER_MISMATCH,
                f"Voiceprint was made by {stored_version}, but the server now runs {self.encoder_version}; "
                "rebuild voiceprints from the clip archive.",
            )

    def rebuild_voiceprints(self) -> dict[str, Enrollment]:
        if self.archive is None:
            raise RuntimeError("rebuild_voiceprints needs archive_root")
        rebuilt = {}
        for student_id, paths in self.archive.clips(ClipKind.ENROLL).items():
            embeddings = [self.encoder.embed(self.archive.load(path)) for path in paths]
            voiceprint = unit_vector(np.mean(embeddings, axis=0))
            rebuilt[student_id] = Enrollment(
                student_id, matcher.voiceprint_to_bytes(voiceprint), self.encoder_version, len(paths)
            )
        return rebuilt

    def closest_pairs(self, voiceprints: Mapping[str, bytes], top_k: int = 10) -> list[tuple[str, str, float]]:
        decoded = {sid: matcher.voiceprint_from_bytes(blob) for sid, blob in voiceprints.items()}
        return matcher.closest_pairs(decoded, top_k)
