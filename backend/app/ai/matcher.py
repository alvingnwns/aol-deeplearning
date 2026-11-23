from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .encoder import EMBEDDING_DIM, unit_vector
from .errors import AudioRejected, RejectionReason
from .settings import VoiceSettings

VOICEPRINT_DTYPE = "<f4"
VOICEPRINT_BYTES = EMBEDDING_DIM * np.dtype(VOICEPRINT_DTYPE).itemsize


class Decision(str, Enum):
    PRESENT = "present"
    RETRY = "retry"
    REJECTED = "rejected"


class VerdictReason(str, Enum):
    MATCH = "match"
    LOW_CONFIDENCE = "low_confidence"
    SIMILAR_VOICE = "similar_voice"
    VOICE_MISMATCH = "voice_mismatch"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: VerdictReason
    student_id: str
    score: float
    runner_up_id: str | None
    runner_up_score: float | None

    @property
    def margin(self) -> float | None:
        return None if self.runner_up_score is None else self.score - self.runner_up_score

    def as_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "student_id": self.student_id,
            "score": round(self.score, 4),
            "runner_up_id": self.runner_up_id,
            "runner_up_score": None if self.runner_up_score is None else round(self.runner_up_score, 4),
            "margin": None if self.margin is None else round(self.margin, 4),
        }


def voiceprint_to_bytes(voiceprint: np.ndarray) -> bytes:
    assert voiceprint.shape == (EMBEDDING_DIM,), voiceprint.shape
    return voiceprint.astype(VOICEPRINT_DTYPE).tobytes()


def voiceprint_from_bytes(blob: bytes) -> np.ndarray:
    if len(blob) != VOICEPRINT_BYTES:
        raise ValueError(f"voiceprint must be {VOICEPRINT_BYTES} bytes, got {len(blob)}")
    return np.frombuffer(blob, dtype=VOICEPRINT_DTYPE).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit_vector(a), unit_vector(b)))


def build_voiceprint(embeddings: Sequence[np.ndarray], settings: VoiceSettings) -> np.ndarray:
    if len(embeddings) < settings.min_enroll_clips:
        raise AudioRejected(
            RejectionReason.NOT_ENOUGH_CLIPS,
            f"Please record at least {settings.min_enroll_clips} clips; got {len(embeddings)}.",
        )
    if len(embeddings) > 1:
        for index, embedding in enumerate(embeddings):
            others = [e for i, e in enumerate(embeddings) if i != index]
            if cosine(embedding, np.mean(others, axis=0)) < settings.accept_threshold:
                raise AudioRejected(
                    RejectionReason.INCONSISTENT_CLIPS,
                    f"Clip {index + 1} does not sound like the others; please record it again.",
                    index,
                )
    return unit_vector(np.mean(embeddings, axis=0))


def verify(
    probe: np.ndarray,
    claimed_id: str,
    voiceprints: Mapping[str, np.ndarray],
    settings: VoiceSettings,
) -> Verdict:
    if claimed_id not in voiceprints:
        raise AudioRejected(RejectionReason.NOT_ENROLLED, f"Student {claimed_id} has not enrolled a voice yet.")
    score = cosine(probe, voiceprints[claimed_id])
    rivals = {sid: cosine(probe, vp) for sid, vp in voiceprints.items() if sid != claimed_id}
    runner_up_id = max(rivals, key=rivals.__getitem__) if rivals else None
    runner_up_score = rivals[runner_up_id] if runner_up_id is not None else None

    def verdict(decision: Decision, reason: VerdictReason) -> Verdict:
        return Verdict(decision, reason, claimed_id, score, runner_up_id, runner_up_score)

    if score < settings.retry_floor:
        return verdict(Decision.REJECTED, VerdictReason.VOICE_MISMATCH)
    if score < settings.accept_threshold:
        return verdict(Decision.RETRY, VerdictReason.LOW_CONFIDENCE)
    if runner_up_score is not None and score - runner_up_score < settings.min_margin:
        return verdict(Decision.RETRY, VerdictReason.SIMILAR_VOICE)
    return verdict(Decision.PRESENT, VerdictReason.MATCH)


def closest_pairs(voiceprints: Mapping[str, np.ndarray], top_k: int = 10) -> list[tuple[str, str, float]]:
    scored = [(a, b, cosine(voiceprints[a], voiceprints[b])) for a, b in combinations(sorted(voiceprints), 2)]
    return sorted(scored, key=lambda pair: pair[2], reverse=True)[:top_k]
