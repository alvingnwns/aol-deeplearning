from enum import Enum


class RejectionReason(str, Enum):
    UNDECODABLE = "undecodable"
    TOO_SHORT = "too_short"
    TOO_QUIET = "too_quiet"
    NOT_ENOUGH_CLIPS = "not_enough_clips"
    INCONSISTENT_CLIPS = "inconsistent_clips"
    NOT_ENROLLED = "not_enrolled"
    ENCODER_MISMATCH = "encoder_mismatch"


class AudioRejected(Exception):
    def __init__(self, reason: RejectionReason, message: str, clip_index: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.clip_index = clip_index

    def as_dict(self) -> dict:
        return {"reason": self.reason.value, "message": self.message, "clip_index": self.clip_index}
