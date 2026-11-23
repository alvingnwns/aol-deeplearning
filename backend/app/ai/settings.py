import os
from dataclasses import dataclass
from pathlib import Path

PRETRAINED_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ENCODER_ALIASES = {"ecapa": PRETRAINED_ECAPA_SOURCE}


@dataclass(frozen=True)
class VoiceSettings:
    accept_threshold: float = 0.30
    retry_floor: float = 0.20
    min_margin: float = 0.05
    min_duration_s: float = 2.0
    min_rms_db: float = -40.0
    min_enroll_clips: int = 3
    encoder: str = "ecapa"
    model_dir: Path = Path("/app/data/models")
    archive_dir: Path | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not -1.0 <= self.retry_floor <= self.accept_threshold <= 1.0:
            raise ValueError("expected -1 <= retry_floor <= accept_threshold <= 1")
        if self.min_margin < 0:
            raise ValueError("min_margin must be >= 0")
        if self.min_enroll_clips < 1:
            raise ValueError("min_enroll_clips must be >= 1")

    @property
    def encoder_source(self) -> str:
        return ENCODER_ALIASES.get(self.encoder, self.encoder)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "VoiceSettings":
        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            accept_threshold=float(env.get("THRESHOLD", defaults.accept_threshold)),
            retry_floor=float(env.get("RETRY_FLOOR", defaults.retry_floor)),
            min_margin=float(env.get("MIN_MARGIN", defaults.min_margin)),
            min_duration_s=float(env.get("MIN_DURATION_S", defaults.min_duration_s)),
            min_rms_db=float(env.get("MIN_RMS_DB", defaults.min_rms_db)),
            min_enroll_clips=int(env.get("MIN_ENROLL_CLIPS", defaults.min_enroll_clips)),
            encoder=env.get("ENCODER", defaults.encoder),
            model_dir=Path(env.get("MODEL_DIR", str(defaults.model_dir))),
            archive_dir=Path(env["ARCHIVE_DIR"]) if env.get("ARCHIVE_DIR") else None,
            device=env.get("ENCODER_DEVICE", defaults.device),
        )
