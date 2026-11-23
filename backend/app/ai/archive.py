import re
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from .audio import decode, to_wav_bytes

SAFE_STUDENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ClipKind(str, Enum):
    ENROLL = "enroll"
    ATTEND = "attend"


def safe_student_id(student_id: str) -> str:
    if not SAFE_STUDENT_ID.fullmatch(student_id):
        raise ValueError("student_id may only contain letters, digits, '_' or '-' (max 64)")
    return student_id


class ClipArchive:
    def __init__(self, root: Path):
        self.root = Path(root)

    def save(self, student_id: str, samples: np.ndarray, kind: ClipKind) -> Path:
        folder = self.root / safe_student_id(student_id)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"{stamp}-{kind.value}.wav"
        path.write_bytes(to_wav_bytes(samples))
        return path

    def clips(self, kind: ClipKind | None = None) -> dict[str, list[Path]]:
        by_student: dict[str, list[Path]] = defaultdict(list)
        if not self.root.is_dir():
            return {}
        suffix = f"-{kind.value}.wav" if kind else ".wav"
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and SAFE_STUDENT_ID.fullmatch(p.name)):
            by_student[folder.name] = sorted(p for p in folder.iterdir() if p.name.endswith(suffix))
        return {sid: paths for sid, paths in by_student.items() if paths}

    def load(self, path: Path) -> np.ndarray:
        return decode(path.read_bytes())
