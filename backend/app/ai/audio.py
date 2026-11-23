import io
import shutil
import subprocess
import wave

import numpy as np

from .errors import AudioRejected, RejectionReason
from .settings import VoiceSettings

SAMPLE_RATE = 16_000
PCM16_FULL_SCALE = 32768.0
SILENCE_FLOOR_DB = -120.0
FFMPEG_TIMEOUT_S = 30


def decode(raw: bytes) -> np.ndarray:
    if not raw:
        raise AudioRejected(RejectionReason.UNDECODABLE, "The recording is empty.")
    if raw[:4] == b"RIFF":
        try:
            return _decode_pcm_wav(raw)
        except (wave.Error, EOFError, ValueError):
            pass
    return _decode_with_ffmpeg(raw)


def _decode_pcm_wav(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw)) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM is decoded natively")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / PCM16_FULL_SCALE
    mono = samples.reshape(-1, channels).mean(axis=1)
    return _resample(mono, rate)


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return samples
    import torch
    import torchaudio.functional

    resampled = torchaudio.functional.resample(torch.from_numpy(samples), rate, SAMPLE_RATE)
    return resampled.numpy().astype(np.float32)


def _decode_with_ffmpeg(raw: bytes) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioRejected(
            RejectionReason.UNDECODABLE,
            "Only 16-bit PCM WAV can be read on this server (ffmpeg is not installed).",
        )
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=raw,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise AudioRejected(RejectionReason.UNDECODABLE, "The recording could not be read as audio.")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def duration_s(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


def rms_db(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return SILENCE_FLOOR_DB
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(SILENCE_FLOOR_DB, 20.0 * np.log10(rms)) if rms > 0 else SILENCE_FLOOR_DB


def check_quality(samples: np.ndarray, settings: VoiceSettings, clip_index: int | None = None) -> None:
    length = duration_s(samples)
    if length < settings.min_duration_s:
        raise AudioRejected(
            RejectionReason.TOO_SHORT,
            f"The recording is {length:.1f}s long; please speak for at least {settings.min_duration_s:.0f}s.",
            clip_index,
        )
    loudness = rms_db(samples)
    if loudness < settings.min_rms_db:
        raise AudioRejected(
            RejectionReason.TOO_QUIET,
            "The recording is too quiet; move closer to the microphone and try again.",
            clip_index,
        )


def to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0 - 1.0 / PCM16_FULL_SCALE)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((pcm * PCM16_FULL_SCALE).astype("<i2").tobytes())
    return buffer.getvalue()
