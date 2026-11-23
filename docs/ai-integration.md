# VoiceAttend AI module: integration guide

`backend/app/ai` turns voice recordings into voiceprints and decides whether a student is present.
It owns **no HTTP routes, no database and no UI**. The backend calls three methods, stores the bytes it gets back, and returns the JSON it gets back to the frontend.

```
frontend ──audio──▶ backend route ──bytes──▶ VoiceAI.enroll / VoiceAI.verify
                         │                          │
                         ◀── voiceprint bytes ──────┘   (backend stores them in SQLite)
                         ◀── verdict dict ──────────    (backend returns it to the frontend)
```

See `diagrams/voiceattend-system-flow.html` for the whole system and `diagrams/ecapa-tdnn-flow.html` for the model.

## 1. Setup (backend owner)

- `speechbrain` is already in `backend/app/requirements.txt`; it pulls in `torch` and `torchaudio`.
- Install **ffmpeg** in the Docker image so browser recordings (webm/ogg/mp4) can be decoded:
  `RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*`.
  Without ffmpeg only 16-bit PCM WAV is accepted.
- The first `warm_up()` downloads the pretrained ECAPA model (~80 MB) from Hugging Face into `MODEL_DIR`. Mount `/app/data` as a volume so it downloads once.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `THRESHOLD` | `0.30` | Cosine score at or above which a voice counts as the claimed student |
| `RETRY_FLOOR` | `0.20` | Below this the attempt is rejected; between this and `THRESHOLD` the student is asked to retry |
| `MIN_MARGIN` | `0.05` | The claimed student must beat the most similar other student by this much, otherwise retry |
| `MIN_DURATION_S` | `2.0` | Shortest accepted clip (3–5 s recommended) |
| `MIN_RMS_DB` | `-40` | Quietest accepted clip |
| `MIN_ENROLL_CLIPS` | `3` | Clips required to enroll |
| `ENCODER` | `ecapa` | `ecapa` = pretrained VoxCeleb model, or a path / Hugging Face id of a fine-tuned model |
| `MODEL_DIR` | `/app/data/models` | Where model weights are cached |
| `ARCHIVE_DIR` | unset | If set, accepted clips are saved here as training data for retraining |
| `ENCODER_DEVICE` | `cpu` | `cpu` or `cuda` |

## 2. Python API (backend owner)

```python
from app.ai import AudioRejected, Decision, VoiceAI

voice_ai = VoiceAI()
voice_ai.warm_up()

enrollment = voice_ai.enroll("2602123456", [clip1_bytes, clip2_bytes, clip3_bytes])
enrollment.voiceprint
enrollment.encoder_version

verdict = voice_ai.verify("2602123456", clip_bytes, {"2602123456": blob, "2602999999": other_blob})
verdict.decision
verdict.as_dict()
```

| Call | Returns | Raises |
|---|---|---|
| `enroll(student_id, clips: list[bytes])` | `Enrollment(student_id, voiceprint: bytes, encoder_version: str, clip_count)` | `AudioRejected` |
| `verify(student_id, clip: bytes, voiceprints: dict[str, bytes])` | `Verdict` (`.as_dict()` is ready-to-return JSON) | `AudioRejected` |
| `rebuild_voiceprints()` | `dict[student_id, Enrollment]` rebuilt from `ARCHIVE_DIR` | `RuntimeError` without an archive |
| `encoder_version` | string to store next to every voiceprint | |

- `voiceprint` is always **768 bytes** (192 × float32). Store it as a BLOB.
- Pass **every enrolled student of the class** in `voiceprints`, not just the claimed one. The other voiceprints are used for the similar-voice check.
- Only pass voiceprints whose `encoder_version` equals `voice_ai.encoder_version`. After the model is retrained, call `rebuild_voiceprints()` and overwrite the stored rows.
- Both calls are CPU-bound (≈0.1–0.5 s per clip). In async FastAPI routes wrap them in `run_in_threadpool`.
- `student_id` may only contain letters, digits, `_` and `-` (it becomes a folder name in the archive).

### Suggested tables

```sql
CREATE TABLE voiceprints (
  student_id      TEXT PRIMARY KEY REFERENCES students(student_id),
  voiceprint      BLOB NOT NULL,
  encoder_version TEXT NOT NULL,
  clip_count      INTEGER NOT NULL,
  enrolled_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
  session_id  INTEGER NOT NULL REFERENCES class_sessions(id),
  student_id  TEXT NOT NULL REFERENCES students(student_id),
  score       REAL NOT NULL,
  marked_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (session_id, student_id)
);
```

### Example routes

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ai import AudioRejected, Decision, VoiceAI

voice_ai = VoiceAI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(voice_ai.warm_up)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/api/enroll")
async def enroll(student_id: str = Form(...), clips: list[UploadFile] = File(...)):
    raw_clips = [await clip.read() for clip in clips]
    try:
        enrollment = await run_in_threadpool(voice_ai.enroll, student_id, raw_clips)
    except AudioRejected as rejection:
        raise HTTPException(status_code=422, detail=rejection.as_dict())
    db.upsert_voiceprint(enrollment.student_id, enrollment.voiceprint, enrollment.encoder_version, enrollment.clip_count)
    return enrollment.as_dict()


@app.post("/api/attend")
async def attend(session_id: int = Form(...), student_id: str = Form(...), clip: UploadFile = File(...)):
    db.require_open_session(session_id)
    voiceprints = db.voiceprints_for_session(session_id, voice_ai.encoder_version)
    try:
        verdict = await run_in_threadpool(voice_ai.verify, student_id, await clip.read(), voiceprints)
    except AudioRejected as rejection:
        raise HTTPException(status_code=422, detail=rejection.as_dict())
    if verdict.decision is Decision.PRESENT:
        db.mark_present(session_id, student_id, verdict.score)
    return verdict.as_dict()
```

`db.*` are the backend's own helpers. The AI module never touches the database.

## 3. Frontend contract (frontend owner)

**Send** `multipart/form-data`:

- Enroll: `student_id` + 3 or more `clips` files (one per recording).
- Attend: `session_id` + `student_id` + one `clip` file.

Any format `MediaRecorder` produces is fine (webm/opus, ogg, mp4) as long as the server has ffmpeg. Record **3–5 seconds** of the student reading the on-screen phrase.

```js
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const recorder = new MediaRecorder(stream);
const chunks = [];
recorder.ondataavailable = (event) => chunks.push(event.data);
recorder.start();
await new Promise((resolve) => setTimeout(resolve, 4000));
recorder.stop();
await new Promise((resolve) => (recorder.onstop = resolve));
const form = new FormData();
form.append("session_id", sessionId);
form.append("student_id", studentId);
form.append("clip", new Blob(chunks, { type: recorder.mimeType }), "clip.webm");
const response = await fetch("/api/attend", { method: "POST", body: form });
```

**Receive** from `/api/attend` (HTTP 200):

```json
{
  "decision": "present",
  "reason": "match",
  "student_id": "2602123456",
  "score": 0.71,
  "runner_up_id": "2602999999",
  "runner_up_score": 0.22,
  "margin": 0.49
}
```

| `decision` | `reason` | Show the student |
|---|---|---|
| `present` | `match` | "You're marked present." |
| `retry` | `low_confidence` | "We couldn't confirm your voice, please try again." |
| `retry` | `similar_voice` | "Your voice is close to a classmate's, please try again." (after 2 retries, ask the lecturer) |
| `rejected` | `voice_mismatch` | "This voice doesn't match the enrolled student." |

**Receive** on bad audio (HTTP 422), `detail` is `{"reason", "message", "clip_index"}`. `message` is safe to show as is; `clip_index` says which enrollment clip to re-record.

| `reason` | When |
|---|---|
| `undecodable` | File isn't audio, or the server has no ffmpeg for this format |
| `too_short` | Clip shorter than `MIN_DURATION_S` |
| `too_quiet` | Clip quieter than `MIN_RMS_DB` |
| `not_enough_clips` | Fewer than `MIN_ENROLL_CLIPS` at enrollment |
| `inconsistent_clips` | One enrollment clip sounds like a different person or is mostly noise |
| `not_enrolled` | Student has no voiceprint yet: send them to enrollment |

## 4. Tuning and checks (anyone)

Run from `backend/`:

```bash
python -m app.ai score alice_1.wav alice_2.wav
python -m app.ai closest-pairs /app/data/clips --top 10
```

- `score` prints the cosine similarity of two recordings: use it to sanity-check `THRESHOLD` with real classmates.
- `closest-pairs` lists the most similar-sounding enrolled students. If a pair scores near `THRESHOLD`, raise `THRESHOLD` or `MIN_MARGIN` before going live.

## 5. Retraining

Retraining is offline and never needed to add a student. When `ARCHIVE_DIR` holds enough data (rule of thumb: 50+ students × 10+ clips), fine-tune on Colab, point `ENCODER` at the new model, restart, and call `rebuild_voiceprints()`. Archive layout: `ARCHIVE_DIR/<student_id>/<utc-timestamp>-<enroll|attend>.wav` (16 kHz mono PCM16).
