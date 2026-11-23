import argparse
import json
import sys
from pathlib import Path

from . import matcher
from .archive import ClipArchive, ClipKind
from .errors import AudioRejected
from .service import VoiceAI


def score_files(ai: VoiceAI, first: Path, second: Path) -> dict:
    a, b = (ai.encoder.embed(ai.prepare(path.read_bytes())) for path in (first, second))
    similarity = matcher.cosine(a, b)
    return {"score": round(similarity, 4), "accept_threshold": ai.settings.accept_threshold,
            "same_speaker": similarity >= ai.settings.accept_threshold}


def closest_pairs(ai: VoiceAI, archive_root: Path, top_k: int) -> dict:
    ai.archive = ClipArchive(archive_root)
    voiceprints = {sid: e.voiceprint for sid, e in ai.rebuild_voiceprints().items()}
    pairs = ai.closest_pairs(voiceprints, top_k)
    return {"students": len(voiceprints), "accept_threshold": ai.settings.accept_threshold,
            "closest_pairs": [{"a": a, "b": b, "score": round(s, 4)} for a, b, s in pairs]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ai", description="VoiceAttend speaker-verification tools")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="cosine similarity between the speakers of two audio files")
    score.add_argument("first", type=Path)
    score.add_argument("second", type=Path)
    pairs = commands.add_parser("closest-pairs", help="most similar-sounding enrolled students in a clip archive")
    pairs.add_argument("archive_root", type=Path)
    pairs.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    ai = VoiceAI()
    try:
        if args.command == "score":
            result = score_files(ai, args.first, args.second)
        else:
            result = closest_pairs(ai, args.archive_root, args.top)
    except AudioRejected as rejection:
        print(json.dumps(rejection.as_dict(), indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
