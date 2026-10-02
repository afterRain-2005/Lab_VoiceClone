#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Iterator


AUDIO_FIELDS = ("audio", "wav", "path", "audio_path", "file", "filename")
TEXT_FIELDS = ("text", "transcript", "sentence", "normalized_text", "content")
SPEAKER_FIELDS = ("speaker_id", "speaker", "spk", "client_id", "voice")
LANGUAGE_FIELDS = ("language", "lang", "locale")


def first(item: dict, fields: tuple[str, ...], default: str = "") -> str:
    for field in fields:
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "unknown"


def duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as handle:
                    return handle.getnframes() / handle.getframerate()
            except (OSError, wave.Error, ZeroDivisionError):
                pass
        return 0.0


def metadata_rows(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = next(
                (payload[key] for key in ("data", "items", "samples", "records") if isinstance(payload.get(key), list)),
                [payload],
            )
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def generic_records(root: Path, source: str, split: str) -> list[dict]:
    records: list[dict] = []
    seen: set[Path] = set()
    metadata_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}
    )
    for metadata in metadata_files:
        try:
            for row_index, row in enumerate(metadata_rows(metadata)):
                audio_value = first(row, AUDIO_FIELDS)
                text = first(row, TEXT_FIELDS)
                if not audio_value or not text:
                    continue
                audio = Path(audio_value).expanduser()
                if not audio.is_absolute():
                    local = metadata.parent / audio
                    audio = local if local.is_file() else root / audio
                if not audio.is_file():
                    continue
                audio = audio.resolve()
                if audio in seen or " " in str(audio):
                    continue
                seen.add(audio)
                speaker = f"{source}_{safe(first(row, SPEAKER_FIELDS, audio.parent.name))}"
                uid = f"{source}_{safe(first(row, ('id', 'uid', 'utt', 'key'), audio.stem))}_{row_index:07d}"
                records.append(
                    {
                        "SourceDataset": source,
                        "Split": split,
                        "Singer": speaker,
                        "Uid": uid,
                        "Path": str(audio),
                        "Text": " ".join(text.split()),
                        "Duration": duration(audio),
                        "Language": first(row, LANGUAGE_FIELDS, "unknown"),
                        "NoiseTag": first(row, ("noise", "noise_tag", "condition"), "unknown"),
                    }
                )
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
            continue

    audio_suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    for audio in sorted(path for path in root.rglob("*") if path.suffix.lower() in audio_suffixes):
        resolved = audio.resolve()
        if resolved in seen or " " in str(resolved):
            continue
        text_path = next(
            (candidate for candidate in (audio.with_suffix(".txt"), audio.with_suffix(".normalized.txt")) if candidate.is_file()),
            None,
        )
        if text_path is None:
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        records.append(
            {
                "SourceDataset": source,
                "Split": split,
                "Singer": f"{source}_{safe(audio.parent.name)}",
                "Uid": f"{source}_{safe('_'.join(audio.relative_to(root).with_suffix('').parts))}",
                "Path": str(resolved),
                "Text": " ".join(text.split()),
                "Duration": duration(audio),
                "Language": "unknown",
                "NoiseTag": "unknown",
            }
        )
    return records


def libritts_records(root: Path) -> list[dict]:
    records: list[dict] = []
    for transcript in sorted(root.rglob("*.trans.tsv")):
        with transcript.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                utt = fields[0]
                text = fields[2] if len(fields) > 2 and fields[2].strip() else fields[1]
                audio = next(
                    (transcript.parent / f"{utt}{suffix}" for suffix in (".wav", ".flac") if (transcript.parent / f"{utt}{suffix}").is_file()),
                    None,
                )
                if audio is None:
                    continue
                records.append(
                    {
                        "SourceDataset": "libritts",
                        "Split": "train",
                        "Singer": f"libritts_{safe(utt.split('_')[0])}",
                        "Uid": f"libritts_{safe(utt)}",
                        "Path": str(audio.resolve()),
                        "Text": " ".join(text.split()),
                        "Duration": duration(audio),
                        "Language": "en",
                        "NoiseTag": "clean" if "clean" in str(audio).lower() else "other",
                    }
                )
    return records


def choose_speaker(records: list[dict], requested: str, min_minutes: float) -> str:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        by_speaker[item["Singer"]].append(item)
    if requested:
        normalized = safe(requested).lower()
        matches = [
            speaker
            for speaker in by_speaker
            if safe(speaker).lower() == normalized or safe(speaker).lower().endswith(f"_{normalized}")
        ]
        if len(matches) != 1:
            raise ValueError(f"target speaker {requested!r} matched {matches}")
        return matches[0]
    candidates = [
        speaker
        for speaker, items in by_speaker.items()
        if len(items) >= 20 and sum(item["Duration"] for item in items) >= min_minutes * 60
    ]
    if not candidates:
        raise RuntimeError("No speaker satisfies the minimum duration and utterance count")
    return max(candidates, key=lambda speaker: sum(item["Duration"] for item in by_speaker[speaker]))


def finalize(items: list[dict], dataset_name: str) -> list[dict]:
    output = []
    for index, item in enumerate(items):
        item = dict(item)
        item["Dataset"] = dataset_name
        item["index"] = index
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Emilia, LibriTTS and CV3-Eval for VALL-E")
    parser.add_argument("--emilia-root", type=Path, required=True)
    parser.add_argument("--libritts-root", type=Path, required=True)
    parser.add_argument("--cv3-root", type=Path, required=True)
    parser.add_argument("--target-speaker", default="")
    parser.add_argument("--min-speaker-minutes", type=float, default=20.0)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for root in (args.emilia_root, args.libritts_root, args.cv3_root):
        if not root.is_dir():
            raise FileNotFoundError(root)

    train_pool = generic_records(args.emilia_root, "emilia", "train")
    train_pool.extend(libritts_records(args.libritts_root))
    cv3_test = generic_records(args.cv3_root, "cv3_eval", "test")
    speaker = choose_speaker(train_pool, args.target_speaker, args.min_speaker_minutes)
    selected = [
        item
        for item in train_pool
        if item["Singer"] == speaker
        and (item["Duration"] == 0.0 or 1.0 <= item["Duration"] <= 20.0)
    ]
    if len(selected) < 2:
        raise RuntimeError(f"Speaker {speaker} has fewer than two usable utterances")
    random.Random(args.seed).shuffle(selected)
    valid_count = max(1, min(len(selected) - 1, round(len(selected) * args.valid_ratio)))
    valid = finalize(selected[:valid_count], "voiceclone")
    train = finalize(selected[valid_count:], "voiceclone")
    test = finalize(cv3_test, "voiceclone")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", train), ("valid", valid), ("test", test)):
        (args.output / f"{name}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.output / "singers.json").write_text(
        json.dumps({speaker: 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "utt2singer").open("w", encoding="utf-8") as handle:
        for item in train + valid:
            handle.write(f"voiceclone_{item['Uid']}\t{speaker}\n")
    report = {
        "target_speaker": speaker,
        "train_utterances": len(train),
        "valid_utterances": len(valid),
        "cv3_test_utterances": len(test),
        "target_minutes": round(sum(item["Duration"] for item in selected) / 60, 2),
        "seed": args.seed,
    }
    (args.output / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
