"""CPU-only lyric transcription, safe alignment, caption import, and freshness metadata."""

from __future__ import annotations

import gc
import hashlib
import html
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


WHISPER_MODEL = "small"
WHISPER_FALLBACK_MODEL = "medium"
ALIGNMENT_VERSION = 2
FALLBACK_COVERAGE = 0.70
TIMESTAMP_RE = re.compile(
    r"^\s*\[(?P<start>\d{1,3}:\d{2}(?:\.\d{1,3})?)\s*--?>\s*"
    r"(?P<end>\d{1,3}:\d{2}(?:\.\d{1,3})?)\]\s*"
)
SECTION_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]|\([^\)]+\)|(?:verse|chorus|bridge|intro|outro|pre[- ]?chorus|"
    r"refrain|instrumental|solo|hook|break)(?:\s+\d+)?\s*:?)\s*$",
    re.IGNORECASE,
)
_TIME_TOKEN = r"(?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d+)?"
_ARROW_TIME_RE = re.compile(
    rf"^\s*(?P<start>{_TIME_TOKEN})\s*-->\s*(?P<end>{_TIME_TOKEN})(?:\s+.*)?$"
)
_SBV_TIME_RE = re.compile(
    rf"^\s*(?P<start>{_TIME_TOKEN})\s*,\s*(?P<end>{_TIME_TOKEN})\s*$"
)
_LRC_RE = re.compile(r"\[(?P<time>\d{1,3}:\d{2}(?:[.,]\d{1,3})?)\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class LyricTimingError(RuntimeError):
    pass


def _text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_timestamp(line: str) -> str:
    return TIMESTAMP_RE.sub("", str(line), count=1)


def strip_lyric_timestamps(text: str) -> str:
    return "\n".join(strip_timestamp(line) for line in str(text or "").splitlines())


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold().replace("’", "'")
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", normalized, flags=re.UNICODE)
    return [word.replace("'", "") for word in words if word.replace("'", "")]


def _text_similarity(left: str | list[str], right: str | list[str]) -> float:
    left_tokens = _tokens(left) if isinstance(left, str) else left
    right_tokens = _tokens(right) if isinstance(right, str) else right
    if not left_tokens or not right_tokens:
        return 0.0
    token_score = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()
    char_score = SequenceMatcher(
        None, "".join(left_tokens), "".join(right_tokens), autojunk=False
    ).ratio()
    return 0.55 * token_score + 0.45 * char_score


def _lyric_lines(lyrics: str) -> tuple[str, list[str], list[dict]]:
    raw_lyrics = strip_lyric_timestamps(lyrics)
    lines = raw_lyrics.splitlines()
    eligible = []
    for line_index, line in enumerate(lines):
        tokens = [] if not line.strip() or SECTION_RE.match(line) else _tokens(line)
        if tokens:
            eligible.append({"line_index": line_index, "text": line, "tokens": tokens})
    if not eligible:
        raise LyricTimingError("The Lyrics field has no lyric lines to timestamp.")
    return raw_lyrics, lines, eligible


def _heard_words(transcript_words: list[dict]) -> list[dict]:
    heard = []
    for item in transcript_words:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        probability = float(item.get("probability", 0.5) or 0.0)
        segment_index = int(item.get("segment_index", -1))
        for word in _tokens(item.get("word", "")):
            heard.append({
                "word": word,
                "start": start,
                "end": max(start, end),
                "probability": max(0.0, min(1.0, probability)),
                "segment_index": segment_index,
            })
    heard.sort(key=lambda item: (item["start"], item["end"]))
    return heard


def _candidate_windows(line: dict, heard: list[dict], rank: int, line_count: int,
                       audio_duration: float) -> list[dict]:
    target = line["tokens"]
    target_count = len(target)
    minimum_words = max(1, int(target_count * 0.5))
    maximum_words = min(len(heard), max(target_count + 5, int(target_count * 1.7)))
    maximum_span = min(22.0, max(9.0, target_count * 2.8))
    expected_midpoint = audio_duration * ((rank + 0.75) / (line_count + 1.5))
    candidates = []

    for start_index in range(len(heard)):
        upper = min(len(heard), start_index + maximum_words)
        for end_index in range(start_index + minimum_words, upper + 1):
            window = heard[start_index:end_index]
            span = window[-1]["end"] - window[0]["start"]
            if span > maximum_span:
                break
            gaps = [
                max(0.0, window[index]["start"] - window[index - 1]["end"])
                for index in range(1, len(window))
            ]
            maximum_gap = max(gaps, default=0.0)
            if maximum_gap > 4.5:
                continue

            window_tokens = [item["word"] for item in window]
            similarity = _text_similarity(target, window_tokens)
            if similarity < 0.48:
                continue
            probability = sum(item["probability"] for item in window) / len(window)
            length_quality = min(len(window), target_count) / max(len(window), target_count)
            midpoint = (window[0]["start"] + window[-1]["end"]) / 2
            temporal_error = abs(midpoint - expected_midpoint) / max(audio_duration, 1.0)
            temporal_quality = max(0.0, 1.0 - min(1.0, temporal_error * 2.5))
            gap_quality = max(0.0, 1.0 - maximum_gap / 4.5)
            confidence = (
                0.58 * similarity
                + 0.18 * probability
                + 0.12 * length_quality
                + 0.08 * temporal_quality
                + 0.04 * gap_quality
            )
            if confidence < 0.56:
                continue
            candidates.append({
                "line_rank": rank,
                "line_index": line["line_index"],
                "start_word": start_index,
                "end_word": end_index - 1,
                "start": window[0]["start"],
                "end": window[-1]["end"],
                "duration": span,
                "max_gap": maximum_gap,
                "similarity": similarity,
                "word_probability": probability,
                "temporal_quality": temporal_quality,
                "confidence": confidence,
                "transcript_text": " ".join(window_tokens),
            })

    best_by_start = {}
    for candidate in candidates:
        key = candidate["start_word"]
        if key not in best_by_start or candidate["confidence"] > best_by_start[key]["confidence"]:
            best_by_start[key] = candidate
    return sorted(
        best_by_start.values(), key=lambda item: item["confidence"], reverse=True
    )[:80]


def _select_monotonic_candidates(candidates_by_line: list[list[dict]], line_count: int) -> dict[int, dict]:
    nodes = [candidate for candidates in candidates_by_line for candidate in candidates]
    nodes.sort(key=lambda item: (item["line_rank"], item["end_word"], item["start_word"]))
    best_scores = []
    predecessors = []
    for index, node in enumerate(nodes):
        reward = 0.55 + node["confidence"]
        best_score = reward - 0.012 * node["line_rank"]
        predecessor = None
        for prior_index in range(index):
            prior = nodes[prior_index]
            if prior["line_rank"] >= node["line_rank"]:
                continue
            if prior["end_word"] >= node["start_word"]:
                continue
            skipped = node["line_rank"] - prior["line_rank"] - 1
            score = best_scores[prior_index] + reward - 0.012 * skipped
            if score > best_score:
                best_score = score
                predecessor = prior_index
        best_scores.append(best_score)
        predecessors.append(predecessor)

    if not nodes:
        return {}
    final_index = max(
        range(len(nodes)),
        key=lambda index: best_scores[index] - 0.012 * (line_count - nodes[index]["line_rank"] - 1),
    )
    selected = {}
    while final_index is not None:
        node = nodes[final_index]
        selected[node["line_rank"]] = node
        final_index = predecessors[final_index]
    return selected


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def align_lyric_lines(lyrics: str, transcript_words: list[dict],
                      audio_duration: float | None = None) -> dict:
    raw_lyrics, lines, eligible = _lyric_lines(lyrics)
    heard = _heard_words(transcript_words)
    if not heard:
        raise LyricTimingError("Whisper did not detect any words in the isolated vocals track.")
    duration = max(float(audio_duration or 0.0), heard[-1]["end"], 1.0)
    candidates_by_line = [
        _candidate_windows(line, heard, rank, len(eligible), duration)
        for rank, line in enumerate(eligible)
    ]
    selected = _select_monotonic_candidates(candidates_by_line, len(eligible))

    output_lines = list(lines)
    records = []
    confidences = []
    for rank, line in enumerate(eligible):
        candidate = selected.get(rank)
        record = {
            "line_number": line["line_index"] + 1,
            "text": line["text"],
            "matched": bool(candidate),
            "candidate_count": len(candidates_by_line[rank]),
        }
        if candidate:
            output_lines[line["line_index"]] = (
                f"[{format_timestamp(candidate['start'])} --> {format_timestamp(candidate['end'])}] "
                f"{line['text']}"
            )
            record.update({
                key: round(candidate[key], 4) if isinstance(candidate[key], float) else candidate[key]
                for key in (
                    "start", "end", "duration", "max_gap", "similarity",
                    "word_probability", "temporal_quality", "confidence", "transcript_text",
                )
            })
            confidences.append(candidate["confidence"])
        records.append(record)

    matched = len(selected)
    return {
        "source": "whisper",
        "raw_lyrics": raw_lyrics,
        "timestamped_lyrics": "\n".join(output_lines),
        "records": records,
        "matched_lines": matched,
        "eligible_lines": len(eligible),
        "coverage": matched / len(eligible),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def _parse_time(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes, seconds = "0", parts[0], parts[1]
        else:
            raise ValueError
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError as exc:
        raise LyricTimingError(f"Invalid caption timestamp: {value}") from exc


def _clean_caption_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return html.unescape(_HTML_TAG_RE.sub("", text)).strip()


def parse_timed_captions(path: str) -> tuple[list[dict], str]:
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = Path(path).read_text(encoding="cp1252")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    lrc_entries = []
    for line in lines:
        matches = list(_LRC_RE.finditer(line))
        if not matches:
            continue
        text = _clean_caption_text(line[matches[-1].end():])
        if not text:
            continue
        for match in matches:
            lrc_entries.append({"start": _parse_time(match.group("time")), "text": text})
    if lrc_entries:
        lrc_entries.sort(key=lambda item: item["start"])
        cues = []
        for index, entry in enumerate(lrc_entries):
            next_start = lrc_entries[index + 1]["start"] if index + 1 < len(lrc_entries) else entry["start"] + 5.0
            cues.append({"start": entry["start"], "end": max(entry["start"] + 0.5, next_start),
                         "text": entry["text"]})
        return cues, "LRC"

    cues = []
    index = 0
    detected_format = "timed captions"
    while index < len(lines):
        timing = _ARROW_TIME_RE.match(lines[index])
        if timing:
            detected_format = "VTT" if raw.lstrip().startswith("WEBVTT") else "SRT"
        else:
            timing = _SBV_TIME_RE.match(lines[index])
            if timing:
                detected_format = "SBV"
        if not timing:
            index += 1
            continue
        start, end = _parse_time(timing.group("start")), _parse_time(timing.group("end"))
        index += 1
        text_lines = []
        while index < len(lines) and lines[index].strip():
            if _ARROW_TIME_RE.match(lines[index]) or _SBV_TIME_RE.match(lines[index]):
                break
            text_lines.append(lines[index])
            index += 1
        cleaned_lines = [
            cleaned for cleaned in (_clean_caption_text(item) for item in text_lines) if cleaned
        ]
        for text in cleaned_lines:
            cues.append({"start": start, "end": end, "text": text})
    if not cues:
        raise LyricTimingError("No SBV, SRT, VTT, or LRC caption cues were found in the selected file.")
    return cues, detected_format


def align_caption_cues(lyrics: str, cues: list[dict]) -> dict:
    raw_lyrics, lines, eligible = _lyric_lines(lyrics)
    usable_cues = [cue for cue in cues if _tokens(cue.get("text", ""))]
    if not usable_cues:
        raise LyricTimingError("The caption file does not contain any lyric text.")

    rows, cols = len(eligible), len(usable_cues)
    gap = -0.45
    scores = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    trace = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        scores[row][0], trace[row][0] = row * gap, 2
    for col in range(1, cols + 1):
        scores[0][col], trace[0][col] = col * gap, 3
    similarities = {}
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            similarity = _text_similarity(eligible[row - 1]["text"], usable_cues[col - 1]["text"])
            similarities[(row - 1, col - 1)] = similarity
            options = (
                (scores[row - 1][col - 1] + 2.2 * similarity - 0.8, 1),
                (scores[row - 1][col] + gap, 2),
                (scores[row][col - 1] + gap, 3),
            )
            scores[row][col], trace[row][col] = max(options, key=lambda item: item[0])

    selected = {}
    row, col = rows, cols
    while row or col:
        direction = trace[row][col]
        if direction == 1:
            similarity = similarities[(row - 1, col - 1)]
            if similarity >= 0.50:
                selected[row - 1] = (usable_cues[col - 1], similarity)
            row, col = row - 1, col - 1
        elif direction == 2:
            row -= 1
        else:
            col -= 1

    output_lines = list(lines)
    records = []
    for rank, line in enumerate(eligible):
        match = selected.get(rank)
        record = {
            "line_number": line["line_index"] + 1,
            "text": line["text"],
            "matched": bool(match),
        }
        if match:
            cue, similarity = match
            output_lines[line["line_index"]] = (
                f"[{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}] {line['text']}"
            )
            record.update({
                "start": cue["start"], "end": cue["end"],
                "confidence": round(similarity, 4), "caption_text": cue["text"],
            })
        records.append(record)
    matched = len(selected)
    return {
        "source": "caption_import",
        "raw_lyrics": raw_lyrics,
        "timestamped_lyrics": "\n".join(output_lines),
        "records": records,
        "matched_lines": matched,
        "eligible_lines": len(eligible),
        "coverage": matched / len(eligible),
        "mean_confidence": (
            sum(match[1] for match in selected.values()) / matched if matched else 0.0
        ),
    }


def transcribe_audio(audio_path: str, model_name: str = WHISPER_MODEL) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise LyricTimingError(
            "faster-whisper is not installed. Run the launcher dependency update and try again."
        ) from exc

    model = None
    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segment_iterator, info = model.transcribe(
            audio_path,
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        words = []
        segments = []
        for segment_index, segment in enumerate(segment_iterator):
            segment_words = []
            for word in segment.words or []:
                item = {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability or 0.0),
                    "segment_index": segment_index,
                }
                segment_words.append(item)
                words.append(item)
            segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text).strip(),
                "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                "words": segment_words,
            })
        return {
            "model": model_name,
            "language": str(getattr(info, "language", "unknown")),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "words": words,
            "segments": segments,
            "raw_text": " ".join(segment["text"] for segment in segments).strip(),
        }
    except Exception as exc:
        raise LyricTimingError(
            f"Whisper transcription failed: {exc}. The first use of the '{model_name}' model "
            "requires internet access to download it."
        ) from exc
    finally:
        del model
        gc.collect()


def transcribe_words(audio_path: str, ground_truth_lyrics: str = "",
                     model_name: str = WHISPER_MODEL) -> tuple[list[dict], str]:
    """Backward-compatible wrapper; ground truth is intentionally not decoder context."""
    transcription = transcribe_audio(audio_path, model_name)
    return transcription["words"], transcription["language"]


def preferred_fallback_model(language: str) -> str:
    return "medium.en" if str(language).lower().startswith("en") else WHISPER_FALLBACK_MODEL


def alignment_needs_fallback(alignment: dict) -> bool:
    return float(alignment.get("coverage", 0.0)) < FALLBACK_COVERAGE


def better_alignment(first: tuple[dict, dict], second: tuple[dict, dict]) -> tuple[dict, dict]:
    def score(candidate):
        alignment, _ = candidate
        return (int(alignment.get("matched_lines", 0)), float(alignment.get("mean_confidence", 0.0)))
    return max((first, second), key=score)


def save_alignment(pm, vocals_path: str, alignment: dict, language: str,
                   model_name: str = WHISPER_MODEL, transcription: dict | None = None,
                   source: str | None = None) -> dict:
    project_dir = Path(pm.base_dir) / pm.current_project
    raw_path = project_dir / "lyrics_untimestamped.txt"
    metadata_path = project_dir / "lyrics_alignment.json"
    raw_path.write_text(alignment["raw_lyrics"], encoding="utf-8")
    pm.save_lyrics(alignment["timestamped_lyrics"])
    actual_source = source or alignment.get("source", "whisper")
    metadata = {
        "version": ALIGNMENT_VERSION,
        "source": actual_source,
        "model": model_name,
        "device": "cpu" if actual_source != "caption_import" else "not applicable",
        "compute_type": "int8" if actual_source != "caption_import" else "not applicable",
        "language": language,
        "vocals_sha256": _file_hash(vocals_path),
        "raw_lyrics_sha256": _text_hash(alignment["raw_lyrics"]),
        "timestamped_lyrics_sha256": _text_hash(alignment["timestamped_lyrics"]),
        "matched_lines": alignment["matched_lines"],
        "eligible_lines": alignment["eligible_lines"],
        "coverage": alignment.get("coverage", 0.0),
        "mean_confidence": alignment.get("mean_confidence", 0.0),
        "records": alignment["records"],
    }
    if transcription is not None:
        metadata["transcription"] = transcription
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_valid_alignment(pm) -> tuple[dict | None, str | None]:
    if not pm or not pm.current_project:
        return None, "No project is loaded."
    project_dir = Path(pm.base_dir) / pm.current_project
    metadata_path = project_dir / "lyrics_alignment.json"
    vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
    if not metadata_path.is_file():
        return None, "Lyrics have not been timestamped yet."
    if not vocals_path or not os.path.isfile(vocals_path):
        return None, "The isolated vocals track is missing."
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Lyric alignment metadata is unreadable: {exc}"
    current_lyrics = pm.get_lyrics()
    if metadata.get("timestamped_lyrics_sha256") != _text_hash(current_lyrics):
        return None, "Lyrics changed after timestamping. Rescan or import timed captions again."
    if not any(TIMESTAMP_RE.match(line) for line in current_lyrics.splitlines()):
        return None, "No timestamped lyric lines were found. Rescan or import timed captions again."
    if metadata.get("vocals_sha256") != _file_hash(vocals_path):
        return None, "The isolated vocals track changed after timestamping. Run Add Lyric Timestamps again."
    if int(metadata.get("matched_lines", 0)) < 1:
        return None, "No lyric lines were matched confidently."
    return metadata, None
