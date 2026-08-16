"""Build a validated, backend-aware external video-director handoff bundle."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

import config
from lyric_timing import load_valid_alignment


class ExternalExportError(RuntimeError):
    pass


def _validate_timeline(pm) -> None:
    if pm.df is None or pm.df.empty:
        raise ExternalExportError("Build the timeline before exporting external-LLM templates.")
    required = ("Shot_ID", "Type", "Start_Time", "End_Time", "Duration")
    missing = [column for column in required if column not in pm.df.columns]
    if missing:
        raise ExternalExportError(f"Timeline is missing required columns: {', '.join(missing)}.")
    if pm.df[list(required)].isnull().any().any():
        raise ExternalExportError("Timeline contains blank IDs, types, or timing values.")
    if pm.df["Shot_ID"].astype(str).str.strip().eq("").any():
        raise ExternalExportError("Timeline contains a blank Shot_ID.")


def _project_context(settings: dict) -> str:
    fields = [
        ("Video backend", config.VIDEO_BACKEND),
        ("Video mode", settings.get("video_mode", "Intercut")),
        ("Rough concept", settings.get("rough_concept", "")),
        ("Plot", settings.get("plot", "")),
        ("Singer / performance / venue description", settings.get("performance_desc", "")),
        ("Singer gender", settings.get("singer_gender", "")),
        ("Visual style", settings.get("last_style", "None")),
        ("Director", settings.get("last_director", "None")),
    ]
    if config.VIDEO_BACKEND == "MiniMax H3":
        fields.extend([
            ("H3 aspect", settings.get("h3_aspect", "3:4 - Photo")),
            ("H3 lead singer", settings.get("h3_lead_character", "Not selected")),
            ("Vocal prompt mode", settings.get("vocal_prompt_mode", "Use Singer/Band Description")),
        ])
    return "\n".join(f"{label}: {value or 'Not provided'}" for label, value in fields)


def _backend_instructions(settings: dict) -> str:
    if config.VIDEO_BACKEND == "MiniMax H3":
        return """MINIMAX H3-SPECIFIC RULES
- Include the lead singer in character_bibles.csv. H3 requires that Character Bible entry to generate the singer's face and full-body identity references.
- Give the lead singer one unique first name and use that exact name consistently in relevant Video_Prompt cells.
- Action prompts may name up to four recurring Character Bible characters per shot.
- Vocal prompts should describe the desired venue, composition, visible performance, action, camera movement, and ending state. When Vocal Shot Prompt Mode is Use Storyboard Prompt, the prompt also drives the generated target first frame.
- Do not add MiniMax reference labels such as <Picture 1> or <Subject 1>; Synesthesia's H3 rewrite stage adds those labels.
"""
    return """LTX-SPECIFIC RULES
- The lead singer may remain in the separate singer/performance description instead of character_bibles.csv.
- Recurring narrative characters should be placed in character_bibles.csv and referenced by exact first name in Video_Prompt.
- Fill Vocal Video_Prompt cells as useful storyboard alternatives even when the project is currently configured to use the shared singer/band description.
"""


def build_instructions(pm, alignment: dict) -> str:
    settings = pm.load_project_settings()
    matched = int(alignment.get("matched_lines", 0))
    eligible = int(alignment.get("eligible_lines", 0))
    coverage = 100 * matched / eligible if eligible else 0.0
    timing_source = alignment.get("source", "legacy lyric scan").replace("_", " ")
    return f"""SYNESTHESIA EXTERNAL VIDEO DIRECTOR HANDOFF

PROJECT CONTEXT
{_project_context(settings)}

YOUR TASK
Edit and return exactly these two CSV files:
1. shot_list.csv
2. character_bibles.csv

SHOT LIST RULES
- Preserve every Shot_ID, row, Type, Start_Time, End_Time, Duration, Start_Frame, End_Frame, and Total_Frames exactly.
- Do not add, delete, reorder, or retime shots.
- Fill or improve Video_Prompt. Do not edit internal cache, path, status, or render columns.
- Keep valid CSV quoting. Video_Prompt contains commas and must remain one CSV field.
- Use present tense and describe a coherent beginning, action progression, camera relationship, and ending composition.
- Align important visual beats to the shot intervals and timestamped lyrics below.
- Use recurring characters by one exact first name only. Do not repeat a character's physical description inside every prompt; Synesthesia injects Character Bible descriptions.
- Do not put markdown fences, explanations, or commentary into either CSV.

CHARACTER BIBLE RULES
- Keep exactly two columns: character_name,description.
- Include every recurring named character needed by the prompts.
- Names must be unique case-insensitively and should be distinctive first names.
- Each description must be a dense visual identity specification covering apparent age, presentation, ethnicity/skin tone where relevant, facial structure, hair, body proportions, wardrobe, colors, and distinguishing features.
- Preserve useful existing entries and improve them when necessary.

{_backend_instructions(settings)}
IMPORT ORDER
The user will import shot_list.csv first and character_bibles.csv second. Synesthesia then recalculates each shot's Characters column from exact names found in Video_Prompt.

LYRIC TIMING QUALITY
Timing source: {timing_source}. Matched {matched} of {eligible} lyric lines ({coverage:.1f}%). Lines without timestamp prefixes could not be aligned confidently. Treat their timing as unknown; do not guess that they occur near adjacent matched lines when chorus notation or ad-libs make that ambiguous.

TIMESTAMPED LYRICS
{pm.get_lyrics()}
"""


def create_external_llm_bundle(pm) -> tuple[str, str]:
    if not pm or not pm.current_project:
        raise ExternalExportError("Load a project before exporting external-LLM templates.")
    _validate_timeline(pm)
    alignment, alignment_error = load_valid_alignment(pm)
    if alignment_error:
        raise ExternalExportError(alignment_error)

    project_dir = Path(pm.base_dir) / pm.current_project
    export_dir = project_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = export_dir / "external_llm_director_bundle.zip"

    with tempfile.TemporaryDirectory(prefix="synesthesia_external_llm_") as temp_root:
        temp_dir = Path(temp_root)
        shot_path = temp_dir / "shot_list.csv"
        bible_path = temp_dir / "character_bibles.csv"
        instructions_path = temp_dir / "external_llm_instructions.txt"

        pm.df.to_csv(shot_path, index=False)
        pd.DataFrame(
            list(pm.character_bibles.items()), columns=["character_name", "description"]
        ).to_csv(bible_path, index=False)
        instructions_path.write_text(build_instructions(pm, alignment), encoding="utf-8")

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in (shot_path, bible_path, instructions_path):
                archive.write(path, arcname=path.name)

    matched = int(alignment.get("matched_lines", 0))
    eligible = int(alignment.get("eligible_lines", 0))
    qualifier = ""
    if matched < eligible:
        qualifier = f" Warning: {eligible - matched} lyric line(s) remain untimestamped."
    return f"✅ External-LLM bundle exported.{qualifier}", os.path.abspath(bundle_path)
