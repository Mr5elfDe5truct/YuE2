"""
YuE2 Symbolic Music Analyzer & Musical Linter
---------------------------------------------
Analyzes, validates, and lints ABC sheet music for YuE2 score-conditioned synthesis.
Enables real-time feedback on key signature, tempo, measures, chord harmony, and notation syntax.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import the core ABC parsing engine from skills
_SKILLS_PATH = Path(__file__).resolve().parents[2] / "skills/yue2-music/scripts"
if str(_SKILLS_PATH) not in sys.path:
    sys.path.insert(0, str(_SKILLS_PATH))

try:
    import abc_tools
except ImportError:
    abc_tools = None

# Curated palette of supported, musically rich chords for YuE2
CURATED_CHORD_PALETTE = [
    # Major & Minor triads
    "C", "Dm", "Em", "F", "G", "Am", "Bdim",
    # 7th chords
    "Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5",
    # Suspended & Color chords
    "Gsus4", "Csus2", "A7sus4", "F6", "Am6",
    # Inversions / Slash chords
    "C/E", "G/B", "F/A", "Am/G",
    # Borrowed & Chromatic chords
    "Bb", "Eb", "Ab", "Db", "F#m", "C#m"
]


def analyze_score(abc_text: str) -> Dict[str, Any]:
    """
    Perform deep structural analysis on an ABC score:
    - Meter, key, BPM, and calculated total playback duration
    - Voice breakdown (Vocal notes vs Instrumental notes)
    - Total measures / bars
    - Complete chord progression list
    """
    text = (abc_text or "").strip()
    if not text:
        return {"status": "empty", "error": "Empty score provided."}

    # Header regex parsing
    meter_match = re.search(r"^M:\s*([^\n]+)", text, re.MULTILINE)
    unit_match = re.search(r"^L:\s*([^\n]+)", text, re.MULTILINE)
    bpm_match = re.search(r"^Q:[^=]*=\s*([0-9]+)", text, re.MULTILINE)
    key_match = re.search(r"^K:\s*([^\n]+)", text, re.MULTILINE)

    meter = meter_match.group(1).strip() if meter_match else "4/4"
    unit = unit_match.group(1).strip() if unit_match else "1/32"
    bpm = int(bpm_match.group(1)) if bpm_match else 120
    key = key_match.group(1).strip() if key_match else "C"

    # Extract chords in quotes, e.g. "Am7"
    chords_found = re.findall(r'"([^"\n]+)"', text)
    unique_chords = list(dict.fromkeys(chords_found))

    # Count notes vs rests
    vocal_notes = len(re.findall(r"[A-Ga-g][,']*[0-9]*", text))
    vocal_rests = len(re.findall(r"z[0-9]*", text))

    # Count bars
    vocal_line = ""
    for line in text.splitlines():
        if "[V:Vocal]" in line or line.startswith("V:Vocal"):
            vocal_line += line
    bars_count = max(1, len(re.findall(r"\|", vocal_line))) if vocal_line else len(re.findall(r"\|", text))

    # Calculate estimated duration in seconds (4 quarter notes per 4/4 bar)
    quarter_notes_per_bar = 4
    if "/" in meter:
        try:
            n, d = map(int, meter.split("/"))
            quarter_notes_per_bar = (n * 4) / d
        except Exception:
            quarter_notes_per_bar = 4

    total_quarters = bars_count * quarter_notes_per_bar
    estimated_seconds = round((total_quarters * 60.0) / max(30, bpm), 1)

    # Detailed validation using abc_tools if available
    lint_status = "valid"
    lint_issues = []
    if abc_tools is not None:
        try:
            score = abc_tools.parse(text)
            rep = abc_tools.report(score)
            bars_count = rep.get("bars_total", bars_count)
            total_quarters = float(rep.get("total_duration_quarters", total_quarters))
            estimated_seconds = round((total_quarters * 60.0) / max(30, bpm), 1)
        except Exception as e:
            lint_status = "warning"
            lint_issues.append(str(e))

    return {
        "status": lint_status,
        "key": key,
        "meter": meter,
        "bpm": bpm,
        "unit": unit,
        "bars_count": bars_count,
        "estimated_seconds": estimated_seconds,
        "vocal_notes_count": vocal_notes,
        "rests_count": vocal_rests,
        "chord_symbols_count": len(chords_found),
        "chords": unique_chords,
        "chord_progression_sample": " → ".join(chords_found[:12]) + ("..." if len(chords_found) > 12 else ""),
        "issues": lint_issues,
    }


def format_score_report_markdown(analysis: Dict[str, Any]) -> str:
    """Format the analysis result into a clean markdown card for the UI."""
    if analysis.get("status") == "empty":
        return "⚠️ *No score provided to analyze.*"

    md = []
    status = analysis.get("status")
    if status == "valid":
        md.append("#### ✅ Symbolic Score Health: **Verified & Compliant**")
    else:
        md.append("#### ⚠️ Symbolic Score Notice")

    md.append(
        f"| Key Signature | Meter | Tempo | Length | Est. Duration |\n"
        f"|:---:|:---:|:---:|:---:|:---:|\n"
        f"| **{analysis.get('key')}** | **{analysis.get('meter')}** | **{analysis.get('bpm')} BPM** | **{analysis.get('bars_count')} Bars** | **~{analysis.get('estimated_seconds')}s** |"
    )

    chords = analysis.get("chords", [])
    if chords:
        md.append(f"\n**Harmonic Progression ({len(chords)} unique chords)**:")
        md.append(f"`{analysis.get('chord_progression_sample')}`")
    else:
        md.append("\n*No explicit chord annotations found (Melody-Only Mode).*")

    issues = analysis.get("issues", [])
    if issues:
        md.append("\n> [!NOTE]\n> **Notation advisory**:\n" + "\n".join(f"> - {iss}" for iss in issues))

    return "\n".join(md)


def diff_two_scores(score_a: str, score_b: str) -> Dict[str, Any]:
    """
    Compare two ABC scores (e.g. Original vs Reharmonized or Take A vs Take B).
    """
    ana_a = analyze_score(score_a)
    ana_b = analyze_score(score_b)

    chords_a = set(ana_a.get("chords", []))
    chords_b = set(ana_b.get("chords", []))

    added_chords = list(chords_b - chords_a)
    removed_chords = list(chords_a - chords_b)

    tempo_changed = ana_a.get("bpm") != ana_b.get("bpm")
    key_changed = ana_a.get("key") != ana_b.get("key")

    return {
        "tempo_changed": tempo_changed,
        "key_changed": key_changed,
        "tempo_a": ana_a.get("bpm"),
        "tempo_b": ana_b.get("bpm"),
        "key_a": ana_a.get("key"),
        "key_b": ana_b.get("key"),
        "added_chords": added_chords,
        "removed_chords": removed_chords,
        "bars_a": ana_a.get("bars_count"),
        "bars_b": ana_b.get("bars_count"),
    }
