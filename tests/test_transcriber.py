"""
Unit tests for Audio Transcriber and Melody Extractor
"""

import numpy as np
from pathlib import Path
import soundfile as sf
import pytest

from yue2.transcriber import (
    hz_to_midi,
    midi_to_abc_pitch,
    quantize_duration_to_units,
    extract_melody_notes,
    format_notes_to_abc,
    transcribe_audio_to_abc,
)


def test_hz_to_midi():
    assert hz_to_midi(440.0) == 69   # A4
    assert hz_to_midi(261.63) == 60  # C4
    assert hz_to_midi(0.0) == 0      # Rest


def test_midi_to_abc_pitch():
    assert midi_to_abc_pitch(60) == "C"    # C4
    assert midi_to_abc_pitch(69) == "A"    # A4
    assert midi_to_abc_pitch(72) == "c"    # C5
    assert midi_to_abc_pitch(81) == "a"    # A5
    assert midi_to_abc_pitch(48) == "C,"   # C3
    assert midi_to_abc_pitch(0) == "z"     # Rest


def test_quantize_duration():
    # At 120 BPM, quarter note is 0.5s -> 8 units
    assert quantize_duration_to_units(0.5, bpm=120) == 8
    # Half note is 1.0s -> 16 units
    assert quantize_duration_to_units(1.0, bpm=120) == 16
    # Eighth note is 0.25s -> 4 units
    assert quantize_duration_to_units(0.25, bpm=120) == 4


def test_transcribe_audio_synthetic(tmp_path: Path):
    sr = 16000
    # 1 second of 440 Hz (A4)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.6 * np.sin(2 * np.pi * 440.0 * t)

    audio_file = tmp_path / "test_a4.wav"
    sf.write(str(audio_file), sig, sr)

    abc = transcribe_audio_to_abc(audio_file, bpm=120, key="C", title="Synthetic Riff")
    assert "X:1" in abc
    assert "M:4/4" in abc
    assert "L:1/32" in abc
    assert "V:Vocal" in abc
    assert "V:Ins" in abc
    assert "A" in abc  # Extracted pitch A4
