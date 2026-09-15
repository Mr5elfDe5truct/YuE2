"""
Unit tests for WebUI Song Library and UI state management
"""

import pandas as pd
from pathlib import Path
import pytest

import webui


def test_load_song_from_id_empty():
    audio, cover, score, details = webui.load_song_from_id("")
    assert audio is None
    assert cover is None
    assert score == ""
    assert "Select a song" in details


def test_load_song_from_id_nonexistent():
    audio, cover, score, details = webui.load_song_from_id("non_existent_song_xyz_123")
    assert audio is None
    assert cover is None
    assert score == ""
    assert "not found" in details.lower()


def test_load_song_from_id_with_existing(tmp_path: Path, monkeypatch):
    fake_outputs = tmp_path / "outputs"
    fake_song = fake_outputs / "song_test_123"
    fake_song.mkdir(parents=True)
    (fake_song / "audio.flac").write_bytes(b"RIFFdummy")
    (fake_song / "cover.png").write_bytes(b"PNGdummy")
    (fake_song / "score.abc").write_text("X:1\nT:Test Song\nK:C\nC D E F|", encoding="utf-8")
    (fake_song / "request.json").write_text('{"style": "synthwave", "seed": 42}', encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    audio, cover, score, details = webui.load_song_from_id("song_test_123")
    assert audio is not None and "audio.flac" in audio
    assert cover is not None and "cover.png" in cover
    assert "Test Song" in score
    assert "synthwave" in details
    assert "42" in details

    # Also test passing label string containing space
    audio2, cover2, score2, details2 = webui.load_song_from_id("song_test_123 (1:00) • synthwave...")
    assert audio2 is not None and "audio.flac" in audio2


def test_webui_create_ui():
    app = webui.create_ui()
    assert app is not None
