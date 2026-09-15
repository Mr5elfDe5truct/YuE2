"""
Tests for YuE2 Album Cover Art Generation Studio
"""

import os
from pathlib import Path
import pytest
from PIL import Image

from yue2.cover_art import (
    build_cover_prompt,
    generate_procedural_cover,
    create_cover_art,
)


def test_build_cover_prompt_genre_and_mood():
    prompt = build_cover_prompt(
        style="synthwave cyberpunk energetic male vocals 122 BPM",
        lyrics="[Verse]\nNeon signs reflecting off the street\nChasing the beat",
        title="Midnight Driver"
    )
    assert "cyberpunk" in prompt.lower() or "synthwave" in prompt.lower()
    assert "album cover" in prompt.lower()
    assert "masterpiece" in prompt.lower()


def test_build_cover_prompt_fallback():
    prompt = build_cover_prompt(
        style="experimental acoustic didgeridoo",
        lyrics="",
        title="Desert Song"
    )
    assert "album cover" in prompt.lower()
    assert "experimental acoustic didgeridoo" in prompt.lower() or "modern" in prompt.lower()


def test_generate_procedural_cover(tmp_path: Path):
    out_file = tmp_path / "test_cover.png"
    result = generate_procedural_cover(
        style="80s synthwave retro electro",
        lyrics="Streets of chrome beneath headlights\nEngine humming into midnight",
        title="Night Pulse",
        output_path=out_file,
        seed=1337
    )
    assert result.exists()
    assert result.stat().st_size > 1000

    # Verify Pillow can open and verify dimensions
    with Image.open(result) as img:
        assert img.size == (800, 800)
        assert img.format == "PNG"


def test_generate_procedural_cover_various_genres(tmp_path: Path):
    genres = [
        "hard rock metal heavy guitars",
        "cozy lo-fi chill hip hop rain",
        "r&b soul smooth romantic",
        "classical orchestra grand piano"
    ]
    for i, g in enumerate(genres):
        out_file = tmp_path / f"cover_{i}.png"
        res = generate_procedural_cover(
            style=g,
            lyrics="Some lyrics line 1\nSome lyrics line 2",
            title=f"Track {i}",
            output_path=out_file,
            seed=i * 10
        )
        assert res.exists()
        assert res.stat().st_size > 500


def test_create_cover_art_disabled(tmp_path: Path):
    res = create_cover_art(
        style="synthwave",
        lyrics="lyrics",
        engine="none",
        output_dir=tmp_path
    )
    assert res is None
    assert not (tmp_path / "cover.png").exists()


def test_create_cover_art_procedural(tmp_path: Path):
    res = create_cover_art(
        style="cyberpunk",
        lyrics="[Verse]\nDigital rain",
        engine="procedural",
        output_dir=tmp_path,
        title="Cyber Matrix",
        seed=42
    )
    assert res is not None
    assert res.exists()
    assert res.name == "cover.png"
    assert res.parent == tmp_path


def test_normalize_cover_engine():
    from yue2.cover_art import normalize_cover_engine

    assert normalize_cover_engine("cloud") == "cloud"
    assert normalize_cover_engine("☁️ Cloud AI Diffusion (Flux/SDXL Quality, 0 MB VRAM)") == "cloud"
    assert normalize_cover_engine("procedural") == "procedural"
    assert normalize_cover_engine("🎨 Procedural Graphic Studio (Vinyl Sleeve, Offline, 0 MB VRAM)") == "procedural"
    assert normalize_cover_engine("local") == "local"
    assert normalize_cover_engine("⚡ Local AI Diffusion (SD-Turbo on GPU, Sequenced)") == "local"
    assert normalize_cover_engine("none") == "none"
    assert normalize_cover_engine("🚫 Disabled (No Cover Art)") == "none"
    assert normalize_cover_engine("disabled") == "none"
    assert normalize_cover_engine(None) == "cloud"
    assert normalize_cover_engine("") == "cloud"

