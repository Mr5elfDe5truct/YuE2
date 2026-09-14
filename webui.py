"""YuE2 Music Studio: Modern 3D Interactive Audio Workstation for Song Generation.

Features:
- 3D depth shadows, neumorphic bevels, and layered glassmorphism cards.
- 4 User-selectable themes: Cyberpunk Neon, Midnight Studio, Synthwave Sunset, and Modern Emerald.
- Interactive style prompt composer with clickable category chips (Genres, Vocals, Instruments, Moods, BPM).
- Lyrics structure builder (+ [Intro], + [Verse], + [Chorus], etc.).
- Reharmonization & Score Editing: Edit ABC chords/melodies and synthesize directly from the score.
- Full model controls: ODE steps (4/8/16/32), CFG guidance (1.0-3.0), temperature, top-p, and repetition penalty.
- Song Library & History explorer to preview, listen, and reload past generations from outputs/.
- Live Hardware Doctor & VRAM monitor.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

# Configure Windows and PyTorch environment
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Install with: pip install gradio", file=sys.stderr)
    sys.exit(1)

from yue2.pipeline import YuE2Pipeline
from yue2.protocol import GenerationConfig, Sampling
from yue2.cli import doctor as cli_doctor

# Global pipeline cache
PIPELINE: YuE2Pipeline | None = None
CURRENT_CONFIG: dict = {}


def get_system_info() -> str:
    parts = []
    parts.append(f"Python {sys.version.split()[0]}")
    parts.append(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        capability = torch.cuda.get_device_capability(0)
        parts.append(f"⚡ {gpu_name} ({vram_gib:.1f} GiB VRAM, sm_{capability[0]}{capability[1]})")
    else:
        parts.append("💻 CPU Mode")
    return " • ".join(parts)


def load_pipeline(device: str = "auto", offload_ar: bool = True) -> YuE2Pipeline:
    global PIPELINE, CURRENT_CONFIG
    desired_config = {"device": device, "offload_ar": offload_ar}
    if PIPELINE is not None and CURRENT_CONFIG == desired_config:
        return PIPELINE

    if PIPELINE is not None:
        PIPELINE.close()
        PIPELINE = None

    print(f"[*] Initializing YuE2Pipeline (device={device}, offload_ar={offload_ar})...")
    PIPELINE = YuE2Pipeline.from_pretrained(
        "m-a-p/YuE2-3B",
        vae="m-a-p/YuE2-Vae",
        device=device,
        offload_ar=offload_ar,
        progress=True,
    )
    CURRENT_CONFIG = desired_config
    return PIPELINE


# Curated Studio Presets
PRESETS = {
    "City Lights (Warm Piano Pop)": {
        "style": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
        "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night",
        "cot": "full",
        "ode_steps": 16,
    },
    "Neon Highway (80s Retrowave / Synthwave)": {
        "style": "English, 80s synthwave, energetic male vocals, analog synthesizer arpeggios, driving punchy drums, gated reverb snare, deep synth bassline, 122 BPM",
        "lyrics": "[Intro]\n(Synth arpeggio sweeps into punchy drums)\n\n[Verse 1]\nStreets of chrome beneath the headlights\nChasing shadows into midnight\nEngine humming like a heartbeat\nLeaving all the ghosts behind me\n\n[Chorus]\nRunning down the neon highway\nNothing in the world can stop us now\nElectric skies are burning sideways\nWe will never let the fire out",
        "cot": "full",
        "ode_steps": 16,
    },
    "Midnight Coffee (Lo-Fi Chill Hop)": {
        "style": "English, cozy lo-fi hip hop, smooth female vocal, jazzy electric piano, tape hiss, muted bass, vinyl crackle, laid-back swung groove, 75 BPM",
        "lyrics": "[Verse]\nRaindrops tapping on the windowsill\nSteam is rising from the coffee cup\nTime is frozen and the world is still\nWaiting for the sun to wake us up\n\n[Chorus]\nStay right here where the shadows soften\nSimple moments that we lose too often\nLet the record spin another round\nSilence is the sweetest sound",
        "cot": "full",
        "ode_steps": 8,
    },
    "Thunder & Rust (Modern Driving Rock)": {
        "style": "English, hard rock, gritty powerful male vocals, distorted electric guitar riffs, heavy pounding drums, aggressive bassline, raw energetic production, 138 BPM",
        "lyrics": "[Intro]\n(Heavy guitar feedback and drum count-in)\n\n[Verse 1]\nDust on the boots and gravel on the road\nCarrying the weight of a heavy load\nSpark in the engine, fire in the veins\nBreaking every single one of these chains\n\n[Chorus]\nHear the thunder rolling in the sky\nWe were born to fight and not to die\nTurn the volume up and let it roar\nWe're not backing down anymore",
        "cot": "melody",
        "ode_steps": 16,
    },
    "Starfall (Atmospheric Dream Pop)": {
        "style": "English, dream pop, ethereal airy female vocals, lush reverb guitar, shimmering synth pads, gentle electronic beat, wide spacious mix, 100 BPM",
        "lyrics": "[Verse]\nConstellations drifting slow\nDrifting where the silent currents flow\nSilver rivers through the dark\nEvery whisper leaves a spark\n\n[Chorus]\nCatch the starfall in your hands\nOver ocean, over golden sands\nFloating where the gravity expires\nEndless heavens, eternal fires",
        "cot": "full",
        "ode_steps": 16,
    },
}

# Tag Palette options
GENRE_TAGS = ["Synthwave", "Contemporary Pop", "Lo-Fi Chill Hop", "Acoustic Pop", "R&B Soul", "Indie Rock", "Cyberpunk EDM", "Cinematic Orchestral", "80s Rock", "Funk Groove"]
VOCAL_TAGS = ["Warm Female Vocal", "Airy Ethereal Female", "Soulful Male Vocal", "Gritty Rock Male", "Vocal Duet", "Dreamy Choir", "Instrumental (No Vocals)"]
INSTRUMENT_TAGS = ["Grand Piano", "Acoustic Guitar", "Electric Guitar Solo", "Warm 808 Bass", "Analog Synth", "Tenor Saxophone", "Brushed Drums", "Punchy Drums", "Orchestral Strings"]
MOOD_TAGS = ["Uplifting & Inspiring", "Melancholic & Moody", "Relaxed & Chill", "Energetic & Driving", "Nostalgic 80s", "Dark & Atmospheric", "Romantic & Tender"]
BPM_PRESETS = ["75 BPM", "88 BPM", "96 BPM", "110 BPM", "122 BPM", "138 BPM"]


def add_style_tag(current_style: str, tag: str) -> str:
    current = current_style.strip() if current_style else ""
    if not current:
        return tag
    parts = [p.strip() for p in current.split(",") if p.strip()]
    if tag not in parts:
        parts.append(tag)
    return ", ".join(parts)


def insert_lyrics_tag(current_lyrics: str, tag: str) -> str:
    current = (current_lyrics or "").rstrip()
    if not current:
        return f"[{tag}]\n"
    return f"{current}\n\n[{tag}]\n"


def load_preset_values(preset_name: str):
    if preset_name in PRESETS:
        p = PRESETS[preset_name]
        return p["style"], p["lyrics"], p["cot"], p["ode_steps"]
    return gr.update(), gr.update(), gr.update(), gr.update()


def generate_music(
    style: str,
    lyrics: str,
    cot_mode: str,
    custom_abc: str,
    seed_input: int,
    cfg_scale: float,
    ode_steps: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stage: str,
    device: str,
    offload_ar: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not style.strip():
        raise gr.Error("Please enter or select a style prompt.")
    if not lyrics.strip():
        raise gr.Error("Please enter song lyrics.")

    seed = int(seed_input) if seed_input >= 0 else random.randint(0, 2**31 - 1)
    timestamp = int(time.time())
    output_dir = Path("outputs") / f"song_{timestamp}_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.05, desc="Initializing YuE2 Pipeline...")
        pipe = load_pipeline(device=device, offload_ar=offload_ar)

        # Update ODE steps in generation config
        steps = int(ode_steps)
        pipe.generation_config = GenerationConfig(
            ode_steps=steps,
            abc=Sampling(temperature=float(temperature), top_p=float(top_p), repetition_penalty=float(repetition_penalty)),
            semantic=Sampling(temperature=float(temperature), top_p=float(top_p), repetition_penalty=float(repetition_penalty)),
        )

        kwargs = {
            "style": style.strip(),
            "lyrics": lyrics.strip(),
            "cot": cot_mode,
            "seed": seed,
            "cfg_scale": float(cfg_scale),
        }

        if custom_abc and custom_abc.strip():
            if cot_mode == "off":
                raise gr.Error("CoT mode cannot be 'off' when supplying a custom ABC score.")
            kwargs["abc"] = custom_abc.strip()

        if stage == "Plan Score Only":
            progress(0.3, desc="Generating symbolic score (ABC)...")
            plan = pipe.plan(**kwargs)
            plan.save(output_dir)
            score_text = plan.abc or "No ABC score generated."
            summary = {
                "stage": "plan_only",
                "status": "complete",
                "cot": cot_mode,
                "seed": seed,
                "output_dir": str(output_dir),
                "plan_tokens": len(plan.abc_ids),
                "truncated": plan.truncated,
            }
            return (
                None,  # No audio yet
                score_text,
                score_text,  # also update the Reharmonization editor
                f"✅ Score plan generated successfully! Output: {output_dir.name}",
                json.dumps(summary, indent=2),
                scan_song_library(),
            )
        else:
            progress(0.2, desc="Planning score & tokens...")
            result = pipe(**kwargs)
            progress(0.85, desc="Saving audio artifacts...")
            result.save_artifacts(output_dir)

            audio_path = str(output_dir / "audio.flac")
            score_text = result.abc or ""
            duration_s = round(len(result.audio) / result.sample_rate, 2)
            summary = {
                "status": "complete",
                "audio_seconds": duration_s,
                "sample_rate": result.sample_rate,
                "seed": seed,
                "cot": cot_mode,
                "ode_steps": steps,
                "cfg_scale": cfg_scale,
                "output_dir": str(output_dir),
                "timing": result.timing,
            }
            status_msg = f"🎵 Song generated: {duration_s}s audio ({steps} ODE steps, seed {seed})"
            return (
                audio_path,
                score_text,
                score_text,  # also update Reharmonization tab
                status_msg,
                json.dumps(summary, indent=2),
                scan_song_library(),
            )
    except Exception as exc:
        raise gr.Error(f"Generation error: {str(exc)}")


def synthesize_from_score(
    style: str,
    lyrics: str,
    custom_abc: str,
    seed_input: int,
    cfg_scale: float,
    ode_steps: int,
    device: str,
    offload_ar: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not custom_abc.strip():
        raise gr.Error("No ABC score provided. Generate or paste a score first.")
    if not style.strip():
        raise gr.Error("Please provide a style prompt.")
    if not lyrics.strip():
        raise gr.Error("Please provide lyrics.")

    seed = int(seed_input) if seed_input >= 0 else random.randint(0, 2**31 - 1)
    timestamp = int(time.time())
    output_dir = Path("outputs") / f"reharmonized_{timestamp}_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.1, desc="Loading YuE2 Pipeline...")
        pipe = load_pipeline(device=device, offload_ar=offload_ar)
        steps = int(ode_steps)
        pipe.generation_config = GenerationConfig(ode_steps=steps)

        progress(0.3, desc="Synthesizing audio conditioned on custom score...")
        result = pipe(
            style=style.strip(),
            lyrics=lyrics.strip(),
            cot="full",
            abc=custom_abc.strip(),
            seed=seed,
            cfg_scale=float(cfg_scale),
        )
        progress(0.9, desc="Saving reharmonized song...")
        result.save_artifacts(output_dir)

        audio_path = str(output_dir / "audio.flac")
        duration_s = round(len(result.audio) / result.sample_rate, 2)
        summary = {
            "status": "complete",
            "type": "reharmonized_from_score",
            "audio_seconds": duration_s,
            "seed": seed,
            "ode_steps": steps,
            "output_dir": str(output_dir),
            "timing": result.timing,
        }
        return (
            audio_path,
            f"✨ Successfully synthesized audio from custom score! ({duration_s}s)",
            json.dumps(summary, indent=2),
            scan_song_library(),
        )
    except Exception as exc:
        raise gr.Error(f"Synthesis failed: {str(exc)}")


def scan_song_library():
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []

    entries = []
    for item in sorted(outputs_dir.iterdir(), reverse=True):
        if not item.is_dir():
            continue
        audio_file = item / "audio.flac"
        if not audio_file.exists():
            continue

        title = item.name
        req_file = item / "request.json"
        res_file = item / "result.json"

        style_snippet = "No style prompt"
        duration_str = "--:--"
        seed_str = "-"

        if req_file.exists():
            try:
                req_data = json.loads(req_file.read_text(encoding="utf-8"))
                style_snippet = req_data.get("style", "")[:60] + "..."
                seed_str = str(req_data.get("seed", "-"))
            except Exception:
                pass

        if res_file.exists():
            try:
                res_data = json.loads(res_file.read_text(encoding="utf-8"))
                secs = float(res_data.get("audio_seconds", 0))
                mins = int(secs // 60)
                rem = int(secs % 60)
                duration_str = f"{mins}:{rem:02d}"
            except Exception:
                pass

        mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        entries.append({
            "Folder": item.name,
            "Created": mtime,
            "Duration": duration_str,
            "Seed": seed_str,
            "Style": style_snippet,
            "Path": str(audio_file),
        })

    return entries


def select_library_song(evt: gr.SelectData, table_data: list):
    if not table_data:
        return None, "", ""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None or idx >= len(table_data):
        return None, "", ""
    row = table_data[idx]
    folder_name = row["Folder"] if isinstance(row, dict) else row[0]
    folder_path = Path("outputs") / folder_name
    audio_path = str(folder_path / "audio.flac")

    score_text = ""
    score_file = folder_path / "score.abc"
    if score_file.exists():
        try:
            score_text = score_file.read_text(encoding="utf-8")
        except Exception:
            pass

    info_text = ""
    res_file = folder_path / "result.json"
    if res_file.exists():
        try:
            info_text = res_file.read_text(encoding="utf-8")
        except Exception:
            pass

    return audio_path, score_text, info_text


def load_song_to_studio(evt: gr.SelectData, table_data: list):
    if not table_data:
        return gr.update(), gr.update(), gr.update()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None or idx >= len(table_data):
        return gr.update(), gr.update(), gr.update()
    row = table_data[idx]
    folder_name = row["Folder"] if isinstance(row, dict) else row[0]
    folder_path = Path("outputs") / folder_name
    req_file = folder_path / "request.json"
    score_file = folder_path / "score.abc"

    style = gr.update()
    lyrics = gr.update()
    score = ""

    if req_file.exists():
        try:
            d = json.loads(req_file.read_text(encoding="utf-8"))
            style = d.get("style", "")
            lyrics = d.get("lyrics", "")
        except Exception:
            pass

    if score_file.exists():
        try:
            score = score_file.read_text(encoding="utf-8")
        except Exception:
            pass

    return style, lyrics, score


def run_system_doctor():
    class Args:
        vae = "standard"
        model = None
        verify_hashes = False
        offline = False
        output = None

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_doctor(Args())
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# CSS STYLING & 3D DEPTH DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

STUDIO_CSS = """
/* Root Studio Themes & CSS Variables */
:root {
  --studio-bg: #090b12;
  --studio-card: rgba(16, 20, 30, 0.85);
  --studio-card-border: rgba(0, 243, 255, 0.22);
  --studio-primary: #00f3ff;
  --studio-secondary: #ff007f;
  --studio-accent: #a855f7;
  --studio-glow: rgba(0, 243, 255, 0.35);
  --studio-btn-text: #05070c;
  --studio-text: #e2e8f0;
  --studio-muted: #94a3b8;
  --studio-shadow-depth: 0 16px 36px -8px rgba(0, 0, 0, 0.75), 0 0 24px rgba(0, 243, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 20px rgba(0, 243, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}

.theme-cyberpunk {
  --studio-bg: #07090e;
  --studio-card: rgba(14, 18, 28, 0.9);
  --studio-card-border: rgba(0, 243, 255, 0.25);
  --studio-primary: #00f3ff;
  --studio-secondary: #ff007f;
  --studio-accent: #b026ff;
  --studio-glow: rgba(0, 243, 255, 0.4);
  --studio-btn-text: #05070c;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 243, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(0, 243, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

.theme-midnight {
  --studio-bg: #0a0b0e;
  --studio-card: rgba(20, 22, 28, 0.9);
  --studio-card-border: rgba(245, 158, 11, 0.28);
  --studio-primary: #f59e0b;
  --studio-secondary: #fbbf24;
  --studio-accent: #d97706;
  --studio-glow: rgba(245, 158, 11, 0.35);
  --studio-btn-text: #0b0c10;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(245, 158, 11, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.12);
  --studio-btn-depth: 0 6px 22px rgba(245, 158, 11, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}

.theme-synthwave {
  --studio-bg: #0c0818;
  --studio-card: rgba(24, 18, 44, 0.88);
  --studio-card-border: rgba(244, 63, 94, 0.28);
  --studio-primary: #f43f5e;
  --studio-secondary: #8b5cf6;
  --studio-accent: #ec4899;
  --studio-glow: rgba(244, 63, 94, 0.38);
  --studio-btn-text: #0e071a;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(244, 63, 94, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(244, 63, 94, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

.theme-emerald {
  --studio-bg: #060e0c;
  --studio-card: rgba(12, 26, 22, 0.88);
  --studio-card-border: rgba(16, 185, 129, 0.28);
  --studio-primary: #10b981;
  --studio-secondary: #06b6d4;
  --studio-accent: #34d399;
  --studio-glow: rgba(16, 185, 129, 0.35);
  --studio-btn-text: #04120e;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(16, 185, 129, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

/* Studio Global Container */
#studio-wrapper {
  background: var(--studio-bg);
  min-height: 100vh;
  padding: 1rem;
  color: var(--studio-text);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  transition: all 0.3s ease;
}

/* 3D Depth Card Containers */
.depth-card {
  background: var(--studio-card) !important;
  border: 1px solid var(--studio-card-border) !important;
  border-radius: 16px !important;
  box-shadow: var(--studio-shadow-depth) !important;
  backdrop-filter: blur(16px) !important;
  padding: 1.25rem !important;
  margin-bottom: 1.25rem !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.depth-card:hover {
  border-color: var(--studio-primary) !important;
}

/* 3D Tactile Buttons */
.btn-3d-primary {
  background: linear-gradient(135deg, var(--studio-primary), var(--studio-secondary)) !important;
  color: var(--studio-btn-text) !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  border: none !important;
  border-radius: 12px !important;
  box-shadow: var(--studio-btn-depth) !important;
  cursor: pointer !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.btn-3d-primary:hover {
  transform: translateY(-2px) !important;
  filter: brightness(1.12) !important;
  box-shadow: 0 10px 28px var(--studio-glow) !important;
}

.btn-3d-primary:active {
  transform: translateY(2px) !important;
  filter: brightness(0.92) !important;
}

.btn-3d-secondary {
  background: rgba(255, 255, 255, 0.07) !important;
  color: var(--studio-text) !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}

.btn-3d-secondary:hover {
  border-color: var(--studio-primary) !important;
  color: var(--studio-primary) !important;
  transform: translateY(-1px) !important;
}

/* Chip Tag Palette */
.chip-btn {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 20px !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.25rem 0.65rem !important;
  color: var(--studio-muted) !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
}

.chip-btn:hover {
  background: var(--studio-primary) !important;
  color: var(--studio-btn-text) !important;
  border-color: var(--studio-primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px var(--studio-glow) !important;
}

/* Header & Status Banner */
.studio-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0 1.25rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  margin-bottom: 1.25rem;
}

.studio-title h1 {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--studio-primary), var(--studio-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

.studio-title p {
  color: var(--studio-muted);
  font-size: 0.95rem;
  margin: 0.2rem 0 0 0;
}

.hardware-pill {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 30px;
  padding: 0.4rem 1rem;
  font-size: 0.85rem;
  color: var(--studio-muted);
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hardware-pill .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
}

.tag-section-title {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--studio-primary);
  margin: 0.4rem 0 0.2rem 0;
}
"""


def create_ui():
    with gr.Blocks(title="YuE2 Music Studio") as app:
        gr.HTML(f"""
        <style>{STUDIO_CSS}</style>
        <script>
        function setStudioTheme(themeName) {{
            const wrapper = document.getElementById('studio-wrapper');
            if (wrapper) {{
                wrapper.className = 'theme-' + themeName.toLowerCase().split(' ')[0];
            }}
        }}
        </script>
        """)

        with gr.Column(elem_id="studio-wrapper", elem_classes=["theme-cyberpunk"]):
            # Header
            with gr.Row(elem_classes=["studio-header"]):
                with gr.Column(scale=8):
                    gr.HTML(f"""
                    <div class="studio-title">
                        <h1>🎵 YuE2 Music Studio</h1>
                        <p>Advanced Neural Music Generation • Symbolic Score Planning • Flow-Matching Acoustic Engine</p>
                    </div>
                    """)
                with gr.Column(scale=4):
                    theme_dropdown = gr.Dropdown(
                        label="🎨 Studio Theme",
                        choices=["Cyberpunk Neon", "Midnight Studio", "Synthwave Sunset", "Modern Emerald"],
                        value="Cyberpunk Neon",
                        interactive=True,
                    )
                    gr.HTML(f"""
                    <div class="hardware-pill">
                        <span class="dot"></span>
                        <span>{get_system_info()}</span>
                    </div>
                    """)

            with gr.Tabs():
                # TAB 1: STUDIO COMPOSITION
                with gr.TabItem("🎛️ Composition Studio"):
                    with gr.Row():
                        with gr.Column(scale=6):
                            with gr.Group(elem_classes=["depth-card"]):
                                with gr.Row():
                                    preset_dropdown = gr.Dropdown(
                                        label="✨ Instant Presets (Click to load)",
                                        choices=list(PRESETS.keys()),
                                        value="Neon Highway (80s Retrowave / Synthwave)",
                                        scale=8,
                                    )
                                    clear_btn = gr.Button("Clear", scale=2, size="sm", variant="secondary")

                                style_input = gr.Textbox(
                                    label="Musical Style & Instrumentation Prompt",
                                    placeholder="English, 80s synthwave, driving electronic drums, analog synthesizer, warm bass, 122 BPM",
                                    value=PRESETS["Neon Highway (80s Retrowave / Synthwave)"]["style"],
                                    lines=2,
                                )

                                with gr.Accordion("🏷️ Interactive Style Tag Palette (Click to add)", open=True):
                                    gr.HTML('<div class="tag-section-title">Genre & Sound</div>')
                                    with gr.Row():
                                        for tag in GENRE_TAGS[:5]:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for tag in GENRE_TAGS[5:]:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Vocals</div>')
                                    with gr.Row():
                                        for tag in VOCAL_TAGS:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Instruments</div>')
                                    with gr.Row():
                                        for tag in INSTRUMENT_TAGS[:5]:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for tag in INSTRUMENT_TAGS[5:]:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Mood & Tempo</div>')
                                    with gr.Row():
                                        for tag in MOOD_TAGS[:4]:
                                            b = gr.Button(tag, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for bpm in BPM_PRESETS:
                                            b = gr.Button(bpm, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(bpm)], outputs=style_input)

                            with gr.Group(elem_classes=["depth-card"]):
                                gr.Markdown("#### 📝 Lyrics & Song Structure")
                                lyrics_input = gr.Textbox(
                                    label="Lyrics (Structured with section tags)",
                                    placeholder="[Verse]\nWalking through the neon light...\n\n[Chorus]\nTake me back to yesterday...",
                                    value=PRESETS["Neon Highway (80s Retrowave / Synthwave)"]["lyrics"],
                                    lines=9,
                                )
                                with gr.Row():
                                    for section in ["Intro", "Verse 1", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Solo", "Outro"]:
                                        sb = gr.Button(f"+ [{section}]", size="sm", elem_classes=["chip-btn"])
                                        sb.click(fn=insert_lyrics_tag, inputs=[lyrics_input, gr.State(section)], outputs=[lyrics_input])

                        with gr.Column(scale=6):
                            with gr.Group(elem_classes=["depth-card"]):
                                with gr.Row():
                                    cot_choice = gr.Dropdown(
                                        label="CoT Reasoning Mode",
                                        choices=[
                                            ("Full (Chord-Annotated ABC Score Plan)", "full"),
                                            ("Melody (Melody plan only, free harmony)", "melody"),
                                            ("Off (Direct audio without symbolic score)", "off"),
                                        ],
                                        value="full",
                                        scale=6,
                                    )
                                    stage_choice = gr.Dropdown(
                                        label="Stage",
                                        choices=["Full Song (Audio + Score)", "Plan Score Only"],
                                        value="Full Song (Audio + Score)",
                                        scale=4,
                                    )

                                with gr.Row():
                                    ode_steps_choice = gr.Radio(
                                        label="Acoustic ODE Steps",
                                        choices=[
                                            ("4 (Draft Preview ~25s)", 4),
                                            ("8 (Fast ~45s)", 8),
                                            ("16 (Standard ~90s)", 16),
                                            ("32 (Studio Master ~180s)", 32),
                                        ],
                                        value=16,
                                    )

                                with gr.Row():
                                    seed_box = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, scale=4)
                                    cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=3.0, value=1.0, step=0.1, scale=4)

                                with gr.Row():
                                    generate_song_btn = gr.Button(
                                        "⚡ Generate Song",
                                        variant="primary",
                                        size="lg",
                                        elem_classes=["btn-3d-primary"],
                                        scale=7,
                                    )
                                    plan_only_btn = gr.Button(
                                        "🎼 Plan Score Only",
                                        variant="secondary",
                                        size="lg",
                                        elem_classes=["btn-3d-secondary"],
                                        scale=5,
                                    )

                            with gr.Group(elem_classes=["depth-card"]):
                                gr.Markdown("### 🎧 Audio Playback & Output")
                                audio_output = gr.Audio(label="Rendered FLAC Audio", type="filepath")
                                status_output = gr.Textbox(label="Status", interactive=False, max_lines=2)

                                with gr.Accordion("🎼 Symbolic ABC Music Sheet", open=True):
                                    score_display = gr.Code(label="ABC Music Score", language=None, lines=6)

                                with gr.Accordion("📊 Generation Diagnostics & Receipts", open=False):
                                    metadata_output = gr.Code(label="Result Manifest", language="json", lines=5)

                # TAB 2: SCORE REHARMONIZATION & COVER
                with gr.TabItem("🎼 Score & Reharmonize"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("""
                        ### 🎼 Score-Conditioned Synthesis & Reharmonization
                        YuE2's symbolic CoT architecture allows you to **edit musical chords, change notes, or adjust the tempo header**, and then synthesize high-fidelity audio directly from your revised score!
                        """)
                        with gr.Row():
                            with gr.Column(scale=7):
                                reharmonize_score_input = gr.Code(
                                    label="ABC Sheet Music (Edit chords in quotes, e.g. \"Am7\", \"F#m\", or modify note durations)",
                                    language=None,
                                    lines=14,
                                    value="""X:1
M:4/4
L:1/32
Q:1/4=122
K:G
%%stretchstaff 1
V:Vocal clef=treble
V:Chords clef=treble
[V:Vocal] z32 | z8 B2 B2 B4 c2 c2 c2 B2 A4 | z8 B2 B2 B4 c2 c2 c2 B2 A4 |]
[V:Chords] "G" [G4B4d4] z28 | "Em" [E4G4B4] z28 | "C" [C4E4G4] z28 |]""",
                                )
                                synthesize_score_btn = gr.Button(
                                    "✨ Synthesize Audio from this Score",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["btn-3d-primary"],
                                )

                            with gr.Column(scale=5):
                                reharmonized_audio_output = gr.Audio(label="Reharmonized Song Audio", type="filepath")
                                reharmonized_status = gr.Textbox(label="Status", interactive=False)
                                reharmonized_meta = gr.Code(label="Receipt", language="json", lines=5)

                # TAB 3: ACOUSTIC & SAMPLING ENGINE
                with gr.TabItem("⚙️ Acoustic & Sampling Engine"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("### 🎛️ Flow Matching & Autoregressive Sampling Parameters")
                        with gr.Row():
                            temp_slider = gr.Slider(label="Sampling Temperature", minimum=0.4, maximum=1.6, value=1.0, step=0.05)
                            topp_slider = gr.Slider(label="Top-P (Nucleus Sampling)", minimum=0.7, maximum=1.0, value=0.95, step=0.01)
                            rep_slider = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=1.6, value=1.2, step=0.05)

                        with gr.Row():
                            device_select = gr.Radio(
                                label="Execution Device",
                                choices=["cuda", "cpu"],
                                value="cuda" if torch.cuda.is_available() else "cpu",
                            )
                            offload_toggle = gr.Checkbox(
                                label="Phase-Based AR/NAR VRAM Offloading (Recommended for ≤ 12GB GPUs)",
                                value=True,
                            )

                        gr.Markdown("""
                        > [!TIP]
                        > **Memory Guide**:
                        > - **Phase-Based VRAM Offloading**: Automatically swaps AR weights (4.03 GiB) and NAR weights (2.63 GiB) between GPU and CPU during planning and ODE flow matching. Keeps peak GPU memory under **4.8 GiB** at all times!
                        > - **ODE Steps**: 4 steps provides a very fast 25-second preview. 16 steps delivers studio-quality dynamics and crisp acoustic separation.
                        """)

                # TAB 4: SONG LIBRARY & HISTORY
                with gr.TabItem("📂 Song Library & History"):
                    with gr.Group(elem_classes=["depth-card"]):
                        with gr.Row():
                            gr.Markdown("### 🗃️ Generated Song Explorer")
                            refresh_library_btn = gr.Button("🔄 Refresh Library", size="sm", elem_classes=["chip-btn"])

                        library_table = gr.Dataframe(
                            headers=["Folder", "Created", "Duration", "Seed", "Style", "Path"],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            value=scan_song_library(),
                            interactive=False,
                        )

                        with gr.Row():
                            with gr.Column(scale=6):
                                library_audio = gr.Audio(label="Song Audio Player", type="filepath")
                            with gr.Column(scale=6):
                                library_score = gr.Code(label="ABC Score", language=None, lines=6)

                # TAB 5: SYSTEM & HARDWARE DOCTOR
                with gr.TabItem("🩺 System Doctor"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("### 🔍 Hardware Status & Dependency Diagnostics")
                        doctor_btn = gr.Button("Run Environment Health Check", variant="secondary", elem_classes=["btn-3d-secondary"])
                        doctor_output = gr.Code(label="Diagnostic Report", language="json", lines=16)

            gr.Markdown("""
            ---
            *YuE2 open-source music model by M-A-P. Windows One-Click & Modern Studio UI by Mr5elfDe5truct.*
            """)

        # Theme switcher client-side action
        theme_dropdown.change(
            fn=None,
            inputs=[theme_dropdown],
            js="""(theme) => {
                const wrapper = document.getElementById('studio-wrapper');
                if (wrapper) {
                    wrapper.className = 'theme-' + theme.toLowerCase().split(' ')[0];
                }
            }""",
        )

        # Preset loader
        preset_dropdown.change(
            fn=load_preset_values,
            inputs=[preset_dropdown],
            outputs=[style_input, lyrics_input, cot_choice, ode_steps_choice],
        )

        # Clear button
        clear_btn.click(
            fn=lambda: ("", "", "full", 16),
            outputs=[style_input, lyrics_input, cot_choice, ode_steps_choice],
        )

        # Generate Full Song button
        generate_song_btn.click(
            fn=lambda s, l, c, a, se, cfg, o, t, tp, rp, dev, off: generate_music(
                s, l, c, a, se, cfg, o, t, tp, rp, "Full Song (Audio + Score)", dev, off
            ),
            inputs=[
                style_input,
                lyrics_input,
                cot_choice,
                reharmonize_score_input,
                seed_box,
                cfg_slider,
                ode_steps_choice,
                temp_slider,
                topp_slider,
                rep_slider,
                device_select,
                offload_toggle,
            ],
            outputs=[
                audio_output,
                score_display,
                reharmonize_score_input,
                status_output,
                metadata_output,
                library_table,
            ],
        )

        # Plan Score Only button
        plan_only_btn.click(
            fn=lambda s, l, c, a, se, cfg, o, t, tp, rp, dev, off: generate_music(
                s, l, c, a, se, cfg, o, t, tp, rp, "Plan Score Only", dev, off
            ),
            inputs=[
                style_input,
                lyrics_input,
                cot_choice,
                reharmonize_score_input,
                seed_box,
                cfg_slider,
                ode_steps_choice,
                temp_slider,
                topp_slider,
                rep_slider,
                device_select,
                offload_toggle,
            ],
            outputs=[
                audio_output,
                score_display,
                reharmonize_score_input,
                status_output,
                metadata_output,
                library_table,
            ],
        )

        # Synthesize from Score button
        synthesize_score_btn.click(
            fn=synthesize_from_score,
            inputs=[
                style_input,
                lyrics_input,
                reharmonize_score_input,
                seed_box,
                cfg_slider,
                ode_steps_choice,
                device_select,
                offload_toggle,
            ],
            outputs=[
                reharmonized_audio_output,
                reharmonized_status,
                reharmonized_meta,
                library_table,
            ],
        )

        # Refresh Library
        refresh_library_btn.click(fn=scan_song_library, outputs=[library_table])

        # Select song from table
        library_table.select(
            fn=select_library_song,
            inputs=[library_table],
            outputs=[library_audio, library_score, metadata_output],
        )

        # Doctor
        doctor_btn.click(fn=run_system_doctor, outputs=[doctor_output])

    return app


def main():
    parser = argparse.ArgumentParser(description="YuE2 Music Studio")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--listen", action="store_true", help="Listen on all network interfaces (0.0.0.0)")
    args = parser.parse_args()

    app = create_ui()
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    print(f"[*] Launching YuE2 Music Studio on http://{server_name}:{args.port}...")
    app.launch(
        server_name=server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
