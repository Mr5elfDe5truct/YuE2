"""YuE2 Music Studio: Modern 3D Interactive Audio Workstation for Song Generation.

Features:
- 3D depth shadows, neumorphic bevels, and layered glassmorphism cards.
- 4 User-selectable themes: Cyberpunk Neon, Midnight Studio, Synthwave Sunset, and Modern Emerald.
- Interactive style prompt composer with glowing clickable category chips (Genres, Vocals, Instruments, Moods, BPM).
- Lyrics structure builder (+ [Intro], + [Verse], + [Chorus], etc.).
- Reharmonization & Score Editing: Edit ABC chords/melodies and synthesize directly from the score.
- Full model controls: ODE steps (4/8/16/32), CFG guidance (1.0-3.0), temperature, top-p, and repetition penalty.
- Song Library & History explorer to preview, listen, delete, and reload past generations from outputs/.
- Live Hardware Doctor & VRAM monitor.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Configure Windows UTF-8 and PyTorch environment
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Install with: pip install gradio", file=sys.stderr)
    sys.exit(1)

from yue2.pipeline import YuE2Pipeline
from yue2.protocol import GenerationConfig, Sampling
from yue2.cli import doctor as cli_doctor
from yue2.cover_art import create_cover_art, build_cover_prompt
from yue2.transcriber import transcribe_audio_to_abc
from yue2.score_analyzer import (
    analyze_score,
    format_score_report_markdown,
    diff_two_scores,
    CURATED_CHORD_PALETTE,
)

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
    "Neon Highway (80s Retrowave / Synthwave)": {
        "style": "English, 80s synthwave, energetic male vocals, analog synthesizer arpeggios, driving punchy drums, gated reverb snare, deep synth bassline, 122 BPM",
        "lyrics": "[Intro]\n(Synth arpeggio sweeps into punchy drums)\n\n[Verse 1]\nStreets of chrome beneath the headlights\nChasing shadows into midnight\nEngine humming like a heartbeat\nLeaving all the ghosts behind me\n\n[Chorus]\nRunning down the neon highway\nNothing in the world can stop us now\nElectric skies are burning sideways\nWe will never let the fire out",
        "cot": "full",
        "ode_steps": 16,
    },
    "City Lights (Warm Piano Pop)": {
        "style": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
        "lyrics": "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night",
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

# Tag Palette options with musical icons (display label -> raw tag)
GENRE_TAGS = [
    ("🌆 Synthwave", "Synthwave"),
    ("☕ Piano Pop", "Contemporary Pop"),
    ("📻 Lo-Fi Chill", "Lo-Fi Chill Hop"),
    ("🎸 Acoustic Pop", "Acoustic Pop"),
    ("🏙️ R&B Soul", "R&B Soul"),
    ("⚡ Cyberpunk EDM", "Cyberpunk EDM"),
    ("🎬 Cinematic", "Cinematic Orchestral"),
    ("🕶️ 80s Rock", "80s Rock"),
    ("🕺 Funk Groove", "Funk Groove"),
]

VOCAL_TAGS = [
    ("🎤 Warm Female", "Warm Female Vocal"),
    ("✨ Airy Ethereal", "Airy Ethereal Female"),
    ("🎙️ Soulful Male", "Soulful Male Vocal"),
    ("⚡ Rock Male", "Gritty Rock Male"),
    ("👥 Vocal Duet", "Vocal Duet"),
    ("🌌 Dreamy Choir", "Dreamy Choir"),
    ("🎧 Instrumental", "Instrumental (No Vocals)"),
]

INSTRUMENT_TAGS = [
    ("🎹 Grand Piano", "Grand Piano"),
    ("🎸 Acoustic Guitar", "Acoustic Guitar"),
    ("⚡ Electric Guitar", "Electric Guitar Solo"),
    ("🔊 808 Bass", "Warm 808 Bass"),
    ("🎛️ Analog Synth", "Analog Synth"),
    ("🎷 Tenor Sax", "Tenor Saxophone"),
    ("🥁 Brushed Drums", "Brushed Drums"),
    ("💥 Punchy Drums", "Punchy Live Drums"),
    ("🎻 Strings", "Orchestral Strings"),
]

MOOD_TAGS = [
    ("🌟 Uplifting", "Uplifting & Inspiring"),
    ("🌧️ Melancholic", "Melancholic & Moody"),
    ("☕ Chill & Relaxed", "Relaxed & Chill"),
    ("⚡ Energetic", "Energetic & Driving"),
    ("📼 Nostalgic 80s", "Nostalgic 80s"),
    ("🌑 Atmospheric", "Dark & Atmospheric"),
    ("💖 Romantic", "Romantic & Tender"),
]

BPM_PRESETS = [
    ("⏱️ 75 BPM", "75 BPM"),
    ("⏱️ 88 BPM", "88 BPM"),
    ("⏱️ 96 BPM", "96 BPM"),
    ("⏱️ 110 BPM", "110 BPM"),
    ("⏱️ 122 BPM", "122 BPM"),
    ("⏱️ 138 BPM", "138 BPM"),
]


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
    cover_engine: str = "cloud",
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
                None,  # No cover yet
                score_text,
                score_text,  # also update the Reharmonization editor
                f"✅ Score plan generated successfully! Saved to: {output_dir.name}",
                json.dumps(summary, indent=2),
                get_library_table_data(),
                gr.update(choices=get_library_choices()),
            )
        else:
            progress(0.2, desc="Planning score & tokens...")
            result = pipe(**kwargs)
            progress(0.85, desc="Saving audio artifacts...")
            result.save_artifacts(output_dir)

            audio_path = str(output_dir / "audio.flac")
            score_text = result.abc or ""
            duration_s = round(len(result.audio) / result.sample_rate, 2)

            cover_file = None
            if cover_engine and cover_engine != "none":
                progress(0.92, desc=f"Synthesizing album cover art ({cover_engine})...")
                try:
                    cover_file = create_cover_art(
                        style=style.strip(),
                        lyrics=lyrics.strip(),
                        engine=cover_engine,
                        output_dir=output_dir,
                        title=output_dir.name,
                        seed=seed,
                    )
                except Exception as ce:
                    print(f"[!] Cover art generation note: {ce}")

            cover_path = str(cover_file) if cover_file and Path(cover_file).exists() else None

            summary = {
                "status": "complete",
                "audio_seconds": duration_s,
                "sample_rate": result.sample_rate,
                "seed": seed,
                "cot": cot_mode,
                "ode_steps": steps,
                "cfg_scale": cfg_scale,
                "cover_art": cover_path,
                "output_dir": str(output_dir),
                "timing": result.timing,
            }
            status_msg = f"🎵 Song generated: {duration_s}s audio ({steps} ODE steps, seed {seed})"
            return (
                audio_path,
                cover_path,
                score_text,
                score_text,  # also update Reharmonization tab
                status_msg,
                json.dumps(summary, indent=2),
                get_library_table_data(),
                gr.update(choices=get_library_choices()),
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
    cover_engine: str = "cloud",
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
        progress(0.85, desc="Saving reharmonized song...")
        result.save_artifacts(output_dir)

        audio_path = str(output_dir / "audio.flac")
        duration_s = round(len(result.audio) / result.sample_rate, 2)

        cover_file = None
        if cover_engine and cover_engine != "none":
            progress(0.92, desc=f"Synthesizing album cover art ({cover_engine})...")
            try:
                cover_file = create_cover_art(
                    style=style.strip(),
                    lyrics=lyrics.strip(),
                    engine=cover_engine,
                    output_dir=output_dir,
                    title=output_dir.name,
                    seed=seed,
                )
            except Exception as ce:
                print(f"[!] Cover art generation note: {ce}")

        cover_path = str(cover_file) if cover_file and Path(cover_file).exists() else None

        summary = {
            "status": "complete",
            "type": "reharmonized_from_score",
            "audio_seconds": duration_s,
            "seed": seed,
            "ode_steps": steps,
            "cover_art": cover_path,
            "output_dir": str(output_dir),
            "timing": result.timing,
        }
        return (
            audio_path,
            cover_path,
            f"✨ Successfully synthesized audio from custom score! ({duration_s}s)",
            json.dumps(summary, indent=2),
            get_library_table_data(),
            gr.update(choices=get_library_choices()),
        )
    except Exception as exc:
        raise gr.Error(f"Synthesis failed: {str(exc)}")



# ══════════════════════════════════════════════════════════════════════════════
# SONG LIBRARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_library_choices() -> list[tuple[str, str]]:
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []
    choices = []
    for item in sorted(outputs_dir.iterdir(), reverse=True):
        if not item.is_dir() or not (item / "audio.flac").exists():
            continue
        req_file = item / "request.json"
        res_file = item / "result.json"
        dur = ""
        snippet = ""
        if res_file.exists():
            try:
                s = float(json.loads(res_file.read_text(encoding="utf-8")).get("audio_seconds", 0))
                dur = f" ({int(s // 60)}:{int(s % 60):02d})"
            except Exception:
                pass
        if req_file.exists():
            try:
                st = json.loads(req_file.read_text(encoding="utf-8")).get("style", "")
                snippet = " • " + st[:35] + "..." if st else ""
            except Exception:
                pass
        label = f"{item.name}{dur}{snippet}"
        choices.append((label, item.name))
    return choices


def get_library_table_data() -> list[list[str]]:
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []
    rows = []
    for item in sorted(outputs_dir.iterdir(), reverse=True):
        if not item.is_dir() or not (item / "audio.flac").exists():
            continue
        mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        req_file = item / "request.json"
        res_file = item / "result.json"
        dur = "--:--"
        seed = "-"
        snippet = "No prompt"
        if res_file.exists():
            try:
                s = float(json.loads(res_file.read_text(encoding="utf-8")).get("audio_seconds", 0))
                dur = f"{int(s // 60)}:{int(s % 60):02d}"
            except Exception:
                pass
        if req_file.exists():
            try:
                d = json.loads(req_file.read_text(encoding="utf-8"))
                seed = str(d.get("seed", "-"))
                snippet = d.get("style", "")[:55]
            except Exception:
                pass
        rows.append([item.name, mtime, dur, seed, snippet])
    return rows


def load_song_from_id(folder_name: str):
    if not folder_name:
        return None, None, "", "Select a song from the library above.", ""
    folder_path = Path("outputs") / folder_name
    audio_path = str(folder_path / "audio.flac") if (folder_path / "audio.flac").exists() else None
    cover_path = str(folder_path / "cover.png") if (folder_path / "cover.png").exists() else None
    score_text = ""
    score_file = folder_path / "score.abc"
    if score_file.exists():
        try:
            score_text = score_file.read_text(encoding="utf-8")
        except Exception:
            pass

    details_md = f"### 🎵 Song: `{folder_name}`\n"
    req_file = folder_path / "request.json"
    res_file = folder_path / "result.json"
    if req_file.exists():
        try:
            req = json.loads(req_file.read_text(encoding="utf-8"))
            details_md += f"- **Style**: `{req.get('style', '')}`\n"
            details_md += f"- **Seed**: `{req.get('seed', '')}` | **CoT Mode**: `{req.get('cot', '')}`\n"
            details_md += f"- **Lyrics Preview**:\n```\n{req.get('lyrics', '')[:250]}...\n```\n"
        except Exception:
            pass
    if res_file.exists():
        try:
            res = json.loads(res_file.read_text(encoding="utf-8"))
            details_md += f"- **Duration**: {res.get('audio_seconds', 0):.2f}s | **Sample Rate**: {res.get('sample_rate', 48000)} Hz\n"
        except Exception:
            pass

    return audio_path, cover_path, score_text, details_md, folder_name


def load_selected_song_into_studio(folder_name: str):
    if not folder_name:
        raise gr.Error("No song selected to load into studio.")
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

    gr.Info(f"Loaded '{folder_name}' into Composition Studio and Reharmonize tabs!")
    return style, lyrics, score


def delete_selected_song(folder_name: str):
    if not folder_name:
        raise gr.Error("No song selected to delete.")
    folder_path = Path("outputs") / folder_name
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
    new_choices = get_library_choices()
    new_rows = get_library_table_data()
    first_choice = new_choices[0][1] if new_choices else None
    audio, cover, score, details, _ = load_song_from_id(first_choice) if first_choice else (None, None, "", "No songs left.", "")
    gr.Info(f"Deleted '{folder_name}' from library.")
    return audio, cover, score, details, gr.update(choices=new_choices, value=first_choice), new_rows


def regenerate_cover_art_for_song(folder_name: str, engine: str):
    if not folder_name:
        raise gr.Error("Please select a song from the library first.")
    folder_path = Path("outputs") / folder_name
    req_file = folder_path / "request.json"
    style = "modern music"
    lyrics = ""
    seed = random.randint(0, 2**31 - 1)
    if req_file.exists():
        try:
            req = json.loads(req_file.read_text(encoding="utf-8"))
            style = req.get("style", style)
            lyrics = req.get("lyrics", "")
            seed = req.get("seed", seed)
        except Exception:
            pass
    cov = create_cover_art(
        style=style,
        lyrics=lyrics,
        engine=engine,
        output_dir=folder_path,
        title=folder_name,
        seed=seed,
    )
    if cov and Path(cov).exists():
        gr.Info(f"✨ New album cover generated for {folder_name}!")
        return str(cov)
    else:
        raise gr.Error("Failed to generate cover art. Please check connection or try Procedural Studio.")


def transcribe_sample_action(audio_path: str, bpm: int, key: str, offset_sec: float, max_sec: float):
    if not audio_path:
        raise gr.Error("Please upload or record an audio sample first.")
    try:
        abc = transcribe_audio_to_abc(
            audio_path=audio_path,
            bpm=int(bpm),
            key=key,
            offset_seconds=float(offset_sec),
            max_seconds=float(max_sec),
            title="Audio Sample Transcription",
        )
        analysis = analyze_score(abc)
        report_md = format_score_report_markdown(analysis)
        gr.Info(f"Transcribed audio sample ({analysis.get('bars_count')} bars at {bpm} BPM)!")
        return abc, report_md
    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")


def lint_score_action(score_text: str):
    if not score_text or not score_text.strip():
        raise gr.Error("Please provide an ABC score to lint.")
    analysis = analyze_score(score_text)
    return format_score_report_markdown(analysis)


def insert_chord_into_score(current_score: str, chord: str) -> str:
    if not current_score:
        return f'"{chord}" '
    chord_tag = f'"{chord}"'
    if '[V:Chords]' in current_score:
        idx = current_score.find('[V:Chords]')
        head = current_score[:idx + len('[V:Chords]')]
        tail = current_score[idx + len('[V:Chords]'):]
        return f"{head} {chord_tag} {tail.lstrip()}"
    return f'{current_score.rstrip()} {chord_tag}'


def compare_two_songs(song_a_id: str, song_b_id: str):
    if not song_a_id or not song_b_id:
        return None, None, "", None, None, "", "Select both Song A and Song B to compare."

    audio_a, cover_a, score_a, details_a, _ = load_song_from_id(song_a_id)
    audio_b, cover_b, score_b, details_b, _ = load_song_from_id(song_b_id)

    diff_data = diff_two_scores(score_a or "", score_b or "")

    tempo_status = "⚠️ Changed" if diff_data.get("tempo_changed") else "✅ Identical"
    key_status = "⚠️ Changed" if diff_data.get("key_changed") else "✅ Identical"
    bars_status = "Different length" if diff_data.get("bars_a") != diff_data.get("bars_b") else "Matched length"

    diff_md = f"""### 🎧 A/B Studio Comparison: `{song_a_id}` vs `{song_b_id}`

| Metric | Song A (`{song_a_id}`) | Song B (`{song_b_id}`) | Difference |
|:---|:---:|:---:|:---:|
| **Tempo** | {diff_data.get('tempo_a', '-')} BPM | {diff_data.get('tempo_b', '-')} BPM | {tempo_status} |
| **Key** | {diff_data.get('key_a', '-')} | {diff_data.get('key_b', '-')} | {key_status} |
| **Measures** | {diff_data.get('bars_a', '-')} Bars | {diff_data.get('bars_b', '-')} Bars | {bars_status} |
"""

    if diff_data.get("added_chords"):
        diff_md += f"\n- **Chords unique to Song B**: `{', '.join(diff_data['added_chords'])}`\n"
    if diff_data.get("removed_chords"):
        diff_md += f"\n- **Chords unique to Song A**: `{', '.join(diff_data['removed_chords'])}`\n"

    return audio_a, cover_a, score_a, audio_b, cover_b, score_b, diff_md


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
/* Theme 1: Cyberpunk Neon */
body.theme-cyberpunk, .gradio-container.theme-cyberpunk, #studio-wrapper.theme-cyberpunk {
  --studio-bg: #07090e;
  --studio-card: rgba(14, 18, 28, 0.95);
  --studio-card-border: rgba(0, 243, 255, 0.3);
  --studio-primary: #00f3ff;
  --studio-secondary: #ff007f;
  --studio-accent: #b026ff;
  --studio-glow: rgba(0, 243, 255, 0.45);
  --studio-tag-bg: rgba(0, 243, 255, 0.08);
  --studio-tag-border: rgba(0, 243, 255, 0.35);
  --studio-tag-hover: #00f3ff;
  --studio-btn-text: #05070c;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 243, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(0, 243, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

/* Theme 2: Midnight Studio */
body.theme-midnight, .gradio-container.theme-midnight, #studio-wrapper.theme-midnight {
  --studio-bg: #0a0b0e;
  --studio-card: rgba(20, 22, 28, 0.95);
  --studio-card-border: rgba(245, 158, 11, 0.35);
  --studio-primary: #f59e0b;
  --studio-secondary: #fbbf24;
  --studio-accent: #d97706;
  --studio-glow: rgba(245, 158, 11, 0.45);
  --studio-tag-bg: rgba(245, 158, 11, 0.08);
  --studio-tag-border: rgba(245, 158, 11, 0.35);
  --studio-tag-hover: #f59e0b;
  --studio-btn-text: #0b0c10;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(245, 158, 11, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.12);
  --studio-btn-depth: 0 6px 22px rgba(245, 158, 11, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}

/* Theme 3: Synthwave Sunset */
body.theme-synthwave, .gradio-container.theme-synthwave, #studio-wrapper.theme-synthwave {
  --studio-bg: #0c0818;
  --studio-card: rgba(24, 18, 44, 0.95);
  --studio-card-border: rgba(244, 63, 94, 0.35);
  --studio-primary: #f43f5e;
  --studio-secondary: #8b5cf6;
  --studio-accent: #ec4899;
  --studio-glow: rgba(244, 63, 94, 0.45);
  --studio-tag-bg: rgba(244, 63, 94, 0.08);
  --studio-tag-border: rgba(244, 63, 94, 0.35);
  --studio-tag-hover: #f43f5e;
  --studio-btn-text: #0e071a;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(244, 63, 94, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(244, 63, 94, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

/* Theme 4: Modern Emerald */
body.theme-emerald, .gradio-container.theme-emerald, #studio-wrapper.theme-emerald {
  --studio-bg: #060e0c;
  --studio-card: rgba(12, 26, 22, 0.95);
  --studio-card-border: rgba(16, 185, 129, 0.35);
  --studio-primary: #10b981;
  --studio-secondary: #06b6d4;
  --studio-accent: #34d399;
  --studio-glow: rgba(16, 185, 129, 0.45);
  --studio-tag-bg: rgba(16, 185, 129, 0.08);
  --studio-tag-border: rgba(16, 185, 129, 0.35);
  --studio-tag-hover: #10b981;
  --studio-btn-text: #04120e;
  --studio-shadow-depth: 0 18px 40px -8px rgba(0, 0, 0, 0.8), 0 0 25px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  --studio-btn-depth: 0 6px 22px rgba(16, 185, 129, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

/* Base Container */
body, .gradio-container, #studio-wrapper {
  background: var(--studio-bg, #07090e) !important;
  color: #e2e8f0 !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  transition: background 0.3s ease, color 0.3s ease;
}

#studio-wrapper {
  min-height: 100vh;
  padding: 1rem;
}

/* 3D Depth Card Containers */
.depth-card {
  background: var(--studio-card, rgba(14, 18, 28, 0.95)) !important;
  border: 1px solid var(--studio-card-border, rgba(0, 243, 255, 0.3)) !important;
  border-radius: 16px !important;
  box-shadow: var(--studio-shadow-depth) !important;
  backdrop-filter: blur(16px) !important;
  padding: 1.25rem !important;
  margin-bottom: 1.25rem !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.depth-card:hover {
  border-color: var(--studio-primary, #00f3ff) !important;
}

/* 3D Tactile Buttons */
.btn-3d-primary {
  background: linear-gradient(135deg, var(--studio-primary, #00f3ff), var(--studio-secondary, #ff007f)) !important;
  color: var(--studio-btn-text, #05070c) !important;
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
  filter: brightness(1.15) !important;
  box-shadow: 0 10px 28px var(--studio-glow) !important;
}

.btn-3d-primary:active {
  transform: translateY(2px) !important;
  filter: brightness(0.92) !important;
}

.btn-3d-secondary {
  background: rgba(255, 255, 255, 0.08) !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(255, 255, 255, 0.2) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}

.btn-3d-secondary:hover {
  border-color: var(--studio-primary, #00f3ff) !important;
  color: var(--studio-primary, #00f3ff) !important;
  transform: translateY(-1px) !important;
}

/* Glowing Chip Tag Buttons */
.chip-btn {
  background: var(--studio-tag-bg, rgba(0, 243, 255, 0.08)) !important;
  border: 1px solid var(--studio-tag-border, rgba(0, 243, 255, 0.35)) !important;
  border-radius: 24px !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  padding: 0.3rem 0.75rem !important;
  color: #f1f5f9 !important;
  cursor: pointer !important;
  transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 3px 8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
}

.chip-btn:hover {
  background: var(--studio-primary, #00f3ff) !important;
  color: var(--studio-btn-text, #05070c) !important;
  border-color: var(--studio-primary, #00f3ff) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 16px var(--studio-glow) !important;
}

.chip-btn:active {
  transform: translateY(1px) !important;
}

/* Lyric Section Tag Buttons */
.lyric-chip-btn {
  background: rgba(168, 85, 247, 0.12) !important;
  border: 1px solid rgba(168, 85, 247, 0.4) !important;
  border-radius: 20px !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  padding: 0.25rem 0.65rem !important;
  color: #e9d5ff !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
}

.lyric-chip-btn:hover {
  background: #a855f7 !important;
  color: #ffffff !important;
  border-color: #a855f7 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(168, 85, 247, 0.4) !important;
}

/* Header & Status Banner */
.studio-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0 1.25rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 1.25rem;
}

.studio-title h1 {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--studio-primary, #00f3ff), var(--studio-secondary, #ff007f));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

.studio-title p {
  color: #94a3b8;
  font-size: 0.95rem;
  margin: 0.2rem 0 0 0;
}

.hardware-pill {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 30px;
  padding: 0.4rem 1rem;
  font-size: 0.85rem;
  color: #cbd5e1;
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
  color: var(--studio-primary, #00f3ff);
  margin: 0.5rem 0 0.25rem 0;
}

/* 3D Vinyl Album Cover Art Card */
.cover-art-card {
  border-radius: 16px !important;
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.75), 0 0 20px var(--studio-glow) !important;
  border: 1px solid var(--studio-card-border, rgba(0, 243, 255, 0.3)) !important;
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease !important;
  overflow: hidden !important;
  background: #000000 !important;
}

.cover-art-card:hover {
  transform: translateY(-3px) scale(1.01) !important;
  box-shadow: 0 22px 48px rgba(0, 0, 0, 0.85), 0 0 28px var(--studio-primary, #00f3ff) !important;
}
"""



def create_ui():
    initial_library_choices = get_library_choices()
    initial_choice = initial_library_choices[0][1] if initial_library_choices else None
    initial_audio, initial_cover, initial_score, initial_details, _ = load_song_from_id(initial_choice) if initial_choice else (None, None, "", "Select a song above.", "")

    with gr.Blocks(title="YuE2 Music Studio") as app:
        # Client-side style injection & Theme wrapper
        gr.HTML(f"""
        <style>{STUDIO_CSS}</style>
        <script>
        function setStudioTheme(themeName) {{
            const themeClass = 'theme-' + themeName.toLowerCase().split(' ')[0];
            const classes = ['theme-cyberpunk', 'theme-midnight', 'theme-synthwave', 'theme-emerald'];
            [document.body, document.querySelector('.gradio-container'), document.getElementById('studio-wrapper')].forEach(el => {{
                if (el) {{
                    classes.forEach(c => el.classList.remove(c));
                    el.classList.add(themeClass);
                }}
            }});
        }}
        </script>
        """)

        with gr.Column(elem_id="studio-wrapper", elem_classes=["theme-cyberpunk"]):
            # Studio Header
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
                # ──────────────────────────────────────────────────────────────
                # TAB 1: STUDIO COMPOSITION
                # ──────────────────────────────────────────────────────────────
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
                                        for label, tag in GENRE_TAGS[:5]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for label, tag in GENRE_TAGS[5:]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Vocals</div>')
                                    with gr.Row():
                                        for label, tag in VOCAL_TAGS:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Instruments</div>')
                                    with gr.Row():
                                        for label, tag in INSTRUMENT_TAGS[:5]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for label, tag in INSTRUMENT_TAGS[5:]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)

                                    gr.HTML('<div class="tag-section-title">Mood & Tempo</div>')
                                    with gr.Row():
                                        for label, tag in MOOD_TAGS[:4]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[style_input, gr.State(tag)], outputs=style_input)
                                    with gr.Row():
                                        for label, bpm in BPM_PRESETS:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
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
                                        sb = gr.Button(f"+ [{section}]", size="sm", elem_classes=["lyric-chip-btn"])
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
                                    cover_engine_choice = gr.Dropdown(
                                        label="🎨 Auto-Generated Album Cover Art Engine",
                                        choices=[
                                            ("☁️ Cloud AI Diffusion (Flux/SDXL Quality, 0 MB VRAM)", "cloud"),
                                            ("🎨 Procedural Graphic Studio (Vinyl Sleeve, Offline, 0 MB VRAM)", "procedural"),
                                            ("⚡ Local AI Diffusion (SD-Turbo on GPU, Sequenced)", "local"),
                                            ("🚫 Disabled (No Cover Art)", "none"),
                                        ],
                                        value="cloud",
                                        scale=12,
                                    )

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
                                gr.Markdown("### 🎧 Audio Playback & Album Cover")
                                with gr.Row():
                                    with gr.Column(scale=5):
                                        cover_output = gr.Image(label="Vinyl Album Jacket", type="filepath", elem_classes=["cover-art-card"])
                                    with gr.Column(scale=7):
                                        audio_output = gr.Audio(label="Rendered FLAC Audio", type="filepath")
                                        status_output = gr.Textbox(label="Status", interactive=False, max_lines=2)

                                with gr.Accordion("🎼 Symbolic ABC Music Sheet", open=True):
                                    score_display = gr.Code(label="ABC Music Score", language=None, lines=6)

                                with gr.Accordion("📊 Generation Diagnostics & Receipts", open=False):
                                    metadata_output = gr.Code(label="Result Manifest", language="json", lines=5)

                # ──────────────────────────────────────────────────────────────
                # ──────────────────────────────────────────────────────────────
                # TAB 2: AUDIO SAMPLE & REMIX STUDIO
                # ──────────────────────────────────────────────────────────────
                with gr.TabItem("🎙️ Sample & Remix Studio"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("""
                        ### 🎙️ Zero-Shot Audio Covers & Sample Remakes
                        Upload any audio sample (`.wav`, `.mp3`, `.flac`) or record a melody/riff via microphone.
                        YuE2's symbolic transcriber extracts the melodic pitch contours into native ABC sheet music, then synthesizes a brand-new complete production in your target style!
                        """)
                        with gr.Row():
                            with gr.Column(scale=5):
                                sample_audio_input = gr.Audio(
                                    label="Upload Audio Sample or Record (.wav, .mp3, .flac)",
                                    sources=["upload", "microphone"],
                                    type="filepath"
                                )
                                with gr.Row():
                                    sample_bpm = gr.Number(label="Target Tempo (BPM)", value=122, precision=0, scale=6)
                                    sample_key = gr.Dropdown(label="Estimated Key", choices=["C", "G", "D", "A", "E", "F", "Bb", "Am", "Em", "Dm"], value="C", scale=6)
                                with gr.Row():
                                    sample_offset = gr.Number(label="Start Offset (sec)", value=0.0, precision=1, scale=6)
                                    sample_max_sec = gr.Slider(label="Max Duration (sec)", minimum=5.0, maximum=45.0, value=30.0, step=1.0, scale=6)

                                transcribe_btn = gr.Button("🎼 Transcribe Sample to ABC Melody", variant="primary", elem_classes=["btn-3d-primary"])

                            with gr.Column(scale=7):
                                sample_extracted_abc = gr.Code(label="Extracted Symbolic ABC Sheet Music (Preview & Edit)", language=None, lines=8)
                                sample_analysis_display = gr.Markdown(value="*Upload an audio file and click Transcribe to inspect its musical structure.*")

                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("#### 🎨 Target Re-creation Style & Production Settings")
                        with gr.Row():
                            with gr.Column(scale=6):
                                sample_target_style = gr.Textbox(
                                    label="Target Musical Style Prompt",
                                    placeholder="English, 80s synthwave, driving electronic drums, analog synthesizer, warm bass, 122 BPM",
                                    value="English, 80s synthwave, driving electronic drums, analog synthesizer, warm bass, 122 BPM",
                                    lines=2,
                                )
                                with gr.Accordion("🏷️ Style Tag Palette (Click to add)", open=False):
                                    with gr.Row():
                                        for label, tag in GENRE_TAGS[:5]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[sample_target_style, gr.State(tag)], outputs=sample_target_style)
                                    with gr.Row():
                                        for label, tag in VOCAL_TAGS[:4]:
                                            b = gr.Button(label, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=add_style_tag, inputs=[sample_target_style, gr.State(tag)], outputs=sample_target_style)

                                sample_lyrics_input = gr.Textbox(
                                    label="Lyrics (Aligned with the sample's melodic phrasing)",
                                    placeholder="[Verse]\nEchoes in the neon light...\n\n[Chorus]\nTake me back to yesterday...",
                                    value="[Verse]\nEchoes calling through the night\nFootsteps fading out of sight\n\n[Chorus]\nRunning down the neon highway\nNothing in the world can stop us now",
                                    lines=5,
                                )

                            with gr.Column(scale=6):
                                with gr.Row():
                                    sample_remake_mode = gr.Radio(
                                        label="Remake Mode",
                                        choices=[
                                            ("Melody Cover (Free Style/Harmony, cot='melody')", "melody"),
                                            ("Harmonic Remake (Preserve Score & Chords, cot='full')", "full"),
                                        ],
                                        value="melody",
                                    )
                                with gr.Row():
                                    sample_ode_steps = gr.Radio(
                                        label="ODE Steps",
                                        choices=[("8 (Fast)", 8), ("16 (Standard)", 16), ("32 (Studio Master)", 32)],
                                        value=16,
                                    )
                                    sample_cover_engine = gr.Dropdown(
                                        label="Cover Art Engine",
                                        choices=[
                                            ("☁️ Cloud AI", "cloud"),
                                            ("🎨 Procedural", "procedural"),
                                            ("⚡ Local AI", "local"),
                                            ("🚫 None", "none"),
                                        ],
                                        value="cloud",
                                    )

                                sample_generate_btn = gr.Button(
                                    "⚡ Recreate Track from Sample",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["btn-3d-primary"],
                                )

                                with gr.Row():
                                    with gr.Column(scale=5):
                                        sample_rendered_cover = gr.Image(label="Vinyl Album Jacket", type="filepath", elem_classes=["cover-art-card"])
                                    with gr.Column(scale=7):
                                        sample_rendered_audio = gr.Audio(label="Recreated Song Audio", type="filepath")
                                        sample_rendered_status = gr.Textbox(label="Status", interactive=False)

                # ──────────────────────────────────────────────────────────────
                # TAB 3: SCORE REHARMONIZATION & MUSICAL LINTER
                # ──────────────────────────────────────────────────────────────
                with gr.TabItem("🎼 Score & Reharmonize"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("""
                        ### 🎼 Score-Conditioned Synthesis, Reharmonization & Musical Linter
                        YuE2's symbolic CoT architecture allows you to **edit musical chords, change notes, or adjust the tempo header**, and then synthesize high-fidelity audio directly from your revised score!
                        """)
                        with gr.Row():
                            with gr.Column(scale=7):
                                reharmonize_score_input = gr.Code(
                                    label="ABC Sheet Music (Edit chords in quotes, e.g. \"Am7\", \"F#m\", or modify note durations)",
                                    language=None,
                                    lines=13,
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
                                with gr.Row():
                                    lint_score_btn = gr.Button("🔍 Check & Lint Score", variant="secondary", elem_classes=["btn-3d-secondary"], scale=5)
                                    synthesize_score_btn = gr.Button("✨ Synthesize Audio from this Score", variant="primary", size="lg", elem_classes=["btn-3d-primary"], scale=7)

                                with gr.Accordion("🎹 Quick Chord Palette (Click to insert chord into score)", open=False):
                                    with gr.Row():
                                        for chord in CURATED_CHORD_PALETTE[:7]:
                                            b = gr.Button(chord, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=insert_chord_into_score, inputs=[reharmonize_score_input, gr.State(chord)], outputs=[reharmonize_score_input])
                                    with gr.Row():
                                        for chord in CURATED_CHORD_PALETTE[7:14]:
                                            b = gr.Button(chord, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=insert_chord_into_score, inputs=[reharmonize_score_input, gr.State(chord)], outputs=[reharmonize_score_input])
                                    with gr.Row():
                                        for chord in CURATED_CHORD_PALETTE[14:21]:
                                            b = gr.Button(chord, size="sm", elem_classes=["chip-btn"])
                                            b.click(fn=insert_chord_into_score, inputs=[reharmonize_score_input, gr.State(chord)], outputs=[reharmonize_score_input])

                                score_lint_output = gr.Markdown(value="*Click 'Check & Lint Score' to verify key signature, bar lengths, and chord syntax.*")

                            with gr.Column(scale=5):
                                reharmonized_cover_output = gr.Image(label="Vinyl Album Jacket", type="filepath", elem_classes=["cover-art-card"])
                                reharmonized_audio_output = gr.Audio(label="Reharmonized Song Audio", type="filepath")
                                reharmonized_status = gr.Textbox(label="Status", interactive=False)
                                reharmonized_meta = gr.Code(label="Receipt", language="json", lines=5)

                # ──────────────────────────────────────────────────────────────
                # TAB 4: A/B COMPARISON LAB
                # ──────────────────────────────────────────────────────────────
                with gr.TabItem("🎧 A/B Comparison Lab"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("""
                        ### 🎧 Side-by-Side A/B Studio Comparison
                        Compare any two generated songs, reharmonized versions, or sample remakes side-by-side. Inspect differences in tempo, key, measures, and harmonic chord progressions!
                        """)
                        with gr.Row():
                            compare_a_select = gr.Dropdown(
                                label="Track A (Reference / Original)",
                                choices=initial_library_choices,
                                value=initial_choice,
                                scale=5,
                            )
                            compare_b_select = gr.Dropdown(
                                label="Track B (Remake / Variation)",
                                choices=initial_library_choices,
                                value=initial_library_choices[1][1] if len(initial_library_choices) > 1 else initial_choice,
                                scale=5,
                            )
                            run_compare_btn = gr.Button("⚖️ Compare Tracks", variant="primary", scale=2, elem_classes=["btn-3d-primary"])

                        with gr.Row():
                            with gr.Column(scale=6):
                                gr.Markdown("#### 🎵 Track A")
                                compare_cover_a = gr.Image(label="Track A Cover", value=initial_cover, type="filepath", elem_classes=["cover-art-card"])
                                compare_audio_a = gr.Audio(label="Track A Audio", value=initial_audio, type="filepath")
                                with gr.Accordion("Track A Score", open=False):
                                    compare_score_a = gr.Code(label="Track A ABC Score", value=initial_score, language=None, lines=6)

                            with gr.Column(scale=6):
                                gr.Markdown("#### 🎵 Track B")
                                compare_cover_b = gr.Image(label="Track B Cover", type="filepath", elem_classes=["cover-art-card"])
                                compare_audio_b = gr.Audio(label="Track B Audio", type="filepath")
                                with gr.Accordion("Track B Score", open=False):
                                    compare_score_b = gr.Code(label="Track B ABC Score", language=None, lines=6)

                        with gr.Group(elem_classes=["depth-card"]):
                            compare_diff_display = gr.Markdown(value="*Select two songs above and click 'Compare Tracks' for a musical and structural breakdown.*")


                # ──────────────────────────────────────────────────────────────
                # TAB 3: ACOUSTIC & SAMPLING ENGINE
                # ──────────────────────────────────────────────────────────────
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

                # ──────────────────────────────────────────────────────────────
                # TAB 4: SONG LIBRARY & HISTORY
                # ──────────────────────────────────────────────────────────────
                with gr.TabItem("📂 Song Library & History"):
                    with gr.Group(elem_classes=["depth-card"]):
                        with gr.Row():
                            library_select = gr.Dropdown(
                                label="🎵 Select Song to Audition / Play",
                                choices=initial_library_choices,
                                value=initial_choice,
                                scale=8,
                            )
                            refresh_library_btn = gr.Button("🔄 Refresh", scale=2, size="sm", elem_classes=["chip-btn"])

                        with gr.Row():
                            with gr.Column(scale=5):
                                library_cover = gr.Image(label="Album Cover Jacket", value=initial_cover, type="filepath", elem_classes=["cover-art-card"])
                                with gr.Row():
                                    regen_cover_engine = gr.Dropdown(
                                        label="Engine",
                                        choices=[
                                            ("☁️ Cloud AI", "cloud"),
                                            ("🎨 Procedural", "procedural"),
                                            ("⚡ Local AI", "local"),
                                        ],
                                        value="cloud",
                                        scale=7,
                                    )
                                    regen_cover_btn = gr.Button("🎨 Regen Cover", scale=5, size="sm", elem_classes=["chip-btn"])
                            with gr.Column(scale=7):
                                library_audio = gr.Audio(label="Audio Player", value=initial_audio, type="filepath")
                                library_details = gr.Markdown(value=initial_details)
                                with gr.Row():
                                    load_to_studio_btn = gr.Button("📥 Load into Studio", variant="primary", elem_classes=["btn-3d-primary"])
                                    delete_song_btn = gr.Button("🗑️ Delete Song", variant="secondary")

                                with gr.Accordion("🎼 Symbolic ABC Music Sheet", open=False):
                                    library_score = gr.Code(label="ABC Music Score", value=initial_score, language=None, lines=6)

                        with gr.Accordion("📋 All Generated Songs Archive", open=True):
                            library_table = gr.Dataframe(
                                headers=["Song ID", "Created", "Duration", "Seed", "Style Preview"],
                                datatype=["str", "str", "str", "str", "str"],
                                value=get_library_table_data(),
                                interactive=False,
                            )


                # ──────────────────────────────────────────────────────────────
                # TAB 5: SYSTEM & HARDWARE DOCTOR
                # ──────────────────────────────────────────────────────────────
                with gr.TabItem("🩺 System Doctor"):
                    with gr.Group(elem_classes=["depth-card"]):
                        gr.Markdown("### 🔍 Hardware Status & Dependency Diagnostics")
                        doctor_btn = gr.Button("Run Environment Health Check", variant="secondary", elem_classes=["btn-3d-secondary"])
                        doctor_output = gr.Code(label="Diagnostic Report", language="json", lines=16)

            gr.Markdown("""
            ---
            *YuE2 open-source music model by M-A-P. Windows One-Click & Modern Studio UI by Mr5elfDe5truct.*
            """)

        # ──────────────────────────────────────────────────────────────────────
        # EVENT HANDLERS & CALLBACKS
        # ──────────────────────────────────────────────────────────────────────

        # Theme switcher client-side action
        theme_dropdown.change(
            fn=None,
            inputs=[theme_dropdown],
            js="""(theme) => {
                const themeClass = 'theme-' + theme.toLowerCase().split(' ')[0];
                const classes = ['theme-cyberpunk', 'theme-midnight', 'theme-synthwave', 'theme-emerald'];
                [document.body, document.querySelector('.gradio-container'), document.getElementById('studio-wrapper')].forEach(el => {
                    if (el) {
                        classes.forEach(c => el.classList.remove(c));
                        el.classList.add(themeClass);
                    }
                });
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
            fn=lambda s, l, c, a, se, cfg, o, t, tp, rp, dev, off, cov: generate_music(
                s, l, c, a, se, cfg, o, t, tp, rp, "Full Song (Audio + Score)", dev, off, cov
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
                cover_engine_choice,
            ],
            outputs=[
                audio_output,
                cover_output,
                score_display,
                reharmonize_score_input,
                status_output,
                metadata_output,
                library_table,
                library_select,
            ],
        )

        # Plan Score Only button
        plan_only_btn.click(
            fn=lambda s, l, c, a, se, cfg, o, t, tp, rp, dev, off, cov: generate_music(
                s, l, c, a, se, cfg, o, t, tp, rp, "Plan Score Only", dev, off, cov
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
                cover_engine_choice,
            ],
            outputs=[
                audio_output,
                cover_output,
                score_display,
                reharmonize_score_input,
                status_output,
                metadata_output,
                library_table,
                library_select,
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
                cover_engine_choice,
            ],
            outputs=[
                reharmonized_audio_output,
                reharmonized_cover_output,
                reharmonized_status,
                reharmonized_meta,
                library_table,
                library_select,
            ],
        )

        # Song Library Selection via Dropdown
        library_select.change(
            fn=load_song_from_id,
            inputs=[library_select],
            outputs=[library_audio, library_cover, library_score, library_details, library_select],
        )

        # Table Row Select updates Dropdown and loads song
        def on_table_select(evt: gr.SelectData, table_data: list):
            if not table_data or evt is None or not hasattr(evt, "index"):
                return None, None, "", "", gr.update()
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row_idx is not None and row_idx < len(table_data):
                row = table_data[row_idx]
                folder = row[0]
                return load_song_from_id(folder)
            return None, None, "", "", gr.update()

        library_table.select(
            fn=on_table_select,
            inputs=[library_table],
            outputs=[library_audio, library_cover, library_score, library_details, library_select],
        )

        # Transcribe Audio Sample Button
        transcribe_btn.click(
            fn=transcribe_sample_action,
            inputs=[sample_audio_input, sample_bpm, sample_key, sample_offset, sample_max_sec],
            outputs=[sample_extracted_abc, sample_analysis_display],
        )

        # Recreate Song from Sample Button
        sample_generate_btn.click(
            fn=lambda s, l, c, a, se, cfg, o, t, tp, rp, dev, off, cov: generate_music(
                s, l, c, a, se, cfg, o, t, tp, rp, "Full Song (Audio + Score)", dev, off, cov
            ),
            inputs=[
                sample_target_style,
                sample_lyrics_input,
                sample_remake_mode,
                sample_extracted_abc,
                seed_box,
                cfg_slider,
                sample_ode_steps,
                temp_slider,
                topp_slider,
                rep_slider,
                device_select,
                offload_toggle,
                sample_cover_engine,
            ],
            outputs=[
                sample_rendered_audio,
                sample_rendered_cover,
                score_display,
                reharmonize_score_input,
                sample_rendered_status,
                metadata_output,
                library_table,
                library_select,
            ],
        )

        # Lint Score Button
        lint_score_btn.click(
            fn=lint_score_action,
            inputs=[reharmonize_score_input],
            outputs=[score_lint_output],
        )

        # A/B Compare Songs Button
        run_compare_btn.click(
            fn=compare_two_songs,
            inputs=[compare_a_select, compare_b_select],
            outputs=[
                compare_audio_a,
                compare_cover_a,
                compare_score_a,
                compare_audio_b,
                compare_cover_b,
                compare_score_b,
                compare_diff_display,
            ],
        )

        # Refresh Library Button
        def refresh_library():
            new_choices = get_library_choices()
            new_rows = get_library_table_data()
            first_val = new_choices[0][1] if new_choices else None
            second_val = new_choices[1][1] if len(new_choices) > 1 else first_val
            audio, cover, score, details, _ = load_song_from_id(first_val) if first_val else (None, None, "", "No songs found.", "")
            return (
                gr.update(choices=new_choices, value=first_val),
                new_rows,
                audio,
                cover,
                score,
                details,
                gr.update(choices=new_choices, value=first_val),
                gr.update(choices=new_choices, value=second_val),
            )

        refresh_library_btn.click(
            fn=refresh_library,
            outputs=[
                library_select,
                library_table,
                library_audio,
                library_cover,
                library_score,
                library_details,
                compare_a_select,
                compare_b_select,
            ],
        )

        # Load Selected Song into Studio
        load_to_studio_btn.click(
            fn=load_selected_song_into_studio,
            inputs=[library_select],
            outputs=[style_input, lyrics_input, reharmonize_score_input],
        )

        # Delete Selected Song
        delete_song_btn.click(
            fn=delete_selected_song,
            inputs=[library_select],
            outputs=[library_audio, library_cover, library_score, library_details, library_select, library_table],
        )

        # Regenerate Cover Art Button
        regen_cover_btn.click(
            fn=regenerate_cover_art_for_song,
            inputs=[library_select, regen_cover_engine],
            outputs=[library_cover],
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
