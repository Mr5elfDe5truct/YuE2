"""Gradio Web UI for YuE2: Compose in symbols, create in sound."""
from __future__ import annotations

import argparse
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
from yue2.cli import doctor as cli_doctor

# Global pipeline cache
PIPELINE: YuE2Pipeline | None = None
CURRENT_CONFIG: dict = {}


def get_system_info():
    info = []
    info.append(f"**Python**: `{sys.version.split()[0]}`")
    info.append(f"**PyTorch**: `{torch.__version__}`")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        capability = torch.cuda.get_device_capability(0)
        info.append(f"**GPU**: `{gpu_name}` ({vram_gib:.1f} GiB VRAM, Compute {capability[0]}.{capability[1]})")
    else:
        info.append("**GPU**: None (CPU Mode)")
    return " | ".join(info)


def load_pipeline(device: str = "auto", offload_ar: bool = True):
    global PIPELINE, CURRENT_CONFIG
    desired_config = {"device": device, "offload_ar": offload_ar}
    if PIPELINE is not None and CURRENT_CONFIG == desired_config:
        return PIPELINE

    if PIPELINE is not None:
        PIPELINE.close()
        PIPELINE = None

    print(f"Loading YuE2Pipeline (device={device}, offload_ar={offload_ar})...")
    PIPELINE = YuE2Pipeline.from_pretrained(
        "m-a-p/YuE2-3B",
        vae="m-a-p/YuE2-Vae",
        device=device,
        offload_ar=offload_ar,
        progress=True,
    )
    CURRENT_CONFIG = desired_config
    return PIPELINE


def generate_song(
    style: str,
    lyrics: str,
    cot_mode: str,
    custom_abc: str,
    seed_input: int,
    cfg_scale: float,
    device: str,
    stage: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not style.strip():
        raise gr.Error("Please enter a style prompt.")
    if not lyrics.strip():
        raise gr.Error("Please enter lyrics.")

    seed = int(seed_input) if seed_input >= 0 else random.randint(0, 2**31 - 1)

    output_dir = Path("outputs") / f"song_{int(time.time())}_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "style": style.strip(),
        "lyrics": lyrics.strip(),
        "cot": cot_mode,
        "seed": seed,
        "cfg_scale": float(cfg_scale),
    }

    if custom_abc and custom_abc.strip():
        if cot_mode == "off":
            raise gr.Error("COT mode cannot be 'off' when providing custom ABC score.")
        kwargs["abc"] = custom_abc.strip()

    try:
        progress(0.1, desc="Loading YuE2 Pipeline...")
        pipe = load_pipeline(device=device, offload_ar=True)

        if stage == "Plan only (ABC Score)":
            progress(0.4, desc="Planning score...")
            plan = pipe.plan(**kwargs)
            plan.save(output_dir)
            score_text = plan.abc or "No score generated."
            return (
                None,  # No audio
                score_text,
                f"Planned successfully. Output saved to: {output_dir}",
                json.dumps({"plan_status": "complete", "truncated": plan.truncated, "seed": seed}, indent=2),
            )
        else:
            progress(0.3, desc="Generating song (Planning + Synthesis)...")
            result = pipe(**kwargs)
            progress(0.9, desc="Saving artifacts...")
            result.save_artifacts(output_dir)
            audio_path = str(output_dir / "audio.flac")
            score_text = result.abc or (output_dir / "score.abc").read_text(encoding="utf-8") if (output_dir / "score.abc").exists() else ""
            summary = {
                "status": "complete",
                "audio_seconds": round(len(result.audio) / result.sample_rate, 2),
                "seed": seed,
                "output_dir": str(output_dir),
                "truncated": result.truncated,
                "timing": result.timing,
            }
            return (
                audio_path,
                score_text,
                f"Generated {summary['audio_seconds']}s audio! Saved to: {output_dir}",
                json.dumps(summary, indent=2),
            )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {str(exc)}")


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


CUSTOM_CSS = """
.main-header { text-align: center; margin-bottom: 1.5rem; }
.main-header h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
.main-header p { color: #888; font-size: 1.05rem; }
.badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 4px; background: #2b2b2b; font-size: 0.85rem; }
"""


def create_ui():
    with gr.Blocks(title="YuE2 Music Studio") as app:
        with gr.Column(elem_classes="main-header"):
            gr.Markdown("# 🎵 YuE2 Music Studio")
            gr.Markdown("### Compose in symbols. Create in sound.")
            gr.Markdown(f"<div class='badge'>{get_system_info()}</div>")

        with gr.Tabs():
            with gr.TabItem("Create Song"):
                with gr.Row():
                    with gr.Column(scale=5):
                        style_input = gr.Textbox(
                            label="Style & Instruments Prompt",
                            placeholder="e.g. English, acoustic pop, female vocal, gentle piano and acoustic guitar, warm and uplifting",
                            value="English, 80s synthwave, driving electronic drums, expressive male vocal, energetic and nostalgic",
                            lines=2,
                        )
                        lyrics_input = gr.Textbox(
                            label="Lyrics",
                            placeholder="[Verse]\nWalking through the neon light...\n\n[Chorus]\nTake me back to yesterday...",
                            value="[Verse]\nWalking through the city neon glow\nMidnight shadows dancing down below\nElectric dreams are calling in the dark\n\n[Chorus]\nTake me back to yesterday\nWhere the synth lines used to play\nNever fading, never far away",
                            lines=8,
                        )

                        with gr.Row():
                            cot_choice = gr.Dropdown(
                                label="CoT Planning Mode",
                                choices=[
                                    ("Full (Melody + Harmony Plan - Recommended)", "full"),
                                    ("Melody (Melody plan only, free harmony)", "melody"),
                                    ("Off (Direct audio without score)", "off"),
                                ],
                                value="full",
                            )
                            stage_choice = gr.Dropdown(
                                label="Stage",
                                choices=["Full Song (Audio + Score)", "Plan only (ABC Score)"],
                                value="Full Song (Audio + Score)",
                            )

                        with gr.Accordion("Advanced & Score Conditioning", open=False):
                            abc_input = gr.Textbox(
                                label="Custom ABC Score (Optional - overrides auto planning)",
                                placeholder="X:1\nM:4/4\nL:1/32\nQ:1/4=88\n...",
                                lines=5,
                            )
                            with gr.Row():
                                seed_box = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                                cfg_slider = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=3.0, value=1.0, step=0.1)
                                device_choice = gr.Radio(
                                    label="Device",
                                    choices=["auto", "cuda", "cpu"],
                                    value="auto" if torch.cuda.is_available() else "cpu",
                                )

                        generate_btn = gr.Button("🚀 Generate Song", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        audio_output = gr.Audio(label="Generated Song Audio", type="filepath")
                        status_output = gr.Textbox(label="Status", interactive=False)
                        score_output = gr.Code(label="ABC Music Score", language=None, lines=8)
                        with gr.Accordion("Technical Metadata & Diagnostics", open=False):
                            metadata_output = gr.Code(label="Result JSON", language="json", lines=6)

                generate_btn.click(
                    fn=generate_song,
                    inputs=[
                        style_input,
                        lyrics_input,
                        cot_choice,
                        abc_input,
                        seed_box,
                        cfg_slider,
                        device_choice,
                        stage_choice,
                    ],
                    outputs=[audio_output, score_output, status_output, metadata_output],
                )

            with gr.TabItem("System & Hardware Diagnostics"):
                doctor_btn = gr.Button("Run Environment Doctor", variant="secondary")
                doctor_output = gr.Code(label="Doctor Report", language="json", lines=15)
                doctor_btn.click(fn=run_system_doctor, outputs=[doctor_output])

        gr.Markdown(
            "---\n*YuE2 open-source music model by M-A-P. Windows one-click package by Mr5elfDe5truct.*"
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="YuE2 Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--listen", action="store_true", help="Listen on all network interfaces (0.0.0.0)")
    args = parser.parse_args()

    app = create_ui()
    server_name = "0.0.0.0" if args.listen else "127.0.0.1"
    print(f"Launching YuE2 Web UI on http://{server_name}:{args.port}...")
    app.launch(
        server_name=server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
