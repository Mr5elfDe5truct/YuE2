"""
YuE2 Smart Album Cover Art Studio
---------------------------------
Generates customized, visually congruent album cover artwork for generated songs
based on musical style, genre tags, and lyrical themes.

Supports 3 generation engines:
1. Cloud AI Diffusion (Serverless Flux/SDXL quality, 0 MB VRAM, ~3s)
2. Procedural Graphic Studio (Algorithmic Vinyl Sleeve via Pillow, 100% offline, 0 MB VRAM, instant)
3. Local AI Diffusion (SD-Turbo via diffusers on CUDA, sequenced after audio offload)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Genre-to-visual aesthetic mapping dictionary
GENRE_VISUAL_MAP = {
    "cyberpunk": "futuristic cyberpunk neon cityscape, holographic glow, rainy asphalt reflections, synth aesthetic, octane render, 8k wallpaper",
    "synthwave": "retro 80s synthwave, chrome grid, neon sunset over horizon, purple and magenta palm trees, retrowave aesthetic, highly detailed",
    "vaporwave": "vaporwave aesthetic, classical statue bust, pastel gradients, lo-fi aesthetic, nostalgic 90s anime backdrop, dreamy surrealism",
    "metal": "dark fantasy, dramatic crimson and charcoal lighting, intricate gothic motifs, heavy metal album art style, gritty cinematic atmosphere",
    "rock": "vintage electric guitar, amplifiers, dynamic stage smoke, warm amber spotlight, analog film grain, classic rock poster art",
    "pop": "vibrant colorful pop art, energetic bold gradients, hyper-stylized modern studio lighting, clean elegant graphic design, commercial cover art",
    "hip hop": "urban skyline, golden hour rim light, bold street art graffiti elements, gritty asphalt texture, contemporary album cover typography aesthetic",
    "trap": "dark trap aesthetic, neon purple ambient glow, smoke plumes, deep shadows, cinematic minimalism, 3D chrome details",
    "r&b": "smooth moody midnight atmosphere, velvety deep blue and violet tones, soft bokeh reflections, sultry soul album aesthetic",
    "jazz": "smoky vintage jazz lounge, warm brass trumpet silhouette, moody sepia lighting, classic Blue Note record jacket design, analog texture",
    "classical": "grand symphony concert hall, dramatic gilded architectural lighting, ethereal sheet music vortex, dark baroque oil painting style",
    "orchestral": "epic cinematic orchestral atmosphere, sweeping cosmic vistas, gold and obsidian color palette, Hans Zimmer soundtrack artwork vibe",
    "folk": "misty rustic woodland, acoustic guitar, autumn leaves, warm lantern glow, cozy nostalgic watercolor painting aesthetic",
    "country": "dusty desert highway at sunset, lone weathered porch, warm amber horizon, vintage Americana album sleeve",
    "ambient": "ethereal ambient soundscape, floating geometric crystal monoliths, celestial nebula aurora, vast meditative horizon, minimalist elegance",
    "lo-fi": "cozy bedroom studio at midnight, rain streaks on windowpane, warm desk lamp, lo-fi anime chill aesthetic, soft pastel tones",
    "electronic": "pulsing digital soundwaves, abstract iridescent fractal geometry, glowing fiber-optic streams, high-tech electronic music visual",
    "edm": "massive festival lasers piercing dark clouds, euphoric confetti explosion, electric cyan and ultraviolet energy, hyper-detailed",
    "phonk": "drift phonk aesthetic, vintage Japanese sports car in rainy Tokyo backstreet, neon purple underglow, gritty VHS distortion filter",
    "piano": "grand piano keys reflecting starlight, minimalist dark studio, subtle water ripples, emotive cinematic poster"
}

MOOD_VISUAL_MAP = {
    "melancholic": "somber desaturated tones, gentle raindrops, poignant quiet atmosphere",
    "sad": "poignant deep blue shadows, solitude, soft cinematic rim light",
    "energetic": "explosive dynamic composition, high contrast, vibrant vivid sparks",
    "dark": "ominous shadows, deep black and crimson contrast, mysterious silhouettes",
    "romantic": "soft warm rosy glow, gentle bokeh, dreamy impressionistic brushwork",
    "uplifting": "golden sunrise rays breaking through clouds, bright optimistic palette",
    "epic": "monumental scale, dramatic volumetric lighting, cinematic wide-angle majesty",
    "chill": "relaxed pastel twilight, calm soothing reflections, peaceful minimalism",
    "intense": "high voltage tension, stark dramatic lighting, intense hyper-focus",
    "nostalgic": "faded warm 35mm film grain, Polaroid color shift, bittersweet aesthetic"
}


def build_cover_prompt(style: str, lyrics: str = "", title: str = "YuE2 Track") -> str:
    """
    Synthesize an evocative, visually congruent text-to-image prompt from music style tags
    and lyrical themes.
    """
    style_lower = style.lower()
    lyrics_lower = lyrics.lower()

    visual_elements = []

    # 1. Detect matching genre aesthetics
    matched_genres = []
    for genre, aesthetic in GENRE_VISUAL_MAP.items():
        if genre in style_lower:
            matched_genres.append(aesthetic)
            if len(matched_genres) >= 2:
                break

    if matched_genres:
        visual_elements.append(", ".join(matched_genres))
    else:
        # Fallback to style string keywords if no explicit genre match
        clean_style = re.sub(r"[^\w\s,]", "", style).strip()
        if clean_style:
            visual_elements.append(f"album cover art inspired by {clean_style}")
        else:
            visual_elements.append("stunning modern vinyl album cover art, minimalist modern graphic design")

    # 2. Detect mood aesthetics
    matched_moods = []
    for mood, mood_desc in MOOD_VISUAL_MAP.items():
        if mood in style_lower or mood in lyrics_lower:
            matched_moods.append(mood_desc)
            if len(matched_moods) >= 2:
                break
    if matched_moods:
        visual_elements.append(", ".join(matched_moods))

    prompt_core = ", ".join(visual_elements)
    full_prompt = (
        f"Professional album cover art, vinyl sleeve jacket, square format. "
        f"{prompt_core}. "
        f"High quality graphic design, award-winning concept art, 8k resolution, cinematic lighting, masterpiece."
    )
    return full_prompt


def generate_cloud_cover(prompt: str, output_path: str | Path, seed: int = 42) -> Path:
    """
    Serverless Cloud AI Diffusion.
    Uses high-speed serverless pollinations endpoint (Flux/SDXL quality, 0 MB VRAM).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_prompt = urllib.parse.quote(prompt[:400])
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=800&nologo=true&seed={seed}&model=flux"

    headers = {
        "User-Agent": "YuE2-Music-Studio/1.1 (Windows NT 10.0; Win64; x64)"
    }
    req = urllib.request.Request(url, headers=headers)
    
    # 20 second timeout
    with urllib.request.urlopen(req, timeout=20.0) as response:
        content = response.read()
        if len(content) < 1000:
            raise ValueError(f"Response too small ({len(content)} bytes), likely an error image.")
        with open(output_path, "wb") as f:
            f.write(content)

    return output_path


def generate_procedural_cover(
    style: str,
    lyrics: str = "",
    title: str = "YuE2 Studio",
    output_path: str | Path = "cover.png",
    seed: int = 42
) -> Path:
    """
    Procedural Graphic Studio:
    Renders an 800x800 modern vinyl record sleeve cover using Pillow.
    Features:
    - Dynamic gradient meshes based on genre color palette
    - Sleek vinyl record grooves and soundwave concentric geometry
    - Stylized typographic layout with track title and genre badge
    - 100% offline, 0 MB VRAM, instant (<50ms)
    """
    from PIL import Image, ImageDraw, ImageFilter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 800, 800
    
    # Determine color palette based on style
    s_low = style.lower()
    if any(k in s_low for k in ["cyber", "synth", "retro", "electronic", "electro"]):
        # Cyberpunk / Synthwave: deep purple to electric cyan / neon magenta
        c_top = (15, 8, 38)
        c_mid = (88, 28, 135)
        c_accent = (0, 242, 254)
        c_secondary = (255, 0, 128)
    elif any(k in s_low for k in ["metal", "rock", "dark", "industrial", "punk"]):
        # Dark Crimson & Charcoal
        c_top = (18, 18, 20)
        c_mid = (45, 12, 18)
        c_accent = (225, 29, 72)
        c_secondary = (245, 158, 11)
    elif any(k in s_low for k in ["lo-fi", "chill", "ambient", "acoustic", "folk", "indie"]):
        # Warm twilight / Emerald & Gold
        c_top = (13, 30, 28)
        c_mid = (22, 60, 52)
        c_accent = (16, 185, 129)
        c_secondary = (245, 158, 11)
    elif any(k in s_low for k in ["pop", "dance", "r&b", "soul"]):
        # Sunset violet / Coral
        c_top = (24, 12, 45)
        c_mid = (76, 29, 149)
        c_accent = (236, 72, 153)
        c_secondary = (251, 146, 60)
    else:
        # Midnight Blue / Sapphire
        c_top = (10, 15, 30)
        c_mid = (30, 58, 138)
        c_accent = (56, 189, 248)
        c_secondary = (129, 140, 248)

    # 1. Base Gradient Canvas
    img = Image.new("RGB", (width, height), c_top)
    draw = ImageDraw.Draw(img)

    for y in range(height):
        t = y / height
        # Smooth cubic gradient transition
        r = int(c_top[0] * (1 - t) + c_mid[0] * t)
        g = int(c_top[1] * (1 - t) + c_mid[1] * t)
        b = int(c_top[2] * (1 - t) + c_mid[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 2. Draw Geometric Soundwave / Celestial Circles
    cx, cy = width // 2, height // 2 - 40
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Concentric vinyl rings
    for radius in range(90, 310, 25):
        alpha = int(40 + 60 * math.sin(radius * 0.05 + seed))
        ring_color = (*c_accent[:3], max(20, min(140, alpha)))
        overlay_draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=ring_color,
            width=2 if radius % 50 != 0 else 3
        )

    # Inner vinyl record core disc
    core_r = 85
    overlay_draw.ellipse(
        [cx - core_r, cy - core_r, cx + core_r, cy + core_r],
        fill=(10, 10, 16, 220),
        outline=(*c_secondary[:3], 200),
        width=3
    )

    # Spindle hole
    spindle_r = 16
    overlay_draw.ellipse(
        [cx - spindle_r, cy - spindle_r, cx + spindle_r, cy + spindle_r],
        fill=(0, 0, 0, 255),
        outline=(*c_accent[:3], 240),
        width=2
    )

    # Radial frequency bars
    num_bars = 48
    for i in range(num_bars):
        angle = (2 * math.pi / num_bars) * i
        # Procedural bar height based on hash
        h_val = 30 + (hash(f"{style}_{seed}_{i}") % 80)
        x1 = cx + int((core_r + 8) * math.cos(angle))
        y1 = cy + int((core_r + 8) * math.sin(angle))
        x2 = cx + int((core_r + 8 + h_val) * math.cos(angle))
        y2 = cy + int((core_r + 8 + h_val) * math.sin(angle))
        bar_alpha = int(90 + 130 * (i / num_bars))
        overlay_draw.line([(x1, y1), (x2, y2)], fill=(*c_accent[:3], bar_alpha), width=2)

    # Blend overlay onto background
    img.paste(overlay, (0, 0), overlay)

    # 3. Typography Layout
    draw = ImageDraw.Draw(img)

    # Header Bar: YuE2 Hi-Fi Studio Badge
    draw.rectangle([50, 48, 210, 80], fill=(0, 0, 0), outline=c_accent, width=1)
    draw.text((64, 56), "YuE2 MASTER HI-FI", fill=c_accent)

    # Top Right: Stereo / 44.1kHz badge
    draw.text((width - 170, 56), "STEREO · 44.1 kHz", fill=(180, 190, 210))

    # Footer Card Backdrop
    draw.rectangle([40, height - 190, width - 40, height - 45], fill=(8, 12, 22))
    draw.rectangle([40, height - 190, width - 40, height - 45], outline=(*c_accent[:3],), width=1)

    # Clean Song Title
    display_title = (title or "ORIGINAL COMPOSITION").upper()[:28]
    draw.text((60, height - 175), display_title, fill=(255, 255, 255))

    # Clean Subtitle (Style / Genres)
    clean_style = style.replace("\n", " ").strip()
    if len(clean_style) > 42:
        clean_style = clean_style[:39] + "..."
    draw.text((60, height - 128), clean_style.upper(), fill=c_accent)

    # Lyric Snippet quote
    lines = [l.strip() for l in lyrics.split("\n") if l.strip() and not l.strip().startswith("[")]
    lyric_quote = f'"{lines[0][:50]}..."' if lines else "AI Neural Acoustic Synthesizer"
    draw.text((60, height - 90), lyric_quote, fill=(160, 175, 195))

    # Footer Audio Spectrum accent line
    for bar_x in range(width - 180, width - 60, 8):
        h = 10 + (hash(f"{bar_x}_{seed}") % 24)
        draw.line([(bar_x, height - 85), (bar_x, height - 85 - h)], fill=c_secondary, width=4)

    # Save final image
    img.save(output_path, format="PNG", optimize=True)
    return output_path


_LOCAL_DIFFUSER_PIPE = None

def generate_local_ai_cover(
    prompt: str,
    output_path: str | Path,
    seed: int = 42,
    model_id: str = "stabilityai/sd-turbo"
) -> Path:
    """
    Local AI Diffusion:
    Runs SD-Turbo on CUDA using diffusers.
    Requires ~2.2 GB VRAM in float16.
    Designed to be called after YuE2 audio pipeline offloads to CPU.
    """
    global _LOCAL_DIFFUSER_PIPE
    import torch
    from diffusers import AutoPipelineForText2Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if _LOCAL_DIFFUSER_PIPE is None:
        logger.info(f"Loading local image diffusion model {model_id} onto {device}...")
        _LOCAL_DIFFUSER_PIPE = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None
        )
        _LOCAL_DIFFUSER_PIPE.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    # SD-Turbo generates crisp images in 1-2 steps
    image = _LOCAL_DIFFUSER_PIPE(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=generator,
        width=512,
        height=512
    ).images[0]

    image.save(output_path, format="PNG")
    return output_path


def create_cover_art(
    style: str,
    lyrics: str = "",
    engine: str = "cloud",
    output_dir: str | Path = "outputs",
    title: str = "YuE2 Track",
    seed: int = 42
) -> Optional[Path]:
    """
    Unified high-level dispatcher for cover art generation.

    Parameters:
    - style: Song genre / style prompt
    - lyrics: Lyrics content
    - engine: 'cloud' | 'procedural' | 'local' | 'none'
    - output_dir: Directory where cover.png should be written
    - title: Track title or folder name
    - seed: Integer seed

    Returns:
    Path to cover.png or None if disabled or failed.
    """
    engine = (engine or "cloud").lower()
    if engine in ("none", "disabled", "off"):
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cover_path = out_dir / "cover.png"

    prompt = build_cover_prompt(style=style, lyrics=lyrics, title=title)

    # 1. Procedural Engine
    if "procedural" in engine:
        try:
            return generate_procedural_cover(
                style=style,
                lyrics=lyrics,
                title=title,
                output_path=cover_path,
                seed=seed
            )
        except Exception as e:
            logger.error(f"Procedural cover art generation failed: {e}", exc_info=True)
            return None

    # 2. Local AI Diffusion
    elif "local" in engine:
        try:
            return generate_local_ai_cover(
                prompt=prompt,
                output_path=cover_path,
                seed=seed
            )
        except Exception as e:
            logger.warning(f"Local AI diffusion failed ({e}), falling back to Procedural Graphic Studio...")
            return generate_procedural_cover(
                style=style,
                lyrics=lyrics,
                title=title,
                output_path=cover_path,
                seed=seed
            )

    # 3. Cloud AI Diffusion (Default)
    else:
        try:
            return generate_cloud_cover(
                prompt=prompt,
                output_path=cover_path,
                seed=seed
            )
        except Exception as e:
            logger.warning(f"Cloud AI cover art request failed or timed out ({e}). Falling back to Procedural Graphic Studio...")
            # Seamless fallback to procedural so user always gets a stunning cover!
            return generate_procedural_cover(
                style=style,
                lyrics=lyrics,
                title=title,
                output_path=cover_path,
                seed=seed
            )
