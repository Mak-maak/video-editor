"""FFmpeg command construction and execution."""
from __future__ import annotations

import subprocess
from typing import List, Optional

from .schema import AudioInstructions, Segment


def build_command(video: str, segments: List[Segment], out: str,
                  audio: AudioInstructions, commentary: Optional[str] = None,
                  bg_music: Optional[str] = None, encoder: str = "h264_nvenc") -> List[str]:
    if not segments:
        raise ValueError("no segments to render")
    use_comm = audio.use_external_commentary and commentary is not None
    use_bg = audio.use_background_music and bg_music is not None

    cmd = ["ffmpeg", "-y", "-i", video]
    comm_idx = bg_idx = None
    if use_comm:
        comm_idx = 1
        cmd += ["-i", commentary]
    if use_bg:
        bg_idx = 1 + int(use_comm)
        cmd += ["-stream_loop", "-1", "-i", bg_music]

    a_src = comm_idx if use_comm else 0  # custom commentary replaces embedded audio
    parts, labels = [], ""
    for i, s in enumerate(segments):
        parts.append(f"[0:v]trim=start={s.start:.3f}:end={s.end:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[{a_src}:a]atrim=start={s.start:.3f}:end={s.end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels += f"[v{i}][a{i}]"
    n = len(segments)
    main_out = "main" if use_bg else "outa"
    parts.append(f"{labels}concat=n={n}:v=1:a=1[outv][m0]")
    parts.append(f"[m0]volume={audio.commentary_volume}[{main_out}]")

    if use_bg:
        parts.append(f"[{bg_idx}:a]volume={audio.bg_music_volume}[bg]")
        if audio.ducking_enabled:
            parts.append("[main]asplit=2[mix][sc]")
            parts.append("[bg][sc]sidechaincompress=threshold=0.02:ratio=10[duck]")
            parts.append("[mix][duck]amix=inputs=2:duration=first:normalize=0[outa]")
        else:
            parts.append("[main][bg]amix=inputs=2:duration=first:normalize=0[outa]")

    cmd += ["-filter_complex", ";".join(parts), "-map", "[outv]", "-map", "[outa]",
            "-c:v", encoder, "-preset", "fast", "-c:a", "aac", out]
    return cmd


def render(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)
