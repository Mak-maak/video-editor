"""Sequential job pipeline (one heavy model resident at a time)."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import audio, llm, render, timeline, vision


@dataclass
class Job:
    id: str
    video: str
    prompt: str
    workdir: str
    commentary: Optional[str] = None
    bg_music: Optional[str] = None
    status: str = "queued"
    progress: float = 0.0
    output: Optional[str] = None
    error: Optional[str] = None
    intent: Optional[dict] = None


# A single lock serialises GPU work across jobs.
GPU_LOCK = threading.Lock()


def _probe_duration(path: str) -> float:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def run_job(job: Job, encoder: str = "h264_nvenc") -> Job:
    def step(status: str, p: float) -> None:
        job.status, job.progress = status, p

    try:
        with GPU_LOCK:
            step("parsing_intent", 0.05)
            intent = llm.parse_intent(job.prompt, bool(job.commentary), bool(job.bg_music))
            job.intent = intent.model_dump()

            step("preprocessing", 0.15)
            wav = audio.extract_wav(job.commentary or job.video, os.path.join(job.workdir, "audio.wav"))

            step("speech_indexing", 0.3)
            transcript = audio.transcribe(wav)
            with open(os.path.join(job.workdir, "transcript.json"), "w") as f:
                json.dump(transcript, f)

            step("visual_indexing", 0.55)
            states = vision.index_scorecard(job.video)
            events = vision.detect_events(states)

            step("timeline", 0.8)
            segs = timeline.build_timeline(intent, events, transcript, _probe_duration(job.video))

            step("rendering", 0.85)
            out = os.path.join(job.workdir, "output_highlight.mp4")
            cmd = render.build_command(job.video, segs, out, intent.audio_instructions,
                                       job.commentary, job.bg_music, encoder)
            render.render(cmd)
        job.output = out
        step("done", 1.0)
    except Exception as e:  # surfaced to the UI
        job.error = str(e)
        job.status = "failed"
    return job
