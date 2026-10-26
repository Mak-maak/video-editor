"""Intent parsing via a local Ollama server, with a deterministic fallback."""
from __future__ import annotations

import json
import re

import requests

from .schema import AudioInstructions, EditIntent

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:8b-instruct-q4_K_M"

SYSTEM = (
    "You are an expert Video Editing Assistant API. Your job is to convert user "
    "editing requests into a valid JSON schema. Do NOT return markdown commentary "
    "or extra text, ONLY return the raw JSON object."
)

TEMPLATE = """{system}

Input User Query: "{prompt}"
Secondary Audio Uploaded: {has_audio}
Background Music Uploaded: {has_bg}

Target Schema Structure:
{{
  "target_player": string or null,
  "events": array of strings (subset of ["six", "four", "wicket", "bowling_spell"]),
  "over_range": [start_over, end_over] or null,
  "audio_instructions": {{
    "use_external_commentary": boolean,
    "use_background_music": boolean,
    "ducking_enabled": boolean,
    "commentary_volume": float (0.0 to 1.0),
    "bg_music_volume": float (0.0 to 1.0)
  }}
}}
"""


def build_prompt(user_prompt: str, has_audio: bool, has_bg: bool) -> str:
    return TEMPLATE.format(
        system=SYSTEM,
        prompt=user_prompt.replace('"', "'"),
        has_audio=str(has_audio).lower(),
        has_bg=str(has_bg).lower(),
    )


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in LLM output")
    return json.loads(text[start : end + 1])


def heuristic_parse(user_prompt: str, has_audio: bool, has_bg: bool) -> EditIntent:
    p = user_prompt.lower()
    events = [e for e, pat in (("six", r"\bsix(es)?\b"), ("four", r"\bfours?\b"),
                               ("wicket", r"\bwickets?\b"),
                               ("bowling_spell", r"\bspell\b")) if re.search(pat, p)]
    over_range = None
    m = re.search(r"overs?\s*(\d+)\s*(?:-|to)\s*(\d+)", p)
    if m:
        over_range = (int(m.group(1)), int(m.group(2)))
    else:
        m = re.search(r"last\s*(\d+)\s*overs?", p)
        if m:
            # Resolved against match length by the timeline engine.
            over_range = (-int(m.group(1)), -1)
    player = None
    m = re.search(r"(?:by|of|from)\s+((?:[A-Z][a-z]+\s?){1,3})", user_prompt)
    if m:
        player = m.group(1).strip()
    return EditIntent(
        target_player=player,
        events=events,
        over_range=over_range if over_range and over_range[0] >= 0 else None,
        audio_instructions=AudioInstructions(
            use_external_commentary=has_audio,
            use_background_music=has_bg,
        ),
    )


def parse_intent(user_prompt: str, has_audio: bool = False, has_bg: bool = False,
                 timeout: float = 120.0) -> EditIntent:
    """Ask the local LLM; fall back to rules if it is unavailable or invalid."""
    body = {"model": MODEL, "prompt": build_prompt(user_prompt, has_audio, has_bg),
            "stream": False, "format": "json"}
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
        r.raise_for_status()
        intent = EditIntent.model_validate(_extract_json(r.json()["response"]))
    except Exception:
        return heuristic_parse(user_prompt, has_audio, has_bg)
    # Never let the model enable audio that was not uploaded.
    ai = intent.audio_instructions
    ai.use_external_commentary = ai.use_external_commentary and has_audio
    ai.use_background_music = ai.use_background_music and has_bg
    return intent
