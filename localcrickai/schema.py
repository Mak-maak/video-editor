from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

ALLOWED_EVENTS = {"six", "four", "wicket", "bowling_spell"}


class AudioInstructions(BaseModel):
    use_external_commentary: bool = False
    use_background_music: bool = False
    ducking_enabled: bool = True
    commentary_volume: float = Field(1.0, ge=0.0, le=1.0)
    bg_music_volume: float = Field(0.25, ge=0.0, le=1.0)


class EditIntent(BaseModel):
    target_player: Optional[str] = None
    events: List[str] = Field(default_factory=list)
    over_range: Optional[Tuple[int, int]] = None
    audio_instructions: AudioInstructions = Field(default_factory=AudioInstructions)

    @field_validator("events")
    @classmethod
    def _check_events(cls, v: List[str]) -> List[str]:
        bad = [e for e in v if e not in ALLOWED_EVENTS]
        if bad:
            raise ValueError(f"unsupported events: {bad}")
        return v

    @field_validator("over_range")
    @classmethod
    def _check_range(cls, v):
        if v is not None and (v[0] < 0 or v[1] < v[0]):
            raise ValueError("over_range must be [start, end] with 0 <= start <= end")
        return v


class Segment(BaseModel):
    start: float
    end: float
    label: str = ""
