"""Timeline decision engine: cross-references OCR events and STT transcript."""
from __future__ import annotations

from typing import Dict, List

from .schema import EditIntent, Segment

PRE_ROLL = 8.0
POST_ROLL = 6.0
MERGE_GAP = 2.0


def _player_mentioned(player: str, transcript: List[Dict], t: float, window: float = 20.0) -> bool:
    names = [p.lower() for p in player.split()]
    for seg in transcript:
        if seg["end"] >= t - window and seg["start"] <= t + window:
            text = seg["text"].lower()
            if any(n in text for n in names):
                return True
    return False


def merge_segments(segs: List[Segment], gap: float = MERGE_GAP) -> List[Segment]:
    out: List[Segment] = []
    for s in sorted(segs, key=lambda s: s.start):
        if out and s.start <= out[-1].end + gap:
            out[-1].end = max(out[-1].end, s.end)
            if s.label and s.label not in out[-1].label:
                out[-1].label += f"+{s.label}"
        else:
            out.append(s.model_copy())
    return out


def build_timeline(intent: EditIntent, events: List[Dict], transcript: List[Dict],
                   duration: float) -> List[Segment]:
    wanted = set(intent.events) - {"bowling_spell"}
    max_over = max((e["over"] for e in events if e.get("over") is not None), default=None)
    segs: List[Segment] = []
    for e in events:
        if wanted and e["kind"] not in wanted:
            continue
        if intent.over_range and e.get("over") is not None:
            lo, hi = intent.over_range
            if not (lo <= e["over"] <= hi):
                continue
        elif intent.over_range and e.get("over") is None:
            continue
        if intent.target_player and not _player_mentioned(intent.target_player, transcript, e["t"]):
            continue
        segs.append(Segment(start=max(e["t"] - PRE_ROLL, 0.0),
                            end=min(e["t"] + POST_ROLL, duration), label=e["kind"]))
    return merge_segments(segs)
