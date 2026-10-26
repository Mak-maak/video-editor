"""Scorecard OCR indexing (1 FPS sampling)."""
from __future__ import annotations

import gc
import re
from typing import Dict, List, Optional, Tuple

SCORE_RE = re.compile(r"(\d{1,3})\s*[-/]\s*(\d{1,2})")
OVER_RE = re.compile(r"(\d{1,2})\.([0-5])")

# (x, y, w, h) as fractions of the frame; broadcast-specific, override as needed.
DEFAULT_SCORE_BOX = (0.0, 0.88, 0.5, 0.12)


def parse_scorecard_text(text: str) -> Optional[Dict]:
    """Extract runs/wickets/over from OCR text like 'PAK 134-3 (15.2)'."""
    m = SCORE_RE.search(text)
    if not m:
        return None
    state = {"runs": int(m.group(1)), "wickets": int(m.group(2)), "over": None}
    rest = text[m.end():]
    o = OVER_RE.search(rest)
    if o:
        state["over"] = int(o.group(1))
    return state


def index_scorecard(video_path: str, box: Tuple[float, float, float, float] = DEFAULT_SCORE_BOX,
                    fps: float = 1.0) -> List[Dict]:
    """Return [{t, runs, wickets, over}] sampled at `fps`. Models are freed afterwards."""
    import cv2  # lazy imports
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(native / fps), 1)
    states: List[Dict] = []
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    x, y, bw, bh = box
                    crop = frame[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
                    res = ocr.ocr(crop, cls=False) or [[]]
                    text = " ".join(line[1][0] for line in (res[0] or []))
                    st = parse_scorecard_text(text)
                    if st:
                        st["t"] = idx / native
                        states.append(st)
            idx += 1
    finally:
        cap.release()
        del ocr
        gc.collect()
    return states


def detect_events(states: List[Dict]) -> List[Dict]:
    """Turn consecutive scorecard states into six/four/wicket events."""
    events: List[Dict] = []
    for a, b in zip(states, states[1:]):
        dr, dw = b["runs"] - a["runs"], b["wickets"] - a["wickets"]
        if dw == 1:
            kind = "wicket"
        elif dr == 6:
            kind = "six"
        elif dr == 4:
            kind = "four"
        else:
            continue
        events.append({"t": b["t"], "kind": kind, "over": b.get("over") if b.get("over") is not None else a.get("over")})
    return events
