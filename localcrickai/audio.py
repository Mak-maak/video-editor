"""Audio preprocessing, speech indexing and loudness analysis."""
from __future__ import annotations

import gc
import subprocess
from typing import Dict, List


def extract_wav(src: str, dst: str) -> str:
    """Extract 16kHz mono wav from any media file."""
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
                   check=True, capture_output=True)
    return dst


def transcribe(wav_path: str, model_size: str = "small") -> List[Dict]:
    """Run faster-whisper, then release the model so VRAM is freed."""
    from faster_whisper import WhisperModel  # lazy: optional heavy dependency

    model = WhisperModel(model_size, device="auto", compute_type="int8_float16")
    try:
        segments, _ = model.transcribe(wav_path, word_timestamps=True)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    finally:
        del model
        gc.collect()


def rms_spikes(wav_path: str, window_s: float = 1.0, z: float = 2.0) -> List[float]:
    """Return timestamps (s) of windows whose RMS exceeds mean + z*std (crowd roars)."""
    import array
    import math
    import wave

    with wave.open(wav_path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit wav")
        win = int(rate * window_s)
        rms: List[float] = []
        while True:
            raw = w.readframes(win)
            if not raw:
                break
            s = array.array("h", raw)
            rms.append(math.sqrt(sum(x * x for x in s) / len(s)))
    if not rms:
        return []
    mean = sum(rms) / len(rms)
    std = math.sqrt(sum((x - mean) ** 2 for x in rms) / len(rms))
    return [i * window_s for i, x in enumerate(rms) if std and x > mean + z * std]
