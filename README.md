# LocalCrickAI Studio
Offline, AI-powered cricket highlight editor (see FDD). Pipeline: Ollama intent parsing -> faster-whisper STT -> PaddleOCR scorecard -> timeline engine -> FFmpeg render.

```
pip install -r requirements.txt   # plus faster-whisper, paddleocr, opencv-python on the GPU box
uvicorn localcrickai.api:app --host 127.0.0.1 &
streamlit run ui/app.py
pytest
```
Requires `ffmpeg` and a local Ollama with the model set in `localcrickai/llm.py`. If Ollama is unreachable, a rule-based parser is used. Use `encoder="libx264"` (or `h264_videotoolbox`) in `pipeline.run_job` when NVENC is unavailable.
