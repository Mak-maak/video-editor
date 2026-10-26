"""FastAPI orchestration layer. Run: uvicorn localcrickai.api:app --host 127.0.0.1"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import pipeline

app = FastAPI(title="LocalCrickAI Studio")
JOBS: Dict[str, pipeline.Job] = {}
VIDEO_EXT = {".mp4", ".mkv", ".mov"}
AUDIO_EXT = {".mp3", ".wav", ".m4a"}


def _save(up: UploadFile, workdir: str, allowed: set) -> str:
    ext = os.path.splitext(up.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    path = os.path.join(workdir, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return path


@app.post("/jobs")
def create_job(prompt: str = Form(...), video: UploadFile = File(...),
               commentary: Optional[UploadFile] = File(None),
               bg_music: Optional[UploadFile] = File(None)):
    job_id = uuid.uuid4().hex
    workdir = tempfile.mkdtemp(prefix=f"crickai_{job_id}_")
    job = pipeline.Job(
        id=job_id, prompt=prompt, workdir=workdir,
        video=_save(video, workdir, VIDEO_EXT),
        commentary=_save(commentary, workdir, AUDIO_EXT) if commentary else None,
        bg_music=_save(bg_music, workdir, AUDIO_EXT) if bg_music else None,
    )
    JOBS[job_id] = job
    threading.Thread(target=pipeline.run_job, args=(job,), daemon=True).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {"id": job.id, "status": job.status, "progress": job.progress,
            "error": job.error, "intent": job.intent}


@app.get("/jobs/{job_id}/output")
def get_output(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.output:
        raise HTTPException(404, "output not ready")
    return FileResponse(job.output, media_type="video/mp4")
