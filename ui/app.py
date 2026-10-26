"""Streamlit front-end. Run: streamlit run ui/app.py"""
import time

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="LocalCrickAI Studio", layout="wide")
st.title("LocalCrickAI Studio")

video = st.file_uploader("Match video", type=["mp4", "mkv", "mov"])
commentary = st.file_uploader("Optional external commentary", type=["mp3", "wav", "m4a"])
bg = st.file_uploader("Optional background music", type=["mp3", "wav", "m4a"])
prompt = st.chat_input("e.g. Extract all sixes in the last 5 overs and add my background track")

if prompt and video:
    files = {"video": (video.name, video.getvalue())}
    if commentary:
        files["commentary"] = (commentary.name, commentary.getvalue())
    if bg:
        files["bg_music"] = (bg.name, bg.getvalue())
    with st.chat_message("user"):
        st.write(prompt)
    job = requests.post(f"{API}/jobs", data={"prompt": prompt}, files=files).json()["id"]
    bar, label = st.progress(0.0), st.empty()
    while True:
        info = requests.get(f"{API}/jobs/{job}").json()
        bar.progress(info["progress"])
        label.write(f"Status: {info['status']}")
        if info["status"] in ("done", "failed"):
            break
        time.sleep(2)
    if info["status"] == "done":
        st.json(info["intent"])
        st.video(requests.get(f"{API}/jobs/{job}/output").content)
    else:
        st.error(info["error"])
elif prompt:
    st.warning("Upload a match video first.")
