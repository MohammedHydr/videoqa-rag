# app/streamlit_app.py

import streamlit as st
import time
import sys
import pathlib

# Add project root to Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.retrieve import retrieve

st.set_page_config(page_title="Ask‑the‑Video", page_icon="🎬", layout="wide")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar settings
st.sidebar.header("Retrieval Mode")
mode = st.sidebar.radio(
    "Choose retrieval strategy:",
    ("Fused (default)", "Semantic Only", "BM25 Only", "TF‑IDF Only"),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **Try questions like:**  \n"
    "- Who is the speaker?  \n"
    "- What is the topic?  \n"
    "- What example does he give about **Truman**?"
)

st.title("🎬 Ask‑the‑Video")

# User query input
query = st.chat_input("Type your question about the video…")
if query:
    st.chat_message("user").write(query)

    # Perform retrieval
    start = time.time()
    result = retrieve(query, mode=mode)
    latency = time.time() - start

    # Render answer
    if result is None:
        st.chat_message("assistant").error("❌ I couldn’t find that in the video.")
    else:
        ts = int(result["start"])
        end_ts = int(result["end"])
        mm, ss = divmod(ts, 60)
        timecode = f"{mm:02d}:{ss:02d}"

        with st.chat_message("assistant"):
            st.markdown(f"**@{timecode}** — {result['text']}")

            # Embed segment link if local clips not used
            clip_url = f"https://www.youtube.com/embed/dARr3lGKwk8?start={ts}&end={end_ts}&autoplay=1"
            st.video(clip_url)

    # Store chat
    st.session_state.history.append(("user", query))
    response_text = result["text"] if result else "No answer found in video."
    st.session_state.history.append(("assistant", response_text))

# Display full chat history
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)
