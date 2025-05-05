# app/streamlit_app.py
import streamlit as st
import time
import sys
import pathlib

# Add project root to Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.retrieve import retrieve

st.set_page_config(page_title="Ask‑the‑Video", page_icon="🎬", layout="wide")

# Chat history state
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Retrieval mode")
mode = st.sidebar.radio(
    "Choose retrieval strategy:",
    ("Fused (default)", "Semantic Only", "BM25 Only", "TF‑IDF Only"),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **Try questions like:**  \n"
    "- Who is the speaker?  \n"
    "- What example does he give about **Truman**?  \n"
    "- How long is the talk?"
)

st.title("🎬 Ask‑the‑Video")

# Input box
query = st.chat_input("Type your question about the video…")
if query:
    st.chat_message("user").write(query)

    # Perform retrieval
    start = time.time()
    result = retrieve(query, mode=mode)
    latency = time.time() - start

    if result is None:
        st.chat_message("assistant").error("❌ I couldn’t find that in the video.")
    else:
        ts = int(result["start"])
        mm, ss = divmod(ts, 60)
        timecode = f"{mm:02d}:{ss:02d}"

        with st.chat_message("assistant"):
            st.markdown(f"**@{timecode}**  ")
            st.markdown(f"{result['text']}")
            st.video(f"https://www.youtube.com/embed/dARr3lGKwk8?start={ts}")

    # Update chat history
    st.session_state.history.append(("user", query))
    response_text = result["text"] if result else "No answer"
    st.session_state.history.append(("assistant", response_text))

# Render full chat history
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)
