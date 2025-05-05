# Ask-the-Video: Retrieval-Augmented Video Question Answering 🎬

This project implements a multimodal Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions and get answers based on the content of a YouTube video. It integrates audio (speech) and visual (frames) information to retrieve the most relevant video segments using both semantic and lexical retrieval methods.

## 🎯 Objective

To create a system that lets users ask open-ended questions about a video, and intelligently returns:

* A timestamp where the answer is mentioned.
* A specific segment of the video.
* A friendly chat interface.
* A clear “not found” message if no answer exists.

## 🚀 Pipeline Overview

### 1. **Data Preparation**

* **Video Download**: Using `yt-dlp`, we fetch the video and extract a 16kHz mono audio track with `ffmpeg`.
* **Speech-to-Text**: We transcribe the video using `faster-whisper`, generating timestamped segments.
* **Frame Extraction**: We extract keyframes every 3 seconds using `ffmpeg`, aligned with transcript segments.

### 2. **Embedding Generation**

* **Text Embeddings**: Using `BAAI/bge-large-en-v1.5` for accurate sentence-level semantic embeddings.
* **Image Embeddings**: Using `CLIP ViT-B/32` (OpenAI) for visual encoding of frames.

### 3. **Indexing and Retrieval**

* **FAISS**: Vector search engine for text and image embeddings.
* **BM25 + TF-IDF**: Lexical baselines for keyword-based retrieval.
* **Fusion**: Weighted scoring to combine similarity from text, image, and lexical matches.

### 4. **Web Application**

* **Streamlit** interface that lets users chat with the system and receive results that:

  * Embed a playable video starting from the retrieved timestamp.
  * Show the matched text.
  * Provide fallback messages when the answer is not found.

### 5. **Evaluation**

* Accuracy measured on a 10-question gold set.
* Rejection quality measured on 5 questions not present in the video.
* Latency benchmarked per query.

## 🔧 Technologies

* `Streamlit`, `FAISS`, `OpenCLIP`, `SentenceTransformers`, `BM25`, `TfidfVectorizer`, `ffmpeg`, `yt-dlp`

## 📁 Folder Structure

```bash
videoqa-rag/
├── app/                   # Streamlit interface
├── data/
│   └── raw/               # Downloaded video/audio
│   └── processed/         # Embeddings, transcripts
├── logs/                 # Slurm job logs
├── scripts/              # All preprocessing, retrieval and indexing code
│   ├── download.py
│   ├── transcribe.py
│   ├── extract_frames.py
│   ├── embed.py
│   ├── retrieve.py
│   └── common.py
└── requirements.txt
```

## ⚡ Slurm Tips

Use Slurm shell scripts to submit batch jobs on the HPC cluster (Octopus). Example:

```bash
sbatch scripts/transcribe.sh
```

Make sure to load modules like `ffmpeg`, `python/3.10`, `cuda/12.4`, and activate the right venv.

## 🧪 Gold Test Set

Located in `data/eval/gold.json`, the dataset contains 15 questions (10 answerable, 5 unanswerable) with timestamps. Evaluation results will be logged and visualized.

## 📊 Results (WIP)

| Method                | Accuracy | Rejection | Avg Latency |
| --------------------- | -------- | --------- | ----------- |
| Fused (CLIP+BGE+BM25) | 7/10     | 4/5       | 0.24s       |
| BM25                  | 4/10     | 3/5       | 0.05s       |
| Semantic Only         | 6/10     | 2/5       | 0.11s       |

