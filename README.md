# PulseTube AI (v4.0) — Full-Stack YouTube Lecture, Exam Revision & Technical Interview Summarizer

**PulseTube AI** is a full-stack web application designed for university students revising lectures before exams and developers preparing for technical interviews. Instead of re-watching hours of video lectures, PulseTube AI extracts spoken transcripts, parses playlist metadata via the **YouTube Data API v3**, and uses **Google Gemini / OpenAI** to teach the actual concepts, working code examples, formulas, exam questions, and interview flashcards inside any video or multi-video course playlist.

## Why This Project Exists (Problem & Solution)

- **The Problem**: Standard video summarizers only output vague 2-line descriptions ("In this video the speaker talks about X") and fail on large 40+ video playlists due to LLM rate limits or missing English captions on Indian/Hinglish lectures.
- **The Solution**:
  1. **Teaches What Is Actually Taught**: Generates textbook-quality concept explanations, working code examples (`Java/Spring`, `Python`, `C++`, `SQL`), mathematical formulas, worked walkthroughs, and comparison matrices.
  2. **Built for Exams & Interviews**: Includes dedicated tabs for **🎓 University Exam Revision** (cheat sheets, common exam traps, and full-credit model answers) and **💼 Technical Interview Prep** (30-second elevator pitches, deep-dive internals, production trade-offs, and interactive active-recall flashcards).
  3. **Scales to 200-Video Playlists**: Fetches up to 200 videos per playlist via paginated YouTube Data API v3 calls, computes playback speed savings (`1x` to `2x`), builds a Master Course Rollup, and summarizes individual lectures on demand (`/api/summarize-video`) to prevent `HTTP 429` API rate-limit errors.

## System Architecture & Engineering Highlights (For Future Reference & Interviews)

| Component                                       | Technical Implementation                                                                                                                                                                                                                                 | Why It Was Designed This Way                                                                                                                                                           |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Metadata & Playlist Engine**               | Python `FastAPI` + `httpx` async client calling YouTube Data API v3 (`playlists.list`, paginated `playlistItems.list`, and batched `videos.list` in chunks of 50 IDs).                                                                                   | `playlistItems.list` only returns video IDs/snippets; batching 50 IDs into `videos.list` fetches exact ISO-8601 durations (`PT1H22M42S`) for a 200-video playlist in ~1 second.        |
| **2. Multi-Layer Transcript Extractor**         | **Layer 1**: Direct YouTube Watch-Page `captionTracks` parser using `json.JSONDecoder().raw_decode` + Innertube Player API (`/youtubei/v1/player`). **Layer 2**: `youtube-transcript-api` (`en`, `en-IN`, `hi`). **Layer 3**: Syllabus/Chapter fallback. | Handles nested JSON arrays in YouTube's player response, supports Hindi/Hinglish auto-captions natively, and still teaches all chapter topics even when a video has captions disabled. |
| **3. Dynamic LLM Model Discovery**              | Calls `GET /v1beta/models` (`ListModels`) to discover active Gemini models (`gemini-2.0-flash`, `gemini-2.5-flash`), filters out `"thought": true` tokens, and sets `maxOutputTokens: 16384`.                                                            | Prevents `HTTP 404 model not found` errors when Google retires older model versions and prevents thinking tokens from truncating structured JSON output.                               |
| **4. On-Demand + Batch Playlist Summarization** | `/api/analyze` summarizes Lecture #1 + the full Course Rollup immediately; `/api/summarize-video` summarizes Lectures #2–#N on click or in batches of 3.                                                                                                 | Free-tier LLM APIs allow ~15 requests/minute; on-demand summarization loads a 42-video playlist in 5 seconds without hitting `HTTP 429` rate limits.                                   |
| **5. Export Engine**                            | Client-side + Server-side Markdown (`.md`) and PDF (`/api/export/pdf` via `fpdf2` + `@media print` CSS).                                                                                                                                                 | Lets students download portable revision notes for Notion, Obsidian, GitHub, or printing before an exam.                                                                               |

## Project Structure

```text
pulsetube-ai/
├── backend_server.py               # FastAPI backend & Vercel Python Serverless entrypoint
├── youtube_summarizer_webapp.html  # Responsive Single-Page Frontend (Tailwind CSS + Vanilla JS)
├── vercel.json                     # Vercel routing (/api/* -> backend_server.py, / -> HTML app)
├── requirements.txt                # Python backend dependencies
├── README.md                       # Project documentation & architecture guide
└── .gitignore                      # Excludes venv/, .env, and __pycache__/
```

## Local Setup Guide

1. **Clone the repository and enter the folder**:

   ```bash
   git clone https://github.com/vishal2045/pulsetube-ai.git
   cd pulsetube-ai
   ```

2. **Create a virtual environment and install dependencies**:

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys** (in a `.env` file or via the **API & Gemini Key Config** button in the web UI):

   ```env
   YOUTUBE_API_KEY=your_youtube_data_api_v3_key
   GEMINI_API_KEY=your_google_ai_studio_gemini_key
   ```

4. **Run the server**:
   ```bash
   uvicorn backend_server:app --reload --port 8000
   ```
   Open `http://localhost:8000` in your browser.

## Deploying on Vercel

1. Push this repository (including `README.md`, `vercel.json`, `backend_server.py`, `youtube_summarizer_webapp.html`, and `requirements.txt`) to GitHub.
2. Import the repository at [vercel.com/new](https://www.google.com/search?q=https%3A%2F%2Fvercel.com%2Fnew).
3. Add `YOUTUBE_API_KEY` and `GEMINI_API_KEY` under **Environment Variables** and click **Deploy**.

## Author

- **Vishal** — [GitHub (@vishal2045)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fvishal2045) • `vishal2045ks@gmail.com`
