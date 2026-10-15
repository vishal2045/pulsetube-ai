# PulseTube AI (v5.0) — Full-Stack YouTube Lecture, Exam Revision & Technical Interview Summarizer

**PulseTube AI** is a full-stack web application built for university students revising video lectures before exams and developers preparing for technical interviews. Instead of watching hours of long lectures, PulseTube AI extracts spoken transcripts, fetches course metadata via the **YouTube Data API v3**, and uses **Google Gemini / Gemma 3 / Groq LLMs** to teach the actual concepts, working code examples, formulas, exam questions, and interview flashcards inside any video or 200-video playlist.

🔗 **Live Web App:** [https://pulsetube-ai.vercel.app](https://www.google.com/search?q=https%3A%2F%2Fpulsetube-ai.vercel.app)  
💻 **GitHub Repository:** [https://github.com/vishal2045/pulsetube-ai](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fvishal2045%2Fpulsetube-ai)

## Key Features

- **📚 Deep Lecture Notes & Working Code Examples**: Explains every core concept, under-the-hood mechanism, mathematical formula, and working code snippet (`Java/Spring`, `Python`, `C++`, `SQL`) step by step, paired with clickable timestamp deep links (`?v={id}&t={seconds}s`).
- **🎓 University Exam Revision Mode**: Generates last-minute exam cheat sheets, common student exam traps to avoid, and practice university exam questions with expandable full-credit model answers.
- **💼 Technical Interview Prep & Active-Recall Flashcards**: Structures topics into **30-Second Elevator Pitches**, **Technical Deep-Dives**, **Production Trade-Off Matrices**, and interactive flip flashcards with mastery tracking.
- **🗂️ 200-Video Playlist Support & Playback Speed Calculator**: Fetches full course playlists (40–200+ videos) via paginated YouTube Data API v3 calls, computes exact watch-time savings across `1x`, `1.25x`, `1.5x`, `1.75x`, and `2x` speeds, and summarizes individual lectures on demand.
- **📄 One-Click Study Guide Exports**: Exports complete lecture notes, code blocks, and Q&As to **Markdown (`.md`)** and printable **PDF (`.pdf`)** reports.

## System Architecture & Engineering Decisions

| Layer                                   | How It Works Under the Hood                                                                                                                                                                                                                                                  | Engineering Problem Solved                                                                                                                                                                           |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. YouTube Data API v3 Batching**     | Uses async `httpx` to call `playlists.list`, paginated `playlistItems.list` (50 items/page), and batched `videos.list` (50 video IDs per request) with regex ISO-8601 duration parsing (`PT1H22M42S`).                                                                       | Reduces 200 individual video metadata lookups down to just 4 batched HTTP requests (~1.2 seconds total).                                                                                             |
| **2. 3-Layer Transcript Pipeline**      | **Layer 1**: `youtube-transcript-api` (`en`, `en-IN`, `hi`). **Layer 2**: Watch-Page `captionTracks` parser (`json.JSONDecoder().raw_decode`) + Innertube API (`/youtubei/v1/player`). **Layer 3**: Syllabus/Chapter fallback (`asyncio.wait_for` 1.8s cloud timeout).       | Supports English and Hindi/Hinglish lectures natively and degrades gracefully to chapter/syllabus synthesis when cloud IPs (AWS/Vercel) are rate-limited by YouTube.                                 |
| **3. Free-Tier Multi-Model LLM Pool**   | Dynamically queries `GET /v1beta/models` (`ListModels`) and rotates across `gemini-3.8-flash` → `gemini-flash-lite-latest` → `gemma-3-27b-it` / `gemma-3-12b-it` (14,400 free req/day on the same key), plus optional **Groq** (`llama-3.3-70b-versatile`) & **OpenRouter**. | Bypasses the 20 req/day free-tier cap on `gemini-3.8-flash` by failing over to high-quota free models (`Flash-Lite` & `Gemma 3`) and caching generated summaries in memory (`_VIDEO_SUMMARY_CACHE`). |
| **4. On-Demand Playlist Summarization** | `/api/analyze` loads all playlist videos + duration stats + Lecture #1 summary immediately; `/api/summarize-video` summarizes Lectures #2–#N on click or in batches of 3.                                                                                                    | Prevents Vercel Serverless `HTTP 504` timeouts and `HTTP 429` LLM rate-limit crashes on 40+ video playlists.                                                                                         |

## Project Structure

```text
pulsetube-ai/
├── backend_server.py               # FastAPI backend & Vercel Python Serverless entrypoint
├── youtube_summarizer_webapp.html  # Single-Page Frontend UI (Tailwind CSS + Vanilla JS)
├── vercel.json                     # Vercel routing (/api/* -> backend_server.py, / -> HTML UI)
├── requirements.txt                # Python dependencies (fastapi, uvicorn, httpx, youtube-transcript-api, fpdf2)
├── README.md                       # Architecture & setup documentation
└── .gitignore                      # Excludes venv/, .env, and __pycache__/
```

## Local Setup & Running the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/vishal2045/pulsetube-ai.git
   cd pulsetube-ai
   ```

2. **Create a virtual environment and install dependencies**:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure API Keys** (in a `.env` file or via the **API & Gemini Key Config** modal in the UI):

   ```env
   YOUTUBE_API_KEY=your_youtube_data_api_v3_key
   GEMINI_API_KEY=your_google_ai_studio_key
   GROQ_API_KEY=your_optional_free_groq_key
   ```

4. **Start the FastAPI server**:
   ```bash
   uvicorn backend_server:app --reload --port 8000
   ```
   Open `http://localhost:8000` in your browser.

## Author

- **Vishal Kumar** — [GitHub (@vishal2045)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fvishal2045) • [LinkedIn](https://www.google.com/search?q=https%3A%2F%2Flinkedin.com%2Fin%2Fvishal2045ks) • `vishal2045ks@gmail.com`
