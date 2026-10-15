#!/usr/bin/env python3
"""
PulseTube AI (v4.0) — Full-Stack YouTube Lecture, Code, Exam & Interview Summarizer
===================================================================================
Major Fixes in v4.0:
1. Dynamic Gemini `ListModels` Discovery (Fixes `models/gemini-1.5-flash HTTP 404`):
   - Automatically queries `GET https://generativelanguage.googleapis.com/v1beta/models?key=...`
     to discover the exact models enabled for your API key that support `generateContent`
     (such as `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-flash-latest`).
   - Properly handles Gemini 2.5 thinking parts (`thought: true`) and sets `maxOutputTokens: 16384`
     so JSON output never truncates.
2. Deep Concept, Code & Example Teaching (Even When Captions Are Missing or in Hindi/Hinglish!):
   - Extracts English (`en`, `en-IN`, `en-US`) and Hindi (`hi` auto-captions) directly without
     breaking when YouTube blocks `&tlang=en`.
   - Even if a video has 0 captions on YouTube, Gemini uses the video title, description, and
     all chapter timestamps (e.g. `00:00 Intro`, `15:40 Circular Dependency in Core Java`,
     `36:15 Bean Scopes`, `1:03:09 @Lazy Annotation`) to thoroughly teach every concept, provide
     real working code examples (`code_example`), explain internal mechanisms, and generate
     exam & interview Q&As!
3. Fixed Chapter Title Parenthesis Bug:
   - Preserves balanced parentheses in chapter titles like `Bean Scopes (Singleton & Prototype)`.
"""

import asyncio
import html
import io
import json
import math
import os
import pathlib
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_TRANSCRIPT_API = True
except ImportError:
    HAS_TRANSCRIPT_API = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

load_dotenv()
load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="PulseTube AI v4.0 — Deep Concept, Code, Exam & Interview Summarizer API",
    version="4.0.0",
    description="Full-stack backend with dynamic Gemini ListModels discovery, Hindi/English caption extraction, and code-grounded lecture teaching.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist URL")
    mode: str = Field("auto", description="'auto', 'video', or 'playlist'")
    study_goal: str = Field("master", description="'master', 'exam', or 'interview'")
    youtube_api_key: Optional[str] = Field(None, description="Optional override for YOUTUBE_API_KEY")
    llm_provider: str = Field("gemini", description="'gemini' or 'openai'")
    llm_api_key: Optional[str] = Field(None, description="Optional override for GEMINI_API_KEY / OPENAI_API_KEY")
    llm_model: Optional[str] = Field(None, description="Optional model override")
    max_playlist_videos: int = Field(200, ge=1, le=250, description="Max videos to fetch in a playlist")
    initial_deep_summarize_count: int = Field(1, ge=1, le=10, description="Videos to summarize immediately on playlist load")
    summary_depth: str = Field("comprehensive", description="'concise', 'detailed', or 'comprehensive'")


class SummarizeSingleVideoRequest(BaseModel):
    video: Dict[str, Any] = Field(..., description="Video metadata object")
    study_goal: str = Field("master", description="'master', 'exam', or 'interview'")
    llm_provider: str = Field("gemini", description="'gemini' or 'openai'")
    llm_api_key: Optional[str] = Field(None, description="Optional override for GEMINI_API_KEY / OPENAI_API_KEY")
    llm_model: Optional[str] = Field(None, description="Optional model override")
    summary_depth: str = Field("comprehensive", description="'concise', 'detailed', or 'comprehensive'")


class DurationCalcRequest(BaseModel):
    durations_seconds: List[int] = Field(..., description="List of video durations in seconds")
    start_index: int = Field(1, ge=1, description="1-based start video index")
    end_index: Optional[int] = Field(None, description="1-based inclusive end video index")


class ExportRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Analyzed video or playlist payload to export")


# ---------------------------------------------------------------------------
# URL, Duration & Text Utilities
# ---------------------------------------------------------------------------

ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

DESCRIPTION_TIMESTAMP_RE = re.compile(
    r"(?:^|\n)\s*(?:\[|\()?(\d{1,2}:\d{2}(?::\d{2})?)(?:\]|\))?\s*[-–—:•|]?\s*([^\n\r]{2,110})"
)


def clean_api_key(raw_key: Optional[str]) -> str:
    if not raw_key:
        return ""
    return raw_key.strip().strip('"').strip("'").strip()


def parse_youtube_url(url: str) -> Dict[str, Optional[str]]:
    cleaned = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", cleaned):
        return {"video_id": cleaned, "playlist_id": None, "detected_type": "video"}
    if re.fullmatch(r"(?:PL|UU|LL|FL|OLAK5uy_|RD)[A-Za-z0-9_-]{8,}", cleaned):
        return {"video_id": None, "playlist_id": cleaned, "detected_type": "playlist"}

    parsed = urlparse(cleaned)
    query = parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    playlist_id = query.get("list", [None])[0]

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if "youtu.be" in host:
        slug = path.lstrip("/").split("/")[0]
        if slug:
            video_id = slug
    elif "/shorts/" in path or "/embed/" in path or "/live/" in path:
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            video_id = parts[1]

    if path.startswith("/playlist") or (playlist_id and not video_id):
        detected_type = "playlist"
    elif video_id:
        detected_type = "video"
    elif playlist_id:
        detected_type = "playlist"
    else:
        detected_type = "unknown"

    return {
        "video_id": video_id,
        "playlist_id": playlist_id,
        "detected_type": detected_type,
    }


def parse_iso8601_duration(iso_str: str) -> int:
    if not iso_str:
        return 0
    match = ISO8601_DURATION_RE.match(iso_str.strip())
    if not match:
        return 0
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_seconds_hms(total_seconds: int) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_seconds_human(total_seconds: int) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def timestamp_str_to_seconds(ts_str: str) -> int:
    parts = [int(p) for p in ts_str.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def clean_description_text(description: str) -> str:
    if not description:
        return ""
    cleaned_lines = []
    for line in description.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if re.search(r"https?://|www\.|\b(?:subscribe|instagram|twitter|linkedin|discord|telegram|patreon|sponsor|coupon|discount|merch)\b", ln, re.I):
            continue
        if re.match(r"^(?:\[|\()?\d{1,2}:\d{2}", ln):
            continue
        if ln.startswith("#") or re.match(r"^[-=_*•]{3,}$", ln):
            continue
        cleaned_lines.append(ln)
    return " ".join(cleaned_lines)


def split_into_clean_sentences(text: str, max_words_per_chunk: int = 24) -> List[str]:
    if not text:
        return []
    cleaned = re.sub(r"\[(?:Music|Applause|Laughter)\]", "", text, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 12]

    if len(raw_sentences) <= 1 and len(cleaned.split()) > max_words_per_chunk:
        words = cleaned.split()
        chunks = []
        for i in range(0, len(words), max_words_per_chunk):
            piece = " ".join(words[i : i + max_words_per_chunk]).strip(" ,;-")
            if piece:
                if piece[-1] not in ".!?":
                    piece += "."
                chunks.append(piece[0].upper() + piece[1:])
        return chunks

    final_sentences = []
    for s in raw_sentences:
        words = s.split()
        if len(words) > 40:
            short_s = " ".join(words[:34]).rstrip(" ,;-") + "."
            final_sentences.append(short_s[0].upper() + short_s[1:])
        else:
            final_sentences.append(s[0].upper() + s[1:])
    return final_sentences


def calculate_playlist_durations(
    durations_seconds: List[int],
    start_index: int = 1,
    end_index: Optional[int] = None,
) -> Dict[str, Any]:
    if not durations_seconds:
        return {
            "video_count": 0,
            "total_seconds": 0,
            "total_formatted": "0m 0s",
            "average_seconds": 0,
            "average_formatted": "0m 0s",
            "speeds": {},
        }

    n = len(durations_seconds)
    s_idx = max(1, min(start_index, n)) - 1
    e_idx = max(s_idx + 1, min(end_index or n, n))
    subset = durations_seconds[s_idx:e_idx]

    total_sec = sum(subset)
    count = len(subset)
    avg_sec = int(round(total_sec / count)) if count else 0

    speeds = {}
    for sp in [1.0, 1.25, 1.5, 1.75, 2.0]:
        adj_sec = int(round(total_sec / sp))
        saved_sec = max(0, total_sec - adj_sec)
        clean_key = f"{sp:g}x"
        speeds[clean_key] = {
            "speed": sp,
            "watch_seconds": adj_sec,
            "watch_formatted": format_seconds_human(adj_sec),
            "watch_hms": format_seconds_hms(adj_sec),
            "saved_seconds": saved_sec,
            "saved_formatted": format_seconds_human(saved_sec),
        }

    est_reading_seconds = min(total_sec, count * 120 + 60)

    return {
        "video_count": count,
        "range": {"start_index": s_idx + 1, "end_index": e_idx, "total_available": n},
        "total_seconds": total_sec,
        "total_formatted": format_seconds_human(total_sec),
        "total_hms": format_seconds_hms(total_sec),
        "average_seconds": avg_sec,
        "average_formatted": format_seconds_human(avg_sec),
        "shortest_seconds": min(subset) if subset else 0,
        "shortest_formatted": format_seconds_human(min(subset)) if subset else "0s",
        "longest_seconds": max(subset) if subset else 0,
        "longest_formatted": format_seconds_human(max(subset)) if subset else "0s",
        "estimated_summary_reading_seconds": est_reading_seconds,
        "estimated_summary_reading_formatted": format_seconds_human(est_reading_seconds),
        "time_saved_by_reading_formatted": format_seconds_human(max(0, total_sec - est_reading_seconds)),
        "speeds": speeds,
    }


# ---------------------------------------------------------------------------
# Multi-Language Transcript Extraction (English + Hindi/Hinglish Support)
# ---------------------------------------------------------------------------

def parse_description_chapters(description: str, total_duration: int) -> List[Dict[str, Any]]:
    """Extracts creator timestamps from description while preserving balanced parentheses like `(Singleton & Prototype)`."""
    if not description:
        return []
    matches = DESCRIPTION_TIMESTAMP_RE.findall(description)
    raw_chapters = []
    seen_starts = set()
    for ts_str, title in matches:
        sec = timestamp_str_to_seconds(ts_str)
        clean_title = re.sub(r"https?://\S+", "", title).strip(" -–—:•|[]")
        # Only strip outer parentheses if the entire title is wrapped in ( ... )
        if clean_title.startswith("(") and clean_title.endswith(")") and "(" not in clean_title[1:-1]:
            clean_title = clean_title[1:-1].strip()
        if len(clean_title) >= 2 and sec not in seen_starts and (total_duration == 0 or sec < total_duration):
            seen_starts.add(sec)
            raw_chapters.append({"start_seconds": sec, "timestamp": format_seconds_hms(sec), "title": clean_title})

    raw_chapters.sort(key=lambda x: x["start_seconds"])
    if len(raw_chapters) < 2:
        return []

    for i, ch in enumerate(raw_chapters):
        if i + 1 < len(raw_chapters):
            ch["end_seconds"] = raw_chapters[i + 1]["start_seconds"]
        else:
            ch["end_seconds"] = total_duration if total_duration > ch["start_seconds"] else ch["start_seconds"] + 600
    return raw_chapters


def extract_caption_tracks_from_html(html_text: str) -> List[Dict[str, Any]]:
    marker = '"captionTracks":'
    idx = html_text.find(marker)
    if idx == -1:
        return []
    arr_start = html_text.find("[", idx + len(marker))
    if arr_start == -1:
        return []
    try:
        decoder = json.JSONDecoder()
        tracks, _ = decoder.raw_decode(html_text, arr_start)
        if isinstance(tracks, list):
            return tracks
    except Exception:
        pass
    return []


def _parse_caption_payload(raw_body: str) -> List[Dict[str, Any]]:
    raw_body = (raw_body or "").strip()
    if not raw_body:
        return []

    if raw_body.startswith("{"):
        try:
            data = json.loads(raw_body)
            events = data.get("events", [])
            items = []
            for ev in events:
                segs = ev.get("segs")
                if not segs:
                    continue
                text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
                if not text:
                    continue
                start_ms = float(ev.get("tStartMs", 0))
                dur_ms = float(ev.get("dDurationMs", 2000))
                items.append({
                    "text": html.unescape(text),
                    "start": round(start_ms / 1000.0, 2),
                    "duration": round(dur_ms / 1000.0, 2),
                })
            if items:
                return items
        except Exception:
            pass

    try:
        root = ET.fromstring(raw_body)
        items = []
        for elem in root.iter():
            if elem.tag == "text":
                txt = "".join(elem.itertext()).replace("\n", " ").strip()
                if txt:
                    items.append({
                        "text": html.unescape(txt),
                        "start": float(elem.attrib.get("start", 0.0)),
                        "duration": float(elem.attrib.get("dur", 2.0)),
                    })
            elif elem.tag == "p":
                txt = "".join(elem.itertext()).replace("\n", " ").strip()
                if txt and ("t" in elem.attrib or "start" in elem.attrib):
                    start_val = float(elem.attrib.get("t", elem.attrib.get("start", 0.0)))
                    dur_val = float(elem.attrib.get("d", elem.attrib.get("dur", 2000.0)))
                    if "t" in elem.attrib:
                        start_val /= 1000.0
                        dur_val /= 1000.0
                    items.append({
                        "text": html.unescape(txt),
                        "start": round(start_val, 2),
                        "duration": round(dur_val, 2),
                    })
        return items
    except Exception:
        return []


def _select_candidate_caption_urls(tracks: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Returns ordered candidate `(base_url, label)` pairs to try.
    Includes raw track URLs (English, Hindi, etc.) BEFORE `&tlang=en` so that if YouTube
    blocks automated `&tlang=en` translation, we still get the full Hindi/English transcript
    (which Gemini translates & summarizes into English natively!).
    """
    if not tracks:
        return []

    preferred_prefixes = ["en", "en-IN", "en-US", "en-GB", "hi"]
    ordered_tracks = []
    seen_urls = set()

    for prefix in preferred_prefixes:
        for tr in tracks:
            code = str(tr.get("languageCode", ""))
            if code.lower().startswith(prefix.lower()) and tr.get("baseUrl") not in seen_urls:
                seen_urls.add(tr.get("baseUrl"))
                ordered_tracks.append(tr)

    for tr in tracks:
        if tr.get("baseUrl") not in seen_urls:
            seen_urls.add(tr.get("baseUrl"))
            ordered_tracks.append(tr)

    candidates: List[Tuple[str, str]] = []
    for tr in ordered_tracks[:4]:
        base_url = tr.get("baseUrl", "")
        if not base_url:
            continue
        lang_code = str(tr.get("languageCode", "en"))
        name_obj = tr.get("name", {})
        lang_label = (
            name_obj.get("simpleText")
            or (name_obj.get("runs", [{}])[0].get("text") if isinstance(name_obj.get("runs"), list) else None)
            or lang_code
        )
        # Try direct caption track first (works reliably for both English and Hindi!)
        candidates.append((base_url, str(lang_label)))
        candidates.append((base_url + "&fmt=json3", str(lang_label)))
        if not lang_code.lower().startswith("en") and tr.get("isTranslatable"):
            candidates.append((base_url + "&tlang=en", f"{lang_label} (Translated to EN)"))

    return candidates


async def fetch_transcript_via_innertube(video_id: str) -> Tuple[List[Dict[str, Any]], str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    proxy_url = clean_api_key(os.getenv("YOUTUBE_PROXY_URL") or os.getenv("PROXY_URL")) or None

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers, proxy=proxy_url) as client:
        innertube_key = None
        try:
            watch_resp = await client.get(
                f"https://www.youtube.com/watch?v={video_id}",
                cookies={"CONSENT": "YES+cb"},
            )
            if watch_resp.status_code == 200:
                html_text = watch_resp.text
                key_match = re.search(r'"INNERTUBE_API_KEY":\s*"([^"]+)"', html_text)
                if key_match:
                    innertube_key = key_match.group(1)

                tracks = extract_caption_tracks_from_html(html_text)
                for url_cand, label in _select_candidate_caption_urls(tracks):
                    try:
                        cap_resp = await client.get(url_cand)
                        if cap_resp.status_code == 200 and cap_resp.text.strip():
                            parsed_items = _parse_caption_payload(cap_resp.text)
                            if parsed_items:
                                return parsed_items, label
                    except Exception:
                        continue
        except Exception:
            pass

        innertube_clients = [
            {"clientName": "ANDROID", "clientVersion": "19.09.37", "androidSdkVersion": 30, "hl": "en", "gl": "IN"},
            {"clientName": "IOS", "clientVersion": "19.09.3", "deviceModel": "iPhone14,3", "hl": "en", "gl": "IN"},
            {"clientName": "WEB", "clientVersion": "2.20240726.00.00", "hl": "en", "gl": "IN"},
        ]
        player_url = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
        if innertube_key:
            player_url += f"&key={innertube_key}"

        for ctx_client in innertube_clients:
            try:
                resp = await client.post(
                    player_url,
                    json={"context": {"client": ctx_client}, "videoId": video_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    tracks = (
                        data.get("captions", {})
                        .get("playerCaptionsTracklistRenderer", {})
                        .get("captionTracks", [])
                    )
                    for url_cand, label in _select_candidate_caption_urls(tracks):
                        try:
                            cap_resp = await client.get(url_cand)
                            if cap_resp.status_code == 200 and cap_resp.text.strip():
                                parsed_items = _parse_caption_payload(cap_resp.text)
                                if parsed_items:
                                    return parsed_items, label
                        except Exception:
                            continue
            except Exception:
                continue

    return [], "unavailable"


def _normalize_transcript_items(raw_items: Any) -> List[Dict[str, Any]]:
    normalized = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).replace("\n", " ").strip()
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
        else:
            text = str(getattr(item, "text", "")).replace("\n", " ").strip()
            start = float(getattr(item, "start", 0.0))
            duration = float(getattr(item, "duration", 0.0))
        if text:
            normalized.append({"text": html.unescape(text), "start": start, "duration": duration})
    return normalized


async def fetch_video_transcript(video_id: str) -> Tuple[List[Dict[str, Any]], str]:
    # 1. Try youtube-transcript-api first (fast when installed & working, supports raw 'hi' + 'en' natively)
    if HAS_TRANSCRIPT_API:
        preferred_langs = ["en", "en-IN", "en-US", "en-GB", "hi"]
        # v1.x+ interface (`YouTubeTranscriptApi().fetch(video_id, languages=...)`)
        try:
            ytt = YouTubeTranscriptApi()
            if hasattr(ytt, "fetch"):
                fetched = ytt.fetch(video_id, languages=preferred_langs)
                norm = _normalize_transcript_items(fetched)
                if norm:
                    print(f"[Transcript] Extracted {len(norm)} segments for {video_id} via youtube-transcript-api v1+ fetch")
                    return norm, "Spoken Transcript Extracted"
        except Exception:
            pass

        # v0.6.x interface (`YouTubeTranscriptApi.get_transcript(video_id, languages=...)`)
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_langs)
                norm = _normalize_transcript_items(raw)
                if norm:
                    print(f"[Transcript] Extracted {len(norm)} segments for {video_id} via youtube-transcript-api get_transcript")
                    return norm, "Spoken Transcript Extracted"
        except Exception:
            pass

        # Try listing any available language transcript without forcing translation
        try:
            if hasattr(YouTubeTranscriptApi, "list_transcripts"):
                t_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for tr in t_list:
                    norm = _normalize_transcript_items(tr.fetch())
                    if norm:
                        return norm, getattr(tr, "language", "Spoken Transcript")
        except Exception:
            pass

    # 2. Try Watch Page + Innertube extractor
    items, label = await fetch_transcript_via_innertube(video_id)
    if items:
        print(f"[Transcript] Extracted {len(items)} segments for {video_id} via Watch/Innertube ({label})")
        return items, label

    print(f"[Transcript] Captions unavailable on YouTube for {video_id}; AI will teach from title, description & chapter syllabus.")
    return [], "AI Deep Syllabus & Topic Synthesis"


def group_transcript_into_chapters(
    transcript_items: List[Dict[str, Any]],
    video_title: str,
    description: str,
    total_duration: int,
    target_chapters: int = 7,
) -> List[Dict[str, Any]]:
    native_chapters = parse_description_chapters(description, total_duration)
    if len(native_chapters) > 12:
        step_idx = max(1, len(native_chapters) // 10)
        sampled = [native_chapters[i] for i in range(0, len(native_chapters), step_idx)][:12]
        for i in range(len(sampled)):
            sampled[i]["end_seconds"] = (
                sampled[i + 1]["start_seconds"] if i + 1 < len(sampled) else max(total_duration, sampled[i]["start_seconds"] + 300)
            )
        native_chapters = sampled

    safe_duration = max(total_duration, 600)

    if not transcript_items:
        if native_chapters:
            return [
                {
                    "start_seconds": ch["start_seconds"],
                    "timestamp": ch["timestamp"],
                    "title": ch["title"],
                    "transcript_text": "",
                }
                for ch in native_chapters
            ]

        step = max(60, safe_duration // 4)
        fallback_titles = [
            f"Foundations & Problem Statement ({video_title[:45]})",
            "Core Architecture, Internal Mechanism & Theory",
            "Code Implementation & Step-by-Step Walkthrough",
            "Edge Cases, Best Practices & Interview Takeaways",
        ]
        return [
            {
                "start_seconds": i * step,
                "timestamp": format_seconds_hms(i * step),
                "title": fallback_titles[i],
                "transcript_text": "",
            }
            for i in range(4)
        ]

    max_end = max(
        total_duration,
        int(transcript_items[-1]["start"] + transcript_items[-1].get("duration", 5)),
    )

    if native_chapters:
        buckets = []
        for ch in native_chapters:
            s_sec = ch["start_seconds"]
            e_sec = ch["end_seconds"]
            lines = [
                item["text"]
                for item in transcript_items
                if s_sec <= item["start"] < e_sec
            ]
            buckets.append({
                "start_seconds": s_sec,
                "timestamp": ch["timestamp"],
                "title": ch["title"],
                "transcript_text": " ".join(lines).strip(),
            })
        return buckets

    num_segments = max(4, min(8, target_chapters if max_end > 300 else 4))
    window = max(60, int(math.ceil(max_end / num_segments)))

    buckets = []
    for idx in range(num_segments):
        s_sec = idx * window
        e_sec = min(max_end + 10, (idx + 1) * window) if idx < num_segments - 1 else max_end + 60
        lines = [
            item["text"]
            for item in transcript_items
            if s_sec <= item["start"] < e_sec
        ]
        if not lines:
            continue
        actual_start = int(next((item["start"] for item in transcript_items if s_sec <= item["start"] < e_sec), s_sec))
        first_sents = split_into_clean_sentences(" ".join(lines[:6]), max_words_per_chunk=8)
        snippet_label = first_sents[0].rstrip(".")[:45] if first_sents else f"Segment {len(buckets) + 1}"
        buckets.append({
            "start_seconds": actual_start,
            "timestamp": format_seconds_hms(actual_start),
            "title": f"Part {len(buckets) + 1}: {snippet_label}",
            "transcript_text": " ".join(lines).strip(),
        })
    return buckets


# ---------------------------------------------------------------------------
# YouTube Data API v3 Client
# ---------------------------------------------------------------------------

async def fetch_youtube_video_metadata(video_ids: List[str], api_key: str) -> List[Dict[str, Any]]:
    results = []
    async with httpx.AsyncClient(timeout=25.0) as client:
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            resp = await client.get(
                f"{YOUTUBE_API_BASE}/videos",
                params={
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(batch),
                    "key": api_key,
                },
            )
            if resp.status_code != 200:
                detail = resp.json().get("error", {}).get("message", resp.text)
                raise HTTPException(status_code=resp.status_code, detail=f"YouTube Data API v3 error: {detail}")
            payload = resp.json()
            for item in payload.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                stats = item.get("statistics", {})
                dur_iso = content_details.get("duration", "PT0S")
                dur_sec = parse_iso8601_duration(dur_iso)
                thumbs = snippet.get("thumbnails", {})
                thumb_url = (
                    thumbs.get("maxres", {}).get("url")
                    or thumbs.get("high", {}).get("url")
                    or thumbs.get("medium", {}).get("url")
                    or f"https://i.ytimg.com/vi/{item['id']}/hqdefault.jpg"
                )
                results.append({
                    "video_id": item["id"],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "title": snippet.get("title", "Untitled Video"),
                    "channel_title": snippet.get("channelTitle", "Unknown Channel"),
                    "published_at": snippet.get("publishedAt", "")[:10],
                    "description": snippet.get("description", ""),
                    "tags": snippet.get("tags", [])[:15],
                    "thumbnail": thumb_url,
                    "duration_iso": dur_iso,
                    "duration_seconds": dur_sec,
                    "duration_formatted": format_seconds_hms(dur_sec),
                    "duration_human": format_seconds_human(dur_sec),
                    "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                    "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0,
                })
    return results


async def fetch_youtube_playlist_metadata(
    playlist_id: str,
    api_key: str,
    max_videos: int = 200,
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        pl_resp = await client.get(
            f"{YOUTUBE_API_BASE}/playlists",
            params={
                "part": "snippet,contentDetails",
                "id": playlist_id,
                "key": api_key,
            },
        )
        if pl_resp.status_code != 200:
            detail = pl_resp.json().get("error", {}).get("message", pl_resp.text)
            raise HTTPException(status_code=pl_resp.status_code, detail=f"YouTube Playlist API error: {detail}")

        pl_items = pl_resp.json().get("items", [])
        if not pl_items:
            raise HTTPException(status_code=404, detail=f"Playlist '{playlist_id}' not found or is private.")

        pl_info = pl_items[0]
        pl_snippet = pl_info.get("snippet", {})
        pl_total_count = pl_info.get("contentDetails", {}).get("itemCount", 0)

        video_ids: List[str] = []
        next_page_token: Optional[str] = None

        while len(video_ids) < max_videos:
            params: Dict[str, Any] = {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": min(50, max_videos - len(video_ids)),
                "key": api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            items_resp = await client.get(f"{YOUTUBE_API_BASE}/playlistItems", params=params)
            if items_resp.status_code != 200:
                break
            items_data = items_resp.json()
            for it in items_data.get("items", []):
                vid = it.get("contentDetails", {}).get("videoId")
                if vid and vid not in video_ids:
                    video_ids.append(vid)
            next_page_token = items_data.get("nextPageToken")
            if not next_page_token:
                break

    videos = await fetch_youtube_video_metadata(video_ids, api_key)
    return {
        "playlist_id": playlist_id,
        "url": f"https://www.youtube.com/playlist?list={playlist_id}",
        "title": pl_snippet.get("title", "Untitled Playlist"),
        "channel_title": pl_snippet.get("channelTitle", "Unknown Channel"),
        "description": pl_snippet.get("description", ""),
        "published_at": pl_snippet.get("publishedAt", "")[:10],
        "total_item_count": max(pl_total_count, len(videos)),
        "fetched_video_count": len(videos),
        "videos": videos,
    }


# ---------------------------------------------------------------------------
# Dynamic Gemini `ListModels` Discovery & Bulletproof LLM JSON Engine
# ---------------------------------------------------------------------------

_CACHED_GEMINI_MODELS: Dict[str, List[str]] = {}


async def discover_available_gemini_models(client: httpx.AsyncClient, api_key: str) -> List[str]:
    """
    Calls `GET https://generativelanguage.googleapis.com/v1beta/models?key=...`
    to discover the exact model names supported by the user's API key!
    Fixes `HTTP 404: models/gemini-1.5-flash is not found for API version v1beta`.
    """
    cache_key = api_key[-8:]
    if cache_key in _CACHED_GEMINI_MODELS:
        return _CACHED_GEMINI_MODELS[cache_key]

    preferred_order = [
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-3.8-pro",
        "gemini-pro-latest",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
    ]

    try:
        resp = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key, "pageSize": 100},
        )
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            valid_names = []
            for m in models_data:
                methods = m.get("supportedGenerationMethods", [])
                name = str(m.get("name", "")).replace("models/", "")
                if "generateContent" not in methods:
                    continue
                # Skip specialized non-text models
                if any(skip in name.lower() for skip in ["tts", "imagen", "embedding", "aqa", "vision", "audio", "robotics", "computer-use", "thinking-exp"]):
                    continue
                if name.startswith("gemini-"):
                    valid_names.append(name)

            if valid_names:
                ranked = [p for p in preferred_order if p in valid_names]
                for v in valid_names:
                    if v not in ranked:
                        ranked.append(v)
                print(f"[Gemini ListModels] Discovered available models: {ranked[:6]}")
                _CACHED_GEMINI_MODELS[cache_key] = ranked[:6]
                return ranked[:6]
    except Exception as exc:
        print(f"[Gemini ListModels] Warning: {exc}")

    return preferred_order[:5]


def _parse_or_repair_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx : end_idx + 1]
        candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate_fixed)
        except Exception:
            pass
    return None


async def call_llm_json(
    prompt: str,
    provider: str,
    api_key: Optional[str],
    model: Optional[str] = None,
    youtube_url_for_native_ingest: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    api_key = clean_api_key(api_key)
    if not api_key:
        return None, "No Gemini/OpenAI API key configured (Click 'API & Gemini Key Config' to add your key)"

    errors_seen: List[str] = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        if provider.lower() == "openai":
            models_to_try = [model] if model else ["gpt-4o-mini", "gpt-4o"]
            for m in models_to_try:
                try:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": m,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert Computer Science & University Professor. "
                                        "Teach every concept in depth with clear explanations, code examples, and exam/interview Q&As. "
                                        "Always output valid JSON."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 8192,
                        },
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        parsed = _parse_or_repair_json(content)
                        if parsed:
                            print(f"[LLM] OpenAI ({m}) generated study JSON successfully.")
                            return parsed, f"AI Deep Study Guide ({m})"
                        errors_seen.append(f"{m}: JSON parse failed")
                    else:
                        err_msg = resp.json().get("error", {}).get("message", resp.text[:140])
                        errors_seen.append(f"{m} HTTP {resp.status_code}: {err_msg}")
                except Exception as exc:
                    errors_seen.append(f"{m}: {str(exc)[:100]}")
        else:
            # Use gemini-3.8-flash and gemini-flash-latest first (since gemini-2.0-flash and 2.5-flash return 404)
            fast_defaults = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-pro-latest"]
            models_to_try = [model] + [m for m in fast_defaults if m != model] if model else fast_defaults

            for m in models_to_try:
                # Try first with responseMimeType="application/json", and if 400 or parse error, retry standard
                for use_json_mime in [True, False]:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
                        gen_cfg: Dict[str, Any] = {
                            "temperature": 0.2,
                            "maxOutputTokens": 16384,
                        }
                        if use_json_mime:
                            gen_cfg["responseMimeType"] = "application/json"

                        # If transcript scraping was blocked on Vercel/AWS IP, first try passing the YouTube URL
                        # directly via `fileData.fileUri` so Google's own servers ingest the YouTube video!
                        parts_list: List[Dict[str, Any]] = []
                        if youtube_url_for_native_ingest and use_json_mime:
                            parts_list.append({
                                "fileData": {
                                    "mimeType": "video/mp4",
                                    "fileUri": youtube_url_for_native_ingest,
                                }
                            })
                        parts_list.append({"text": prompt})

                        resp = await client.post(
                            url,
                            params={"key": api_key},
                            json={
                                "contents": [{"parts": parts_list}],
                                "generationConfig": gen_cfg,
                            },
                        )
                        # If native YouTube fileUri isn't supported for this video length/model, immediately retry text-only
                        if resp.status_code != 200 and youtube_url_for_native_ingest and len(parts_list) > 1:
                            resp = await client.post(
                                url,
                                params={"key": api_key},
                                json={
                                    "contents": [{"parts": [{"text": prompt}]}],
                                    "generationConfig": gen_cfg,
                                },
                            )
                        if resp.status_code == 200:
                            candidates = resp.json().get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                # Filter out internal thinking parts (`thought: true`) from Gemini 2.5 models!
                                non_thought_parts = [p.get("text", "") for p in parts if not p.get("thought")]
                                text = "".join(non_thought_parts) or "".join(p.get("text", "") for p in parts)
                                parsed = _parse_or_repair_json(text)
                                if parsed:
                                    print(f"[LLM] Gemini ({m}) generated study JSON successfully!")
                                    return parsed, f"AI Deep Study Guide ({m})"
                                errors_seen.append(f"{m}: JSON truncated/invalid")
                        elif resp.status_code == 400 and use_json_mime:
                            # Retry same model without responseMimeType
                            continue
                        else:
                            err_msg = resp.json().get("error", {}).get("message", resp.text[:140])
                            errors_seen.append(f"{m} HTTP {resp.status_code}: {err_msg}")
                            print(f"[LLM Warning] {m} -> HTTP {resp.status_code}: {err_msg}")
                            break
                    except Exception as exc:
                        errors_seen.append(f"{m}: {str(exc)[:100]}")
                        break

    summary_err = " | ".join(errors_seen) if errors_seen else "LLM request failed"
    return None, summary_err


def build_extractive_study_summary(
    video: Dict[str, Any],
    chapter_buckets: List[Dict[str, Any]],
    transcript_items: List[Dict[str, Any]],
    llm_status_note: str = "",
) -> Dict[str, Any]:
    clean_desc = clean_description_text(video.get("description", ""))
    desc_sentences = split_into_clean_sentences(clean_desc, max_words_per_chunk=22)

    chapters_out = []
    concepts_taught = []
    distinct_takeaways = []

    for idx, bucket in enumerate(chapter_buckets):
        seg_text = bucket.get("transcript_text", "").strip()
        seg_sents = split_into_clean_sentences(seg_text, max_words_per_chunk=22)

        if seg_sents:
            short_ts_summary = " ".join(seg_sents[:2])
            ts_bullets = seg_sents[2:4] if len(seg_sents) > 2 else [seg_sents[0]]
            full_concept_explanation = " ".join(seg_sents[:6])
            concept_bullets = seg_sents[2:6] if len(seg_sents) > 2 else seg_sents[:2]
            distinct_takeaways.append(f"[{bucket['timestamp']}] {bucket['title']}: {seg_sents[0]}")
        else:
            short_ts_summary = f"Topic taught at {bucket['timestamp']}: {bucket['title']}."
            ts_bullets = [f"Section covering {bucket['title']}."]
            full_concept_explanation = (
                f"Section {idx + 1} (@ {bucket['timestamp']}): {bucket['title']}. "
                + (desc_sentences[idx] if idx < len(desc_sentences) else "Check your Gemini API key in settings for full AI concept & code explanations.")
            )
            concept_bullets = [f"Timestamp: {bucket['timestamp']} — {bucket['title']}"]
            distinct_takeaways.append(f"[{bucket['timestamp']}] {bucket['title']}")

        chapters_out.append({
            "chapter_number": idx + 1,
            "title": bucket["title"],
            "timestamp": bucket["timestamp"],
            "start_seconds": bucket["start_seconds"],
            "deep_link": f"https://www.youtube.com/watch?v={video['video_id']}&t={bucket['start_seconds']}s",
            "summary": short_ts_summary,
            "what_is_taught": short_ts_summary,
            "takeaways": ts_bullets[:2],
        })

        concepts_taught.append({
            "concept": bucket["title"],
            "timestamp": bucket["timestamp"],
            "start_seconds": bucket["start_seconds"],
            "explanation": full_concept_explanation,
            "code_example": "",
            "key_points": concept_bullets[:4],
        })

    if transcript_items:
        segment_digest_parts = []
        for bucket in chapter_buckets:
            b_sents = split_into_clean_sentences(bucket.get("transcript_text", ""), max_words_per_chunk=22)
            if b_sents:
                segment_digest_parts.append(f"({bucket['timestamp']} - {bucket['title']}) {' '.join(b_sents[:2])}")
        exec_summary = " ".join(segment_digest_parts) if segment_digest_parts else " ".join(desc_sentences[:6])
    elif desc_sentences:
        exec_summary = " ".join(desc_sentences[:8])
    else:
        exec_summary = f"Lecture on '{video['title']}' by {video['channel_title']} ({video['duration_human']})."

    return {
        "is_ai_generated": False,
        "llm_status": llm_status_note or "No Gemini API Key Set",
        "executive_summary": exec_summary,
        "key_takeaways": distinct_takeaways[:8],
        "action_items": [
            "Ensure your Gemini API key is saved in 'API & Gemini Key Config' to generate full code examples, concept explanations, and exam/interview Q&As.",
        ],
        "chapters": chapters_out,
        "lecture_notes": {
            "core_concepts_taught": concepts_taught,
            "formulas_and_algorithms": [],
            "worked_examples": [],
            "comparison_table": [],
        },
        "exam_prep": {
            "cheat_sheet_points": distinct_takeaways[:6],
            "common_exam_traps": [],
            "exam_questions": [],
        },
        "interview_prep": {
            "interview_questions": [],
            "flashcards": [],
        },
    }


def build_lightweight_playlist_item_placeholder(video: Dict[str, Any], index_1based: int) -> Dict[str, Any]:
    native_chapters = parse_description_chapters(video.get("description", ""), video.get("duration_seconds", 0))
    clean_desc = clean_description_text(video.get("description", ""))
    desc_sents = split_into_clean_sentences(clean_desc, max_words_per_chunk=22)
    short_overview = " ".join(desc_sents[:3]) if desc_sents else f"Lecture #{index_1based}: {video['title']} ({video['duration_human']})."

    chapters_preview = [
        {
            "chapter_number": i + 1,
            "title": ch["title"],
            "timestamp": ch["timestamp"],
            "start_seconds": ch["start_seconds"],
            "deep_link": f"https://www.youtube.com/watch?v={video['video_id']}&t={ch['start_seconds']}s",
            "summary": f"Covers {ch['title']} (click '✨ Deep AI Summarize' to generate full concept notes & code).",
            "what_is_taught": f"Covers {ch['title']}.",
            "takeaways": [],
        }
        for i, ch in enumerate(native_chapters[:12])
    ]

    return {
        **video,
        "playlist_index": index_1based,
        "transcript_status": "Click lecture to auto-summarize",
        "transcript_word_count": 0,
        "transcript_snippets": [],
        "summary": {
            "is_ai_generated": False,
            "needs_deep_summary": True,
            "llm_status": "Click to Summarize",
            "executive_summary": short_overview,
            "key_takeaways": [f"[{c['timestamp']}] {c['title']}" for c in chapters_preview[:8]] or [short_overview],
            "action_items": [],
            "chapters": chapters_preview,
            "lecture_notes": {
                "core_concepts_taught": [],
                "formulas_and_algorithms": [],
                "worked_examples": [],
                "comparison_table": [],
            },
            "exam_prep": {
                "cheat_sheet_points": [],
                "common_exam_traps": [],
                "exam_questions": [],
            },
            "interview_prep": {
                "interview_questions": [],
                "flashcards": [],
            },
        },
    }


async def summarize_single_video(
    video: Dict[str, Any],
    provider: str,
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    summary_depth: str = "comprehensive",
    study_goal: str = "master",
) -> Dict[str, Any]:
    is_vercel = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
    tr_timeout = 1.8 if is_vercel else 4.0
    try:
        transcript_items, lang_label = await asyncio.wait_for(
            fetch_video_transcript(video["video_id"]),
            timeout=tr_timeout,
        )
    except Exception:
        transcript_items, lang_label = [], "AI Deep Syllabus & Topic Synthesis (Fast Cloud Mode)" 
    chapter_buckets = group_transcript_into_chapters(
        transcript_items=transcript_items,
        video_title=video.get("title", "Untitled Video"),
        description=video.get("description", ""),
        total_duration=video.get("duration_seconds", 0),
    )

    full_transcript_text = " ".join(item.get("text", "") for item in transcript_items).strip()
    per_bucket_char_limit = max(2500, int(45000 / max(1, len(chapter_buckets))))
    segments_payload = [
        {
            "timestamp": b["timestamp"],
            "start_seconds": b["start_seconds"],
            "chapter_title": b["title"],
            "spoken_transcript": b["transcript_text"][:per_bucket_char_limit],
        }
        for b in chapter_buckets
    ]

    clean_desc = clean_description_text(video.get("description", ""))[:1800]

    prompt = f"""You are a Master Computer Science & University Professor and Senior Technical Interview Coach.
A student is reading your study guide instead of watching this entire {video['duration_human']} lecture video so they can understand EVERY concept, mechanism, code example, and interview/exam question fast!

Video Title: {video['title']}
Channel / Instructor: {video['channel_title']}
Duration: {video['duration_human']}
Video Description: {clean_desc}
Spoken Transcript Status: {"Extracted (" + str(len(full_transcript_text.split())) + " words, may be in English, Hindi, or Hinglish—translate and teach in clear English)" if full_transcript_text else "No captions uploaded on YouTube—use the Video Title, Description, and the exact Chapter Timestamps/Topics below to thoroughly teach every concept, code example, and mechanism covered in this lecture!"}

Lecture Chapters & Transcript Segments:
{json.dumps(segments_payload, ensure_ascii=False)}

MANDATORY TEACHING RULES (READ CAREFULLY):
1. NEVER WRITE META-FILLER:
   - Forbidden: "In this video we will understand X" or "Explains X at 15:40" or "Covers Bean Scopes".
   - Required: Directly TEACH the actual concepts! Explain WHAT each concept is, WHY the problem occurs under the hood, HOW the internal mechanism works step-by-step, and HOW to write the code or formula.
2. COMPREHENSIVE VIDEO SUMMARY (`executive_summary` & `key_takeaways`):
   - `executive_summary`: Write a rich, comprehensive 8–12 sentence technical synthesis that walks through ALL major concepts taught in the video from start to finish (explaining the actual theory, why issues like errors/exceptions happen, how annotations/algorithms solve them, and key design principles).
   - `key_takeaways`: 6–8 high-density, concrete technical takeaways (each teaching a real rule, behavior, or trade-off).
3. DEEP CONCEPT CARDS WITH REAL CODE EXAMPLES (`lecture_notes.core_concepts_taught`):
   - For EVERY major topic/chapter in the video (5–8 concepts), provide:
     - `concept`: Clear title of the concept.
     - `explanation`: 4–6 sentences teaching the exact concept and under-the-hood mechanism in plain, crystal-clear English.
     - `code_example`: A clean, realistic 6–15 line code snippet (e.g. Java/Spring, Python, C++, SQL, or mathematical derivation) illustrating the concept in action!
     - `key_points`: 3 crisp bullet points covering edge cases, rules, or interview gotchas.
4. SHORT, TO-THE-POINT TIMESTAMP SUMMARIES (`chapters`):
   - For each timestamp in `chapters`, keep `summary` and `what_is_taught` SHORT and TO THE POINT (1–2 crisp sentences, max 35 words) teaching the core point of that specific timestamp, plus 2 short bullet `takeaways`.
5. DISTINCT, HIGH-YIELD EXAM & INTERVIEW Q&As:
   - Every question in `exam_questions` and `interview_questions` must test a DIFFERENT concept from the video with a concrete, accurate answer.

Return ONLY valid JSON matching this exact schema:
{{
  "executive_summary": "Comprehensive 8-12 sentence technical synthesis teaching all major concepts, internal mechanisms, and design trade-offs from the lecture.",
  "key_takeaways": [
    "Concrete technical takeaway #1",
    "Concrete technical takeaway #2",
    "Concrete technical takeaway #3",
    "Concrete technical takeaway #4",
    "Concrete technical takeaway #5",
    "Concrete technical takeaway #6"
  ],
  "action_items": [
    "Practical coding or problem-solving exercise #1",
    "Practical coding or problem-solving exercise #2"
  ],
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "Chapter Topic Title",
      "timestamp": "MM:SS",
      "start_seconds": 0,
      "summary": "1-2 short, direct sentences explaining the exact concept taught at this timestamp.",
      "what_is_taught": "1-2 short, direct sentences explaining the exact concept taught at this timestamp.",
      "takeaways": [
        "Short technical point 1",
        "Short technical point 2"
      ]
    }}
  ],
  "lecture_notes": {{
    "core_concepts_taught": [
      {{
        "concept": "Concept Name (e.g. Constructor vs Setter Circular Dependency)",
        "timestamp": "MM:SS",
        "start_seconds": 0,
        "explanation": "4-6 sentences explaining what this concept is, how it works internally, and why it matters.",
        "code_example": "// Clean 6-15 line code snippet or mathematical derivation illustrating this concept",
        "key_points": [
          "Key rule or internal behavior 1",
          "Key rule or internal behavior 2",
          "Key edge case or best practice 3"
        ]
      }}
    ],
    "formulas_and_algorithms": [
      {{
        "name": "Annotation / Algorithm / Formula / Pattern Name",
        "expression": "Exact syntax, annotation, or complexity O(...)",
        "when_to_use": "When to use or avoid this in real projects/exams",
        "explanation": "2-3 sentences explaining how it works under the hood"
      }}
    ],
    "worked_examples": [
      {{
        "title": "Concrete Code / Scenario Walkthrough",
        "timestamp": "MM:SS",
        "problem": "The concrete problem scenario demonstrated in the lecture",
        "step_by_step_solution": "Step-by-step trace of what happens at runtime and how to resolve it"
      }}
    ],
    "comparison_table": [
      {{
        "aspect": "Feature / Comparison Dimension",
        "approach_a": "Option A (e.g. Singleton / Eager / Constructor Injection)",
        "approach_b": "Option B (e.g. Prototype / @Lazy / Setter Injection)",
        "verdict": "Clear rule on when to choose which"
      }}
    ]
  }},
  "exam_prep": {{
    "cheat_sheet_points": [
      "Must-remember exam/revision fact #1",
      "Must-remember exam/revision fact #2",
      "Must-remember exam/revision fact #3",
      "Must-remember exam/revision fact #4"
    ],
    "common_exam_traps": [
      "Common misconception or bug #1 and why it happens",
      "Common misconception or bug #2 and why it happens"
    ],
    "exam_questions": [
      {{
        "question": "Conceptual or code-output exam question #1?",
        "difficulty": "Medium",
        "model_answer": "Clear, complete 3-4 sentence answer explaining the exact reason and output."
      }},
      {{
        "question": "Conceptual or code-output exam question #2?",
        "difficulty": "Hard",
        "model_answer": "Clear, complete 3-4 sentence answer explaining the exact reason and output."
      }},
      {{
        "question": "Conceptual or code-output exam question #3?",
        "difficulty": "Medium",
        "model_answer": "Clear, complete 3-4 sentence answer explaining the exact reason and output."
      }}
    ]
  }},
  "interview_prep": {{
    "interview_questions": [
      {{
        "question": "High-probability technical interview question #1?",
        "topic_tag": "Core Topic",
        "elevator_pitch_30s": "Crisp 2-sentence 30-second answer.",
        "deep_dive_answer": "2-3 sentences explaining the internal lifecycle/mechanism.",
        "follow_up_tradeoffs": "1-2 sentences on design trade-offs and best practices."
      }},
      {{
        "question": "High-probability technical interview question #2?",
        "topic_tag": "Architecture",
        "elevator_pitch_30s": "Crisp 2-sentence 30-second answer.",
        "deep_dive_answer": "2-3 sentences explaining the internal lifecycle/mechanism.",
        "follow_up_tradeoffs": "1-2 sentences on design trade-offs and best practices."
      }},
      {{
        "question": "High-probability technical interview question #3?",
        "topic_tag": "Edge Cases",
        "elevator_pitch_30s": "Crisp 2-sentence 30-second answer.",
        "deep_dive_answer": "2-3 sentences explaining the internal lifecycle/mechanism.",
        "follow_up_tradeoffs": "1-2 sentences on design trade-offs and best practices."
      }}
    ],
    "flashcards": [
      {{
        "front": "Question 1?",
        "back": "Concise answer 1.",
        "tag": "Core"
      }},
      {{
        "front": "Question 2?",
        "back": "Concise answer 2.",
        "tag": "Code"
      }},
      {{
        "front": "Question 3?",
        "back": "Concise answer 3.",
        "tag": "Interview"
      }}
    ]
  }}
}}"""

    llm_res, llm_status = await call_llm_json(
        prompt,
        provider=provider,
        api_key=llm_api_key,
        model=llm_model,
        youtube_url_for_native_ingest=(video.get("url") if (not is_vercel and not transcript_items and video.get("duration_seconds", 0) <= 600) else None),
    )

    if not llm_res or not isinstance(llm_res, dict) or not llm_res.get("executive_summary"):
        summary_final = build_extractive_study_summary(
            video=video,
            chapter_buckets=chapter_buckets,
            transcript_items=transcript_items,
            llm_status_note=llm_status,
        )
    else:
        llm_res["is_ai_generated"] = True
        llm_res["needs_deep_summary"] = False
        llm_res["llm_status"] = llm_status
        for idx, ch in enumerate(llm_res.get("chapters", [])):
            sec = int(ch.get("start_seconds", 0))
            ch["deep_link"] = f"https://www.youtube.com/watch?v={video['video_id']}&t={sec}s"
            ch["chapter_number"] = idx + 1
            if not ch.get("what_is_taught"):
                ch["what_is_taught"] = ch.get("summary", "")
        llm_res.setdefault("lecture_notes", {})
        llm_res.setdefault("exam_prep", {})
        llm_res.setdefault("interview_prep", {})
        summary_final = llm_res

    word_count = len(full_transcript_text.split()) if full_transcript_text else 0
    return {
        **video,
        "transcript_status": lang_label,
        "transcript_word_count": word_count,
        "transcript_snippets": transcript_items[:500],
        "summary": summary_final,
    }


async def generate_playlist_rollup(
    playlist_meta: Dict[str, Any],
    all_videos: List[Dict[str, Any]],
    duration_stats: Dict[str, Any],
    provider: str,
    llm_api_key: Optional[str],
    llm_model: Optional[str],
) -> Dict[str, Any]:
    catalog_digest = [
        {
            "lecture": idx + 1,
            "title": v["title"],
            "duration": v["duration_formatted"],
        }
        for idx, v in enumerate(all_videos)
    ]

    prompt = f"""Summarize this {len(all_videos)}-video YouTube Playlist for university exam revision and technical interview prep.
Playlist Title: {playlist_meta['title']}
Channel: {playlist_meta['channel_title']}
Total Videos: {len(all_videos)} ({duration_stats['total_formatted']})

Videos:
{json.dumps(catalog_digest, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "master_thesis": "4-6 sentences summarizing the complete curriculum taught across all {len(all_videos)} videos and how the topics connect.",
  "learning_progression": [
    {{
      "phase": "Module 1 (Videos 1-N): Topic Name",
      "focus": "2-3 sentences on the exact concepts covered in these lectures."
    }}
  ],
  "cross_video_takeaways": [
    "Important course takeaway #1",
    "Important course takeaway #2",
    "Important course takeaway #3",
    "Important course takeaway #4",
    "Important course takeaway #5"
  ],
  "recommended_watch_strategy": "2 sentences on how to study this playlist efficiently at 1.25x-1.5x.",
  "glossary_terms": [
    {{"term": "Key Term 1", "definition": "Clear 1-sentence definition"}},
    {{"term": "Key Term 2", "definition": "Clear 1-sentence definition"}},
    {{"term": "Key Term 3", "definition": "Clear 1-sentence definition"}}
  ]
}}"""

    llm_rollup, _ = await call_llm_json(prompt, provider=provider, api_key=llm_api_key, model=llm_model)

    if not llm_rollup or not isinstance(llm_rollup, dict) or "master_thesis" not in llm_rollup:
        total_v = len(all_videos)
        chunk_size = max(1, math.ceil(total_v / 4))
        phases = []
        for start_i in range(0, total_v, chunk_size):
            end_i = min(total_v, start_i + chunk_size)
            chunk_titles = ", ".join(f"#{j+1} {all_videos[j]['title']}" for j in range(start_i, min(end_i, start_i + 6)))
            if end_i - start_i > 6:
                chunk_titles += f" (+{end_i - start_i - 6} more)"
            phases.append({
                "phase": f"Module {len(phases)+1}: Videos {start_i+1}–{end_i}",
                "focus": chunk_titles,
            })

        llm_rollup = {
            "master_thesis": (
                f"'{playlist_meta['title']}' by {playlist_meta['channel_title']} covers {total_v} videos "
                f"({duration_stats['total_formatted']}). Click any video below to generate its AI study notes, code examples, timestamp takeaways, and Q&As."
            ),
            "learning_progression": phases,
            "cross_video_takeaways": [
                f"#{i+1} ({v['duration_formatted']}) — {v['title']}"
                for i, v in enumerate(all_videos)
            ],
            "recommended_watch_strategy": (
                f"Total watch time is {duration_stats['total_formatted']}, or "
                f"{duration_stats['speeds'].get('1.5x', {}).get('watch_formatted', '')} at 1.5x speed "
                f"(saving {duration_stats['speeds'].get('1.5x', {}).get('saved_formatted', '')})."
            ),
            "glossary_terms": [
                {"term": f"Video #{i+1}", "definition": f"{v['title']} ({v['duration_human']})"}
                for i, v in enumerate(all_videos[:6])
            ],
        }
    return llm_rollup


# ---------------------------------------------------------------------------
# Markdown & PDF Study Guide Export Builders
# ---------------------------------------------------------------------------

def _append_video_study_markdown(lines: List[str], vid: Dict[str, Any], prefix: str = "##") -> None:
    vsum = vid.get("summary", {})
    notes = vsum.get("lecture_notes", {})
    exam = vsum.get("exam_prep", {})
    interview = vsum.get("interview_prep", {})

    lines.append(f"{prefix} Comprehensive Video Summary")
    lines.append(vsum.get("executive_summary", ""))
    lines.append("")

    if vsum.get("key_takeaways"):
        lines.append(f"{prefix} Key Takeaways")
        for tk in vsum.get("key_takeaways", []):
            lines.append(f"- {tk}")
        lines.append("")

    if notes.get("core_concepts_taught"):
        lines.append(f"{prefix} Core Concepts, Explanations & Code Examples")
        for c in notes.get("core_concepts_taught", []):
            lines.append(f"### {c.get('concept')} (`{c.get('timestamp', '00:00')}`)")
            lines.append(c.get("explanation", ""))
            if c.get("code_example"):
                lines.append("```")
                lines.append(str(c.get("code_example")).strip())
                lines.append("```")
            for kp in c.get("key_points", []):
                lines.append(f"- {kp}")
            lines.append("")

    lines.append(f"{prefix} Short Timestamped Takeaways")
    for ch in vsum.get("chapters", []):
        lines.append(f"- **[{ch.get('timestamp')}]({ch.get('deep_link')}) — {ch.get('title')}**: {ch.get('summary')}")
        for tk in ch.get("takeaways", []):
            lines.append(f"  - {tk}")
    lines.append("")

    if exam.get("exam_questions"):
        lines.append(f"{prefix} Exam Revision Q&A")
        for idx, eq in enumerate(exam.get("exam_questions", []), start=1):
            lines.append(f"- **Q{idx}**: {eq.get('question')}")
            lines.append(f"  - *Answer*: {eq.get('model_answer')}")
        lines.append("")

    if interview.get("interview_questions"):
        lines.append(f"{prefix} Interview Prep Q&A")
        for idx, iq in enumerate(interview.get("interview_questions", []), start=1):
            lines.append(f"- **Q{idx} [{iq.get('topic_tag', 'Core')}]**: {iq.get('question')}")
            lines.append(f"  - *30s Answer*: {iq.get('elevator_pitch_30s')}")
            lines.append(f"  - *Deep Dive*: {iq.get('deep_dive_answer')}")
            lines.append(f"  - *Trade-Offs*: {iq.get('follow_up_tradeoffs')}")
        lines.append("")


def render_markdown_export(data: Dict[str, Any]) -> str:
    kind = data.get("kind", "video")
    lines: List[str] = []

    if kind == "playlist":
        pl = data.get("playlist", {})
        dur = data.get("duration_stats", {})
        rollup = data.get("playlist_rollup", {})
        lines.append(f"# Playlist Study Guide: [{pl.get('title', 'YouTube Playlist')}]({pl.get('url', '#')})")
        lines.append(f"**Channel:** {pl.get('channel_title', 'Unknown')} | **Total Videos:** {dur.get('video_count', 0)} | **Total Duration:** {dur.get('total_formatted', '0m')}")
        lines.append("")
        lines.append("## Playlist Overview")
        lines.append(rollup.get("master_thesis", ""))
        lines.append("")
        for idx, vid in enumerate(data.get("videos", []), start=1):
            lines.append(f"## {idx}. [{vid.get('title')}]({vid.get('url')}) ({vid.get('duration_formatted')})")
            _append_video_study_markdown(lines, vid, prefix="###")
    else:
        vid = data.get("video", {})
        lines.append(f"# Video Study Guide: [{vid.get('title', 'YouTube Video')}]({vid.get('url', '#')})")
        lines.append(f"**Channel:** {vid.get('channel_title', 'Unknown')} | **Duration:** {vid.get('duration_human', '')} (`{vid.get('duration_formatted', '')}`)")
        lines.append("")
        _append_video_study_markdown(lines, vid, prefix="##")

    return "\n".join(lines).strip() + "\n"


def render_pdf_export(data: Dict[str, Any]) -> bytes:
    if not HAS_FPDF:
        raise HTTPException(status_code=500, detail="fpdf2 is not installed in the backend environment.")

    def safe_latin(text: str) -> str:
        return (
            str(text)
            .replace("—", "-")
            .replace("–", "-")
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("•", "-")
            .replace("→", "->")
            .replace("≥", ">=")
            .replace("≤", "<=")
            .replace("≠", "!=")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    kind = data.get("kind", "video")
    videos = data.get("videos", []) if kind == "playlist" else [data.get("video", {})]

    if kind == "playlist":
        pl = data.get("playlist", {})
        dur = data.get("duration_stats", {})
        rollup = data.get("playlist_rollup", {})

        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 8, safe_latin(f"Study Guide: {pl.get('title', 'YouTube Playlist')}"))
        pdf.set_font("Helvetica", "", 9.5)
        pdf.cell(
            0,
            6,
            safe_latin(f"Channel: {pl.get('channel_title', '')} | Videos: {dur.get('video_count', 0)} | Duration: {dur.get('total_formatted', '')}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)
        pdf.multi_cell(0, 5, safe_latin(rollup.get("master_thesis", "")))
        pdf.ln(3)

    for idx, vid in enumerate(videos, start=1):
        vsum = vid.get("summary", {})
        notes = vsum.get("lecture_notes", {})
        exam = vsum.get("exam_prep", {})
        interview = vsum.get("interview_prep", {})

        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, safe_latin(f"{idx}. {vid.get('title', '')} ({vid.get('duration_formatted', '')})"))
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5, safe_latin(vsum.get("executive_summary", "")))
        pdf.ln(2)

        for c in notes.get("core_concepts_taught", []):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 5.5, safe_latin(f"[{c.get('timestamp', '00:00')}] {c.get('concept', '')}"))
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.8, safe_latin(c.get("explanation", "")))
            if c.get("code_example"):
                pdf.set_font("Courier", "", 8.5)
                pdf.multi_cell(0, 4.5, safe_latin(str(c.get("code_example"))))
            pdf.ln(1.5)

        for q_idx, eq in enumerate(exam.get("exam_questions", []), start=1):
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.multi_cell(0, 5, safe_latin(f"Exam Q{q_idx}: {eq.get('question', '')}"))
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.8, safe_latin(f"  Ans: {eq.get('model_answer', '')}"))

        for q_idx, iq in enumerate(interview.get("interview_questions", []), start=1):
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.multi_cell(0, 5, safe_latin(f"Interview Q{q_idx}: {iq.get('question', '')}"))
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.8, safe_latin(f"  Pitch: {iq.get('elevator_pitch_30s', '')}"))
        pdf.ln(3)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# FastAPI Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    here = pathlib.Path(__file__).resolve().parent
    html_file = here / "youtube_summarizer_webapp.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>PulseTube AI v4.0 Running</h1><p>Open <code>youtube_summarizer_webapp.html</code> or visit <a href='/docs'>/docs</a>.</p>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "4.0.0",
        "service": "PulseTube AI v4.0 (Dynamic Gemini ListModels + Code & Concept Teaching Engine)",
        "has_youtube_transcript_api": HAS_TRANSCRIPT_API,
        "has_env_youtube_key": bool(clean_api_key(os.getenv("YOUTUBE_API_KEY"))),
        "has_env_gemini_key": bool(clean_api_key(os.getenv("GEMINI_API_KEY"))),
        "has_env_openai_key": bool(clean_api_key(os.getenv("OPENAI_API_KEY"))),
    }


@app.post("/api/calculate-duration")
async def api_calculate_duration(req: DurationCalcRequest):
    return calculate_playlist_durations(
        durations_seconds=req.durations_seconds,
        start_index=req.start_index,
        end_index=req.end_index,
    )


@app.post("/api/summarize-video")
async def api_summarize_single_video_on_demand(req: SummarizeSingleVideoRequest):
    llm_key = clean_api_key(
        req.llm_api_key
        or (os.getenv("OPENAI_API_KEY") if req.llm_provider == "openai" else os.getenv("GEMINI_API_KEY"))
        or os.getenv("NEXT_PUBLIC_GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    sv = await summarize_single_video(
        video=req.video,
        provider=req.llm_provider,
        llm_api_key=llm_key or None,
        llm_model=req.llm_model,
        summary_depth=req.summary_depth,
        study_goal=req.study_goal,
    )
    return {
        "status": "ok",
        "used_ai_model": bool(sv.get("summary", {}).get("is_ai_generated")),
        "llm_status": sv.get("summary", {}).get("llm_status", ""),
        "video": sv,
    }


@app.post("/api/analyze")
async def analyze_youtube_url(req: AnalyzeRequest):
    parsed = parse_youtube_url(req.url)
    yt_key = clean_api_key(req.youtube_api_key or os.getenv("YOUTUBE_API_KEY") or os.getenv("NEXT_PUBLIC_YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    llm_key = clean_api_key(
        req.llm_api_key
        or (os.getenv("OPENAI_API_KEY") if req.llm_provider == "openai" else os.getenv("GEMINI_API_KEY"))
        or os.getenv("NEXT_PUBLIC_GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not yt_key:
        raise HTTPException(
            status_code=400,
            detail="Missing YouTube Data API v3 key. Pass `youtube_api_key` in 'API & Gemini Key Config' or set `YOUTUBE_API_KEY` in `.env`.",
        )

    target_mode = req.mode if req.mode in ("video", "playlist") else parsed["detected_type"]

    if target_mode == "playlist" and parsed["playlist_id"]:
        pl_meta = await fetch_youtube_playlist_metadata(
            playlist_id=parsed["playlist_id"],
            api_key=yt_key,
            max_videos=req.max_playlist_videos,
        )
        videos = pl_meta["videos"]
        durations = [v["duration_seconds"] for v in videos]
        duration_stats = calculate_playlist_durations(durations)

        if videos:
            first_video_task = summarize_single_video(
                video=videos[0],
                provider=req.llm_provider,
                llm_api_key=llm_key or None,
                llm_model=req.llm_model,
                summary_depth=req.summary_depth,
                study_goal=req.study_goal,
            )
            rollup_task = generate_playlist_rollup(
                playlist_meta=pl_meta,
                all_videos=videos,
                duration_stats=duration_stats,
                provider=req.llm_provider,
                llm_api_key=llm_key or None,
                llm_model=req.llm_model,
            )
            first_sv, rollup = await asyncio.gather(first_video_task, rollup_task)
            first_sv["playlist_index"] = 1
            summarized_videos = [first_sv] + [
                build_lightweight_playlist_item_placeholder(v, idx + 1)
                for idx, v in enumerate(videos[1:], start=1)
            ]
        else:
            summarized_videos = []
            rollup = {}

        first_llm_status = summarized_videos[0].get("summary", {}).get("llm_status", "") if summarized_videos else ""
        used_ai = bool(summarized_videos and summarized_videos[0].get("summary", {}).get("is_ai_generated"))

        return {
            "kind": "playlist",
            "study_goal": req.study_goal,
            "used_ai_model": used_ai,
            "llm_status": first_llm_status,
            "playlist": {k: v for k, v in pl_meta.items() if k != "videos"},
            "duration_stats": duration_stats,
            "playlist_rollup": rollup,
            "videos": summarized_videos,
        }

    if parsed["video_id"]:
        videos = await fetch_youtube_video_metadata([parsed["video_id"]], yt_key)
        if not videos:
            raise HTTPException(status_code=404, detail=f"Video '{parsed['video_id']}' not found.")
        sv = await summarize_single_video(
            video=videos[0],
            provider=req.llm_provider,
            llm_api_key=llm_key or None,
            llm_model=req.llm_model,
            summary_depth=req.summary_depth,
            study_goal=req.study_goal,
        )
        duration_stats = calculate_playlist_durations([sv["duration_seconds"]])
        return {
            "kind": "video",
            "study_goal": req.study_goal,
            "used_ai_model": bool(sv.get("summary", {}).get("is_ai_generated")),
            "llm_status": sv.get("summary", {}).get("llm_status", ""),
            "video": sv,
            "duration_stats": duration_stats,
        }

    raise HTTPException(
        status_code=400,
        detail="Could not detect a valid YouTube video ID (`v=...`) or playlist ID (`list=...`) from the URL.",
    )


@app.post("/api/export/markdown")
async def export_markdown_endpoint(req: ExportRequest):
    md_text = render_markdown_export(req.data)
    return Response(
        content=md_text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="youtube_study_guide.md"'},
    )


@app.post("/api/export/pdf")
async def export_pdf_endpoint(req: ExportRequest):
    pdf_bytes = render_pdf_export(req.data)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="youtube_study_guide.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=True)