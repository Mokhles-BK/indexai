from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import time

from indexer import DocumentIndexer

app = FastAPI(title="IndexAI", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_AUDIO_DIR = Path("uploaded_audio")
UPLOAD_BOOK_DIR  = Path("uploaded_books")
UPLOAD_DOC_DIR   = Path("uploaded_docs")
UPLOAD_IMAGE_DIR = Path("uploaded_images")
UPLOAD_VIDEO_DIR = Path("uploaded_videos")

for d in [UPLOAD_AUDIO_DIR, UPLOAD_BOOK_DIR, UPLOAD_DOC_DIR,
          UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR]:
    d.mkdir(exist_ok=True)

indexer = DocumentIndexer()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


# ── Upload: Audio ─────────────────────────────────────────────────────────────

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}")

    dest = UPLOAD_AUDIO_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    result = indexer.index_audio_file(str(dest))
    result["elapsed"] = round(time.time() - t0, 2)
    result["filename"] = file.filename
    return result


# ── Upload: Book ──────────────────────────────────────────────────────────────

@app.post("/upload-book")
async def upload_book(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, "Only .txt and .pdf book files are supported")

    dest = UPLOAD_BOOK_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    result = indexer.index_book_file(str(dest), category="book")
    result["elapsed"] = round(time.time() - t0, 2)
    result["filename"] = file.filename
    return result


# ── Upload: Document ──────────────────────────────────────────────────────────

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, "Only .txt and .pdf document files are supported")

    dest = UPLOAD_DOC_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    result = indexer.index_book_file(str(dest), category="doc")
    result["elapsed"] = round(time.time() - t0, 2)
    result["filename"] = file.filename
    return result


# ── Upload: Image ─────────────────────────────────────────────────────────────

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Index an image file using two strategies:
    - OCR (pytesseract): extracts any visible text in the image → Whoosh + MiniLM FAISS.
    - CLIP visual embedding: maps the image into a semantic vector space → visual FAISS.
      Searching "certificate" will match images that visually look like certificates,
      even if the word does not appear in them.
    """
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported image format: {ext}")

    dest = UPLOAD_IMAGE_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    result = indexer.index_image_file(str(dest))
    result["elapsed"] = round(time.time() - t0, 2)
    result["filename"] = file.filename
    return result


# ── Upload: Video ─────────────────────────────────────────────────────────────

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    frame_every_sec: int = Query(5, ge=1, le=60),
):
    """
    Index a video file using two strategies:
    1. Audio track transcribed with Whisper → text chunks indexed in Whoosh + FAISS.
    2. Frames sampled every N seconds → ResNet50 caption → indexed in Whoosh + FAISS.
    """
    allowed = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported video format: {ext}")

    dest = UPLOAD_VIDEO_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    result = indexer.index_video_file(str(dest), frame_every_sec=frame_every_sec)
    result["elapsed"] = round(time.time() - t0, 2)
    result["filename"] = file.filename
    return result


# ── Bulk CSV import ───────────────────────────────────────────────────────────

@app.post("/index-books")
async def index_books(csv_path: str = Query(...)):
    if not Path(csv_path).exists():
        raise HTTPException(400, "CSV file not found")
    t0 = time.time()
    result = indexer.index_books_csv(csv_path)
    result["elapsed"] = round(time.time() - t0, 2)
    return result


# ── Search ────────────────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    query: str
    method: str
    elapsed: float
    results: list[dict]


def deduplicate_by_file(results: list[dict]) -> list[dict]:
    """
    Group all chunks by their parent file and compute an aggregated score:
      - best_score  : highest single-chunk similarity (pure best match)
      - match_count : how many chunks matched (breadth of relevance)
      - final score : best_score * (1 + 0.05 * extra_matching_chunks), capped at 1.0
    This means a book where many chunks match ranks higher than a file where
    only one lucky chunk happened to score similarly.
    The representative chunk shown in the result card is always the best one.
    """
    groups = {}
    for r in results:
        parent_id = r.get("parent_id") or r.get("doc_id")
        if parent_id not in groups:
            groups[parent_id] = {"best": r, "scores": [r.get("score", 0)]}
        else:
            groups[parent_id]["scores"].append(r.get("score", 0))
            if r.get("score", 0) > groups[parent_id]["best"].get("score", 0):
                groups[parent_id]["best"] = r

    output = []
    for parent_id, g in groups.items():
        rep = dict(g["best"])  # copy so we don't mutate original
        best  = max(g["scores"])
        extra = len(g["scores"]) - 1          # extra chunks beyond the best one
        # Boost score slightly for each additional matching chunk (max +25%)
        aggregated = min(1.0, best * (1 + 0.05 * extra))
        rep["score"]       = round(aggregated, 4)
        rep["match_count"] = len(g["scores"])  # exposed to UI for transparency
        output.append(rep)

    return sorted(output, key=lambda x: x["score"], reverse=True)


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    method: str = Query("both", pattern="^(classic|ai|both)$"),
    top_k: int = Query(10, ge=1, le=50),
):
    t0 = time.time()

    if method == "classic":
        results = indexer.search_classic(q, top_k * 3)
    elif method == "ai":
        results = indexer.search_ai(q, top_k * 3)
    else:
        classic = {
            r["doc_id"]: {**r, "classic_score": r["score"], "ai_score": None}
            for r in indexer.search_classic(q, top_k * 3)
        }
        ai = {r["doc_id"]: r for r in indexer.search_ai(q, top_k * 3)}
        merged = dict(classic)

        for did, r in ai.items():
            if did in merged:
                merged[did]["ai_score"] = r["score"]
                merged[did]["score"] = max(merged[did]["score"], r["score"])
            else:
                merged[did] = {**r, "classic_score": None, "ai_score": r["score"]}

        results = list(merged.values())

    results = deduplicate_by_file(results)
    results = results[:top_k]

    return SearchResponse(
        query=q,
        method=method,
        elapsed=round(time.time() - t0, 4),
        results=results,
    )


# ── Delete / Stats ────────────────────────────────────────────────────────────

@app.get("/file/{doc_id}")
async def get_file(doc_id: str):
    """
    Return the original uploaded file for a given doc_id or parent_id.
    The frontend can link directly to this endpoint from search results.
    """
    # Try text metadata first, then visual metadata
    meta = indexer.metadata.get(doc_id) or indexer.visual_metadata.get(doc_id)

    # If not found by chunk id, try matching as parent_id
    if meta is None:
        meta = next(
            (m for m in list(indexer.metadata.values()) + list(indexer.visual_metadata.values())
             if m.get("parent_id") == doc_id),
            None,
        )

    if meta is None:
        raise HTTPException(404, "Document not found")

    # For video frame chunks, path points to the extracted .jpg frame.
    # Use video_path (the original video file) so the browser can play the video.
    raw_path = meta.get("video_path") or meta.get("path", "")
    file_path = Path(raw_path)
    if not file_path.exists():
        raise HTTPException(404, f"File no longer exists on disk: {file_path.name}")

    mime_map = {
        ".pdf":  "application/pdf",
        ".txt":  "text/plain; charset=utf-8",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
        ".bmp":  "image/bmp",
        ".tiff": "image/tiff",
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".m4a":  "audio/mp4",
        ".mp4":  "video/mp4",
        ".webm": "video/webm",
        ".mov":  "video/quicktime",
        ".avi":  "video/x-msvideo",
        ".mkv":  "video/x-matroska",
    }
    ext = file_path.suffix.lower()
    media_type = mime_map.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
    )


@app.delete("/delete/{doc_id}")
async def delete_document(doc_id: str):
    # doc_id can be either a chunk ID or a parent_id.
    # The indexer resolves the parent and removes ALL chunks belonging to that file.
    result = indexer.delete_document(doc_id)
    if not result["success"]:
        raise HTTPException(404, result["reason"])
    return result


@app.get("/stats")
async def stats():
    return indexer.stats()
