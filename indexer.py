import json
import hashlib
import csv
import shutil
from pathlib import Path

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED, NUMERIC
from whoosh.qparser import MultifieldParser

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ── CLIP (replaces ResNet50) ──────────────────────────────────────────────────
# CLIP maps images AND text into the same embedding space, so searching
# "certificate" will genuinely match an image containing a certificate.
try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# ── OCR (extract real text from images / video frames) ───────────────────────
try:
    import pytesseract
    from PIL import Image as PILImage
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

INDEX_DIR = Path("index_store/whoosh")
FAISS_PATH = Path("index_store/faiss.index")
META_PATH = Path("index_store/metadata.json")

# Two FAISS indexes:
#   - Text index (384-dim MiniLM) for books / audio / docs / text
#   - Visual index (512-dim CLIP) for images and video frames
FAISS_VISUAL_PATH = Path("index_store/faiss_visual.index")
VISUAL_META_PATH = Path("index_store/visual_metadata.json")

EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-dim, fast, widely used

SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    parent_id=ID(stored=True),
    chunk_index=NUMERIC(stored=True),
    path=STORED,
    doc_type=KEYWORD(stored=True),
    title=TEXT(stored=True),
    authors=TEXT(stored=True),
    categories=TEXT(stored=True),
    description=TEXT(stored=True),
    content=TEXT(stored=True),
)


def make_id(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def extract_audio(path: str) -> str:
    if not WHISPER_AVAILABLE:
        return f"[Audio: {Path(path).name}]"
    if not shutil.which("ffmpeg"):
        return f"[Audio: {Path(path).name}]"
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ─── CLIP helpers ─────────────────────────────────────────────────────────────

_CLIP_MODEL = None
_CLIP_PROCESSOR = None

def get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        if not CLIP_AVAILABLE:
            return None, None
        try:
            _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _CLIP_MODEL.eval()
        except Exception:
            _CLIP_MODEL = None
            _CLIP_PROCESSOR = None
    return _CLIP_MODEL, _CLIP_PROCESSOR


def embed_image_clip(pil_img):
    model, processor = get_clip()
    if model is None:
        return None
    try:
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        import numpy as np
        vec = features.squeeze().numpy().astype("float32")
        norm = float((vec ** 2).sum() ** 0.5)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    except Exception:
        return None


def embed_text_clip(text: str):
    model, processor = get_clip()
    if model is None:
        return None
    try:
        inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=77)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        import numpy as np
        vec = features.squeeze().numpy().astype("float32")
        norm = float((vec ** 2).sum() ** 0.5)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    except Exception:
        return None


# ─── OCR helper ───────────────────────────────────────────────────────────────

def ocr_image(pil_img) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(pil_img, timeout=15)
        return text.strip()
    except Exception:
        return ""


# ─── Unified image analysis ───────────────────────────────────────────────────

def analyse_image(path: str) -> dict:
    name = Path(path).stem.replace("_", " ").replace("-", " ")
    ext = Path(path).suffix.lower().lstrip(".")
    result = {"clip_vec": None, "ocr_text": "", "caption": f"{name} image {ext}"}

    try:
        img = PILImage.open(path).convert("RGB")
    except Exception:
        return result

    result["clip_vec"] = embed_image_clip(img)
    ocr_text = ocr_image(img)
    result["ocr_text"] = ocr_text

    combined = f"{name} {ocr_text}".strip()
    combined = " ".join(combined.split())
    result["caption"] = combined if combined else f"{name} image {ext}"
    return result


# ─── Video frame extraction ───────────────────────────────────────────────────

def extract_video_frames(path: str, every_n_seconds: int = 5) -> list[str]:
    if not CV2_AVAILABLE:
        return []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(fps * every_n_seconds)
    frames_dir = Path(path).parent / (Path(path).stem + "_frames")
    frames_dir.mkdir(exist_ok=True)
    saved = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out = frames_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(out), frame)
            saved.append(str(out))
            saved_idx += 1
        frame_idx += 1
    cap.release()
    return saved


# ─── Main indexer ─────────────────────────────────────────────────────────────

class DocumentIndexer:
    def __init__(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

        if index.exists_in(str(INDEX_DIR)):
            self.ix = index.open_dir(str(INDEX_DIR))
        else:
            self.ix = index.create_in(str(INDEX_DIR), SCHEMA)

        if FAISS_PATH.exists():
            self.faiss_index = faiss.read_index(str(FAISS_PATH))
        else:
            self.faiss_index = faiss.IndexFlatIP(384)

        if FAISS_VISUAL_PATH.exists():
            self.faiss_visual = faiss.read_index(str(FAISS_VISUAL_PATH))
        else:
            self.faiss_visual = faiss.IndexFlatIP(512)

        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        if VISUAL_META_PATH.exists():
            with open(VISUAL_META_PATH, "r", encoding="utf-8") as f:
                self.visual_metadata = json.load(f)
        else:
            self.visual_metadata = {}

        self.model = SentenceTransformer(EMBED_MODEL)
        self.faiss_ids = list(self.metadata.keys())
        self.visual_ids = list(self.visual_metadata.keys())

    # ── Audio ──────────────────────────────────────────────────────────────

    def index_audio_file(self, path: str) -> dict:
        ext = Path(path).suffix.lower()
        if ext not in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
            return {"status": "skipped", "reason": f"Unsupported audio format: {ext}"}

        parent_id = make_id(f"audio::{path}")
        try:
            content = extract_audio(path)
        except Exception as e:
            return {"status": "error", "reason": str(e)}

        title = Path(path).stem.replace("_", " ").replace("-", " ").title()
        chunks = chunk_text(content, chunk_size=1000, overlap=120)
        transcription_warning = None
        if not chunks or content.startswith("[Audio:"):
            chunks = [f"[Audio transcript unavailable] {title}"]
            transcription_warning = "Whisper or ffmpeg not available; file indexed by filename only."

        self._delete_parent(parent_id)
        for i, chunk in enumerate(chunks):
            did = make_id(f"{parent_id}::chunk::{i}")
            self._add_text_document(did=did, parent_id=parent_id, chunk_index=i, path=path,
                doc_type="audio", title=title, authors="", categories="podcast/audiobook",
                description="", content=chunk, extra={"snippet": chunk[:300]})

        self._save()
        result = {"status": "ok", "parent_id": parent_id, "type": "audio", "title": title, "chunks": len(chunks)}
        if transcription_warning:
            result["warning"] = transcription_warning
        return result

    # ── Books / Docs ───────────────────────────────────────────────────────

    def index_book_file(self, path: str, title: str | None = None, authors: str = "", category: str = "book") -> dict:
        book_path = Path(path)
        if not book_path.exists():
            return {"status": "error", "reason": "Book file not found"}
        ext = book_path.suffix.lower()
        if ext == ".txt":
            raw_text = book_path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            raw_text = extract_pdf_text(str(book_path))
        else:
            return {"status": "error", "reason": "Unsupported book format"}

        chunks = chunk_text(raw_text, chunk_size=1200, overlap=180)
        if not chunks:
            return {"status": "error", "reason": "No readable text found."}

        title = title or book_path.stem.replace("_", " ").replace("-", " ").title()
        parent_id = make_id(f"{category}::{path}")
        self._delete_parent(parent_id)

        for i, chunk in enumerate(chunks):
            did = make_id(f"{parent_id}::chunk::{i}")
            self._add_text_document(did=did, parent_id=parent_id, chunk_index=i, path=str(book_path),
                doc_type=category, title=title, authors=authors, categories=category,
                description="", content=chunk, extra={"snippet": chunk[:300], "source_ext": ext})

        self._save()
        return {"status": "ok", "parent_id": parent_id, "type": category, "title": title,
                "chunks": len(chunks), "source_ext": ext}

    # ── Images ─────────────────────────────────────────────────────────────

    def index_image_file(self, path: str) -> dict:
        """
        Index an image using CLIP (visual semantic search) + OCR (text in image).
        - CLIP: searching "certificate" matches images that look like certificates.
        - OCR: any text visible in the image is extracted and made searchable.
        Replaces the old ResNet50 approach that only used filename as caption.
        """
        ext = Path(path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}:
            return {"status": "skipped", "reason": f"Unsupported image format: {ext}"}

        parent_id = make_id(f"image::{path}")
        title = Path(path).stem.replace("_", " ").replace("-", " ").title()
        self._delete_parent_visual(parent_id)
        self._delete_parent(parent_id)

        analysis = analyse_image(path)
        caption = analysis["caption"]
        ocr_text = analysis["ocr_text"]
        clip_vec = analysis["clip_vec"]

        did = make_id(f"{parent_id}::chunk::0")

        # Whoosh + MiniLM: searchable by OCR text content
        searchable_content = f"{title} {caption} {ocr_text}".strip()
        self._add_text_document(did=did, parent_id=parent_id, chunk_index=0, path=path,
            doc_type="image", title=title, authors="", categories="image/photo",
            description=caption, content=searchable_content,
            extra={"snippet": (ocr_text[:300] if ocr_text else caption[:300]),
                   "source_ext": ext, "ocr_text": ocr_text[:500]})

        # Visual FAISS (CLIP): semantic visual search
        if clip_vec is not None:
            self._add_visual_document(did=did, parent_id=parent_id, chunk_index=0, path=path,
                doc_type="image", title=title, clip_vec=clip_vec,
                meta={"snippet": (ocr_text[:300] if ocr_text else caption[:300]),
                      "source_ext": ext, "ocr_text": ocr_text[:500]})

        self._save()
        return {"status": "ok", "parent_id": parent_id, "type": "image", "title": title,
                "chunks": 1, "source_ext": ext,
                "ocr_extracted": bool(ocr_text), "clip_indexed": clip_vec is not None}

    # ── Videos ─────────────────────────────────────────────────────────────

    def index_video_file(self, path: str, frame_every_sec: int = 5) -> dict:
        """
        Index a video using:
        1. Whisper audio transcription → text chunks (spoken words searchable)
        2. Frame CLIP embedding → visual FAISS (semantic visual search per frame)
        3. Frame OCR → Whoosh + MiniLM (on-screen text searchable)
        Replaces the old ResNet50 frame captions that contained zero visual semantics.
        """
        ext = Path(path).suffix.lower()
        if ext not in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}:
            return {"status": "skipped", "reason": f"Unsupported video format: {ext}"}

        parent_id = make_id(f"video::{path}")
        title = Path(path).stem.replace("_", " ").replace("-", " ").title()
        self._delete_parent(parent_id)
        self._delete_parent_visual(parent_id)

        total_chunks = 0

        # Strategy 1: Audio transcription
        audio_text = ""
        try:
            if WHISPER_AVAILABLE and shutil.which("ffmpeg"):
                model = whisper.load_model("base")
                result = model.transcribe(path)
                audio_text = result["text"]
        except Exception:
            audio_text = ""

        if audio_text:
            a_chunks = chunk_text(audio_text, chunk_size=1000, overlap=120)
            for i, chunk in enumerate(a_chunks):
                did = make_id(f"{parent_id}::audio::chunk::{i}")
                self._add_text_document(did=did, parent_id=parent_id, chunk_index=i, path=path,
                    doc_type="video", title=f"{title} [audio]", authors="",
                    categories="video/audio", description="", content=chunk,
                    extra={"snippet": chunk[:300], "source_type": "audio_track"})
                total_chunks += 1

        # Strategies 2 & 3: Frame CLIP + OCR
        frame_paths = []
        try:
            frame_paths = extract_video_frames(path, every_n_seconds=frame_every_sec)
        except Exception:
            frame_paths = []

        frames_with_clip = 0
        frames_with_ocr = 0

        for i, fp in enumerate(frame_paths):
            analysis = analyse_image(fp)
            caption = analysis["caption"]
            ocr_text = analysis["ocr_text"]
            clip_vec = analysis["clip_vec"]

            if clip_vec:
                frames_with_clip += 1
            if ocr_text:
                frames_with_ocr += 1

            did = make_id(f"{parent_id}::frame::{i}")

            # Whoosh + MiniLM: OCR text + caption
            searchable = f"{title} frame {i} {caption} {ocr_text}".strip()
            self._add_text_document(did=did, parent_id=parent_id, chunk_index=i, path=fp,
                doc_type="video", title=f"{title} [frame {i}]", authors="",
                categories="video/visual", description=caption, content=searchable,
                extra={"snippet": (ocr_text[:300] if ocr_text else caption[:300]),
                       "source_type": "video_frame", "frame_index": i,
                       "frame_path": fp, "video_path": path, "ocr_text": ocr_text[:500]})
            total_chunks += 1

            # Visual FAISS (CLIP)
            if clip_vec is not None:
                self._add_visual_document(did=did, parent_id=parent_id, chunk_index=i, path=fp,
                    doc_type="video", title=f"{title} [frame {i}]", clip_vec=clip_vec,
                    meta={"snippet": (ocr_text[:300] if ocr_text else caption[:300]),
                          "source_type": "video_frame", "frame_index": i,
                          "frame_path": fp, "video_path": path, "ocr_text": ocr_text[:500]})

        # Fallback
        if total_chunks == 0:
            fallback_text = f"{title} video file {ext}"
            did = make_id(f"{parent_id}::fallback")
            self._add_text_document(did=did, parent_id=parent_id, chunk_index=0, path=path,
                doc_type="video", title=title, authors="", categories="video",
                description=fallback_text, content=fallback_text,
                extra={"snippet": fallback_text, "source_type": "fallback"})
            total_chunks = 1

        self._save()
        return {"status": "ok", "parent_id": parent_id, "type": "video", "title": title,
                "chunks": total_chunks, "audio_transcribed": bool(audio_text),
                "frames_indexed": len(frame_paths), "frames_with_clip": frames_with_clip,
                "frames_with_ocr": frames_with_ocr, "source_ext": ext}

    # ── CSV bulk import ────────────────────────────────────────────────────

    def index_books_csv(self, csv_path: str) -> dict:
        count = 0
        errors = 0
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    title = row.get("title", "")
                    subtitle = row.get("subtitle", "")
                    authors = row.get("authors", "").replace(";", ", ")
                    categories = row.get("categories", "")
                    description = row.get("description", "")
                    thumbnail = row.get("thumbnail", "")
                    year = row.get("published_year", "")
                    rating = row.get("average_rating", "")
                    pages = row.get("num_pages", "")
                    full_title = f"{title} {subtitle}".strip()
                    did = make_id("meta::" + row.get("isbn13", full_title))
                    content = f"{full_title} {authors} {categories} {description}"
                    self._add_text_document(did=did, parent_id=did, chunk_index=0, path=thumbnail,
                        doc_type="book_meta", title=full_title, authors=authors,
                        categories=categories, description=description, content=content,
                        extra={"thumbnail": thumbnail, "year": year, "rating": rating,
                               "pages": pages, "snippet": description[:300]})
                    count += 1
                except Exception:
                    errors += 1
        self._save()
        return {"indexed": count, "errors": errors}

    # ── Internal: text indexes ─────────────────────────────────────────────

    def _add_text_document(self, did, parent_id, chunk_index, path, doc_type, title,
                           authors, categories, description, content, extra=None):
        writer = self.ix.writer()
        writer.update_document(doc_id=did, parent_id=parent_id, chunk_index=chunk_index,
            path=str(path), doc_type=doc_type, title=title, authors=authors,
            categories=categories, description=description, content=content)
        writer.commit()

        text_to_embed = f"{title} {content}"[:2000]
        vec = self.model.encode([text_to_embed], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        self.faiss_index.add(vec)

        self.metadata[did] = {
            "parent_id": parent_id, "chunk_index": chunk_index, "path": str(path),
            "type": doc_type, "title": title, "authors": authors, "categories": categories,
            "description": description[:300] if description else "",
            "snippet": content[:300], **(extra or {}),
        }
        if did not in self.faiss_ids:
            self.faiss_ids.append(did)

    # ── Internal: visual FAISS ─────────────────────────────────────────────

    def _add_visual_document(self, did, parent_id, chunk_index, path, doc_type, title, clip_vec, meta=None):
        import numpy as np
        vec = np.array([clip_vec], dtype="float32")
        faiss.normalize_L2(vec)  # required for IndexFlatIP to compute true cosine similarity
        self.faiss_visual.add(vec)

        self.visual_metadata[did] = {
            "parent_id": parent_id, "chunk_index": chunk_index, "path": str(path),
            "type": doc_type, "title": title, "authors": "", "categories": "image/photo",
            "description": "", "snippet": (meta or {}).get("snippet", ""), **(meta or {}),
        }
        if did not in self.visual_ids:
            self.visual_ids.append(did)

    # ── Search ─────────────────────────────────────────────────────────────

    def search_classic(self, query: str, top_k: int = 10) -> list:
        results = []
        with self.ix.searcher() as searcher:
            parser = MultifieldParser(
                ["title", "authors", "categories", "description", "content"],
                schema=self.ix.schema,
                fieldboosts={"title": 2.5, "authors": 1.8, "categories": 1.2,
                             "description": 1.0, "content": 2.0},
            )
            hits = searcher.search(parser.parse(query), limit=top_k)
            for hit in hits:
                meta = self.metadata.get(hit["doc_id"], {})
                results.append(self._format(hit["doc_id"], hit.score, "classic", meta, hit))
        return results

    def search_ai(self, query: str, top_k: int = 10) -> list:
        results = []
        TEXT_SCORE_THRESHOLD = 0.30  # cosine similarity for MiniLM text embeddings

        # MiniLM text FAISS
        if self.faiss_index.ntotal > 0:
            vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(vec)
            distances, indices = self.faiss_index.search(vec, top_k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.faiss_ids):
                    continue
                score = float(score)
                if score < TEXT_SCORE_THRESHOLD:
                    continue  # skip poor matches
                did = self.faiss_ids[idx]
                meta = self.metadata.get(did, {})
                results.append(self._format(did, score, "ai", meta))

        # CLIP visual FAISS
        results.extend(self.search_visual(query, top_k))
        return results

    def search_visual(self, query: str, top_k: int = 10) -> list:
        """
        Encode query text with CLIP and search the visual FAISS index.
        "certificate" matches images that look like certificates.
        "man holding trophy" matches video frames showing that scene.
        A minimum cosine similarity threshold of 0.20 is applied so that
        unrelated or nonsense queries do not return false positives.
        """
        if self.faiss_visual.ntotal == 0:
            return []
        clip_vec = embed_text_clip(query)
        if clip_vec is None:
            return []
        import numpy as np
        vec = np.array([clip_vec], dtype="float32")
        faiss.normalize_L2(vec)  # normalize query for true cosine similarity
        distances, indices = self.faiss_visual.search(vec, top_k)
        VISUAL_SCORE_THRESHOLD = 0.20  # cosine similarity; range [-1, 1]
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.visual_ids):
                continue
            score = float(score)
            if score < VISUAL_SCORE_THRESHOLD:
                continue  # skip results with poor similarity (e.g. nonsense queries)
            did = self.visual_ids[idx]
            meta = self.visual_metadata.get(did, {})
            results.append(self._format(did, score, "visual_clip", meta))
        return results

    def _format(self, did, score, method, meta, hit=None):
        return {
            "doc_id": did, "parent_id": meta.get("parent_id", did),
            "chunk_index": meta.get("chunk_index", 0), "score": round(score, 4),
            "method": method, "path": meta.get("path", ""), "type": meta.get("type", ""),
            "title": meta.get("title", hit["title"] if hit else ""),
            "authors": meta.get("authors", ""), "categories": meta.get("categories", ""),
            "description": meta.get("description", ""), "snippet": meta.get("snippet", ""),
            "thumbnail": meta.get("thumbnail", ""), "year": meta.get("year", ""),
            "rating": meta.get("rating", ""), "pages": meta.get("pages", ""),
            "source_ext": meta.get("source_ext", ""), "frame_path": meta.get("frame_path", ""),
            "source_type": meta.get("source_type", ""), "ocr_text": meta.get("ocr_text", ""),
        }

    def _delete_parent(self, parent_id: str):
        to_delete = [did for did, meta in self.metadata.items() if meta.get("parent_id") == parent_id]
        if not to_delete:
            return
        writer = self.ix.writer()
        for did in to_delete:
            writer.delete_by_term("doc_id", did)
        writer.commit()
        for did in to_delete:
            if did in self.faiss_ids:
                self.faiss_ids.remove(did)
            self.metadata.pop(did, None)
        self._rebuild_faiss()

    def _delete_parent_visual(self, parent_id: str):
        to_delete = [did for did, meta in self.visual_metadata.items() if meta.get("parent_id") == parent_id]
        if not to_delete:
            return
        for did in to_delete:
            if did in self.visual_ids:
                self.visual_ids.remove(did)
            self.visual_metadata.pop(did, None)
        self._rebuild_visual_faiss()

    def delete_document(self, did: str) -> dict:
        # Accept either a chunk doc_id or a parent_id directly.
        if did not in self.metadata and did not in self.visual_metadata:
            # Try treating did as a parent_id
            match = next(
                (m for m in list(self.metadata.values()) + list(self.visual_metadata.values())
                 if m.get("parent_id") == did),
                None,
            )
            if match is None:
                return {"success": False, "reason": "Document not found"}
            parent_id = did
            title = match.get("title", did)
        else:
            parent_id = (self.metadata.get(did) or self.visual_metadata.get(did) or {}).get("parent_id", did)
            title = (self.metadata.get(did) or self.visual_metadata.get(did) or {}).get("title", did)
        self._delete_parent(parent_id)
        self._delete_parent_visual(parent_id)
        self._save()
        return {"success": True, "deleted": parent_id, "title": title}

    def _rebuild_faiss(self):
        remaining_ids = list(self.metadata.keys())
        self.faiss_ids = remaining_ids
        if not remaining_ids:
            self.faiss_index = faiss.IndexFlatIP(384)
            return
        texts = [f"{self.metadata[d].get('title', '')} {self.metadata[d].get('snippet', '')}"[:2000]
                 for d in remaining_ids]
        vecs = self.model.encode(texts, convert_to_numpy=True, batch_size=64,
                                 show_progress_bar=False).astype("float32")
        faiss.normalize_L2(vecs)
        new_index = faiss.IndexFlatIP(384)
        new_index.add(vecs)
        self.faiss_index = new_index

    def _rebuild_visual_faiss(self):
        # CLIP vectors are not persisted to disk separately from the FAISS index file.
        # On deletion we reset the visual index fully; affected images/videos must be
        # re-uploaded to restore their visual embeddings. This is a known limitation.
        self.faiss_visual = faiss.IndexFlatIP(512)
        self.visual_ids = []
        self.visual_metadata = {}

    def _save(self):
        faiss.write_index(self.faiss_index, str(FAISS_PATH))
        faiss.write_index(self.faiss_visual, str(FAISS_VISUAL_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        with open(VISUAL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.visual_metadata, f, ensure_ascii=False, indent=2)

    def stats(self) -> dict:
        type_chunks = {}
        type_documents = {"book": 0, "doc": 0, "text": 0, "pdf": 0,
                          "audio": 0, "image": 0, "video": 0, "book_meta": 0}
        seen_books = set()
        seen_docs = set()
        seen_audio = set()
        seen_images = set()
        seen_videos = set()

        for v in self.metadata.values():
            t = v.get("type", "unknown")
            parent_id = v.get("parent_id")
            source_ext = (v.get("source_ext") or "").lower()
            type_chunks[t] = type_chunks.get(t, 0) + 1
            if t == "book": seen_books.add((parent_id, source_ext))
            elif t == "doc": seen_docs.add((parent_id, source_ext))
            elif t == "audio": seen_audio.add(parent_id)
            elif t == "image": seen_images.add(parent_id)
            elif t == "video": seen_videos.add(parent_id)
            elif t == "book_meta": type_documents["book_meta"] += 1

        type_documents["book"] = len(seen_books)
        type_documents["doc"] = len(seen_docs)
        type_documents["audio"] = len(seen_audio)
        type_documents["image"] = len(seen_images)
        type_documents["video"] = len(seen_videos)

        for _, source_ext in list(seen_books) + list(seen_docs):
            if source_ext == ".txt": type_documents["text"] += 1
            elif source_ext == ".pdf": type_documents["pdf"] += 1

        return {
            "total_chunks": len(self.metadata),
            "total_documents": (type_documents["book"] + type_documents["doc"] +
                                type_documents["audio"] + type_documents["image"] +
                                type_documents["video"] + type_documents["book_meta"]),
            "by_type": type_chunks,
            "by_type_documents": type_documents,
            "faiss_docs": self.faiss_index.ntotal,
            "faiss_visual_docs": self.faiss_visual.ntotal,
            "clip_available": CLIP_AVAILABLE,
            "ocr_available": OCR_AVAILABLE,
        }
