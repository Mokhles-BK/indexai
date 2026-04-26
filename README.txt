IndexAI — Multimedia Search Engine
======================================

Supports: Books (.txt, .pdf), Documents (.txt, .pdf),
          Audio (.mp3, .wav, .m4a, .ogg, .flac),
          Images (.jpg, .jpeg, .png, .bmp, .webp, .gif, .tiff),
          Videos (.mp4, .avi, .mkv, .mov, .webm, .flv)

Run:
  pip install -r requirements.txt
  python -m uvicorn app:app --reload

External dependencies (must be installed separately):
  - ffmpeg       → required for Whisper (audio/video transcription)
      Ubuntu/Debian: sudo apt-get install ffmpeg
      macOS:         brew install ffmpeg
  - tesseract    → required for OCR (text extraction from images/video frames)
      Ubuntu/Debian: sudo apt-get install tesseract-ocr
      macOS:         brew install tesseract

Upload routes:
  POST /upload-book    → indexes as category="book"
  POST /upload-doc     → indexes as category="doc"
  POST /upload-audio   → transcribes with Whisper, then indexes
  POST /upload-image   → OCR (visible text) + CLIP visual embedding, then indexes
  POST /upload-video   → Whisper audio transcription + frame OCR + CLIP, then indexes

Search:
  GET /search?q=<query>&method=classic|ai|both&top_k=10

Delete:
  DELETE /delete/{doc_id}   → removes the document and all its chunks from all indexes

File access:
  GET /file/{doc_id}        → download the original uploaded file for any search result
                              (accepts both chunk doc_id and parent_id)

Stats:
  GET /stats   → returns counts of indexed documents and chunks by type

Architecture:
  - Classical search : Whoosh (TF-IDF keyword index over title, authors, categories,
                       description, and content fields)
  - AI text search   : FAISS (384-dim MiniLM / all-MiniLM-L6-v2 sentence embeddings)
  - AI visual search : FAISS (512-dim CLIP embeddings — openai/clip-vit-base-patch32)
                       Allows text queries like "certificate" to match images that
                       visually look like certificates, without any text in them.
  - Hybrid search    : both indexes queried and results merged by best score

Content processing pipeline:
  Text/PDF  → direct text extraction → chunked (1200 chars, 180 overlap) → Whoosh + FAISS
  Audio     → Whisper transcription → chunked → Whoosh + FAISS
  Images    → OCR (pytesseract) + CLIP visual embedding → Whoosh + visual FAISS
  Videos    → Whisper (audio track) + per-frame OCR + per-frame CLIP → Whoosh + both FAISS

Notes:
  - PDF support works for text-based PDFs only (not scanned/image-only PDFs).
  - If Whisper or ffmpeg is unavailable, audio/video transcription is skipped
    gracefully; files are still indexed using filename metadata.
  - If pytesseract/tesseract is unavailable, OCR is skipped gracefully.
  - If CLIP (transformers/torch) is unavailable, visual FAISS search is disabled
    gracefully; text-based search still works normally.
  - Deleting a document rebuilds the text FAISS index. The visual FAISS index is
    also rebuilt on deletion (CLIP vectors are not persisted separately between sessions).
  - For large collections, consider replacing IndexFlatIP with IndexIVFFlat for
    faster approximate nearest-neighbour search.
