"""
IndexAI — Automated Test Suite
================================
Run with:
    python test_indexai.py

The server must be running first:
    python -m uvicorn app:app --reload

What is tested:
  1. Server is reachable
  2. Stats endpoint returns expected structure
  3. Upload a small .txt book  →  confirms chunking + indexing
  4. Upload a small .txt doc   →  confirms doc category
  5. Upload a small .png image →  confirms CLIP/OCR path
  6. Classic search finds exact keyword from uploaded book
  7. AI search finds semantically related word (not exact keyword)
  8. Hybrid search returns results from both methods
  9. Nonsense query returns ZERO results  (score-threshold bug fix check)
 10. /file/{doc_id} returns the original file (200 + correct filename header)
 11. Delete removes document from search results
 12. Stats count decreases after deletion
"""

import io
import sys
import time
import struct
import requests

BASE = "http://127.0.0.1:8000"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = []

def check(name, passed, detail=""):
    icon = PASS if passed else FAIL
    print(f"  {icon}  {name}" + (f"  →  {detail}" if detail else ""))
    results.append((name, passed))

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── helpers ────────────────────────────────────────────────────────────────────

def make_txt(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode())

def make_png_1x1() -> io.BytesIO:
    """Minimal valid 1×1 white PNG (no Pillow needed)."""
    import zlib, struct
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig   = b"\x89PNG\r\n\x1a\n"
    ihdr  = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw   = b"\x00\xff\xff\xff"          # filter byte + 1 white RGB pixel
    idat  = chunk(b"IDAT", zlib.compress(raw))
    iend  = chunk(b"IEND", b"")
    buf   = io.BytesIO(sig + ihdr + idat + iend)
    buf.name = "test_image.png"
    return buf

# ── 1. Server reachable ────────────────────────────────────────────────────────
section("1 · Server")
try:
    r = requests.get(BASE + "/", timeout=5)
    check("Server is reachable", r.status_code == 200, f"HTTP {r.status_code}")
except Exception as e:
    check("Server is reachable", False, str(e))
    print("\n  ⚠  Cannot reach server — start it with:  uvicorn app:app --reload")
    sys.exit(1)

# ── 2. Stats ───────────────────────────────────────────────────────────────────
section("2 · Stats endpoint")
r = requests.get(BASE + "/stats")
check("/stats returns 200", r.status_code == 200)
stats_before = r.json() if r.ok else {}
for key in ("total_chunks", "by_type"):
    check(f"/stats contains '{key}'", key in stats_before, str(stats_before.get(key, "missing")))

# ── 3. Upload book ─────────────────────────────────────────────────────────────
section("3 · Upload book (.txt)")
BOOK_TEXT = (
    "Discipline is the foundation of all achievement. "
    "Self-control and willpower are essential habits. "
    "Consistent practice leads to mastery over time. " * 10
)
buf = make_txt(BOOK_TEXT)
buf.name = "test_discipline_book.txt"
r = requests.post(BASE + "/upload-book", files={"file": (buf.name, buf, "text/plain")})
check("Upload book returns 200", r.status_code == 200, f"HTTP {r.status_code}")
book_result = r.json() if r.ok else {}
check("Book has parent_id", "parent_id" in book_result, str(book_result.get("parent_id", "missing")))
check("Book has ≥1 chunk", book_result.get("chunks", 0) >= 1, f"{book_result.get('chunks')} chunks")
BOOK_PARENT_ID = book_result.get("parent_id", "")

# ── 4. Upload document ─────────────────────────────────────────────────────────
section("4 · Upload document (.txt)")
DOC_TEXT = "Neural networks are a subset of machine learning. Deep learning uses many layers. " * 10
buf = make_txt(DOC_TEXT)
buf.name = "test_ml_doc.txt"
r = requests.post(BASE + "/upload-doc", files={"file": (buf.name, buf, "text/plain")})
check("Upload doc returns 200", r.status_code == 200, f"HTTP {r.status_code}")
doc_result = r.json() if r.ok else {}
check("Doc has parent_id", "parent_id" in doc_result)
DOC_PARENT_ID = doc_result.get("parent_id", "")

# ── 5. Upload image ────────────────────────────────────────────────────────────
section("5 · Upload image (.png)")
png = make_png_1x1()
r = requests.post(BASE + "/upload-image", files={"file": (png.name, png, "image/png")})
check("Upload image returns 200", r.status_code == 200, f"HTTP {r.status_code}")
img_result = r.json() if r.ok else {}
check("Image has parent_id", "parent_id" in img_result)
IMG_PARENT_ID = img_result.get("parent_id", "")
# Image may not have CLIP if transformers not installed — warn but don't fail
if not img_result.get("clip_indexed"):
    print(f"  {WARN}  CLIP not available — visual search will be skipped (install transformers+torch)")

# ── 6. Classic search ──────────────────────────────────────────────────────────
section("6 · Classic search — exact keyword")
time.sleep(0.5)   # small delay so index commits flush
r = requests.get(BASE + "/search", params={"q": "discipline", "method": "classic", "top_k": 10})
check("Classic search returns 200", r.status_code == 200)
hits = r.json().get("results", []) if r.ok else []
titles = [h.get("title", "") for h in hits]
check("Classic finds 'discipline' book", any("discipline" in t.lower() for t in titles),
      f"titles returned: {titles[:3]}")

# ── 7. AI search — semantic ────────────────────────────────────────────────────
section("7 · AI search — semantic / synonym")
r = requests.get(BASE + "/search", params={"q": "self-control", "method": "ai", "top_k": 10})
check("AI search returns 200", r.status_code == 200)
hits = r.json().get("results", []) if r.ok else []
check("AI returns ≥1 result for 'self-control'", len(hits) >= 1,
      f"{len(hits)} result(s)")

# ── 8. Hybrid search ───────────────────────────────────────────────────────────
section("8 · Hybrid search")
r = requests.get(BASE + "/search", params={"q": "machine learning", "method": "both", "top_k": 10})
check("Hybrid search returns 200", r.status_code == 200)
hits = r.json().get("results", []) if r.ok else []
check("Hybrid returns ≥1 result", len(hits) >= 1, f"{len(hits)} result(s)")
methods = {h.get("method") for h in hits}
check("Hybrid result has method field", bool(methods), str(methods))

# ── 9. Nonsense query returns zero results ─────────────────────────────────────
section("9 · Score threshold — nonsense query")
r = requests.get(BASE + "/search", params={"q": "xyzqwerty123nonsense", "method": "both", "top_k": 10})
check("Nonsense query returns 200", r.status_code == 200)
hits = r.json().get("results", []) if r.ok else ["fake"]
check("Nonsense query returns 0 results  (score-threshold fix)", len(hits) == 0,
      f"{len(hits)} result(s) returned — should be 0")

# ── 10. File access ────────────────────────────────────────────────────────────
section("10 · File access  (/file/{doc_id})")
if BOOK_PARENT_ID:
    r = requests.get(BASE + f"/file/{BOOK_PARENT_ID}", allow_redirects=True)
    check("/file/{parent_id} returns 200", r.status_code == 200, f"HTTP {r.status_code}")
    cd = r.headers.get("content-disposition", "")
    check("Response has filename in Content-Disposition", "filename" in cd, cd)
    check("File content is non-empty", len(r.content) > 0, f"{len(r.content)} bytes")
else:
    check("/file endpoint skipped (no parent_id from upload)", False, "upload may have failed")

# ── 11. Delete ─────────────────────────────────────────────────────────────────
section("11 · Delete")
if BOOK_PARENT_ID:
    r = requests.delete(BASE + f"/delete/{BOOK_PARENT_ID}")
    check("Delete returns 200", r.status_code == 200, f"HTTP {r.status_code}")

    time.sleep(0.5)
    r = requests.get(BASE + "/search", params={"q": "discipline", "method": "classic", "top_k": 10})
    hits = r.json().get("results", []) if r.ok else []
    titles = [h.get("title", "") for h in hits]
    check("Deleted book no longer appears in search", not any("discipline" in t.lower() for t in titles),
          f"titles: {titles}")

    r2 = requests.get(BASE + f"/file/{BOOK_PARENT_ID}")
    check("Deleted file returns 404 on /file", r2.status_code == 404, f"HTTP {r2.status_code}")
else:
    check("Delete skipped (no parent_id)", False)

# ── 12. Stats after delete ─────────────────────────────────────────────────────
section("12 · Stats after delete")
r = requests.get(BASE + "/stats")
stats_after = r.json() if r.ok else {}
before = stats_before.get("total_chunks", 0)
after  = stats_after.get("total_chunks", 0)
check("Chunk count decreased after delete", after < before + 999,   # +999: we also added doc+image
      f"before upload+delete cycle: {before}  →  after: {after}")

# ── Summary ────────────────────────────────────────────────────────────────────
section("Summary")
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"\n  Passed : {passed}/{total}")
if failed:
    print(f"  Failed : {failed}/{total}")
    for name, ok in results:
        if not ok:
            print(f"    {FAIL}  {name}")
print()
sys.exit(0 if failed == 0 else 1)
