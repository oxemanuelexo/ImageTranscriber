import os
import io
import base64
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError, ImageSequence
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import AsyncOpenAI, APIStatusError

# -----------------------------
# App setup
# -----------------------------
load_dotenv()
st.set_page_config(page_title="Image Transcriber · GPT‑5 (No OCR)", layout="wide")
st.title("🖼️→📝 Image Transcriber (GPT‑5, no OCR)")

st.markdown(
    """
This app sends your images (or PDF pages) to **GPT‑5** to transcribe visible text.
It **does not** use OCR libraries.

- Multiple non‑PDF images: **drag the previews** to set order; the final transcript follows that order.  
- PDFs: pages keep the document order (reordering disabled).  
- Exactly **one** image/page is sent per request; requests run **in parallel**.
"""
)

# --- Sidebar: API, model, options ---
api_env = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input(
    "OpenAI API Key (optional if set via .env)",
    value=api_env,
    type="password",
    help='Set OPENAI_API_KEY in a ".env" file or paste your key here for this session.',
)

st.sidebar.subheader("Model & Settings")
model = st.sidebar.text_input("Model", value="gpt-5", help="Use GPT‑5 (with optional Code Interpreter).")
use_ci = st.sidebar.checkbox("Use Code Interpreter", value=True)
max_concurrency = st.sidebar.slider("Parallel requests", 1, 8, 4)
dpi = st.sidebar.slider("PDF render DPI", 120, 300, 200, 20)
st.sidebar.caption("Higher DPI ⇒ sharper PDF pages ⇒ better transcription (slower).")

# --- Reset / Clear ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

def reset_all():
    for k in ("items", "contains_pdf", "uploaded_signature"):
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] += 1

if st.sidebar.button("🔁 Reset / Clear files"):
    reset_all()
    st.rerun()

# --- Session state containers ---
if "items" not in st.session_state:
    st.session_state["items"] = []  # list of dicts: {uid, bytes, mime, label, from_pdf, thumb_b64}
if "contains_pdf" not in st.session_state:
    st.session_state["contains_pdf"] = False


# -----------------------------
# Helpers
# -----------------------------
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif", "pdf"]

def _ensure_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L", "RGBA"):
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return buf.getvalue()

def _make_thumb_bytes(png_bytes: bytes, max_side: int = 160) -> bytes:
    img = Image.open(io.BytesIO(png_bytes))
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_side, max_side))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def _read_single_image(name: str, bytes_data: bytes) -> Tuple[bytes, str, str]:
    try:
        img = Image.open(io.BytesIO(bytes_data))
        # for animated tiff/gif, take first frame
        if getattr(img, "is_animated", False):
            img = ImageSequence.Iterator(img).__next__()
        img = ImageOps.exif_transpose(img)
        png_bytes = _ensure_png(img)
        return png_bytes, "image/png", name
    except UnidentifiedImageError:
        raise
    except Exception:
        raise

def _pdf_to_png_pages(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in doc:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def prepare_items(uploaded_files) -> Tuple[List[Dict], bool]:
    """
    Create items:
      - images: one item each
      - PDFs: one item per page (in order)
    Each item gets a base64 thumbnail for the drag UI.
    """
    items = []
    contains_pdf = False
    for f in uploaded_files:
        name = f.name
        raw = f.read()
        ext = name.split(".")[-1].lower()

        if ext == "pdf":
            contains_pdf = True
            pages = _pdf_to_png_pages(raw, dpi=dpi)
            for i, pbytes in enumerate(pages, start=1):
                thumb = _make_thumb_bytes(pbytes)
                items.append({
                    "uid": uuid.uuid4().hex[:8],
                    "bytes": pbytes,
                    "mime": "image/png",
                    "label": f"{name} - page {i}",
                    "from_pdf": True,
                    "thumb_b64": _b64(thumb),
                })
        else:
            try:
                png_bytes, mime, label = _read_single_image(name, raw)
                thumb = _make_thumb_bytes(png_bytes)
                items.append({
                    "uid": uuid.uuid4().hex[:8],
                    "bytes": png_bytes,
                    "mime": mime,
                    "label": label,
                    "from_pdf": False,
                    "thumb_b64": _b64(thumb),
                })
            except Exception:
                st.warning(f"Couldn't read file: {name}. Unsupported or corrupted.", icon="⚠️")
    return items, contains_pdf

def to_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_b64(image_bytes)}"

def build_instructions(use_code_interpreter: bool) -> str:
    base = (
        "You are a meticulous transcriptionist. "
        "Your job is to transcribe *only* the visible text in an image, preserving reading order and line breaks. "
        "Do not add commentary, labels, or explanations. "
        "Do not describe graphics. Output must be just the transcription text. "
        "If no text is clearly legible, output exactly: [no text]. "
    )
    ci = (
        "You may use the code interpreter to crop, rotate, zoom, or otherwise enhance the image for readability, "
        "but you must not use OCR libraries or external tools."
    )
    return base + (ci if use_code_interpreter else "")

TRANSCRIBE_PROMPT = (
    "Transcribe all visible text from this image as plain UTF‑8 text. "
    "Preserve line breaks and layout where it helps readability. "
    "Do not add any extra words before or after. "
    "If text is partially unreadable, use the character “?” for ambiguous glyphs."
)

async def transcribe_one(client: AsyncOpenAI, item: Dict, model: str, use_code_interpreter: bool) -> str:
    data_url = to_data_url(item["bytes"], item["mime"])
    kwargs = dict(
        model=model,
        instructions=build_instructions(use_code_interpreter),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": TRANSCRIBE_PROMPT},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    if use_code_interpreter:
        kwargs["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]

    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()

async def transcribe_all(items: List[Dict], api_key: str, model: str, concurrency: int, use_code_interpreter: bool) -> List[str]:
    sem = asyncio.Semaphore(concurrency)
    async with AsyncOpenAI(api_key=api_key) as client:
        async def bound_call(it: Dict) -> str:
            async with sem:
                try:
                    return await transcribe_one(client, it, model, use_code_interpreter)
                except APIStatusError as e:
                    code = getattr(e, "status_code", "API")
                    return f"[error {code}] {getattr(e, 'message', '') or 'API error'}"
                except Exception as e:
                    return f"[error] {str(e)}"
        tasks = [asyncio.create_task(bound_call(it)) for it in items]
        return await asyncio.gather(*tasks)

def show_thumbnails(items: List[Dict]):
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            st.image(it["bytes"], caption=it["label"], use_container_width=True)

# -----------------------------
# Custom drag component (no extra pip)
# -----------------------------
# Expect a sibling folder "drag_reorder_component" with index.html
COMP_PATH = Path(__file__).parent / "drag_reorder_component"
drag_reorder = components.declare_component("drag_reorder", path=str(COMP_PATH))

def drag_reorder_ui():
    """
    Render a draggable list of image cards (custom HTML5 DnD).
    Returns nothing; updates st.session_state['items'] when a new order arrives.
    """
    items = st.session_state["items"]
    payload = [
        {
            "uid": it["uid"],
            "label": it["label"],
            "thumb": f"data:image/png;base64,{it['thumb_b64']}",
        }
        for it in items
    ]
    # Ask component to render; it returns a list of UIDs in the current order
    new_order = drag_reorder(
        items=payload,
        key="drag_reorder",
        default=[p["uid"] for p in payload],
        height= max(140, min(560, 110 * len(payload))),  # adaptive height
    )
    if new_order and len(new_order) == len(items):
        uid_to_item = {it["uid"]: it for it in items}
        st.session_state["items"] = [uid_to_item[u] for u in new_order if u in uid_to_item]


# -----------------------------
# Upload & preparation
# -----------------------------
uploaded = st.file_uploader(
    "Upload images or PDFs",
    key=f"uploader_{st.session_state['uploader_key']}",
    type=ACCEPTED_TYPES,
    accept_multiple_files=True,
    help="You can upload multiple images (PNG, JPG, WebP, TIFF, BMP, GIF) or PDFs.",
)

# Only (re)prepare when files actually changed (prevents wiping custom order on rerun)
if uploaded:
    signature = tuple((f.name, getattr(f, "size", None)) for f in uploaded)
    if st.session_state.get("uploaded_signature") != signature:
        items, contains_pdf = prepare_items(uploaded)
        st.session_state["items"] = items
        st.session_state["contains_pdf"] = contains_pdf
        st.session_state["uploaded_signature"] = signature

items = st.session_state["items"]
contains_pdf = st.session_state["contains_pdf"]

if items:
    st.subheader("Preview")
    show_thumbnails(items)

    if contains_pdf:
        st.info("PDF detected. Page order is fixed to the document’s order. Reordering is disabled.", icon="📄")
    else:
        st.subheader("Reorder by dragging the previews")
        # Helpful hint if the static component is missing
        if not COMP_PATH.exists():
            st.error("drag_reorder_component/ not found. Create it next to app.py (see instructions below).")
        else:
            drag_reorder_ui()

    st.divider()

    can_run = bool(api_key or api_env)
    if st.button("Transcribe with GPT‑5", disabled=not can_run, help=None if can_run else "Add your OpenAI API key first."):
        with st.status("Transcribing… running parallel requests", expanded=True) as status:
            # Run concurrently; one image/page per request
            try:
                results = asyncio.run(
                    transcribe_all(st.session_state["items"], api_key or api_env, model, max_concurrency, use_ci)
                )
            except RuntimeError:
                # Fallback if an event loop is already running
                loop = asyncio.new_event_loop()
                results = loop.run_until_complete(
                    transcribe_all(st.session_state["items"], api_key or api_env, model, max_concurrency, use_ci)
                )
                loop.close()

            st.write("Per‑image results")
            for idx, (it, text) in enumerate(zip(st.session_state["items"], results)):
                with st.expander(it["label"], expanded=False):
                    st.text_area("Transcription", value=text, height=200, key=f"transcription_{idx}")

            final_text = "\n\n".join(results).strip()

            st.subheader("Combined transcription")
            st.text_area("All pages/images combined", value=final_text, height=320, key="combined_text")

            st.download_button(
                "Download as .txt",
                data=final_text.encode("utf-8"),
                file_name="transcription.txt",
                mime="text/plain",
                key="download_txt",
            )

            status.update(label="Done!", state="complete", expanded=False)
else:
    st.caption("Upload images or PDFs to begin.")

st.markdown(
    """
---
**Privacy**: Files are sent to OpenAI only for transcription; no OCR libraries are used locally.  
**Drag reorder**: provided by a tiny custom HTML5 component (no extra pip packages).  
"""
)
