import os
import io
import uuid
import base64
import asyncio
import warnings
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageSequence
from dotenv import load_dotenv

import fitz  # PyMuPDF
from openai import AsyncOpenAI, APIStatusError

# Drag & drop UI (MUI + draggable grid)
from streamlit_elements import elements, dashboard, mui, sync

# Silence a harmless SyntaxWarning in streamlit-elements on Python 3.13
warnings.filterwarnings("ignore", category=SyntaxWarning)


# =========================
# App setup
# =========================
load_dotenv()
st.set_page_config(page_title="Image Transcriber · GPT‑5 (No OCR)", layout="wide")
st.title("🖼️→📝 Image Transcriber (GPT‑5, no OCR)")

st.markdown(
    """
This app sends your images (or PDF pages) to **GPT‑5** to transcribe visible text.
It **does not** use OCR libraries.

**Notes**
- Multiple non‑PDF images: **drag the previews** to set order; the final transcript follows that order.  
- PDFs: pages are kept in document order (reordering disabled).  
- Exactly **one** image/page is sent to GPT‑5 per request; all requests run **in parallel**.
"""
)

# --- Sidebar: API, model, options ---
api_env = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input(
    "OpenAI API Key (optional if set via .env)",
    value=api_env,
    type="password",
    help='Add OPENAI_API_KEY to a ".env" file or paste your key here for this session.',
)
st.sidebar.subheader("Model & Settings")
model = st.sidebar.text_input("Model", value="gpt-5", help="Use GPT‑5.")
use_ci = st.sidebar.checkbox("Use Code Interpreter", value=True)
max_concurrency = st.sidebar.slider("Parallel requests", 1, 8, 4)
dpi = st.sidebar.slider("PDF render DPI", 120, 300, 200, 20)
st.sidebar.caption("Higher DPI ⇒ sharper PDF pages ⇒ better transcription (slower).")

# --- Reset / Clear ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

def reset_all():
    for k in ("items", "contains_pdf", "uploaded_signature", "reorder_layout"):
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] += 1

if st.sidebar.button("🔁 Reset / Clear files"):
    reset_all()
    st.rerun()

# --- Session state ---
if "items" not in st.session_state:
    st.session_state["items"] = []  # [{uid, bytes, mime, label, from_pdf, thumb_b64}]
if "contains_pdf" not in st.session_state:
    st.session_state["contains_pdf"] = False


# =========================
# Helpers
# =========================
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif", "pdf"]

def _ensure_png(image: Image.Image) -> bytes:
    """Convert PIL image to PNG bytes."""
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L", "RGBA"):
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return buf.getvalue()

def _make_thumb_bytes(png_bytes: bytes, max_side: int = 160) -> bytes:
    """Create a small PNG thumbnail for the drag UI."""
    img = Image.open(io.BytesIO(png_bytes))
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_side, max_side))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def _read_single_image(name: str, bytes_data: bytes) -> Tuple[bytes, str, str]:
    """Read any supported image and return (png_bytes, mime, label)."""
    try:
        img = Image.open(io.BytesIO(bytes_data))
        # If multi-frame TIFF, take first frame (simple handling)
        if getattr(img, "is_animated", False) and img.format == "TIFF":
            img = ImageSequence.Iterator(img).__next__()
        img = ImageOps.exif_transpose(img)
        png_bytes = _ensure_png(img)
        return png_bytes, "image/png", name
    except UnidentifiedImageError:
        raise
    except Exception:
        raise

def _pdf_to_png_pages(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    """Render each PDF page to PNG bytes using PyMuPDF."""
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
      - PDFs: one item per page (in original order)
    Each item gets a small base64 thumbnail used by the drag UI, plus a stable uid.
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
    """Send exactly one image to the model (optionally with Code Interpreter)."""
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
    """Parallel transcription — one request per item."""
    sem = asyncio.Semaphore(concurrency)
    async with AsyncOpenAI(api_key=api_key) as client:
        async def bound_call(it: Dict) -> str:
            async with sem:
                try:
                    return await transcribe_one(client, it, model, use_code_interpreter)
                except APIStatusError as e:
                    return f"[error {e.status_code}] {getattr(e, 'message', '') or 'API error'}"
                except Exception as e:
                    return f"[error] {str(e)}"
        tasks = [asyncio.create_task(bound_call(it)) for it in items]
        return await asyncio.gather(*tasks)

def show_thumbnails(items: List[Dict]):
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            st.image(it["bytes"], caption=it["label"], use_container_width=True)

def drag_reorder_ui():
    """
    Drag‑and‑drop reordering using streamlit‑elements (MUI cards in a 1‑column draggable grid).
    """
    items = st.session_state["items"]
    if not items:
        return

    # One card per row (y=index)
    layout = [
        dashboard.Item(f"item_{it['uid']}", 0, idx, 1, 1, isResizable=False)
        for idx, it in enumerate(items)
    ]

    with elements("reorder_board"):
        with dashboard.Grid(
            layout,
            cols=1,                  # vertical list
            rowHeight=110,
            compactType="vertical",
            draggableHandle=".drag-handle",
            onLayoutChange=sync("reorder_layout"),
        ):
            for it in items:
                # The key MUST match the dashboard.Item id.
                with mui.Card(
                    key=f"item_{it['uid']}",
                    elevation=1,
                    sx={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": 1.2,
                        "px": 1,
                        "py": 1,
                        "overflow": "hidden",
                    },
                ):
                    # Drag handle (use a simple icon)
                    with mui.Box(className="drag-handle", sx={"cursor": "grab", "display": "flex", "alignItems": "center"}):
                        mui.icon.DragIndicator()

                    # Thumbnail
                    mui.CardMedia(
                        component="img",
                        image=f"data:image/png;base64,{it['thumb_b64']}",
                        sx={
                            "width": 96,
                            "height": 96,
                            "objectFit": "cover",
                            "borderRadius": "8px",
                            "border": "1px solid rgba(0,0,0,.08)",
                            "flexShrink": 0,
                            "mr": 1,
                        },
                    )

                    # Label
                    mui.Typography(
                        it["label"],
                        sx={
                            "fontSize": "0.95rem",
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    )

    # If the layout changed, apply the new order and rerun so the Preview updates.
    layout_update = st.session_state.get("reorder_layout")
    if layout_update:
        # Sort by y then x; extract IDs -> uids
        sorted_uids = [
            entry["i"].replace("item_", "")
            for entry in sorted(layout_update, key=lambda e: (e.get("y", 0), e.get("x", 0)))
        ]
        uid_to_item = {it["uid"]: it for it in items}
        new_items = [uid_to_item[u] for u in sorted_uids if u in uid_to_item]
        if len(new_items) == len(items):
            st.session_state["items"] = new_items
        st.session_state["reorder_layout"] = None
        st.rerun()


# =========================
# Upload & preparation
# =========================
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

# =========================
# UI
# =========================
if items:
    st.subheader("Preview")
    show_thumbnails(items)

    if contains_pdf:
        st.info("PDF detected. Page order is fixed to the document’s order. Reordering is disabled.", icon="📄")
    else:
        st.subheader("Reorder by dragging the previews")
        drag_reorder_ui()

    st.divider()

    # Transcribe button
    can_run = bool(api_key or api_env)
    if st.button("Transcribe with GPT‑5", disabled=not can_run, help=None if can_run else "Add your OpenAI API key first."):
        with st.status("Transcribing… running parallel requests", expanded=True) as status:
            results = asyncio.run(
                transcribe_all(st.session_state["items"], api_key or api_env, model, max_concurrency, use_ci)
            )

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
**Drag reorder**: implemented with `streamlit-elements` MUI cards.  
"""
)