import os
import io
import base64
import asyncio
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageSequence
from dotenv import load_dotenv

# PDF rendering (fast, single wheel on macOS)
import fitz  # PyMuPDF

# Optional HEIC support: only if pillow-heif is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

# OpenAI SDK (Responses API + async client)
from openai import AsyncOpenAI, APIStatusError

# -----------------------------
# App setup
# -----------------------------
load_dotenv()
st.set_page_config(page_title="Image Transcriber · GPT‑5 (No OCR)", layout="wide")
st.title("🖼️→📝 Image Transcriber (GPT‑5, no OCR)")

st.markdown(
    """
This app sends your images (or PDF pages) to **GPT‑5** with **Code Interpreter enabled**  
to transcribe visible text. It **does not** use OCR libraries.

**Notes**
- If you upload multiple images (not PDFs), you can reorder them; the final output joins in that order.  
- If you upload a **PDF**, pages are kept in document order (reordering disabled).  
- For a PDF, each page is sent **one-at-a-time** to GPT‑5; all pages are processed **in parallel**.
"""
)

# Allow user to supply API key via .env (preferred) or text field
api_env = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input(
    "OpenAI API Key (optional if set via .env)",
    value=api_env,
    type="password",
    help='Set OPENAI_API_KEY in a ".env" file or paste your key here for this session.',
)
if not api_key and not api_env:
    st.info("Add your API key in the sidebar or via a .env file to continue.", icon="🔑")

# Model + settings
st.sidebar.subheader("Model & Settings")
model = st.sidebar.text_input("Model", value="gpt-5", help="Uses GPT‑5 with Code Interpreter.")
max_concurrency = st.sidebar.slider(
    "Parallel requests", min_value=1, max_value=8, value=4, help="How many pages/images to process at once."
)
dpi = st.sidebar.slider("PDF render DPI", min_value=120, max_value=300, value=200, step=20)
st.sidebar.caption("Tip: Higher DPI yields crisper PDF images (better transcription) but is slower.")

# Session storage for prepared "items" (individual images/pages)
if "items" not in st.session_state:
    st.session_state["items"] = []  # list of dicts: {bytes, mime, label, from_pdf}
if "contains_pdf" not in st.session_state:
    st.session_state["contains_pdf"] = False


# -----------------------------
# Helpers
# -----------------------------
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif", "pdf"]
if HEIC_OK:
    ACCEPTED_TYPES.append("heic")


def _ensure_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L", "RGBA"):
        image = image.convert("RGB")
    image.save(buf, format="PNG")
    return buf.getvalue()


def _read_single_image(name: str, bytes_data: bytes) -> Tuple[bytes, str, str]:
    try:
        img = Image.open(io.BytesIO(bytes_data))
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


def prepare_items(uploaded_files) -> Tuple[List[Dict], bool]:
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
                items.append({"bytes": pbytes, "mime": "image/png", "label": f"{name} - page {i}", "from_pdf": True})
        else:
            try:
                png_bytes, mime, label = _read_single_image(name, raw)
                items.append({"bytes": png_bytes, "mime": mime, "label": label, "from_pdf": False})
            except Exception:
                st.warning(f"Couldn't read file: {name}. Unsupported or corrupted.", icon="⚠️")
    return items, contains_pdf


def to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


TRANSCRIBE_INSTRUCTIONS = (
    "You are a meticulous transcriptionist. "
    "Your job is to transcribe *only* the visible text in an image, preserving reading order and line breaks. "
    "Do not add commentary, labels, or explanations. "
    "Do not describe graphics. Output must be just the transcription text. "
    "If no text is clearly legible, output exactly: [no text]. "
    "You may use the code interpreter to crop, rotate, zoom, or otherwise enhance the image for readability, "
    "but you must not use OCR libraries or external tools."
)

TRANSCRIBE_PROMPT = (
    "Transcribe all visible text from this image as plain UTF-8 text. "
    "Preserve line breaks and layout where it helps readability. "
    "Do not add any extra words before or after. "
    "If text is partially unreadable, use the character “?” for ambiguous glyphs."
)


async def transcribe_one(client: AsyncOpenAI, item: Dict, model: str) -> str:
    """Send exactly one image to GPT‑5 (Code Interpreter enabled)."""
    data_url = to_data_url(item["bytes"], item["mime"])
    resp = await client.responses.create(
        model=model,
        instructions=TRANSCRIBE_INSTRUCTIONS,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": TRANSCRIBE_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return (resp.output_text or "").strip()


async def transcribe_all(items: List[Dict], api_key: str, model: str, concurrency: int) -> List[str]:
    """Run concurrent transcription tasks—one per image/page."""
    sem = asyncio.Semaphore(concurrency)
    async with AsyncOpenAI(api_key=api_key) as client:

        async def bound_call(item: Dict) -> str:
            async with sem:
                try:
                    return await transcribe_one(client, item, model)
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


def reorder_ui():
    """Simple up/down reordering for image-only batches."""
    items = st.session_state["items"]
    for idx, it in enumerate(items):
        c1, c2, c3 = st.columns([6, 1, 1])
        with c1:
            st.write(f"**{idx+1}.** {it['label']}")
        with c2:
            if st.button("↑", key=f"up_{idx}", help="Move up") and idx > 0:
                items[idx - 1], items[idx] = items[idx], items[idx - 1]
                st.rerun()
        with c3:
            if st.button("↓", key=f"down_{idx}", help="Move down") and idx < len(items) - 1:
                items[idx + 1], items[idx] = items[idx], items[idx + 1]
                st.rerun()


# -----------------------------
# Upload & preparation
# -----------------------------
uploaded = st.file_uploader(
    "Upload images or PDFs",
    type=ACCEPTED_TYPES,
    accept_multiple_files=True,
    help="You can upload multiple images (PNG, JPG, WebP, TIFF, BMP, GIF, HEIC*) or PDFs. HEIC requires optional pillow-heif.",
)

if uploaded:
    items, contains_pdf = prepare_items(uploaded)
    st.session_state["items"] = items
    st.session_state["contains_pdf"] = contains_pdf

items = st.session_state["items"]
contains_pdf = st.session_state["contains_pdf"]

if items:
    st.subheader("Preview")
    show_thumbnails(items)

    if contains_pdf:
        st.info(
            "PDF detected. Page order is fixed to the document’s order. Reordering is disabled to respect the PDF sequence.",
            icon="📄",
        )
    else:
        st.subheader("Reorder images (optional)")
        reorder_ui()

    st.divider()

    # Transcribe button
    can_run = bool(api_key or api_env)
    disabled_reason = None if can_run else "Add your OpenAI API key first."

    if st.button("Transcribe with GPT‑5", disabled=not can_run, help=disabled_reason):
        with st.status("Transcribing… running parallel requests", expanded=True) as status:
            # Run them concurrently; one image per request
            results = asyncio.run(
                transcribe_all(
                    st.session_state["items"], api_key or api_env, model, max_concurrency
                )
            )

            st.write("Per‑image results")
            for idx, (it, text) in enumerate(zip(st.session_state["items"], results)):
                with st.expander(it["label"], expanded=False):
                    # UNIQUE KEYS to avoid StreamlitDuplicateElementId
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
**HEIC**: To support HEIC from iPhones, install the optional dependency `pillow-heif` (see requirements).
"""
)