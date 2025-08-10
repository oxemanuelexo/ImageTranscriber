import os
import io
import base64
import asyncio
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError, ImageSequence
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import AsyncOpenAI, APIStatusError
from streamlit_sortables import sort_items  # drag names to reorder


# ───────────────────────────────────────────────────────────────────────────────
# App setup
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Image Transcriber · GPT-5 (No OCR)", layout="wide")
st.title("🖼️→📝 Image Transcriber (GPT-5, no OCR)")

st.markdown(
    """
Upload images or PDFs. We’ll send **one image/page per request** to **GPT‑5** to transcribe
**visible text** (no OCR libraries).

- Drag the **file names** to set the order.  
- The **image previews** and **combined transcript** follow that order.  
- PDFs keep their original page order (reordering disabled).
"""
)

# Sidebar
api_env = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input(
    "OpenAI API Key (optional if set via .env)",
    value=api_env,
    type="password",
    help='Set OPENAI_API_KEY in a ".env" file or paste your key here for this session.',
)

st.sidebar.subheader("Model & Settings")
model = st.sidebar.text_input("Model", value="gpt-5")
use_ci = st.sidebar.checkbox("Use Code Interpreter", value=True)
max_concurrency = st.sidebar.slider("Parallel requests", 1, 8, 4)
dpi = st.sidebar.slider("PDF render DPI", 120, 300, 200, 20)

# Reset
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

def reset_all():
    for k in ("items", "contains_pdf", "uploaded_signature"):
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] += 1

if st.sidebar.button("🔁 Reset / Clear files"):
    reset_all()
    st.rerun()

# Session state
if "items" not in st.session_state:
    st.session_state["items"] = []   # [{bytes, mime, label, display_label, from_pdf}]
if "contains_pdf" not in st.session_state:
    st.session_state["contains_pdf"] = False


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
ACCEPTED_TYPES = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif", "pdf"]

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

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _ensure_unique_display_labels(items: List[Dict]) -> None:
    """
    Make the draggable names unique even if filenames repeat.
    """
    seen = {}
    for it in items:
        base = it["label"]
        seen[base] = seen.get(base, 0) + 1
        it["display_label"] = base if seen[base] == 1 else f"{base} ({seen[base]})"

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
                items.append({
                    "bytes": pbytes,
                    "mime": "image/png",
                    "label": f"{name} - page {i}",
                    "from_pdf": True,
                })
        else:
            try:
                png_bytes, mime, label = _read_single_image(name, raw)
                items.append({
                    "bytes": png_bytes,
                    "mime": mime,
                    "label": label,
                    "from_pdf": False,
                })
            except Exception:
                st.warning(f"Couldn't read file: {name}. Unsupported or corrupted.", icon="⚠️")

    _ensure_unique_display_labels(items)
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
    "Transcribe all visible text from this image as plain UTF-8 text. "
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
                    return f"[error {e.status_code}] {getattr(e, 'message', '') or 'API error'}"
                except Exception as e:
                    return f"[error] {str(e)}"
        tasks = [asyncio.create_task(bound_call(it)) for it in items]
        return await asyncio.gather(*tasks)

def safe_run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

def show_previews(items: List[Dict]):
    cols = st.columns(4)
    for i, it in enumerate(items):
        with cols[i % 4]:
            st.image(it["bytes"], caption=it["label"], use_container_width=True)

def reorder_names_ui():
    """
    Drag-and-drop list of names (strings) using streamlit-sortables.
    Updates items order + then reruns so previews reflect the new order.
    """
    items = st.session_state["items"]
    if not items:
        return
    names = [it["display_label"] for it in items]
    sorted_names = sort_items(names, direction="vertical", key="name_sort") or names
    if sorted_names != names:
        # Map display label -> item, then reorder
        lookup = {it["display_label"]: it for it in items}
        new_items = [lookup[name] for name in sorted_names if name in lookup]
        if len(new_items) == len(items):
            st.session_state["items"] = new_items
            st.rerun()


# ───────────────────────────────────────────────────────────────────────────────
# Upload & preparation
# ───────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload images or PDFs",
    key=f"uploader_{st.session_state['uploader_key']}",
    type=ACCEPTED_TYPES,
    accept_multiple_files=True,
)

# Only reparse when the upload list actually changes (so order sticks)
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
    if contains_pdf:
        st.info("PDF detected. Page order is fixed to the document’s order. Reordering disabled.", icon="📄")
    else:
        st.subheader("Drag to reorder file names")
        reorder_names_ui()

    st.subheader("Preview (mirrors the order above)")
    show_previews(items)

    st.divider()

    # Transcribe
    can_run = bool(api_key or api_env)
    if st.button("Transcribe with GPT-5", disabled=not can_run, help=None if can_run else "Add your OpenAI API key first."):
        with st.status("Transcribing… running parallel requests", expanded=True) as status:
            results = safe_run(
                transcribe_all(st.session_state["items"], api_key or api_env, model, max_concurrency, use_ci)
            )

            st.write("Per-image results")
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
