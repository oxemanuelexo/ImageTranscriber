import os
import io
import base64
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pypdfium2 as pdfium  # fast PDF -> image on macOS, no poppler
import pandas as pd

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# OpenAI (Responses API) — async client for concurrency
import openai
from openai import AsyncOpenAI


# ---------- App Config ----------
st.set_page_config(page_title="Image Transcriber (GPT‑5)", layout="wide")
load_dotenv()  # reads OPENAI_API_KEY from .env

# ---------- Constants ----------
SUPPORTED_IMG_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}
SUPPORTED_DOC_EXT = {"pdf"}
THUMB_MAX = 240           # px for thumbnail column
SEND_MAX_SIDE = 2200      # px cap before sending to model to keep payloads sane
DEFAULT_CONCURRENCY = 6   # async fan-out limit
MODEL_NAME = "gpt-5"      # as requested

# ---------- Data Models ----------
@dataclass
class Item:
    id: str             # stable id: e.g., filename#p01
    label: str          # human label (filename / page)
    is_pdf_page: bool   # whether originated from a PDF page
    page_num: Optional[int]  # 1-based page number if PDF, else None
    mime: str           # "image/png" / "image/jpeg" ...
    send_data_url: str  # data URL for the *full* image we send to the model
    thumb_data_url: str # small preview used in the reorder UI


# ---------- Utils ----------
def unique_key(prefix: str, *bits: str) -> str:
    return prefix + "::" + "::".join(str(b) for b in bits)

def pil_from_upload(uploaded) -> Image.Image:
    # For GIFs, use first frame
    img = Image.open(uploaded)
    if getattr(img, "is_animated", False):
        img.seek(0)
    return ImageOps.exif_transpose(img.convert("RGBA"))

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def to_data_url(img: Image.Image, preferred: str = "PNG", quality: int = 90) -> Tuple[str, str]:
    """
    Returns (mime, data_url) for given PIL image.
    If preferred == "JPEG", we’ll convert to RGB for smaller payloads.
    """
    bio = io.BytesIO()
    fmt = preferred.upper()
    if fmt == "JPEG":
        img = img.convert("RGB")
        img.save(bio, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(bio, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return mime, f"data:{mime};base64,{b64}"

def make_thumbnail(img: Image.Image, size: int = THUMB_MAX) -> Image.Image:
    t = img.copy()
    t.thumbnail((size, size))
    return t

def render_pdf_to_images(file_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """
    Renders each page to a PIL image.
    pypdfium2 v4+ API: page.render(scale=...).to_pil()
    """
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    pages = []
    # scale: dpi / 72
    scale = dpi / 72.0
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        pil = page.render(scale=scale).to_pil()  # new API
        pages.append(pil.convert("RGBA"))
    return pages

def signature_of_uploads(files) -> Tuple[Tuple[str, int], ...]:
    # Use (name, size) to detect new upload sets
    return tuple((f.name, getattr(f, "size", 0)) for f in files)

def build_items_from_uploads(files) -> List[Item]:
    """
    Convert uploaded images and PDFs into a flat list of Items, keeping
    initial natural order (files in upload order; PDF pages in page order).
    """
    items: List[Item] = []
    for f in files:
        name = f.name
        ext = name.split(".")[-1].lower()
        blob = f.read()
        if ext in SUPPORTED_DOC_EXT:
            pages = render_pdf_to_images(blob, dpi=300)
            for idx, page_img in enumerate(pages, start=1):
                full_img = resize_keep_aspect(page_img, SEND_MAX_SIDE)
                thumb_img = make_thumbnail(full_img)
                # prefer JPEG when possible for smaller payloads
                send_mime, send_url = to_data_url(full_img, preferred="JPEG")
                _, thumb_url = to_data_url(thumb_img, preferred="JPEG")
                items.append(
                    Item(
                        id=f"{name}#p{idx:03d}",
                        label=f"{name} — page {idx}",
                        is_pdf_page=True,
                        page_num=idx,
                        mime=send_mime,
                        send_data_url=send_url,
                        thumb_data_url=thumb_url,
                    )
                )
        elif ext in SUPPORTED_IMG_EXT:
            img = pil_from_upload(io.BytesIO(blob))
            full_img = resize_keep_aspect(img, SEND_MAX_SIDE)
            thumb_img = make_thumbnail(full_img)
            send_mime, send_url = to_data_url(full_img, preferred="JPEG")
            _, thumb_url = to_data_url(thumb_img, preferred="JPEG")
            items.append(
                Item(
                    id=f"{name}",
                    label=name,
                    is_pdf_page=False,
                    page_num=None,
                    mime=send_mime,
                    send_data_url=send_url,
                    thumb_data_url=thumb_url,
                )
            )
        else:
            st.warning(f"Skipping unsupported file type: {name}")
    return items


# ---------- OpenAI ----------
def build_instructions(use_code_interpreter: bool) -> str:
    base = (
        "You are a meticulous visual transcriber. You are given exactly ONE image. "
        "Transcribe the human‑readable text precisely as it appears in the image: keep line breaks, punctuation, emoji, "
        "capitalization, lists, math, tables (as plain text), and spacing when it changes meaning. "
        "Do NOT summarize, comment, guess missing text, or add metadata. "
        "Absolutely do NOT use OCR libraries or any external tools; rely only on visual inspection of the image. "
        "Your entire output MUST be only the transcription. No preface, no labels."
    )
    if use_code_interpreter:
        base += (
            " You may use the Code Interpreter tool to crop, rotate, zoom, or otherwise enhance the image for readability, "
            "but you must not use OCR libraries or external services."
        )
    return base

async def transcribe_one(async_client: AsyncOpenAI, item: Item, use_code_interpreter: bool) -> Tuple[str, str]:
    """
    Send exactly one image to the model. Returns (item.id, transcription_text).
    """
    instructions = build_instructions(use_code_interpreter)
    input_payload = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Transcribe the text in this image. Output only the transcription."},
                {"type": "input_image", "image_url": item.send_data_url},
            ],
        }
    ]

    kwargs = {}
    if use_code_interpreter:
        kwargs["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]

    try:
        resp = await async_client.responses.create(
            model=MODEL_NAME,
            instructions=instructions,
            input=input_payload,
            **kwargs,
        )
        text = (resp.output_text or "").strip()
        return item.id, text
    except openai.APIError as e:
        return item.id, f"[ERROR] {str(e)}"
    except Exception as e:
        return item.id, f"[ERROR] {str(e)}"

async def transcribe_all(items_in_order: List[Item], use_code_interpreter: bool, concurrency: int = DEFAULT_CONCURRENCY) -> Dict[str, str]:
    sem = asyncio.Semaphore(concurrency)

    async def run_one(item: Item):
        async with sem:
            return await transcribe_one(async_client, item, use_code_interpreter)

    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as async_client:
        results = await asyncio.gather(*(run_one(it) for it in items_in_order))
    return dict(results)


# ---------- Session State Setup ----------
if "items" not in st.session_state:
    st.session_state.items: List[Item] = []
if "order" not in st.session_state:
    st.session_state.order: List[str] = []
if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "combined_output" not in st.session_state:
    st.session_state.combined_output = ""
if "per_item_output" not in st.session_state:
    st.session_state.per_item_output: Dict[str, str] = {}

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Options")
    use_code = st.toggle("Use Code Interpreter", value=True, help="Enable/disable the Code Interpreter tool for light image pre-processing (no OCR).")
    concurrency = st.slider("Parallel requests", 1, 12, DEFAULT_CONCURRENCY, help="How many images to transcribe simultaneously.")
    st.caption("Tip: If you see rate limits, lower parallelism.")

    # Reset button – clears everything, including file uploader contents
    if st.button("🔄 Reset app", type="secondary", use_container_width=True, key="reset_btn"):
        st.session_state.items = []
        st.session_state.order = []
        st.session_state.per_item_output = {}
        st.session_state.combined_output = ""
        st.session_state.upload_sig = None
        st.session_state.uploader_key += 1  # force remount of file_uploader
        st.rerun()

st.title("📄➡️🧠 Image Transcriber (GPT‑5)")

# ---------- File Uploader ----------
uploaded = st.file_uploader(
    "Drop images or PDFs",
    type=sorted(list(SUPPORTED_IMG_EXT | SUPPORTED_DOC_EXT)),
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded:
    sig = signature_of_uploads(uploaded)
    if sig != st.session_state.upload_sig:
        st.session_state.items = build_items_from_uploads(uploaded)
        st.session_state.order = [it.id for it in st.session_state.items]
        st.session_state.upload_sig = sig
        st.session_state.per_item_output = {}
        st.session_state.combined_output = ""

# If we have items, show reorder UI
if st.session_state.items:
    items = st.session_state.items
    # Build a DataFrame for AgGrid with thumbnails
    df = pd.DataFrame(
        {
            "id": [it.id for it in items],
            "Preview": [it.thumb_data_url for it in items],
            "Label (drag handle)": [it.label for it in items],
        }
    )

    gb = GridOptionsBuilder.from_dataframe(
        df[["Preview", "Label (drag handle)"]],  # hide the id col inside the grid
        editable=False,
    )

    # Image renderer for preview column (required since direct HTML is blocked in newer Ag-Grid versions)
    # We pass the data URL as the cell value and render it as an <img>.
    img_renderer = JsCode("""
        function(params) {
            var url = params.value;
            if (!url) return '';
            return `<img src="${url}" style="height:72px;object-fit:contain;border-radius:6px;" />`;
        }
    """)
    gb.configure_column("Preview", header_name="Preview", cellRenderer=img_renderer, width=110, pinned="left")

    # Enable row dragging via the label column
    gb.configure_column("Label (drag handle)", header_name="Label (drag to reorder)", rowDrag=True, autoHeight=True)

    # Managed dragging lets the grid reorder the rows as you drag
    gb.configure_grid_options(rowDragManaged=True, animateRows=False, suppressMovableColumns=True)

    grid = AgGrid(
        df[["Preview", "Label (drag handle)"]],  # id is kept outside
        gridOptions=gb.build(),
        allow_unsafe_jscode=True,
        update_mode="MODEL_CHANGED",        # return updated order after drag
        data_return_mode="AS_INPUT",
        height= min(600, 105 + 80 * len(items)),
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        key="reorder_grid",
    )

    # Reflect new order back to session state based on "Label" sequence
    # (We map labels back to ids deterministically.)
    label_to_id = {it.label: it.id for it in items}
    new_labels = [row["Label (drag handle)"] for _, row in grid["data"].iterrows()]
    new_order = [label_to_id[lbl] for lbl in new_labels if lbl in label_to_id]

    # Update if changed
    if new_order and new_order != st.session_state.order:
        st.session_state.order = new_order

    st.caption("Drag rows in the table above to change the order. The final combined transcription follows this order.")

    # Action buttons
    col_a, col_b = st.columns([1, 2])
    with col_a:
        start = st.button("🚀 Transcribe", type="primary", key="go_btn")
    with col_b:
        st.write("")

    # Results zone
    if start:
        # Resolve chosen order into Item list
        id_to_item = {it.id: it for it in items}
        ordered_items = [id_to_item[i] for i in st.session_state.order if i in id_to_item]

        with st.spinner("Transcribing images with GPT‑5..."):
            transcripts_map = asyncio.run(transcribe_all(ordered_items, use_code_interpreter=use_code, concurrency=concurrency))

        st.session_state.per_item_output = transcripts_map
        st.session_state.combined_output = "\n\n".join([transcripts_map.get(it.id, "").strip() for it in ordered_items]).strip()

# Show outputs if available
if st.session_state.per_item_output:
    st.subheader("Per‑image transcriptions")
    id_to_item = {it.id: it for it in st.session_state.items}

    for idx, item_id in enumerate(st.session_state.order, start=1):
        item = id_to_item.get(item_id)
        if not item:
            continue
        with st.expander(f"{idx}. {item.label}", expanded=False):
            st.image(item.thumb_data_url, caption=item.label, use_container_width=False)
            st.text_area(
                "Transcription (read‑only)",
                st.session_state.per_item_output.get(item_id, ""),
                height=180,
                key=unique_key("ta", item_id),  # <-- unique keys fix DuplicateElementId
            )

if st.session_state.combined_output:
    st.subheader("Combined transcription (ordered)")
    st.text_area(
        "Combined",
        st.session_state.combined_output,
        height=240,
        key=unique_key("combined_ta", "final"),
    )
    # Download as .txt
    st.download_button(
        "⬇️ Download .txt",
        data=st.session_state.combined_output,
        file_name="transcription.txt",
        mime="text/plain",
        key="dl_btn",
    )
else:
    st.info("Upload files, optionally reorder them, then click **Transcribe** to generate the combined result.")

# Tiny footer
st.caption("Built for visual transcription with one image per request. No OCR libraries are used. Uses GPT‑5 via the Responses API.")