import os
import io
import base64
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pypdfium2 as pdfium
import pandas as pd

from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

import openai
from openai import AsyncOpenAI

# -------------------- Config --------------------
st.set_page_config(page_title="Image Transcriber (GPT‑5)", layout="wide")
load_dotenv()

SUPPORTED_IMG_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}
SUPPORTED_DOC_EXT = {"pdf"}
THUMB_MAX = 240
SEND_MAX_SIDE = 2200
DEFAULT_CONCURRENCY = 6
MODEL_NAME = "gpt-5"

# -------------------- Models --------------------
@dataclass
class Item:
    id: str
    label: str
    is_pdf_page: bool
    page_num: Optional[int]
    mime: str
    send_data_url: str
    thumb_data_url: str

# -------------------- Utils --------------------
def unique_key(prefix: str, *bits: str) -> str:
    return prefix + "::" + "::".join(str(b) for b in bits)

def pil_from_upload(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    if getattr(img, "is_animated", False):
        img.seek(0)
    # respect EXIF orientation
    return ImageOps.exif_transpose(img.convert("RGBA"))

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def to_data_url(img: Image.Image, preferred: str = "PNG", quality: int = 90) -> Tuple[str, str]:
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
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
    pages = []
    scale = dpi / 72.0
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        pil = page.render(scale=scale).to_pil()
        pages.append(pil.convert("RGBA"))
    return pages

def signature_of_uploads(files) -> Tuple[Tuple[str, int], ...]:
    return tuple((f.name, getattr(f, "size", 0)) for f in files)

def build_items_from_uploads(files) -> List[Item]:
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
                    id=name,
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

# -------------------- OpenAI --------------------
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

async def transcribe_all(items_in_order: List[Item], use_code_interpreter: bool, concurrency: int) -> Dict[str, str]:
    sem = asyncio.Semaphore(concurrency)
    async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as async_client:
        async def run_one(it: Item):
            async with sem:
                return await transcribe_one(async_client, it, use_code_interpreter)
        results = await asyncio.gather(*(run_one(it) for it in items_in_order))
    return dict(results)

# -------------------- Session State (use bracket keys!) --------------------
for k, default in {
    "doc_items": [],
    "doc_order": [],
    "upload_sig": None,
    "uploader_key": 0,
    "combined_output": "",
    "per_item_output": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Options")
    use_code = st.toggle("Use Code Interpreter", value=True, help="Enable/disable Code Interpreter for light image pre-processing (no OCR).")
    concurrency = st.slider("Parallel requests", 1, 12, DEFAULT_CONCURRENCY, help="How many images to transcribe simultaneously.")
    if st.button("🔄 Reset app", type="secondary", use_container_width=True, key="reset_btn"):
        st.session_state["doc_items"] = []
        st.session_state["doc_order"] = []
        st.session_state["per_item_output"] = {}
        st.session_state["combined_output"] = ""
        st.session_state["upload_sig"] = None
        st.session_state["uploader_key"] += 1
        st.rerun()

st.title("📄➡️🧠 Image Transcriber (GPT‑5)")

# -------------------- Upload --------------------
uploaded = st.file_uploader(
    "Drop images or PDFs",
    type=sorted(list(SUPPORTED_IMG_EXT | SUPPORTED_DOC_EXT)),
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

if uploaded:
    sig = signature_of_uploads(uploaded)
    if sig != st.session_state["upload_sig"]:
        st.session_state["doc_items"] = build_items_from_uploads(uploaded)
        st.session_state["doc_order"] = [it.id for it in st.session_state["doc_items"]]
        st.session_state["upload_sig"] = sig
        st.session_state["per_item_output"] = {}
        st.session_state["combined_output"] = ""

# -------------------- Reorder UI (drag & drop) --------------------
if st.session_state["doc_items"]:
    items: List[Item] = st.session_state["doc_items"]

    df = pd.DataFrame(
        {
            "id": [it.id for it in items],
            "Preview": [it.thumb_data_url for it in items],
            "Label": [it.label for it in items],
        }
    )

    gb = GridOptionsBuilder.from_dataframe(df, editable=False)
    img_renderer = JsCode("""
        function(params) {
            var url = params.value;
            if (!url) return '';
            return `<img src="${url}" style="height:72px;object-fit:contain;border-radius:6px;" />`;
        }
    """)
    gb.configure_column("id", header_name="id", hide=True)
    gb.configure_column("Preview", header_name="Preview", cellRenderer=img_renderer, width=110, pinned="left")
    gb.configure_column("Label", header_name="Label (drag to reorder)", rowDrag=True, autoHeight=True)
    gb.configure_grid_options(rowDragManaged=True, animateRows=False, suppressMovableColumns=True)

    grid = AgGrid(
        df,
        gridOptions=gb.build(),
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode="AS_INPUT",
        height=min(600, 105 + 80 * len(items)),
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        key="reorder_grid",
    )

    # Capture new order by the hidden 'id' column (robust even if labels duplicate)
    new_df = grid["data"]
    if "id" in new_df.columns:
        new_order = list(new_df["id"])
        if new_order and new_order != st.session_state["doc_order"]:
            st.session_state["doc_order"] = new_order

    st.caption("Drag rows to change order. The final combined transcription follows this order.")

    col_run, _ = st.columns([1, 3])
    with col_run:
        start = st.button("🚀 Transcribe", type="primary", key="go_btn")
    if start:
        id_to_item = {it.id: it for it in items}
        ordered_items = [id_to_item[i] for i in st.session_state["doc_order"] if i in id_to_item]
        with st.spinner("Transcribing images with GPT‑5..."):
            transcripts_map = asyncio.run(
                transcribe_all(ordered_items, use_code_interpreter=use_code, concurrency=concurrency)
            )
        st.session_state["per_item_output"] = transcripts_map
        st.session_state["combined_output"] = "\n\n".join(
            [transcripts_map.get(it.id, "").strip() for it in ordered_items]
        ).strip()

# -------------------- Outputs --------------------
if st.session_state["per_item_output"]:
    st.subheader("Per‑image transcriptions")
    id_to_item = {it.id: it for it in st.session_state["doc_items"]}
    for idx, item_id in enumerate(st.session_state["doc_order"], start=1):
        it = id_to_item.get(item_id)
        if not it:
            continue
        with st.expander(f"{idx}. {it.label}", expanded=False):
            st.image(it.thumb_data_url, caption=it.label, use_container_width=False)
            st.text_area(
                "Transcription (read‑only)",
                st.session_state["per_item_output"].get(item_id, ""),
                height=180,
                key=unique_key("ta", item_id),
            )

if st.session_state["combined_output"]:
    st.subheader("Combined transcription (ordered)")
    st.text_area(
        "Combined",
        st.session_state["combined_output"],
        height=240,
        key=unique_key("combined_ta", "final"),
    )
    st.download_button(
        "⬇️ Download .txt",
        data=st.session_state["combined_output"],
        file_name="transcription.txt",
        mime="text/plain",
        key="dl_btn",
    )
else:
    st.info("Upload files, drag to reorder, then click **Transcribe** to generate the combined result.")

st.caption("No OCR libraries are used. One image per request. Toggle Code Interpreter on/off from the sidebar.")