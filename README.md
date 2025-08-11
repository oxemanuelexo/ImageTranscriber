# Image Transcriber · GPT‑5 (No OCR)

A tiny Streamlit web app that **transcribes text from images** using **OpenAI’s GPT‑5** (with an optional **Code Interpreter** tool). It **does not use OCR libraries**—the model looks at the pixels.

* Upload **images** or **PDFs**
* Sends **one image/page per request**
* Runs **requests in parallel** for speed
* **Drag the file names** to reorder; **previews and final transcript** follow that order
* **Reset/Clear** button to start fresh
* **Code Interpreter toggle** (on/off). If off, the prompt line about using it is removed and the tool isn’t sent.

---

# Features

* **Multiple file types**: PNG, JPG/JPEG, WEBP, TIFF, BMP, GIF, and PDF

  * PDFs are split into **pages** (rendered via PyMuPDF) and processed page‑by‑page
  * Multi‑frame images (e.g., animated GIF/TIFF) use the **first frame**
* **Drag‑and‑drop ordering (names)**: Uses `streamlit-sortables`.

  * After you drag names, the **image previews** rearrange to match
  * The **combined transcript** respects the shown order
* **Parallel requests**: Adjustable concurrency to balance speed vs. rate limits
* **No OCR**: No OCR libraries are installed or used
* **Mac‑friendly**: Simple Python deps; works great on macOS (also fine on Linux/Windows)

---

# Requirements

* Python **3.9+**
* An OpenAI API key with access to a **GPT‑5** vision‑capable model

  > You can change the model in the app’s sidebar if needed.

---

# Quickstart

```bash
# 1) Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

# 2) Install deps
pip install -r requirements.txt

# 3) Set your API key
cp .env.example .env
# then edit .env and paste your real key (OPENAI_API_KEY=sk-...)

# 4) Run the app
streamlit run app.py
```

Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

# Usage

1. **Upload** images and/or a PDF.
2. *(Optional)* **Drag the file names** to set the order.

   * PDFs keep their original page order (reordering disabled).
   * After dropping, the **preview grid** will reflect your chosen order.
3. *(Optional)* In the sidebar, adjust:

   * **Parallel requests** (concurrency)
   * **PDF render DPI** (higher = crisper pages, slower)
   * **Code Interpreter** (on/off)
   * **Model** (defaults to `gpt-5`)
4. Click **Transcribe with GPT‑5**.
5. Review per‑image results, then copy or **download the combined transcript** as a `.txt`.

Use **🔁 Reset / Clear files** (sidebar) to start over.

---

# How it works (high level)

* Each image or PDF page is converted to **PNG bytes** (for PDFs, via **PyMuPDF** at your chosen DPI).
* The app sends **exactly one** `input_image` per request to the OpenAI **Responses API** using the selected model.
* Requests are **concurrent** (configurable concurrency).
* The system prompt **forces “transcription only”**. If **Code Interpreter** is on, the tool is attached…

  * …and the instructions include: “You may use the code interpreter to crop, rotate, zoom… but not OCR.”
  * If the toggle is **off**, that line is **omitted** and the tool is **not** sent.
* The individual transcripts are **joined (in order)** into one final output.

---

# Project structure

```
.
├─ app.py                 # Streamlit app
├─ requirements.txt       # Minimal, Mac-friendly deps
└─ .env.example           # Template for OPENAI_API_KEY
```

---

# Configuration

* **OPENAI\_API\_KEY**: set in `.env`
* **Model**: change in the sidebar (default `gpt-5`)
* **Parallel requests**: sidebar slider (1–8)
* **PDF DPI**: sidebar slider (120–300). Higher = sharper render (better transcription), but slower.

---

# Supported inputs

* **Images**: `png`, `jpg`, `jpeg`, `webp`, `tif`, `tiff`, `bmp`, `gif`

  * Animated images use the **first frame**
* **PDF**: split into pages (in document order)

> HEIC isn’t included by default. If you need iPhone HEIC support, you can add `pillow-heif` and register it—ping me and I’ll show you how.

---

# Limits & notes

* **No OCR libraries** are used. The model reads pixels; for tiny/blurred text, results may vary.
* **Rate limits**: If you hit 429s, lower **Parallel requests**.
* **Costs**: Each page/image is a separate request—keep an eye on token & image usage.
* **Privacy**: Files are sent to OpenAI to generate transcripts; nothing is stored by this app.

---

# Troubleshooting

* **Nothing happens / button disabled** → Ensure your API key is set (sidebar or `.env`).
* **“Module not found” for PyMuPDF** → Reinstall deps: `pip install -r requirements.txt`.
* **PDF pages look blurry** → Increase **PDF render DPI** (try 220–260).
* **429 / rate limit** → Lower concurrency; try `2` or `3`.
* **Mixed order after dragging** → The **preview grid** should mirror the name list after the app reruns. If not, use **Reset** and try again.

---

# Extending (optional ideas)

* Add HEIC support via `pillow-heif`.
* Export to **.md** with simple heading detection (still no OCR).
* Optional “Unlock PDF order” to manually reorder pages.
* Persist session state across refreshes.

---

# Security & privacy

* Your API key stays **local** (loaded from `.env` or sidebar).
* The app sends your uploaded files to OpenAI **only** to get transcriptions.
* No third‑party OCR or cloud storage is used.

---

# License

Choose a license that fits your needs (MIT is a common choice). If you want, I’ll add an MIT `LICENSE` file.

---

# Credits

* Built with **Streamlit**, **Pillow**, **PyMuPDF**, and the **OpenAI Python SDK** (Responses API).
* Drag‑and‑drop ordering powered by **streamlit‑sortables**.

---