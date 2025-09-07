# app.py — VisionSpeak (Streamlit Cloud • PaddleOCR 3.1.0 • Python 3.13)
# Lightweight build: PIL-only preprocessing, lazy PaddleOCR/pyttsx3 imports

import os
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st

# ---------------- Page & styles ----------------
st.set_page_config(page_title="VisionSpeak • PaddleOCR 3.x", page_icon="🗣️", layout="wide")
st.markdown("""
<style>
/* ===== App-wide look ===== */
.block-container { max-width: 1180px; position: relative; z-index: 2; }
.stApp { background: transparent; }

/* Animated gradient backdrop */
#vs-anim-bg {
  position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
  background: linear-gradient(120deg, #0f1226, #0b213b, #091a2a);
}
#vs-anim-bg::before {
  content: ""; position: absolute; inset: -20% -20%;
  background: radial-gradient(60% 60% at 20% 30%, rgba(0,170,255,0.12) 0%, rgba(0,0,0,0) 60%),
              radial-gradient(55% 55% at 80% 70%, rgba(170,0,255,0.12) 0%, rgba(0,0,0,0) 60%),
              radial-gradient(40% 40% at 40% 80%, rgba(0,255,170,0.10) 0%, rgba(0,0,0,0) 60%);
  filter: blur(40px);
  animation: bg-pan 24s ease-in-out infinite alternate;
}
.vs-blob {
  position: absolute; width: 42vmin; height: 42vmin; border-radius: 50%; opacity: 0.10;
  filter: blur(24px);
  background: radial-gradient(circle at 30% 30%, rgba(0,180,255,0.9), rgba(0,180,255,0) 60%);
  animation: blob-float 26s ease-in-out infinite;
}
.vs-blob.b2 { left: 65%; top: 10%;
  background: radial-gradient(circle at 70% 70%, rgba(190,0,255,0.9), rgba(190,0,255,0) 60%);
  animation-duration: 30s;
}
.vs-blob.b3 { left: 15%; top: 70%;
  background: radial-gradient(circle at 50% 40%, rgba(0,255,190,0.9), rgba(0,255,190,0) 60%);
  animation-duration: 34s;
}

/* Cards & text */
.vs-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px; padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  backdrop-filter: blur(10px);
}
.vs-text {
  font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap;
  line-height: 1.6;
  text-align: left;            /* clean left alignment by default */
  word-break: break-word;
  hyphens: auto;
}
.vs-text.vs-just {             /* optional: justify */
  text-align: justify;
  text-justify: inter-word;
}

/* Buttons */
.stButton>button {
  border-radius: 999px; padding: 0.6rem 1rem;
  border: 1px solid rgba(255,255,255,0.20);
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  transition: transform .15s ease, box-shadow .2s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,.25); }

/* Animated gradient title */
.vs-title {
  font-size: clamp(28px, 5vw, 44px);
  font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #8bd3ff, #c9a2ff, #91ffd8, #8bd3ff);
  background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: hue-slide 10s linear infinite;
}

/* ===== Sidebar glass + accents ===== */
[data-testid="stSidebar"] { background: transparent; }
[data-testid="stSidebar"] > div:first-child {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 20px; margin: 12px; padding: 12px 12px 16px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.25);
  backdrop-filter: blur(12px); position: relative; overflow: hidden;
}
[data-testid="stSidebar"] > div:first-child::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, #8bd3ff, #c9a2ff, #91ffd8, #8bd3ff);
  background-size: 100% 300%; animation: hue-slide 10s linear infinite; opacity: .9;
}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { letter-spacing: -0.02em; margin-top: 8px; }
[data-testid="stSidebar"] .stSlider, 
[data-testid="stSidebar"] .stSelectbox, 
[data-testid="stSidebar"] .stTextInput, 
[data-testid="stSidebar"] .stCheckbox { padding: 6px 8px; border-radius: 12px; transition: background .15s ease; }
[data-testid="stSidebar"] .stSlider:hover, 
[data-testid="stSidebar"] .stSelectbox:hover, 
[data-testid="stSidebar"] .stTextInput:hover, 
[data-testid="stSidebar"] .stCheckbox:hover { background: rgba(255,255,255,0.06); }
[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 6px rgba(139,211,255,0.25); border-radius: 50%; }
[data-testid="stSlider"] .st-emotion-cache-14f2vti, 
[data-testid="stSlider"] .st-emotion-cache-1u8cb6j { height: 4px !important; background: linear-gradient(90deg, #8bd3ff, #c9a2ff, #91ffd8); }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { border-radius: 12px; border: 1px dashed rgba(255,255,255,0.25); padding: 10px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover { background: rgba(255,255,255,0.05); }
.vs-side-foot { font-size: 12px; opacity: .75; text-align: center; margin-top: 10px; }

/* Keyframes */
@keyframes bg-pan { 0%{transform:translate3d(-6%,-4%,0) scale(1.05);} 100%{transform:translate3d(6%,4%,0) scale(1.05);} }
@keyframes blob-float { 0%{transform:translate3d(-5%,0,0) scale(1);} 50%{transform:translate3d(5%,-3%,0) scale(1.08);} 100%{transform:translate3d(-5%,0,0) scale(1);} }
@keyframes hue-slide { 0%{background-position:0% 50%;} 100%{background-position:100% 50%;} }
</style>
""", unsafe_allow_html=True)

# Animated background layer
st.markdown("""
<div id="vs-anim-bg">
  <div class="vs-blob" style="left:-10%; top:10%;"></div>
  <div class="vs-blob b2"></div>
  <div class="vs-blob b3"></div>
</div>
""", unsafe_allow_html=True)

# ---------------- Drawer (no external deps) ----------------
def draw_ocr_simple(np_img: np.ndarray, boxes: List[List[List[float]]], txts: Optional[List[str]] = None) -> Image.Image:
    img = Image.fromarray(np_img.copy())
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.load_default()
    except Exception: font = None
    for i, box in enumerate(boxes):
        pts = [(int(x), int(y)) for x, y in box]
        draw.line(pts + [pts[0]], width=2, fill=(0, 255, 0))
        if txts and i < len(txts) and txts[i]:
            label = str(txts[i]); x, y = pts[0]; y = max(0, y - 16)
            w = 8 * len(label) + 6; h = 16
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
            draw.text((x + 3, y), label, fill=(255, 255, 255), font=font)
    return img

# ---------------- Normalize OCR outputs ----------------
def normalize_paddle_result(result: Any) -> Tuple[List[List[List[float]]], List[str], List[float]]:
    boxes: List[List[List[float]]], txts, scores = [], [], []
    if result is None: return boxes, txts, scores
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page: continue
        if isinstance(page, list) and page and isinstance(page[0], (list, tuple)) and len(page[0]) >= 2:
            for det in page:
                try:
                    box, info = det[0], det[1]
                    text = info[0] if isinstance(info, (list, tuple)) else str(info)
                    score = info[1] if isinstance(info, (list, tuple)) and len(info) > 1 else 1.0
                    boxes.append(box); txts.append(text); scores.append(float(score))
                except Exception: continue
            continue
        if isinstance(page, dict):
            dets = page.get("data") or page.get("result") or page.get("boxes") or []
            for det in dets:
                try:
                    if isinstance(det, dict):
                        box = det.get("box") or det.get("bbox") or []
                        text = det.get("text") or ""; score = det.get("score") or det.get("confidence") or 1.0
                        if box: boxes.append(box); txts.append(str(text)); scores.append(float(score))
                except Exception: continue
    return boxes, txts, scores

# ---------------- Lightweight preprocessing (PIL-only) ----------------
def preprocess_for_ocr(img_pil: Image.Image) -> Image.Image:
    # Upscale if small
    w, h = img_pil.size; short = min(w, h)
    if short < 720:
        scale = 720.0 / max(1, short)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Grayscale + autocontrast + simple threshold
    g = ImageOps.autocontrast(img_pil.convert("L"), cutoff=2)
    arr = np.asarray(g)
    thresh = int(arr.mean())  # simple global threshold
    arr = (arr > thresh).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").convert("RGB")

# Optional: reflow lines for nicer paragraphs (keeps original linebreaks if already present)
def _pretty_wrap(text: str, width_chars: int = 80) -> str:
    # Only wrap long single-line blobs; keep existing newlines intact
    lines = []
    for para in text.split("\n"):
        if len(para.strip()) == 0:
            lines.append("")
            continue
        if " " not in para or len(para) <= width_chars:
            lines.append(para)
            continue
        # greedy wrap by word
        words, cur, cur_len = para.split(), [], 0
        for w in words:
            if cur_len + len(w) + (1 if cur else 0) <= width_chars:
                cur.append(w); cur_len += len(w) + (1 if cur_len else 0)
            else:
                lines.append(" ".join(cur)); cur, cur_len = [w], len(w)
        if cur: lines.append(" ".join(cur))
    return "\n".join(lines)

# ---------------- Sidebar (unchanged design) ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin:6px 2px 6px 2px;">
          <div style="width:30px;height:30px;border-radius:8px;
                      background:linear-gradient(135deg,#8bd3ff,#c9a2ff,#91ffd8);"></div>
          <div style="font-weight:800;letter-spacing:-0.02em;">VisionSpeak</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.header("⚙️ Settings")
    lang = st.selectbox("OCR language", ["en", "latin", "ch", "japan", "korean", "bangla"], index=0)
    justify_text = st.checkbox("Justify extracted text", value=True)  # NEW: better alignment, default ON

    st.markdown("---")
    st.header("🔊 Speech")
    tts_rate = st.slider("Speech rate", 120, 220, 170, 5)
    tts_volume = st.slider("Volume", 0.2, 1.0, 1.0, 0.05)
    st.markdown('<div class="vs-side-foot">© VisionSpeak — OCR → TTS</div>', unsafe_allow_html=True)

# ---------------- Cache OCR (lazy import to speed startup) ----------------
@st.cache_resource(show_spinner=False)
def load_ocr(lang: str):
    # Lazy import so the UI renders instantly; models load on first use
    from paddleocr import PaddleOCR
    return PaddleOCR(lang=lang)

def run_ocr(img: Image.Image, lang: str) -> Tuple[str, Optional[Image.Image]]:
    ocr = load_ocr(lang)

    def _infer(arr: np.ndarray):
        try:
            return ocr(arr)          # PaddleOCR 3.x callable
        except Exception:
            try:
                return ocr.ocr(arr)  # fallback
            except Exception:
                return None

    # Try original
    np_img = np.array(img.convert("RGB"))
    raw = _infer(np_img)

    # If empty, try cleaned + rotations; then latin fallback
    def is_empty(res):
        return (res is None) or (isinstance(res, list) and all(len(p) == 0 for p in res if isinstance(p, list)))

    if is_empty(raw):
        variants = [img, preprocess_for_ocr(img)]
        done = False
        for base in variants:
            for angle in [0, 90, 180, 270]:
                test = base if angle == 0 else base.rotate(angle, expand=True)
                raw2 = _infer(np.array(test.convert("RGB")))
                if not is_empty(raw2):
                    raw = raw2; np_img = np.array(test.convert("RGB")); done = True; break
            if done: break

    if is_empty(raw) and lang != "latin":
        ocr_latin = load_ocr("latin")
        try:
            raw = ocr_latin(np_img) or ocr_latin.ocr(np_img)
        except Exception:
            pass

    boxes, txts, scores = normalize_paddle_result(raw)
    text = "\n".join(txts).strip()
    annotated = draw_ocr_simple(np_img, boxes, txts) if boxes else None
    return text, annotated

def synth_tts_pyttsx3(text: str, rate: int, volume: float) -> bytes:
    # Lazy import for faster startup
    import pyttsx3
    eng = pyttsx3.init()
    eng.setProperty("rate", rate)
    eng.setProperty("volume", volume)
    try:
        voices = eng.getProperty("voices")
        if voices: eng.setProperty("voice", voices[0].id)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        eng.save_to_file(text, wav_path)
        eng.runAndWait()
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try: os.remove(wav_path)
        except Exception: pass

# ---------------- Header ----------------
st.markdown('<div class="vs-title">VisionSpeak</div>', unsafe_allow_html=True)
st.subheader("Upload → OCR → TTS")

# ---------------- Upload & actions ----------------
st.markdown('<div class="vs-card">', unsafe_allow_html=True)
file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
cA, cB = st.columns(2)
with cA: run_btn = st.button("🔎 Extract Text", width="stretch")
with cB: tts_btn = st.button("🗣️ Convert Last Text to Speech", width="stretch")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- State ----------------
st.session_state.setdefault("text", "")
st.session_state.setdefault("img_src", None)
st.session_state.setdefault("img_ann", None)
st.session_state.setdefault("audio", b"")

# ---------------- Actions ----------------
if run_btn:
    if not file:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Initializing OCR engine (first run may download models)…"):
            img = Image.open(file).convert("RGB")
            text, ann = run_ocr(img, lang=lang)
            st.session_state["text"] = text
            st.session_state["img_src"] = img
            st.session_state["img_ann"] = ann
            st.session_state["audio"] = b""

if tts_btn:
    if not st.session_state["text"].strip():
        st.warning("No text to speak; run OCR first.")
    else:
        with st.spinner("Synthesizing speech…"):
            try:
                st.session_state["audio"] = synth_tts_pyttsx3(
                    st.session_state["text"], tts_rate, tts_volume
                )
            except Exception as e:
                st.error(f"TTS failed: {e}")

# ---------------- Results ----------------
if st.session_state["text"] or st.session_state["img_src"]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🖼️ Image")
        st.markdown('<div class="vs-card">', unsafe_allow_html=True)
        if st.session_state["img_ann"] is not None:
            st.image(st.session_state["img_ann"], caption="Detected text (annotated)", width="stretch")
            with st.expander("Show original"):
                st.image(st.session_state["img_src"], width="stretch")
        elif st.session_state["img_src"] is not None:
            st.image(st.session_state["img_src"], caption="Uploaded image", width="stretch")
        else:
            st.info("No image yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### 📝 Extracted Text")
        st.markdown('<div class="vs-card">', unsafe_allow_html=True)
        if st.session_state["text"].strip():
            # Optional pretty wrap to avoid long jagged lines, then justify if selected
            pretty = _pretty_wrap(st.session_state["text"], width_chars=80)
            css_class = "vs-text vs-just" if justify_text else "vs-text"
            st.markdown(f'<div class="{css_class}">{pretty}</div>', unsafe_allow_html=True)
            st.download_button(
                "Download text",
                data=st.session_state["text"].encode("utf-8"),
                file_name=f"visionspeak_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt",
                mime="text/plain",
                width="stretch"
            )
        else:
            st.info("Run OCR to see text here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔊 Speech")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    if st.session_state["audio"]:
        st.audio(st.session_state["audio"], format="audio/wav")
        st.download_button(
            "Download audio (WAV)",
            data=st.session_state["audio"],
            file_name=f"visionspeak_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav",
            mime="audio/wav",
            width="stretch"
        )
    else:
        st.info("Click **Convert Last Text to Speech** to generate audio.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.caption("Upload an image to get started.")
