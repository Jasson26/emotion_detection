import tempfile
from pathlib import Path
import requests

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from emotion_model import load_emotion_model, predict_emotions_on_image_bgr


# ===============================
# MODEL URL (Secrets + Fallback)
# ===============================
try:
    EMOTION_MODEL_URL = str(st.secrets["EMOTION_MODEL_URL"]).strip()
except Exception:
    EMOTION_MODEL_URL = ""

if not EMOTION_MODEL_URL:
    EMOTION_MODEL_URL = (
        "https://huggingface.co/jassont26/emotion-resnet18/"
        "resolve/main/emotion_resnet18_best.pth"
    )

EMOTION_MODEL_LOCAL = Path(tempfile.gettempdir()) / "emotion_resnet18_best.pth"


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def get_model():
    EMOTION_MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)

    if (not EMOTION_MODEL_LOCAL.exists()) or EMOTION_MODEL_LOCAL.stat().st_size == 0:
        with st.spinner("Downloading emotion model (.pth)..."):
            r = requests.get(EMOTION_MODEL_URL, stream=True, timeout=180)
            r.raise_for_status()
            with open(EMOTION_MODEL_LOCAL, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    return load_emotion_model(str(EMOTION_MODEL_LOCAL))


def bgr_to_rgb_for_streamlit(img_bgr):
    if img_bgr is None:
        raise ValueError("Output image is None")

    if not isinstance(img_bgr, np.ndarray):
        raise TypeError(f"Output image is not np.ndarray (type={type(img_bgr)})")

    if img_bgr.size == 0:
        raise ValueError("Output image is empty (size=0)")

    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    if img_bgr.ndim != 3:
        raise ValueError(f"Invalid ndim={img_bgr.ndim}, shape={img_bgr.shape}")

    h, w, c = img_bgr.shape
    if c == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    elif c == 1:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif c != 3:
        raise ValueError(f"Invalid channels={c}, shape={img_bgr.shape}")

    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ===============================
# UI
# ===============================
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎭 Facial Emotion Recognition")
st.write("Upload foto atau ambil snapshot")

conf_thresh = st.slider(
    "Face detection confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

model = get_model()

st.subheader("📸 Snapshot (opsional)")
camera_image = st.camera_input("Ambil foto dari kamera")

st.subheader("📁 Upload Image")
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

image_source = camera_image if camera_image is not None else uploaded_file

if image_source is None:
    st.info("Silakan upload gambar atau ambil foto.")
else:
    image = Image.open(image_source).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        img_bgr_out, results = predict_emotions_on_image_bgr(
            model, img_bgr, conf_threshold=conf_thresh
        )
    except Exception as e:
        st.error(f"Gagal saat predict_emotions_on_image_bgr: {e}")
        st.stop()

    try:
        img_rgb_out = bgr_to_rgb_for_streamlit(img_bgr_out)
        st.subheader("✅ Hasil Deteksi")
        # kompatibel streamlit lama:
        st.image(img_rgb_out, use_column_width=True)
    except Exception as e:
        st.error("Output image dari model tidak bisa ditampilkan.")
        st.code(f"{type(e).__name__}: {e}")
        st.write("DEBUG output:")
        st.write("type:", type(img_bgr_out))
        if isinstance(img_bgr_out, np.ndarray):
            st.write("shape:", img_bgr_out.shape)
            st.write("dtype:", img_bgr_out.dtype)
            st.write("min/max:", float(np.min(img_bgr_out)), float(np.max(img_bgr_out)))
        st.stop()

    if not results:
        st.warning("Tidak ada wajah terdeteksi.")
    else:
        st.markdown("### Detail Prediksi")
        for i, r in enumerate(results, 1):
            st.write(f"Wajah {i}: **{r.get('label','?')}** ({float(r.get('confidence',0))*100:.1f}%)")
