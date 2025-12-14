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


def bgr_to_rgb_uint8(img_bgr):
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Output image invalid")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.ndim != 3:
        raise ValueError(f"Invalid shape: {img_bgr.shape}")
    if img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ===============================
# UI
# ===============================
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎭 Facial Emotion Recognition")
st.write("Upload foto atau ambil snapshot (kamera).")

conf_thresh = st.slider(
    "Face detection confidence threshold (lebih kecil = lebih sensitif)",
    min_value=0.1,
    max_value=0.9,
    value=0.3,     # default lebih sensitif
    step=0.05
)

det_scale = st.slider(
    "Detection upscale (membantu deteksi wajah)",
    min_value=1.0,
    max_value=2.0,
    value=1.5,
    step=0.1
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
            model, img_bgr,
            conf_threshold=conf_thresh,
            det_scale=float(det_scale)
        )
    except Exception as e:
        st.error(f"Gagal saat prediksi: {type(e).__name__}: {e}")
        st.stop()

    try:
        img_rgb_out = bgr_to_rgb_uint8(img_bgr_out)
        st.subheader("✅ Hasil Deteksi")
        st.image(img_rgb_out, use_column_width=True)  # kompatibel streamlit lama
    except Exception as e:
        st.error(f"Gambar output tidak bisa ditampilkan: {type(e).__name__}: {e}")
        st.stop()

    if not results:
        st.warning("Tidak ada wajah terdeteksi.")
    else:
        st.markdown("### Detail Prediksi")
        for i, r in enumerate(results, 1):
            used_fallback = r.get("used_fallback", False)
            extra = " (fallback crop)" if used_fallback else ""
            st.write(
                f"Wajah {i}: **{r.get('label','?')}** "
                f"({float(r.get('confidence',0))*100:.1f}%)"
                f"{extra}"
            )
            # Debug kecil (optional)
            st.caption(
                f"FaceDetConf={float(r.get('face_det_conf',0)):.3f} | "
                f"MaxDetConfSeen={float(r.get('debug_max_face_conf',0)):.3f}"
            )
