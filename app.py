import tempfile
from pathlib import Path
import requests

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from emotion_model import load_emotion_model, predict_emotions_on_image_bgr



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
        with st.spinner("Downloading emotion model "):
            r = requests.get(EMOTION_MODEL_URL, stream=True, timeout=180)
            r.raise_for_status()
            with open(EMOTION_MODEL_LOCAL, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    
    return load_emotion_model(str(EMOTION_MODEL_LOCAL))



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

st.subheader(" Snapshot (opsional)")
camera_image = st.camera_input("Ambil foto dari kamera")

st.subheader(" Upload Image")
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

image_source = camera_image if camera_image is not None else uploaded_file

if image_source is None:
    st.info("Silakan upload gambar atau ambil foto.")
else:
    image = Image.open(image_source).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    img_bgr_out, results = predict_emotions_on_image_bgr(
        model, img_bgr, conf_threshold=conf_thresh
    )

    img_rgb_out = cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2RGB)

    st.subheader(" Hasil Deteksi")
    st.image(img_rgb_out, use_container_width=True)

    if not results:
        st.warning("Tidak ada wajah terdeteksi.")
    else:
        st.markdown(" Detail Prediksi")
        for i, r in enumerate(results, 1):
            st.write(f"Wajah {i}: **{r['label']}** ({r['confidence']*100:.1f}%)")
