import streamlit as st
import numpy as np
import cv2
from PIL import Image

from emotion_model import load_emotion_model, predict_emotions_on_image_bgr

MODEL_PATH = r"C:\Users\MSI Raider\Downloads\DeepLearning_FaceDetection\emotion_resnet18_best.pth"

@st.cache_resource
def get_model():
    return load_emotion_model(MODEL_PATH)

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎭 Facial Emotion Recognition (ResNet18 + DNN Face Detector)")
st.write("Upload foto atau SnapShot")


conf_thresh = st.slider(
    "Face detection confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

model = get_model()

st.subheader(" SnapShot ")
camera_image = st.camera_input("Ambil foto dari webcam (optional)")

st.subheader("Upload Image")
uploaded_file = st.file_uploader("Upload Your Image Down Here", type=["jpg", "jpeg", "png"])

image_source = None
source_name = None

if camera_image is not None:
    image_source = camera_image
    source_name = "kamera"
elif uploaded_file is not None:
    image_source = uploaded_file
    source_name = "upload"

if image_source is None:
    st.info("Silakan ambil foto dari kamera atau upload gambar terlebih dahulu.")
else:
    image = Image.open(image_source).convert("RGB")
    img_np = np.array(image)                 #
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img_bgr_out, results = predict_emotions_on_image_bgr(
        model, img_bgr, conf_threshold=conf_thresh
    )

    img_rgb_out = cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2RGB)

    st.subheader("Hasil Deteksi")
    st.image(img_rgb_out, use_column_width=True, caption=f"Hasil ({source_name})")

    if len(results) == 0:
        st.warning("Tidak ada wajah terdeteksi.")
    else:
        st.markdown("**Detail Prediksi:**")
        for i, r in enumerate(results, 1):
            st.write(f"Wajah {i}: **{r['label']}** ({r['confidence']*100:.1f}%)")
