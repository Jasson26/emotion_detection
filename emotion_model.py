import os
import requests
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models


# ===============================
# CONFIG
# ===============================
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Sumber file face detector OpenCV (auto-download)
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel"

face_net = None


# ===============================
# HELPERS
# ===============================
def _download_if_missing(url: str, dst: str):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _ensure_face_net():
    """Load DNN face detector (download file jika belum ada)."""
    global face_net
    if face_net is not None:
        return face_net

    try:
        _download_if_missing(PROTO_URL, PROTO_PATH)
        _download_if_missing(CAFFE_URL, CAFFE_PATH)
        face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, CAFFE_PATH)
        return face_net
    except Exception:
        face_net = None
        return None


def build_resnet18(num_classes: int):
    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def load_emotion_model(model_path: str):
    model = build_resnet18(num_classes=len(CLASS_NAMES))

    state = torch.load(model_path, map_location=DEVICE)

    # kalau checkpoint punya prefix "backbone."
    if isinstance(state, dict) and any(k.startswith("backbone.") for k in state.keys()):
        state = {k.replace("backbone.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


transform_face = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    return x1, y1, x2, y2


# ===============================
# MAIN PREDICT
# ===============================
def predict_emotions_on_image_bgr(model, img_bgr, conf_threshold: float = 0.3, det_scale: float = 1.5):
    """
    - det_scale: upscale image sebelum deteksi (membantu deteksi di cahaya redup / wajah sedang)
    - kalau wajah tidak ketemu: fallback crop tengah (bukan full image)
    return: (img_bgr_with_box, results_list)
    """
    out_img = img_bgr.copy()
    h, w = out_img.shape[:2]
    results = []

    net = _ensure_face_net()

    faces = []
    max_conf = 0.0

    if net is not None:
        # upscale untuk memperbesar wajah sebelum dideteksi
        if det_scale and det_scale != 1.0:
            img_det = cv2.resize(out_img, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_det = out_img

        hh, ww = img_det.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img_det,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            det_conf = float(detections[0, 0, i, 2])
            max_conf = max(max_conf, det_conf)

            if det_conf < conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([ww, hh, ww, hh])
            x1, y1, x2, y2 = box.astype(int)

            # scale balik ke ukuran gambar asli
            if det_scale and det_scale != 1.0:
                x1 = int(x1 / det_scale); y1 = int(y1 / det_scale)
                x2 = int(x2 / det_scale); y2 = int(y2 / det_scale)

            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            faces.append((x1, y1, x2, y2, det_conf, area))

    # ✅ kalau gagal deteksi wajah → fallback crop tengah (tidak full image)
    if len(faces) == 0:
        cx1 = int(w * 0.2); cy1 = int(h * 0.1)
        cx2 = int(w * 0.8); cy2 = int(h * 0.9)
        cx1, cy1, cx2, cy2 = _clamp_box(cx1, cy1, cx2, cy2, w, h)

        # proses crop tengah sebagai "pseudo-face"
        face_img_bgr = out_img[cy1:cy2, cx1:cx2]
        if face_img_bgr.size == 0:
            return out_img, []

        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = transform_face(face_img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            label = CLASS_NAMES[int(preds.item())]
            confidence = float(conf.item())

        # kita kasih box warna kuning untuk tanda fallback
        color = (0, 255, 255)
        cv2.rectangle(out_img, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(out_img, f"{label} ({confidence*100:.1f}%) [fallback]",
                    (cx1, max(0, cy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "box": (cx1, cy1, cx2 - cx1, cy2 - cy1),
            "label": label,
            "confidence": confidence,
            "face_det_conf": 0.0,
            "debug_max_face_conf": float(max_conf),
            "used_fallback": True,
        })
        return out_img, results

    # ✅ ambil wajah terbesar (stabil)
    faces.sort(key=lambda x: x[5], reverse=True)
    x1, y1, x2, y2, det_conf, _area = faces[0]

    # ✅ margin 10% agar box lebih pas
    mx = int(0.10 * (x2 - x1))
    my = int(0.10 * (y2 - y1))
    x1, y1, x2, y2 = _clamp_box(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)

    face_img_bgr = out_img[y1:y2, x1:x2]
    if face_img_bgr.size == 0:
        return out_img, []

    face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = transform_face(face_img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(face_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        label = CLASS_NAMES[int(preds.item())]
        confidence = float(conf.item())

    color = (0, 255, 0)
    cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out_img, f"{label} ({confidence*100:.1f}%)",
                (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    results.append({
        "box": (x1, y1, x2 - x1, y2 - y1),
        "label": label,
        "confidence": confidence,
        "face_det_conf": float(det_conf),
        "debug_max_face_conf": float(max_conf),
        "used_fallback": False,
    })

    return out_img, results
