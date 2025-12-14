import os
from pathlib import Path
import requests

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

proto_path = os.path.join(MODEL_DIR, "deploy.prototxt")
weights_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel"

face_net = None


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
    
    global face_net

    if face_net is not None:
        return face_net

    try:
        
        _download_if_missing(PROTO_URL, proto_path)
        _download_if_missing(CAFFE_URL, weights_path)

        net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        face_net = net
        return face_net
    except Exception:
        face_net = None
        return None


def build_resnet18(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_emotion_model(model_path: str):
    model = build_resnet18(num_classes=len(CLASS_NAMES))

    state = torch.load(model_path, map_location=DEVICE)

    
    if isinstance(state, dict) and any(k.startswith("backbone.") for k in state.keys()):
        new_state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        state = new_state

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


def predict_emotions_on_image_bgr(model, img_bgr, conf_threshold: float = 0.5):
   
    out_img = img_bgr.copy()
    h, w = out_img.shape[:2]
    results = []

    net = _ensure_face_net()

    faces = []
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            out_img, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            det_conf = float(detections[0, 0, i, 2])
            if det_conf < conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            faces.append((x1, y1, x2, y2, det_conf))

    
    if len(faces) == 0:
        return out_img, []

    
    faces.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    x1, y1, x2, y2, det_conf = faces[0]

    
    mx = int(0.10 * (x2 - x1))
    my = int(0.10 * (y2 - y1))
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx); y2 = min(h - 1, y2 + my)

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
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(out_img, text, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    results.append({
        "box": (x1, y1, x2 - x1, y2 - y1),
        "label": label,
        "confidence": confidence,
        "face_det_conf": float(det_conf),
    })

    return out_img, results
