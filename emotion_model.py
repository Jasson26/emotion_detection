import os
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

proto_path = os.path.join(MODEL_DIR, "deploy.prototxt")
weights_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

face_net = None

print("BASE_DIR :", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("Proto path   :", proto_path)
print("Weights path :", weights_path)
print("Proto exists :", os.path.exists(proto_path))
print("Weights exists:", os.path.exists(weights_path))

try:
    face_net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
    print("DNN face detector loaded.")
except Exception as e:
    print("Failed to load DNN face detector:", repr(e))
    face_net = None



def build_resnet18(num_classes: int):
   
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_emotion_model(model_path: str):
 
    model = build_resnet18(num_classes=len(CLASS_NAMES))

    state = torch.load(model_path, map_location=DEVICE)

    
    if isinstance(state, dict) and any(k.startswith("backbone.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("backbone.", "", 1)] = v
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
    """
    img_bgr: numpy array (H, W, 3) BGR (OpenCV)
    return: (img_bgr_with_boxes, results_list)
    """
    h, w = img_bgr.shape[:2]
    results = []

    
    if face_net is None:
    
        faces = [(0, 0, w, h)]
    else:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            det_conf = float(detections[0, 0, i, 2])
            if det_conf < conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append((x1, y1, x2, y2))

        if len(faces) == 0:
           
            faces = [(0, 0, w, h)]

    
    for (x1, y1, x2, y2) in faces:
        face_img_bgr = img_bgr[y1:y2, x1:x2]
        if face_img_bgr.size == 0:
            continue

        
        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

       
        face_tensor = transform_face(face_img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            label = CLASS_NAMES[int(preds.item())]
            confidence = float(conf.item())

        
        color = (0, 255, 0)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(
            img_bgr, text, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        results.append({
            "box": (x1, y1, x2 - x1, y2 - y1),
            "label": label,
            "confidence": confidence
        })

    return img_bgr, results
