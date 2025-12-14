import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np



model_path = r"C:\Users\MSI Raider\Downloads\DeepLearning_FaceDetection\emotion_cnn_best.pth"
img_size = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = EmotionCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(" Model loaded from:", model_path)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])



face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print(" Error loading Haar cascade")
    raise SystemExit



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Tidak bisa buka webcam")
    raise SystemExit

print("Tekan 'Q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame tidak terbaca")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        
        face_resized = cv2.resize(face_img, (img_size, img_size))

        
        face_tensor = transform(face_resized).unsqueeze(0).to(device)  

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            label = class_names[preds.item()]
            confidence = conf.item()

    
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Emotion Detection - CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
