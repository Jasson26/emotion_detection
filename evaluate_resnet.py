import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_resnet18_best.pth")
VAL_DIR = os.path.join(BASE_DIR, "archive", "images", "validation")

OUT_JSON = os.path.join(BASE_DIR, "class_accuracy.json")
OUT_BAR = os.path.join(BASE_DIR, "accuracy_per_class.png")
OUT_CM = os.path.join(BASE_DIR, "confusion_matrix.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

def build_resnet18(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def save_bar_chart(classes, acc_values, overall_acc, out_path):
    
    plt.figure(figsize=(10, 5))
    x = np.arange(len(classes))
    plt.bar(x, [a * 100 for a in acc_values])
    plt.xticks(x, classes, rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy per Class (Overall: {overall_acc*100:.2f}%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, ha="right")
    plt.yticks(tick_marks, classes)

    
    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation dir not found: {VAL_DIR}")

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    classes = val_ds.classes
    print("Validation classes:", classes)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_resnet18(num_classes=len(classes))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                cm[int(t), int(p)] += 1

    per_class = {}
    acc_values = []
    for i, name in enumerate(classes):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        acc = (correct / total) if total > 0 else 0.0
        acc_values.append(acc)
        per_class[name] = {"accuracy": acc, "correct": correct, "total": total}

    overall_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0

    payload = {
        "overall_accuracy": overall_acc,
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    
    save_bar_chart(classes, acc_values, overall_acc, OUT_BAR)
    save_confusion_matrix(cm, classes, OUT_CM)

    print(" Saved JSON :", OUT_JSON)
    print(" Saved BAR  :", OUT_BAR)
    print(" Saved CM   :", OUT_CM)
    print("Overall accuracy:", overall_acc)

if __name__ == "__main__":
    main()
