import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_dir = r"C:\Users\MSI Raider\Downloads\DeepLearning_FaceDetection\archive\images"
model_save_path = r"C:\Users\MSI Raider\Downloads\DeepLearning_FaceDetection\emotion_cnn_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("data_dir:", data_dir)

if not os.path.isdir(data_dir):
    print("ERROR: data_dir tidak ditemukan!")
    raise SystemExit

img_size = 48  

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "validation")

print("train_dir:", train_dir)
print("val_dir:", val_dir)

if not os.path.isdir(train_dir):
    print("ERROR: Folder train tidak ditemukan!")
    raise SystemExit
if not os.path.isdir(val_dir):
    print(" ERROR: Folder validation tidak ditemukan!")
    raise SystemExit

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Dataset loaded")
print("Classes:", class_names)
print("Train size:", len(train_dataset))
print("Val size:  ", len(val_dataset))



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

model = EmotionCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  



def train_model(num_epochs=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += labels.size(0)

                if (i + 1) % 20 == 0:
                    print(f"{phase} | batch {i+1}/{len(dataloader)}")

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_save_path)
                print(f"✅ New best model saved (val_acc={best_val_acc:.4f})")

    print("\nTraining done. Best val_acc:", best_val_acc)
    model.load_state_dict(best_model_wts)

if __name__ == "__main__":
    train_model(num_epochs=5)
    print("\n Model saved to:", model_save_path)
