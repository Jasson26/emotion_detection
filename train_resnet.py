import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "archive", "images")  
model_save_path = os.path.join(BASE_DIR, "emotion_resnet18_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("data_dir:", data_dir)

if not os.path.isdir(data_dir):
    print("data_dir tidak ditemukan!")
    raise SystemExit

img_size = 224  
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "validation")

print("train_dir:", train_dir)
print("val_dir:", val_dir)

if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    print("Folder train/validation tidak ditemukan!")
    raise SystemExit

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print(" Dataset loaded")
print("Classes:", class_names)
print("Train size:", len(train_dataset))
print("Val size:  ", len(val_dataset))



def build_model(num_classes):
    print("Loading ResNet18 pretrained ImageNet...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    
    for p in model.parameters():
        p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)



def train_model(num_epochs=10):
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

                if (i + 1) % 50 == 0:
                    print(f"{phase} | batch {i+1}/{len(dataloader)}")

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_save_path)
                print(f" New best model saved (val_acc={best_val_acc:.4f})")

    print("\nTraining done. Best val_acc:", best_val_acc)
    model.load_state_dict(best_model_wts)

if __name__ == "__main__":
    train_model(num_epochs=10)  
    print("\n ResNet18 model saved to:", model_save_path)
