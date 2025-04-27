# src/vggfacialrecognition.py
import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from models.recognition_model import FaceDataset

# -------- Configuration --------
data_dir      = 'data/train'
num_classes   = 480
batch_size    = 32
num_epochs    = 10
learning_rate = 1e-4
val_split     = 0.2
seed          = 42
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 2) Chargement des chemins et labels
    classes    = sorted(os.listdir(data_dir))
    class2idx  = {cls: i for i, cls in enumerate(classes)}
    all_paths, all_labels = [], []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            all_paths.append(os.path.join(folder, fname))
            all_labels.append(class2idx[cls])

    # 3) Split train/val
    random.seed(seed)
    idxs = list(range(len(all_paths)))
    random.shuffle(idxs)
    split = int(val_split * len(idxs))
    train_idxs, val_idxs = idxs[split:], idxs[:split]

    train_paths  = [all_paths[i]  for i in train_idxs]
    train_labels = [all_labels[i] for i in train_idxs]
    val_paths    = [all_paths[i]  for i in val_idxs]
    val_labels   = [all_labels[i] for i in val_idxs]

    # 4) Création des Dataset + DataLoader (avec spawn)
    train_ds = FaceDataset(train_paths, train_labels)
    val_ds   = FaceDataset(val_paths,   val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(range(len(train_ds))),
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(range(len(val_ds))),
        num_workers=4
    )

    # 5) Modèle VGG16 adapté
    model = models.vgg16_bn(weights='DEFAULT')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # 6) Criterion / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 7) Boucle d'entraînement & validation
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} — Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1} — Train Loss: {running_loss/len(train_ds):.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} — Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        print(f"Epoch {epoch+1} — Val Acc: {correct/total:.4f}\n")

    # 8) Sauvegarde
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/vgg_face_recognition.pth')
    print("Modèle sauvegardé sous 'models/vgg_face_recognition.pth'")


if __name__ == '__main__':
    main()
