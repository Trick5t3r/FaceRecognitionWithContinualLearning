import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# -------- Configuration --------
data_dir = 'data'  # Structure: data/train/<class>/*.jpg, data/val/<class>/*.jpg
num_classes = 100        # Ajuster selon le nombre d'identités
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Transforms pour les données --------
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- Chargement des datasets --------
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

# -------- Définition du modèle VGG --------
model = models.vgg16_bn(pretrained=True)
# Remplacement de la dernière couche FC pour le nombre de classes défini
def_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(def_features, num_classes)
model = model.to(device)

# -------- Criterion, Optimizer et Scheduler --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -------- Boucle d'entraînement --------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):  
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val  "):  
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}\n")

# -------- Sauvegarde du modèle --------
torch.save(model.state_dict(), 'vgg_face_recognition.pth')
print("Modèle sauvegardé sous 'vgg_face_recognition.pth'")

# -------- Fonction d'inférence --------
def predict(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

# Exemple d'utilisation:
# class_names = train_dataset.classes
# predicted_identity = predict('data/test/person1.jpg', model, class_names)
# print(f"Identité prédite: {predicted_identity}")
