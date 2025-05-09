#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for face recognition with continual learning using iCaRL.
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path if needed
sys.path.append(os.getcwd())

# ============================================================================
# Dataset for face recognition
# ============================================================================
class FaceDataset(Dataset):
    """Custom dataset for face images."""
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.labels_unique = sorted(set(labels))
        self.paths_by_class = {
            cls: [p for p, l in zip(img_paths, labels) if l == cls]
            for cls in self.labels_unique
        }
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

# ============================================================================
# iCaRL classifier with extensible softmax head
# ============================================================================
class FaceRecognitionIcarlClassifier(nn.Module):
    def __init__(self, num_initial_classes, device=None):
        super().__init__()
        backbone = models.vgg16_bn(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(
            *backbone.features,
            backbone.avgpool,
            nn.Flatten()
        )
        self.feat_dim = backbone.classifier[0].in_features
        self.classifier = nn.Linear(self.feat_dim, num_initial_classes)
        self.device = device or torch.device('cpu')
        self.to(self.device)

    def add_class(self):
        old_w = self.classifier.weight.data.clone()
        old_b = self.classifier.bias.data.clone()
        old_n = old_w.size(0)

        new_layer = nn.Linear(self.feat_dim, old_n + 1)
        new_layer.weight.data[:old_n] = old_w
        new_layer.bias.data[:old_n] = old_b

        self.classifier = new_layer.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits, feats

# ============================================================================
# Herding selection of exemplars
# ============================================================================
def herding_selection(paths, m, model):
    feats = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            img, _ = FaceDataset([p], [0])[0]
            img = img.unsqueeze(0).to(model.device)
            f = F.normalize(model.feature_extractor(img).squeeze(0).cpu(), p=2, dim=0)
            feats.append(f)
    feats = torch.stack(feats)
    class_mean = feats.mean(dim=0)
    selected, S = [], torch.zeros_like(class_mean)
    for _ in range(m):
        D = ((class_mean.unsqueeze(0) - (feats + S)/(len(selected)+1))**2).sum(dim=1)
        for idx in selected:
            D[idx] = float('inf')
        i = D.argmin().item()
        selected.append(i)
        S += feats[i]
    return [paths[i] for i in selected]

# ============================================================================
# iCaRL training pipeline
# ============================================================================
def train_icarl(
    model,
    initial_loader,
    new_loaders,
    memory_size,
    epochs=5,
    lr=1e-4,
    temp=2.0,
    alpha=1.0
):
    """
    iCaRL training with built-in validation:
     - Initial phase: validation on initial_loader itself.
     - Incremental phases: validation on current exemplar sets.
    """
    # --- 0) Track which classes have been learned so far ---
    seen_classes = set(initial_loader.dataset.labels_unique)

    # --- 1) Initial training phase ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = total_samples = 0
        for x, y in tqdm(initial_loader, desc=f"Init Train — Epoch {epoch+1}/{epochs}"):
            x, y = x.to(model.device), y.to(model.device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item() * x.size(0)
            total_samples += x.size(0)
        print(f"Init Train — Epoch {epoch+1}/{epochs} — Loss: {running_loss/total_samples:.4f}")

    # --- 2) Build initial exemplars ---
    classes_init = initial_loader.dataset.labels_unique
    exemplar_sets = {}
    k = memory_size // len(classes_init)
    for cls in classes_init:
        paths = initial_loader.dataset.paths_by_class[cls]
        exemplar_sets[cls] = herding_selection(paths, k, model)

    # --- 3) Initial validation on exemplars ---
    model.eval()
    # build loader from exemplar_sets
    val_ds = ConcatDataset([
        FaceDataset(exemplar_sets[cls], [cls] * len(exemplar_sets[cls]))
        for cls in sorted(exemplar_sets.keys())
    ])
    val_loader = DataLoader(
        val_ds,
        batch_size=initial_loader.batch_size,
        shuffle=False
    )
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Init Val"):
            x, y = x.to(model.device), y.to(model.device)
            preds = model(x)[0].argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Init — Val Accuracy: {correct/total:.4f}")

    # --- 4) Incremental phases ---
    for i, loader in enumerate(new_loaders, 1):
        new_classes = loader.dataset.labels_unique
        num_new = len(new_classes)
        seen_classes.update(new_classes)

        # snapshot old model for distillation
        old_model = copy.deepcopy(model).eval()
        old_num = model.classifier.out_features

        # add new heads
        for _ in range(num_new):
            model.add_class()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # combine new data + exemplars
        exemplar_ds = [
            FaceDataset(paths, [cls]*len(paths))
            for cls, paths in exemplar_sets.items()
        ]
        combined_ds = ConcatDataset([loader.dataset] + exemplar_ds)
        comb_loader = DataLoader(
            combined_ds,
            batch_size=loader.batch_size,
            shuffle=True
        )

        # training on combined
        for epoch in range(epochs):
            model.train()
            running_loss = total_samples = 0
            for x, y in tqdm(comb_loader,
                             desc=f"Inc {i}/{len(new_loaders)} — Epoch {epoch+1}/{epochs}"):
                x, y = x.to(model.device), y.to(model.device)
                logits, _ = model(x)

                # classification loss
                loss_ce = F.cross_entropy(logits, y)
                # distillation on old classes
                with torch.no_grad():
                    old_logits, _ = old_model(x)
                    p_old = F.softmax(old_logits / temp, dim=1)
                p_new = F.log_softmax(logits[:, :old_num] / temp, dim=1)
                loss_kd = F.kl_div(p_new, p_old, reduction='batchmean') * (temp**2)

                loss = loss_ce + alpha * loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.detach().item() * x.size(0)
                total_samples += x.size(0)
            print(f"Inc {i}/{len(new_loaders)} — Epoch {epoch+1}/{epochs} — Loss: {running_loss/total_samples:.4f}")

        # --- 5) Update exemplars before validation ---
        total_classes = len(exemplar_sets) + num_new
        k = memory_size // total_classes
        for cls in exemplar_sets:
            exemplar_sets[cls] = exemplar_sets[cls][:k]
        for cls in new_classes:
            paths_new = loader.dataset.paths_by_class[cls]
            exemplar_sets[cls] = herding_selection(paths_new, k, model)

        # --- 6) Validation on updated exemplars ---
        model.eval()
        val_ds = ConcatDataset([
            FaceDataset(exemplar_sets[cls], [cls] * len(exemplar_sets[cls]))
            for cls in sorted(exemplar_sets.keys())
        ])
        val_loader = DataLoader(
            val_ds,
            batch_size=loader.batch_size,
            shuffle=False
        )
        correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Inc {i}/{len(new_loaders)} — Val"):
                x, y = x.to(model.device), y.to(model.device)
                preds = model(x)[0].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Inc {i}/{len(new_loaders)} — Val Accuracy: {correct/total:.4f}")

    return model, exemplar_sets

# ============================================================================
# Evaluation function for a trained iCaRL model
# ============================================================================
def test_icarl(model, test_loader, device=None):
    """Evaluate overall accuracy of an iCaRL model on a test DataLoader."""
    device = device or model.device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc * 100:.2f}% ({correct}/{total})")
    return acc
