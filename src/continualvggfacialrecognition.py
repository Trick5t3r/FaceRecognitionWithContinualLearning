#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for face recognition with continual learning using iCaRL.
Supports grouping multiple new classes per incremental step,
capping total number of classes loaded, and sampling per class for evaluation.
"""

import os
import sys
import torch
import random
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from models.recognition_model import (
    FaceDataset,
    FaceRecognitionIcarlClassifier,
    train_icarl,
    test_icarl
)

# ============================================================================
# Main pipeline: training + evaluation with sampling per class
# ============================================================================
def main(
    data_dir='data/train',
    init_num=20,
    inc_step=3,
    memory_size=2000,
    batch_size=64,
    sample_per_class=100,
    epochs=5,
    lr=1e-4,
    temp=2.0,
    alpha=1.0,
    max_classes=None,
    device=None
):
    print("Loading training data...")

    # List all classes and cap if requested
    all_classes = sorted(os.listdir(data_dir))
    if max_classes and max_classes > 0:
        all_classes = all_classes[:max_classes]

    # Split into initial and new classes
    initial_classes = all_classes[:init_num]
    new_classes = all_classes[init_num:]
    classes_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # Prepare initial DataLoader
    init_paths, init_labels = [], []
    for cls in initial_classes:
        folder = os.path.join(data_dir, cls)
        for img in os.listdir(folder):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                init_paths.append(os.path.join(folder, img))
                init_labels.append(classes_idx[cls])
    init_loader = DataLoader(
        FaceDataset(init_paths, init_labels),
        batch_size=batch_size,
        shuffle=True
    )

    # Group new classes into increments
    new_class_groups = [
        new_classes[i:i + inc_step]
        for i in range(0, len(new_classes), inc_step)
    ]
    new_loaders = []
    for group in new_class_groups:
        paths, labels = [], []
        for cls in group:
            folder = os.path.join(data_dir, cls)
            for img in os.listdir(folder):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(folder, img))
                    labels.append(classes_idx[cls])
        new_loaders.append(
            DataLoader(
                FaceDataset(paths, labels),
                batch_size=batch_size,
                shuffle=True
            )
        )

    # Device setup
    device = device or (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    # Initialize model
    model = FaceRecognitionIcarlClassifier(
        num_initial_classes=len(initial_classes),
        device=device
    )

    print("Training model...")
    model, exemplars = train_icarl(
        model=model,
        initial_loader=init_loader,
        new_loaders=new_loaders,
        memory_size=memory_size,
        epochs=epochs,
        lr=lr,
        temp=temp,
        alpha=alpha
    )

    # Prepare test loader: random sampling of all seen images per class
    # build reverse mapping from index to class name
    idx2class = {v: k for k, v in classes_idx.items()}
    test_paths, test_labels = [], []
    for cls_idx in exemplars.keys():
        cls_name = idx2class[cls_idx]
        folder = os.path.join(data_dir, cls_name)
        all_imgs = [
            os.path.join(folder, img)
            for img in os.listdir(folder)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        sampled = all_imgs
        if len(all_imgs) > sample_per_class:
            sampled = random.sample(all_imgs, sample_per_class)
        for p in sampled:
            test_paths.append(p)
            test_labels.append(cls_idx)
    test_loader = DataLoader(
        FaceDataset(test_paths, test_labels),
        batch_size=batch_size,
        shuffle=False
    )

    print("Evaluating model on random samples per class...")
    test_icarl(model, test_loader)

if __name__ == '__main__':
    main(
        data_dir='data/train',
        init_num=20,
        inc_step=3,
        memory_size=360,
        batch_size=64,
        sample_per_class=10,
        epochs=2,
        lr=1e-4,
        temp=2.0,
        alpha=1.0,
        max_classes=30
    )
