from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

# -------- Définition du Dataset sans détection --------
class FaceDataset(Dataset):
    def __init__(self, img_paths, labels):
        """
        img_paths : list of str, chemins vers les images
        labels    : list of int, étiquettes associées
        """
        self.img_paths = img_paths
        self.labels    = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label    = self.labels[idx]

        # Chargement
        img = Image.open(img_path).convert('RGB')
        # -------- Exemple de pipeline de pré-traitement --------
        face_transforms = transforms.Compose([
            transforms.Resize(256),              # redimensionne le plus petit côté à 256 px
            transforms.CenterCrop(128),          # recadre un carré 128×128
            transforms.RandomHorizontalFlip(),   # augmentation au hasard
            transforms.ToTensor(),               # PIL→Tensor [0,1]
            transforms.Normalize(                # normalize sur la moyenne/std d'ImageNet
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])
        img = face_transforms(img)

        return img, label