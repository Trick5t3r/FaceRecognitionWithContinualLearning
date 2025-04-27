# Face Recognition with VGG and Continual Learning

Ce projet implémente un système de reconnaissance faciale basé sur l'architecture VGG, adapté pour l'apprentissage continu (continual learning). Il permet de reconnaître des visages tout en s'adaptant progressivement à de nouvelles identités sans oublier les précédentes.

## Fonctionnalités

- Reconnaissance faciale utilisant VGG-16 comme backbone
- Support pour l'apprentissage continu (continual learning)
- Entraînement et évaluation sur des ensembles de données personnalisés
- Interface simple pour l'inférence et la prédiction

## Structure du Projet

```
.
├── data/                    # Dossier contenant les données
│   ├── train/              # Images d'entraînement
│   └── val/                # Images de validation
├── src/
│   └── vggfacialrecognition.py  # Code principal
├── requirements.txt        # Dépendances du projet
├── .venv/                  # Environnement virtuel Python
└── README.md              # Documentation
```

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/Trick5t3r/FaceRecognitionWithContinualLearning.git
cd FaceRecognitionWithContinualLearning
```

2. Créez et activez l'environnement virtuel :
```bash
python3 -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
# ou
.venv\Scripts\activate     # Sur Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Préparation des données

Organisez vos données d'images dans la structure suivante :
```
data/
├── train/
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── person1/
    │   └── image1.jpg
    └── person2/
        └── image1.jpg
```

### Entraînement

Pour lancer l'entraînement du modèle :
```bash
python src/vggfacialrecognition.py
```

### Prédiction

Pour utiliser le modèle entraîné pour la reconnaissance faciale :
```python
from src.vggfacialrecognition import predict

# Charger le modèle
model = load_model('vgg_face_recognition.pth')

# Faire une prédiction
predicted_identity = predict('chemin/vers/image.jpg', model, class_names)
print(f"Identité prédite: {predicted_identity}")
```

## Configuration

Les paramètres principaux peuvent être ajustés dans le fichier `vggfacialrecognition.py` :
- `num_classes` : Nombre d'identités à reconnaître
- `batch_size` : Taille des lots pour l'entraînement
- `num_epochs` : Nombre d'époques d'entraînement
- `learning_rate` : Taux d'apprentissage

## Continual Learning

Le système est conçu pour supporter l'apprentissage continu, permettant :
- L'ajout de nouvelles identités sans réentraînement complet
- La préservation des connaissances précédentes
- L'adaptation progressive aux nouvelles données

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 