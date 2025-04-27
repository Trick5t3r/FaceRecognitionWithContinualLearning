# Face Recognition with VGG and Continual Learning

This project implements a face recognition system based on the VGG architecture, adapted for continual learning. It allows for face recognition while progressively adapting to new identities without forgetting previous ones.

## Features

- Face recognition using VGG-16 as backbone
- Support for continual learning
- Training and evaluation on custom datasets
- Simple interface for inference and prediction

## Project Structure

```
.
├── data/                    # Directory containing the data
│   ├── train/              # Training images
│   └── val/                # Validation images
├── src/
│   └── vggfacialrecognition.py  # Main code
├── requirements.txt        # Project dependencies
├── .venv/                  # Python virtual environment
└── README.md              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Trick5t3r/FaceRecognitionWithContinualLearning.git
cd FaceRecognitionWithContinualLearning
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
 - https://www.kaggle.com/datasets/hearfool/vggface2
 - put it in data/train

## Usage

### Data Preparation

Organize your image data in the following structure:
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

### Training

To start training the model:
```bash
python src/vggfacialrecognition.py
```

### Prediction

To use the trained model for face recognition:
```python
from src.vggfacialrecognition import predict

# Load the model
model = load_model('vgg_face_recognition.pth')

# Make a prediction
predicted_identity = predict('path/to/image.jpg', model, class_names)
print(f"Predicted identity: {predicted_identity}")
```

## Configuration

Main parameters can be adjusted in the `vggfacialrecognition.py` file:
- `num_classes`: Number of identities to recognize
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate

## Continual Learning

The system is designed to support continual learning, enabling:
- Adding new identities without complete retraining
- Preserving previous knowledge
- Progressive adaptation to new data
