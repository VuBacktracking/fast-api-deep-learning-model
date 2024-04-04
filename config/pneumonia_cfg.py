import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class PneumoniaDataConfig: 
    N_CLASSES = 2
    IMG_SIZE = 224
    ID2DLABEL = {0: 'Normal', 1: 'Pneumonia'} 
    LABEL2ID = {'Normal': 0, 'Pneumonia': 1} 
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64

class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = 'resnet152'
    MODEL_WEIGHT = ROOT_DIR / 'models' / 'weights' / 'pneumonia_weights.pt' 
    DEVICE = 'cpu'
    LEARNING_RATE = 1e-3  # Learning rate for model training
    WEIGHT_DECAY = 1e-5  # Weight decay for regularization
    NUM_EPOCHS = 15  # Number of training epochs