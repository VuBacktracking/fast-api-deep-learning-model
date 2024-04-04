import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.dataset import PneumoniaDataset
from models.pneumonia_model import PneumoniaModel
from config.pneumonia_cfg import PneumoniaDataConfig, ModelConfig

N_CLASSES = 2
SAVE_PATH = "models/weights/pneumonia_weights.pt"

train_transforms = transforms.Compose([
                    transforms.Resize((PneumoniaDataConfig.IMG_SIZE, PneumoniaDataConfig.IMG_SIZE)),
                    transforms.CenterCrop(PneumoniaDataConfig.IMG_SIZE),
                    transforms.RandomRotation(90),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(PneumoniaDataConfig.NORMALIZE_MEAN,
                                         PneumoniaDataConfig.NORMALIZE_STD)
                ])

val_transforms = transforms.Compose([
                    transforms.Resize((PneumoniaDataConfig.IMG_SIZE, PneumoniaDataConfig.IMG_SIZE)),
                    transforms.CenterCrop(PneumoniaDataConfig.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(PneumoniaDataConfig.NORMALIZE_MEAN,
                                         PneumoniaDataConfig.NORMALIZE_STD)
                ])

test_transforms = transforms.Compose([
                    transforms.Resize((PneumoniaDataConfig.IMG_SIZE, PneumoniaDataConfig.IMG_SIZE)),
                    transforms.CenterCrop(PneumoniaDataConfig.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(PneumoniaDataConfig.NORMALIZE_MEAN,
                                         PneumoniaDataConfig.NORMALIZE_STD)
                ])

# Create train and test datasets
train_dataset = PneumoniaDataset(root_dir='data/train', transforms=train_transforms)
val_dataset = PneumoniaDataset(root_dir='data/val', transforms=val_transforms)
test_dataset = PneumoniaDataset(root_dir='data/test', transforms=test_transforms)

# Create train and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=PneumoniaDataConfig.TRAIN_BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=PneumoniaDataConfig.VAL_BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=PneumoniaDataConfig.TEST_BATCH_SIZE, shuffle=False)

model = PneumoniaModel(N_CLASSES)
model.fit(train_dataloader, 
          val_dataloader, 
          learning_rate = ModelConfig.LEARNING_RATE,
          weight_decay= ModelConfig.WEIGHT_DECAY,
          num_epochs=ModelConfig.NUM_EPOCHS)

torch.save(model.state_dict(), SAVE_PATH)