import os
import numpy as np
from itertools import islice

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PneumoniaDataset(Dataset):
    def __init__(self, data_dir: str, transforms = None):
        '''
        Parameters:
            data_dir (str): path to data directory
            transforms: transform 
        '''
        self.data_dir = data_dir # Assign the root_dir directly
        self.transforms = transforms
        self.image_folder = ImageFolder(root=self.data_dir)
        
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, label = self.image_folder.samples[idx]
        image = self.load_image(os.path.join(img_name))
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label
    
    def load_image(self, path):
        # Here you can implement custom image loading if needed
        return Image.open(path).convert('RGB')

# Example usage:
if __name__ == "__main__":
    # Define transformations
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = PneumoniaDataset(data_dir='data/train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    train_dataloader_iter = iter(train_dataloader)
    first_two_batches = islice(train_dataloader_iter, 2)

    # Iterate over the first two batches and print the information
    print("Printing the first 2 batches:")
    for batch_idx, (data, target) in enumerate(first_two_batches):
        print(f"Train Batch {batch_idx}, Data shape: {data.shape}, Target: {target}")