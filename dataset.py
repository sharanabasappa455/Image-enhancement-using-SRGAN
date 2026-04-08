import os
import torch  # <--- This was missing!
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Only allow standard image formats
        self.image_files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), # Resize all HR images to 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Create High Res (HR)
        hr_image = self.transform(image)
        
        # Create Low Res (LR) by downsampling HR
        # We use torch.nn.functional here, so we needed 'import torch'
        lr_image = torch.nn.functional.interpolate(
            hr_image.unsqueeze(0), 
            scale_factor=0.25, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        return lr_image, hr_image