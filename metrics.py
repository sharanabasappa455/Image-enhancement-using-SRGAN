import torch
import torch.nn as nn
from model import Generator
from dataset import SRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def calculate_psnr(img1, img2):
    mse = nn.MSELoss()(img1, img2)
    if mse == 0:
        return 100
    return 10 * math.log10(1. / mse.item())

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    dataset = SRDataset(root_dir="datatrainimages")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = Generator().to(device)
    model.load_state_dict(torch.load("gen_model.pth", map_location=device))
    model.eval()
    
    total_psnr = 0
    count = 0
    
    print("Calculating PSNR Score...")
    
    with torch.no_grad():
        for lr, hr in tqdm(loader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            fake_hr = model(lr)
            
            # Undo normalization (-1,1) -> (0,1) for accurate measurement
            fake_hr = (fake_hr * 0.5) + 0.5
            hr = (hr * 0.5) + 0.5
            
            psnr = calculate_psnr(fake_hr, hr)
            total_psnr += psnr
            count += 1
            
    avg_psnr = total_psnr / count
    print(f"\n============================")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"============================")
    
    if avg_psnr > 20:
        print("✅ Result: Acceptable quality for a basic GAN.")
    if avg_psnr > 25:
        print("🚀 Result: Good quality! The model learned well.")

if __name__ == "__main__":
    evaluate()