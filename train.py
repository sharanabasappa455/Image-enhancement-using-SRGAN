import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator, VGGLoss  # <--- ENSURE VGGLoss IS IMPORTED
from dataset import SRDataset
from tqdm import tqdm
import os

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.0002
EPOCHS = 5
BATCH_SIZE = 4

def train():
    # 1. Verify path exists before starting
    train_dir = "data/train_images"
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: The folder '{train_dir}' does not exist.")
        print("   Did you run 'python download_data.py'?")
        return

    # 2. Load Dataset (Fixed Path)
    dataset = SRDataset(root_dir=train_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    vgg_loss = VGGLoss().to(DEVICE) # <--- CRITICAL FOR SHARPNESS
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR)
    opt_disc = optim.Adam(disc.parameters(), lr=LR)
    
    bce = nn.BCELoss()
    
    print(f"Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        loop = tqdm(loader)
        for idx, (lr, hr) in enumerate(loop):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            
            # --- Train Discriminator ---
            fake_hr = gen(lr)
            
            d_real = disc(hr)
            # Label smoothing (0.9 instead of 1.0 helps stability)
            loss_d_real = bce(d_real, torch.ones_like(d_real) - 0.1)
            
            d_fake = disc(fake_hr.detach())
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            
            loss_disc = (loss_d_real + loss_d_fake) / 2
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            d_fake_pred = disc(fake_hr)
            loss_gen_adv = bce(d_fake_pred, torch.ones_like(d_fake_pred))
            
            # THE FIX: Use VGG Loss instead of MSE
            loss_content = vgg_loss(fake_hr, hr) 
            
            # Combine losses (0.006 is the standard research weight)
            loss_gen = loss_content + 0.006 * loss_gen_adv
            
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            loop.set_postfix(epoch=epoch, d_loss=loss_disc.item(), g_loss=loss_gen.item())
        
        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(gen.state_dict(), "gen_model.pth")
            torch.save(disc.state_dict(), "disc_model.pth")
            print(f"✅ Models saved at epoch {epoch+1}")
            
    # Final Save
    torch.save(gen.state_dict(), "gen_model.pth")
    torch.save(disc.state_dict(), "disc_model.pth")
    print("Training Complete!")

if __name__ == "__main__":
    train()