import torch
from model import Generator
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def run_inference(image_path):
    # SETTINGS: Tweak this if you still get errors
    # 600px input -> 2400px output (Safe for most GPUs)
    MAX_DIMENSION = 600 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # 1. Load Model
    model = Generator().to(device)
    try:
        model.load_state_dict(torch.load("gen_model.pth", map_location=device))
    except FileNotFoundError:
        print("❌ Error: Model file not found!")
        return
    model.eval()
    
    # 2. Load Image
    img = Image.open(image_path).convert("RGB")
    old_w, old_h = img.size
    print(f"Original Input Size: {old_w} x {old_h}")
    
    # 3. Smart Resize (Prevent Crash)
    # If image is too huge, shrink it slightly to fit in VRAM
    scale_ratio = 1.0
    if max(old_w, old_h) > MAX_DIMENSION:
        scale_ratio = MAX_DIMENSION / max(old_w, old_h)
        new_w = int(old_w * scale_ratio)
        new_h = int(old_h * scale_ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"⚠️ Image too large! Resizing input to {new_w}x{new_h} to prevent crash.")
    
    # 4. Prepare Input Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 5. Generate (Upscale)
    print("AI is upscaling... ⏳")
    try:
        with torch.no_grad():
            generated = model(input_tensor)
    except RuntimeError as e:
        print(f"❌ Error: Your computer ran out of memory. Try reducing MAX_DIMENSION in the code.")
        print(f"Technical error: {e}")
        return

    # 6. Process Output
    generated = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated = (generated * 0.5) + 0.5 # De-normalize
    
    # Clip values to 0-1 range (removes weird color artifacts)
    generated = generated.clip(0, 1)
    
    # Show Stats
    out_h, out_w, _ = generated.shape
    print(f"✅ FINAL OUTPUT SIZE: {out_w} x {out_h}")
    print(f"🚀 Resolution Multiplier: 4x (Total pixels: 16x)")

    # 7. Display Results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Input ({img.width}x{img.height})")
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"SRGAN Output ({out_w}x{out_h})")
    plt.imshow(generated)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference("test_images/test.jpg")