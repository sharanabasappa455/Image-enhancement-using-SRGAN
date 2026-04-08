import torch
from model import Discriminator
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def analyze_image(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the "Police" (Discriminator)
    model = Discriminator().to(device)
    # Ensure you have run train.py recently to create this file!
    try:
        model.load_state_dict(torch.load("disc_model.pth", map_location=device))
    except FileNotFoundError:
        print("❌ Error: 'disc_model.pth' not found. Please run train.py briefly to save the discriminator.")
        return

    model.eval()
    
    # 2. Process the Image (Resize to 128x128 as the model expects)
    img = Image.open(image_path).convert("RGB")
    original_img = img.copy() # Keep for display
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 3. Get the "Realness Score"
    with torch.no_grad():
        score = model(img_tensor).item() # Returns a number between 0 and 1
        
    # 4. Interpret Result
    print(f"\nAnalyzing: {image_path}")
    print(f"Raw Score: {score:.4f}")
    
    verdict = "Unknown"
    color = "black"
    
    # These thresholds depend on how much you trained
    if score > 0.5:
        verdict = "REAL IMAGE"
        color = "green"
        print("✅ Verdict: This looks like a REAL image.")
    else:
        verdict = "AI GENERATED"
        color = "red"
        print("🤖 Verdict: This looks like an AI GENERATED image.")

    # 5. Show Result
    plt.figure(figsize=(6,6))
    plt.imshow(original_img)
    plt.title(f"{verdict}\n(Realness Score: {score:.2f})", color=color, fontweight='bold')
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Test on your generated image
    # Make sure to run inference.py first to create a result, or use any image
    image_to_test = "test_images/test.jpg" 
    analyze_image(image_to_test)