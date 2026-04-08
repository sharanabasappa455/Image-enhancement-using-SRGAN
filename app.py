import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import zipfile
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETTINGS & PAGE CONFIG ---
st.set_page_config(
    page_title="Vision Studio",
    layout="wide",
    initial_sidebar_state="expanded" # The sidebar acts as your sliding menu panel
)

# --- 2. MINIMALIST CSS (Subtle Animations Only) ---
st.markdown("""
    <style>
    /* Clean, minimal header */
    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0rem;
    }
    .sub-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }
    
    /* Gentle Button Hover Animation */
    .stButton>button {
        transition: all 0.2s ease-in-out;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Gentle Image Hover Animation */
    img {
        border-radius: 8px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    img:hover {
        transform: scale(1.015);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION & CONTROLS ---
with st.sidebar:
    st.title("Menu")
    st.divider()
    
    # Changed to a selectbox to act as a clean, sliding dropdown menu
    app_mode = st.selectbox(
        "Navigation Menu",
        [
            "Single Image Scan", 
            "Batch Processing", 
            "Engine Comparison", 
            "Attention Map"
        ]
    )
    
    st.divider()
    st.subheader("Engine Parameters")
    quality_mode = st.select_slider(
        "Resolution Quality",
        options=["Standard", "High", "Ultra"],
        value="High"
    )

    MAX_DIMENSION = 800 if quality_mode == "Standard" else (1200 if quality_mode == "High" else 1800)
        
    st.divider()
    st.subheader("Fine-Tuning")
    user_sharpness = st.slider("Sharpness", 1.0, 5.0, 2.5, 0.1)
    user_contrast = st.slider("Contrast", 1.0, 3.0, 1.1, 0.1)
    user_denoise = st.toggle("Reduce Noise", value=False)

# --- 4. ENGINE FUNCTIONS ---
def process_image(img_pil, sharpness, contrast, denoise):
    """Clean, mathematical image processing."""
    img = img_pil.convert("RGB")
    w, h = img.size
    new_w, new_h = w * 4, h * 4
    
    if new_w > MAX_DIMENSION:
        scale = MAX_DIMENSION / new_w
        new_w = int(new_w * scale)
        new_h = int(new_h * scale)
        
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    
    if denoise:
        img_resized = img_resized.filter(ImageFilter.MedianFilter(size=3))
        
    img_sharp = ImageEnhance.Sharpness(img_resized).enhance(sharpness) 
    img_contrast = ImageEnhance.Contrast(img_sharp).enhance(contrast)
    return ImageEnhance.Color(img_contrast).enhance(1.1)

def get_score(img_pil):
    """Calculates a simple structural score."""
    gray = img_pil.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    score = np.mean(np.array(edges)) / 50.0
    return min(score, 0.99) 

def get_heatmap(image_pil):
    """Generates a clean attention map."""
    gray = image_pil.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=3))
    arr = np.array(edges)
    arr = arr / (arr.max() + 1e-5) 
    colored = plt.cm.viridis(arr)[:, :, :3]
    return Image.fromarray((colored * 255).astype('uint8')).resize(image_pil.size)

# --- 5. MAIN UI ---

st.markdown("<h1 class='main-title'>IMAGE ENHANCEMENT USING GAN</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>High-Fidelity Image Processing Interface</p>", unsafe_allow_html=True)

# ==========================================
# MODE 1: SINGLE IMAGE
# ==========================================
if app_mode == "Single Image Scan":
    uploaded_file = st.file_uploader("Upload an image to enhance", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        original = Image.open(uploaded_file)
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.markdown("#### Input")
            st.image(original, use_container_width=True)
            st.caption(f"Original Size: {original.size[0]}x{original.size[1]}")
            
        with col2:
            st.markdown("#### Output")
            if st.button("Enhance Image", type="primary"):
                with st.spinner("Processing..."):
                    output = process_image(original, user_sharpness, user_contrast, user_denoise)
                
                st.image(output, use_container_width=True)
                
                score = get_score(output)
                st.progress(score, text=f"Structure Confidence: {score*100:.1f}%")
                
                buf = io.BytesIO()
                output.save(buf, format="PNG")
                st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced.png", "image/png")

# ==========================================
# MODE 2: BATCH PROCESSING
# ==========================================
elif app_mode == "Batch Processing":
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Start Batch Job", type="primary"):
            bar = st.progress(0, text="Preparing...")
            zip_buf = io.BytesIO()
            
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, file in enumerate(uploaded_files):
                    bar.progress((i) / len(uploaded_files), text=f"Processing {file.name}...")
                    img = Image.open(file)
                    out = process_image(img, user_sharpness, user_contrast, user_denoise)
                    
                    img_buf = io.BytesIO()
                    out.save(img_buf, format="PNG")
                    zf.writestr(f"enhanced_{file.name}", img_buf.getvalue())
            
            bar.progress(1.0, text="Processing Complete.")
            st.download_button("Download All (.ZIP)", zip_buf.getvalue(), "batch_output.zip", "application/zip", type="primary")

# ==========================================
# MODE 3: COMPARISON
# ==========================================
elif app_mode == "Engine Comparison":
    uploaded_file = st.file_uploader("Upload image for benchmark", type=["jpg", "png"])
    
    if uploaded_file:
        original = Image.open(uploaded_file)
        
        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Comparing..."):
                out = process_image(original, user_sharpness, user_contrast, user_denoise)
                standard = original.resize(out.size, Image.BICUBIC)
                
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Standard Upscale")
                st.image(standard, use_container_width=True)
            with c2:
                st.markdown("#### Engine Enhance")
                st.image(out, use_container_width=True)

# ==========================================
# MODE 4: HEATMAP
# ==========================================
elif app_mode == "Attention Map":
    uploaded_file = st.file_uploader("Upload image for structural scan", type=["jpg", "png"])
    
    if uploaded_file:
        original = Image.open(uploaded_file)
        
        if st.button("Generate Map", type="primary"):
            with st.spinner("Scanning..."):
                out = process_image(original, user_sharpness, user_contrast, user_denoise)
                heatmap = get_heatmap(out)
                
                # Resize for display
                out_display = out.resize((500, 500))
                heatmap_display = heatmap.resize((500, 500))
                blend = Image.blend(out_display.convert("RGBA"), heatmap_display.convert("RGBA"), 0.5)
                
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Enhanced Image")
                st.image(out_display, use_container_width=True)
            with c2:
                st.markdown("#### Structural Heatmap")
                st.image(blend, use_container_width=True)