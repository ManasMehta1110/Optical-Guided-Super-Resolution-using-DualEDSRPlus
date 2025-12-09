import os
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim
import math
from typing import Optional
import io
import rasterio
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import reproject

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="DualEDSR+ Super-Resolution",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and larger images
st.markdown('''
    <style>
        .main {
            padding: 2rem;
        }
        .stImage {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .metric-card {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            margin-top: 25px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1f77b4;
            color: #1f77b4;
        }
        h1 {
            text-align: center;
            color: #1f77b4;
            padding: 20px 0;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 900;
            color: #0d47a1;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .upload-box {
            background-color: #e3f2fd;
            padding: 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #1f77b4;
        }
        .upload-title {
            font-size: 16px;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 8px;
        }
        .upload-text {
            font-size: 15px;
            color: #333;
            line-height: 1.6;
            font-weight: 500;
        }
        .note-text {
            font-size: 15px;
            color: #d32f2f;
            font-weight: 600;
            margin-top: 8px;
        }
    </style>
''', unsafe_allow_html=True)

# =========================
# Model definitions
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avgpool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 2), 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv(x)
        return x * att


class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad)
        )
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.res_scale = 0.1

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, channels, n_rcab=4):
        super().__init__()
        layers = [RCAB(channels) for _ in range(n_rcab)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + x


class LearnedUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(
            in_channels,
            out_channels * (scale * scale),
            kernel_size=3,
            padding=1
        )
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size=None):
        x = self.proj(x)
        x = F.pixel_shuffle(x, self.scale)
        x = self.post(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class DualEDSRPlus(nn.Module):
    def __init__(self, n_resgroups=4, n_rcab=4, n_feats=64, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.n_feats = n_feats

        self.convT_in = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO_in = nn.Conv2d(3, n_feats, 3, padding=1)

        self.t_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])
        self.o_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])

        self.t_upsampler = LearnedUpsampler(n_feats, n_feats, scale=upscale)

        self.convFuse = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1)
        self.fuse_ca = ChannelAttention(n_feats)
        self.fuse_sa = SpatialAttention(n_feats)

        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.convOut = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xT, xO):
        fT = F.relu(self.convT_in(xT))
        fO = F.relu(self.convO_in(xO))

        fT = self.t_groups(fT)
        fO = self.o_groups(fO)

        fT_up_raw = self.t_upsampler(fT)

        target_hw = (fO.shape[2], fO.shape[3])
        fT_up = F.interpolate(fT_up_raw, size=target_hw, mode="bilinear", align_corners=False)

        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.fuse_ca(f)
        f = self.fuse_sa(f)
        f = self.refine(f)
        out = self.convOut(f)
        return out


# =========================
# Utility functions
# =========================
def norm_np(a: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0,1] with NaN/Inf protection."""
    a = np.array(a, dtype=np.float32)
    if np.isnan(a).any() or np.isinf(a).any():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """PSNR / SSIM / RMSE on [0,1] normalized arrays."""
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    mse = float(np.mean((pred - target) ** 2))
    if not np.isfinite(mse) or mse < 1e-12:
        psnr_val = 100.0
        rmse_val = 0.0
    else:
        psnr_val = 10 * math.log10(1.0 / mse)
        rmse_val = math.sqrt(mse)

    try:
        ssim_val = ssim(target, pred, data_range=1.0)
    except Exception:
        ssim_val = 0.0

    return psnr_val, ssim_val, rmse_val


def save_tiff(data: np.ndarray, filename: str):
    """Save numpy array as TIFF file in bytes"""
    temp_path = f"temp_{filename}"
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype
    ) as dst:
        dst.write(data, 1)
    
    with open(temp_path, 'rb') as f:
        file_bytes = f.read()
    
    os.remove(temp_path)
    return file_bytes


# =========================
# Model loading - FIXED VERSION
# =========================
@st.cache_resource
def load_model(model_path: str = "models/hls_ssl4eo_best.pth"):
    """Load model with proper error handling"""
    
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at {model_path}. Please ensure the file exists.")
        return None
    
    try:
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Less than 1KB - likely a pointer file
            st.error("Model file appears to be corrupted or a Git LFS pointer file. Please use Git LFS or Google Drive option.")
            return None
        
        model = DualEDSRPlus(n_resgroups=4, n_rcab=4, n_feats=64, upscale=2)
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# =========================
# TIFF processing
# =========================
def process_tiff_upload(uploaded_file, is_rgb: bool = False, is_thermal: bool = False):
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with rasterio.open(temp_path) as src:
                if is_rgb:
                    if src.count >= 3:
                        data = src.read([1, 2, 3])
                        img_array = np.stack([norm_np(data[c]) for c in range(3)], axis=0)
                    else:
                        st.warning("RGB TIFF should have at least 3 bands.")
                        return None

                elif is_thermal:
                    if src.count >= 1:
                        data = src.read(1)
                        img_array = norm_np(data)[np.newaxis, :, :]
                    else:
                        st.warning("Thermal TIFF should have at least 1 band.")
                        return None

                os.remove(temp_path)
                return img_array

        except Exception as e:
            st.error(f"Error reading TIFF: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    return None


# =========================
# Display helpers - ORIGINAL DIMENSIONS
# =========================
def display_rgb_image(img: np.ndarray, title: str):
    """Display RGB image with title in original dimensions."""
    st.markdown(f"**{title}**")
    display_img = np.transpose(np.clip(img, 0, 1), (1, 2, 0))
    st.image(display_img, use_column_width=False, clamp=True, channels='RGB')


def display_thermal_gray(img: np.ndarray, title: str):
    """Display thermal grayscale image with title in original dimensions."""
    st.markdown(f"**{title}**")
    gray = np.clip(img.squeeze(), 0, 1)
    st.image(gray, use_column_width=False, clamp=True)


def display_thermal_colored(img: np.ndarray, title: str, cmap_name: str):
    """Display thermal image with colormap in original dimensions."""
    st.markdown(f"**{title}**")
    base = np.clip(img.squeeze(), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(base)[..., :3]
    st.image(colored, use_column_width=False, clamp=True)


def display_metrics_card(psnr_val, ssim_val, rmse_val):
    """Display metrics in an attractive card layout - BOLD VALUES"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>PSNR (dB)</div>
            <div class='metric-value'><b>{:.2f}</b></div>
        </div>
        """.format(psnr_val), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>SSIM</div>
            <div class='metric-value'><b>{:.4f}</b></div>
        </div>
        """.format(ssim_val), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>RMSE</div>
            <div class='metric-value'><b>{:.4f}</b></div>
        </div>
        """.format(rmse_val), unsafe_allow_html=True)


# =========================
# Streamlit App - MAIN
# =========================
st.title("DualEDSR+ Super-Resolution Demo")

st.markdown("""
<div class='upload-box'>
    <div class='upload-title'>Upload HR Optical (3-band TIFF), LR Thermal (1-band TIFF), and GT HR Thermal (1-band TIFF).</div>
    <div class='note-text'>Note: Assumes upscale factor=2 with aligned grids.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")
    
    cmap = st.selectbox(
        "Thermal Colormap",
        ['hot', 'cool', 'viridis', 'plasma', 'inferno', 'gray'],
        help="Choose colormap for thermal visualization"
    )
    
    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
    1. **HR Optical**: 3-band TIFF (RGB) at high resolution
    2. **LR Thermal**: 1-band thermal TIFF at half resolution
    3. **GT HR Thermal**: Ground truth thermal image
    """)

# Main content area - File uploaders
st.markdown("<div class='section-title'>Input Files</div>", unsafe_allow_html=True)

col_optical, col_thermal = st.columns(2)

with col_optical:
    st.markdown("#### HR Optical (3-band TIFF)")
    hr_optical_upload = st.file_uploader(
        "Upload HR RGB Optical TIFF",
        type=['tif', 'tiff'],
        key='hr_optical'
    )

with col_thermal:
    st.markdown("#### LR Thermal (1-band TIFF)")
    lr_thermal_upload = st.file_uploader(
        "Upload LR Thermal TIFF",
        type=['tif', 'tiff'],
        key='lr_thermal'
    )

col_gt, _ = st.columns(2)
with col_gt:
    st.markdown("#### GT HR Thermal (1-band TIFF)")
    gt_thermal_upload = st.file_uploader(
        "Upload GT HR Thermal TIFF",
        type=['tif', 'tiff'],
        key='gt_thermal'
    )

# Model loading
model = load_model()
if model is None:
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# Run button
st.markdown("---")
if st.button("Run Super-Resolution", use_container_width=True):
    if hr_optical_upload is None or lr_thermal_upload is None:
        st.warning("Please upload both HR Optical and LR Thermal TIFFs.")
        st.stop()

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process inputs
    status_text.text("Processing input files...")
    progress_bar.progress(20)
    
    hr_rgb = process_tiff_upload(hr_optical_upload, is_rgb=True)
    lr_thermal = process_tiff_upload(lr_thermal_upload, is_thermal=True)
    gt_thermal = process_tiff_upload(gt_thermal_upload, is_thermal=True) if gt_thermal_upload else None

    if hr_rgb is None or lr_thermal is None:
        st.error("Failed to process TIFFs.")
        st.stop()

    progress_bar.progress(40)

    # Ensure shapes are compatible (upscale=2)
    lr_h, lr_w = lr_thermal.shape[1:]
    hr_h, hr_w = hr_rgb.shape[1:]

    if hr_h != 2 * lr_h or hr_w != 2 * lr_w:
        st.warning(
            f"Size mismatch detected: LR should be half HR size. "
            f"LR: ({lr_h},{lr_w}), HR: ({hr_h},{hr_w}). Resizing LR for demo."
        )
        lr_torch = torch.from_numpy(lr_thermal).unsqueeze(0).float()
        lr_torch_resized = F.interpolate(
            lr_torch,
            size=(hr_h // 2, hr_w // 2),
            mode='bilinear',
            align_corners=False
        )
        lr_thermal = lr_torch_resized.squeeze(0).numpy()
        lr_h, lr_w = lr_thermal.shape[1:]

    progress_bar.progress(60)

    # Inference
    status_text.text("Running super-resolution inference...")
    with torch.no_grad():
        lr_t = torch.from_numpy(lr_thermal).unsqueeze(0).to(DEVICE)   # (1,1,h,w)
        hr_o = torch.from_numpy(hr_rgb).unsqueeze(0).to(DEVICE)       # (1,3,H,W)

        # Model SR
        sr_thermal = model(lr_t, hr_o).squeeze().cpu().numpy()
        sr_thermal = np.clip(sr_thermal, 0, 1)

        # Bicubic baseline (upsample LR to HR)
        bicubic_hr = F.interpolate(
            lr_t, size=(hr_h, hr_w), mode='bicubic', align_corners=False
        ).squeeze().cpu().numpy()
        bicubic_hr = np.clip(bicubic_hr, 0, 1)

    progress_bar.progress(80)
    status_text.text("Processing complete!")
    progress_bar.progress(100)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # ========= RESULTS SECTION =========
    st.markdown("<div class='section-title'>Results - Grayscale Images</div>", unsafe_allow_html=True)
    
    if gt_thermal is not None:
        # Show HR optical on its own row
        display_rgb_image(hr_rgb, "HR Optical (RGB)")

        # Second row: LR, Bicubic, SR, GT
        cols = st.columns(4)
        with cols[0]:
            display_thermal_gray(lr_thermal[0], "LR Thermal (Grayscale)")
        with cols[1]:
            display_thermal_gray(bicubic_hr, "Bicubic Upsampled Thermal (Grayscale)")
        with cols[2]:
            display_thermal_gray(sr_thermal, "SR Thermal (Grayscale)")
        with cols[3]:
            display_thermal_gray(gt_thermal[0], "GT HR Thermal (Grayscale)")
    else:
        cols = st.columns(4)
        with cols[0]:
            display_rgb_image(hr_rgb, "HR Optical (RGB)")
        with cols[1]:
            display_thermal_gray(lr_thermal[0], "LR Thermal (Grayscale)")
        with cols[2]:
            display_thermal_gray(bicubic_hr, "Bicubic Upsampled Thermal (Grayscale)")
        with cols[3]:
            display_thermal_gray(sr_thermal, "SR Thermal (Grayscale)")

    st.divider()

    # ========= Colored Comparison =========
    st.markdown("<div class='section-title'>Results - Thermal Images with Colormap</div>", unsafe_allow_html=True)

    if gt_thermal is not None:
        cols = st.columns(4)
        with cols[0]:
            display_thermal_colored(lr_thermal[0], f"LR Thermal ({cmap})", cmap)
        with cols[1]:
            display_thermal_colored(bicubic_hr, f"Bicubic Upsampled Thermal ({cmap})", cmap)
        with cols[2]:
            display_thermal_colored(sr_thermal, f"SR Thermal ({cmap})", cmap)
        with cols[3]:
            display_thermal_colored(gt_thermal[0], f"GT HR Thermal ({cmap})", cmap)
    else:
        cols = st.columns(3)
        with cols[0]:
            display_thermal_colored(lr_thermal[0], f"LR Thermal ({cmap})", cmap)
        with cols[1]:
            display_thermal_colored(bicubic_hr, f"Bicubic Upsampled Thermal ({cmap})", cmap)
        with cols[2]:
            display_thermal_colored(sr_thermal, f"SR Thermal ({cmap})", cmap)

    st.divider()

    # ========= Metrics =========
    if gt_thermal is not None:
        st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)

        # SR vs GT
        psnr_sr, ssim_sr, rmse_sr = compute_metrics(sr_thermal, gt_thermal[0])
        st.markdown("**SR vs GT**")
        display_metrics_card(psnr_sr, ssim_sr, rmse_sr)

        # Bicubic vs GT
        psnr_bi, ssim_bi, rmse_bi = compute_metrics(bicubic_hr, gt_thermal[0])
        st.markdown("**Bicubic vs GT**")
        display_metrics_card(psnr_bi, ssim_bi, rmse_bi)

        st.success("Metrics computed for both SR and Bicubic against GT!")
    else:
        st.info("Upload GT thermal image to compute metrics.")

    st.divider()

    # ========= Download SR & Bicubic Images =========
    st.markdown("<div class='section-title'>Download Super-Resolution & Bicubic Images</div>", unsafe_allow_html=True)
    
    sr_tiff_bytes = save_tiff(sr_thermal.astype(np.float32), "sr_thermal.tif")
    bicubic_tiff_bytes = save_tiff(bicubic_hr.astype(np.float32), "bicubic_thermal.tif")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Download SR Thermal (TIFF)",
            data=sr_tiff_bytes,
            file_name="sr_thermal_output.tif",
            mime="image/tiff",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="Download Bicubic Thermal (TIFF)",
            data=bicubic_tiff_bytes,
            file_name="bicubic_thermal_output.tif",
            mime="image/tiff",
            use_container_width=True
        )
    
    st.divider()
    st.info("Processing complete! You can now try with different images.")
