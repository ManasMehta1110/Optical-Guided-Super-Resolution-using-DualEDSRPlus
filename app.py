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


# =========================
# Model loading
# =========================
@st.cache_resource
def load_model(model_path: str = "models/hls_ssl4eo_best.pth"):
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at {model_path}. Please train the model first.")
        return None
    model = DualEDSRPlus(n_resgroups=4, n_rcab=4, n_feats=64, upscale=2)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =========================
# TIFF processing
# =========================
def process_tiff_upload(uploaded_file, is_rgb: bool = False, is_thermal: bool = False, reference_profile=None):
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with rasterio.open(temp_path) as src:
                if is_rgb:
                    if src.count >= 3:
                        data = src.read([1, 2, 3])  # (3, H, W)
                        img_array = np.stack([norm_np(data[c]) for c in range(3)], axis=0)
                    else:
                        st.warning("RGB TIFF should have at least 3 bands.")
                        return None

                elif is_thermal:
                    if src.count >= 1:
                        data = src.read(1)  # (H, W)
                        img_array = norm_np(data)[np.newaxis, :, :]  # (1, H, W)
                    else:
                        st.warning("Thermal TIFF should have at least 1 band.")
                        return None

                if reference_profile is not None:
                    # Placeholder for reprojection if needed
                    pass

                os.remove(temp_path)
                return img_array

        except Exception as e:
            st.error(f"Error reading TIFF: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    return None


# =========================
# Display helpers
# =========================
def display_rgb_image(img: np.ndarray, title: str):
    st.markdown(f"**{title}**")
    display_img = np.transpose(np.clip(img, 0, 1), (1, 2, 0))
    st.image(display_img, clamp=True, channels='RGB')


def display_thermal_gray(img: np.ndarray, title: str):
    st.markdown(f"**{title}**")
    gray = np.clip(img.squeeze(), 0, 1)
    st.image(gray, clamp=True)


def display_thermal_colored(img: np.ndarray, title: str, cmap_name: str):
    st.markdown(f"**{title}**")
    base = np.clip(img.squeeze(), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(base)[..., :3]  # drop alpha
    st.image(colored, clamp=True)


# =========================
# Streamlit App
# =========================
st.title("HLS SSL4EO Super-Resolution Demo")
st.markdown(
    "Upload HR Optical (3-band TIFF), LR Thermal (1-band TIFF), and optionally GT HR Thermal (1-band TIFF). "
    "Assumes upscale factor=2, aligned grids."
)

# Model loading
model = load_model()
if model is None:
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# File uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("HR Optical (3-band TIFF)")
    hr_optical_upload = st.file_uploader("Upload HR RGB Optical TIFF", type=['tif', 'tiff'])
with col2:
    st.subheader("LR Thermal (1-band TIFF)")
    lr_thermal_upload = st.file_uploader("Upload LR Thermal TIFF", type=['tif', 'tiff'])

gt_col1, _ = st.columns(2)
with gt_col1:
    st.subheader("GT HR Thermal (Optional, 1-band TIFF)")
    gt_thermal_upload = st.file_uploader("Upload GT HR Thermal TIFF", type=['tif', 'tiff'])

# Colormap selector
cmap = st.selectbox("Thermal Colormap", ['hot', 'cool', 'viridis', 'plasma', 'inferno', 'gray'])

if st.button("Run Super-Resolution"):
    if hr_optical_upload is None or lr_thermal_upload is None:
        st.warning("Please upload HR Optical and LR Thermal TIFFs.")
        st.stop()

    # Process inputs
    hr_rgb = process_tiff_upload(hr_optical_upload, is_rgb=True)
    lr_thermal = process_tiff_upload(lr_thermal_upload, is_thermal=True)
    gt_thermal = process_tiff_upload(gt_thermal_upload, is_thermal=True) if gt_thermal_upload else None

    if hr_rgb is None or lr_thermal is None:
        st.error("Failed to process TIFFs.")
        st.stop()

    # Ensure shapes are compatible (upscale=2)
    lr_h, lr_w = lr_thermal.shape[1:]
    hr_h, hr_w = hr_rgb.shape[1:]

    if hr_h != 2 * lr_h or hr_w != 2 * lr_w:
        st.warning(
            f"Size mismatch: LR should be half HR size. "
            f"LR: ({lr_h},{lr_w}), HR: ({hr_h},{hr_w}). Resizing LR for demo."
        )
        lr_torch = torch.from_numpy(lr_thermal).unsqueeze(0).float()  # (1,1,H,W)
        lr_torch_resized = F.interpolate(
            lr_torch,
            size=(hr_h // 2, hr_w // 2),
            mode='bilinear',
            align_corners=False
        )
        lr_thermal = lr_torch_resized.squeeze(0).numpy()  # (1, H, W)
        lr_h, lr_w = lr_thermal.shape[1:]

    # Inference
    with torch.no_grad():
        lr_t = torch.from_numpy(lr_thermal).unsqueeze(0).to(DEVICE)  # (1,1,LRh,LRw)
        hr_o = torch.from_numpy(hr_rgb).unsqueeze(0).to(DEVICE)      # (1,3,HRh,HRw)
        sr_thermal = model(lr_t, hr_o).squeeze().cpu().numpy()       # (HRh,HRw)
        sr_thermal = np.clip(sr_thermal, 0, 1)

    st.header("Results")

    # ========= Grayscale row(s) =========
    if gt_thermal is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            display_rgb_image(hr_rgb, "HR Optical (RGB)")
        with c2:
            display_thermal_gray(lr_thermal[0], "LR Thermal (Grayscale)")
        with c3:
            display_thermal_gray(sr_thermal, "SR Thermal (Grayscale)")
        with c4:
            display_thermal_gray(gt_thermal[0], "GT HR Thermal (Grayscale)")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            display_rgb_image(hr_rgb, "HR Optical (RGB)")
        with c2:
            display_thermal_gray(lr_thermal[0], "LR Thermal (Grayscale)")
        with c3:
            display_thermal_gray(sr_thermal, "SR Thermal (Grayscale)")

    # ========= Colored row(s) =========
    st.subheader("Thermal Images with Colormap")

    if gt_thermal is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            display_thermal_colored(lr_thermal[0], "LR Thermal (Colored)", cmap)
        with c2:
            display_thermal_colored(sr_thermal, "SR Thermal (Colored)", cmap)
        with c3:
            display_thermal_colored(gt_thermal[0], "GT HR Thermal (Colored)", cmap)
    else:
        c1, c2 = st.columns(2)
        with c1:
            display_thermal_colored(lr_thermal[0], "LR Thermal (Colored)", cmap)
        with c2:
            display_thermal_colored(sr_thermal, "SR Thermal (Colored)", cmap)

    # ========= Metrics =========
    if gt_thermal is not None:
        psnr_val, ssim_val, rmse_val = compute_metrics(sr_thermal, gt_thermal[0])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("PSNR (dB)", f"{psnr_val:.2f}")
        with m2:
            st.metric("SSIM", f"{ssim_val:.4f}")
        with m3:
            st.metric("RMSE", f"{rmse_val:.4f}")
        st.success("Metrics computed between SR and GT.")
    else:
        st.info("Upload GT for metrics.")

# Instructions
with st.expander("Instructions"):
    st.markdown(
        """
        - **HR Optical**: 3-band TIFF (e.g., B04/B03/B02 or RGB) at high resolution (e.g., 256x256).
        - **LR Thermal**: 1-band thermal TIFF at low resolution (half size, e.g., 128x128).
        - **GT HR Thermal** (optional): 1-band ground truth high-res thermal TIFF for comparison.
        - Images should be aligned (same spatial extent/CRS).
        - Colormap applies to thermal visuals.
        - Model loaded from `models/hls_ssl4eo_best.pth`.
        - Requires `rasterio` for TIFF reading (add to requirements.txt).
        """
    )
