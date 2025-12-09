# 🌋 InfraNova — Optical-Guided Thermal Super-Resolution

InfraNova is an advanced deep-learning framework designed to **super-resolve low-resolution satellite thermal imagery** using **high-resolution optical guidance**.  
The system reconstructs **10m thermal resolution** from **30m thermal sensors** while maintaining **physical radiometric consistency** and **spatial detail**.

---

## 🚀 Objective

Produce **high-resolution (10m)** thermal maps from **coarse 30m thermal input**, leveraging **Sentinel-2 / HLS optical imagery** for structural guidance.

| Thermal Sensor | Optical Sensor | Result |
|----------------|----------------|--------|
| Landsat-8 TIRS (30m) | Sentinel-2 MSI RGB (10m) | SR 10m Thermal |

Thermal pixels hold radiometric temperature information but lack detail.  
Optical pixels are rich in structure but lack physical temperature.  
**InfraNova intelligently fuses both.**

---

## 🧠 Key Concepts

### 🔥 Thermal Super-Resolution
Enhancing spatial resolution without altering absolute temperature information.

### 🌈 Multi-modal Fusion
Optical channels guide edge and texture sharpening for thermal pixels.

### 🧵 Dual-Stream Encoders
Thermal & optical streams extract domain-specific features independently.

### 🧪 Physics-Aware Loss
Ensures reconstructed temperature averages remain physically valid:

```
downsample(pred_hr) ≈ original_lr
```

### 🏗 EDSR-based Decoder
Deep residual refinement without perceptual artifacts or hallucination.

---

## 📦 Repository Structure

```
InfraNova/\
├── app.py # Streamlit inference app\
├── models/\
│   └── dual_edsr_plus.py # Model architecture\
├── utils/\
│   └── dataset_hls.py # Dataset + patch loader\
├── train_hls_ssl4eo.py # Training script\
├── data_streamlit_semantic/ # Curated demo examples\
├── models/ # .pth weights\
├── requirements.txt\
└── README.md
```

---

## 🏛 Model Architecture

```
Thermal LR ──► Thermal Encoder ─┐\
│──► Fusion + Channel / Spatial Attention ─► EDSR Decoder ─► HR Thermal\
Optical HR ─► Optical Encoder ─┘
```

### Component Breakdown

| Block              | Description                          |
|--------------------|--------------------------------------|
| Thermal encoder    | learns radiometric representation    |
| Optical encoder    | preserves sharp structural boundaries|
| Channel Attention  | weighs important channel-level features |
| Spatial Attention  | focuses on local regions             |
| Learned Upsampler  | PixelShuffle 2×                      |
| EDSR residual blocks | prevents smoothing and detail loss  |

---

## ⚙ Training Setup

### Patch Sampling

| Input             | Size     |
|-------------------|----------|
| LR thermal        | 64×64    |
| HR thermal (GT)   | 128×128  |
| HR Optical        | 128×128  |

### Loss Function

```
L = MSE(pred_hr, gt_hr) + λ * MSE(downsample(pred_hr), lr_input)
```

### Hyperparameters

| Parameter     | Value |
|---------------|-------|
| Learning Rate | 1e-4  |
| Epochs        | 40    |
| Batch Size    | 4     |
| Physics λ     | 0.1   |
| Optimizer     | Adam  |

---

## 📊 Validation Metrics

| Metric | Score   |
|--------|---------|
| PSNR   | 49.5 dB |
| SSIM   | 0.977   |
| RMSE   | 0.0074  |

*Evaluated on HLS 2023 January patch-level testing*

---

## 🖥 Streamlit App (Inference UI)

### Run
```bash
streamlit run app.py
```

### Features
* Upload optical GeoTIFF + thermal GeoTIFF
* Outputs:
  * Optical RGB
  * LR thermal (upsampled)
  * SR predicted thermal
  * Pixel-wise metrics (PSNR / SSIM / RMSE)

---

## 📂 Datasets

| Source       | Usage                          |
|--------------|--------------------------------|
| NASA HLS L30 | thermal                        |
| NASA HLS S30 | optical RGB                    |
| SSL4EO       | benchmark initial warm-start   |
| Custom tiles | Evaluation & wildfires         |

---

## 📸 Example Results

| Optical | LR Thermal | SR Thermal | GT Thermal |
|---------|------------|------------|------------|
| ![Optical](data_streamlit_semantic/optical_example.png) | ![LR Thermal](data_streamlit_semantic/lr_thermal_example.png) | ![SR Thermal](data_streamlit_semantic/sr_thermal_example.png) | ![GT Thermal](data_streamlit_semantic/gt_thermal_example.png) |

(See `/data_streamlit_semantic` in repo for full examples.)

---

## 💬 FAQ

**Why use dual-stream fusion?**  
Prevents optical signal dominance & preserves thermal physics.

**Does it hallucinate features?**  
No — physics-aware loss enforces radiometric validity.

**Can it be applied to ECOSTRESS / UAV sensors?**  
Yes — retrain with adapted scaling.

**Can it run in real-time?**  
Yes with ONNX / TensorRT.

---

## 🤝 Contributing
Contributions & research collaborations are welcome.

---

## 📜 License
MIT License — free to modify and deploy.

---

## ⭐ Support
If this project helps your research, please star ⭐ the repository.
https://optical-guided-super-resolution-using-dualedsrplus-dkpiedfxa7q.streamlit.app/
