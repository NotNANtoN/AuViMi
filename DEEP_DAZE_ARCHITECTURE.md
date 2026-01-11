# Deep Daze: Architecture & Configuration Overview

This document provides a comprehensive overview of the current Deep Daze implementation in AuViMi, including default settings, the SIREN network architecture, and optional enhancements for iteration.

---

## 1. Default Settings (CLI & Runtime)
These values are defined in `utils.py` and govern the standard optimization loop in `server.py`.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `size` | `256` | Resolution of the generated image (256x256). |
| `batch_size` | `32` | Number of random cutouts processed per gradient step. |
| `lr` | `1e-5` | Learning rate for the SIREN weights. |
| `num_layers` | `32` | Depth of the SIREN model. |
| `opt_steps` | `5` | Training steps performed for every single input frame received. |
| `epochs` | `12` | Total number of training epochs. |
| `text_weight` | `0.5` | Balance between the webcam image and text prompt embeddings. |
| `run_avg` | `0.0` | Fraction of the previous image encoding kept (0.0 = no memory). |
| `lower_bound_cutout` | `0.05` | Minimum size of a random cutout (5% of image width). |
| `upper_bound_cutout` | `1.0` | Maximum size of a random cutout (100% of image width). |
| `clip_model` | `ViT-B/32` | The CLIP vision-language model used for guidance. |
| `tv_coef` | `100.0` | Total Variation loss coefficient to encourage spatial smoothness. |

---

## 2. SIREN Network Architecture
The **SIREN** (Sinusoidal Representation Network) acts as an Implicit Neural Representation (INR) that maps 2D coordinates $(x, y)$ to RGB colors.

### Layers & Activation
- **Input**: A coordinate grid normalized to $[-1, 1]$.
- **Hidden Layers**: 32 layers with 256 neurons each.
- **Activation**: $\sin(\omega_0 \cdot x)$, with a frequency $\omega_0 = 30.0$. This allows the network to represent complex, high-frequency signals (like sharp edges) better than ReLU.
- **Initialization**:
  - *First Layer*: Uniform distribution $U(-1/n, 1/n)$.
  - *Hidden Layers*: Uniform distribution $U(-\sqrt{6/n}/\omega_0, \sqrt{6/n}/\omega_0)$.

### Output Transformation
The raw output is in the range $[-1, 1]$ and is normalized to $[0, 1]$ via:
`norm_out = (raw_out + 1) * 0.5`

---

## 3. Standard Pipeline & Augmentations
The core of Deep Daze is the **Random Cutout** mechanism, which allows the CLIP perceptor (which expects 224x224 inputs) to see different parts of the image at different scales.

- **Cutouts**: 32 patches are cropped from the image at random positions and scales (5% to 100%).
- **Interpolation**: Every cutout is resized to the CLIP input resolution using bilinear interpolation.
- **TV Loss**: A Total Variation penalty is applied to the full 256x256 image to prevent high-frequency noise and encourage local smoothness.

---

## 4. Optional Enhancements
These features are implemented in the engine but are either disabled by default or require specific flags to activate.

### Activation Functions
- **Gabor / WIRE Activation**: (`--use_gabor 1`)
  - Replaces Sine with a Gabor wavelet: $\sin(\omega_0 \cdot x) \cdot \exp(-s_0 \cdot x^2)$.
  - Provides "local support" and can sometimes lead to more structured, less repetitive textures.
  - Controlled by `--gabor_scale` (defaults to 10.0).

### Sampling & Cutout Logic
- **Gaussian Sampling**: (`gauss_sampling=True`)
  - Instead of uniform scale selection, cutout sizes are sampled from a Gaussian distribution.
- **Center Bias**: (`--center_bias 1`)
  - Forces more cutouts to be taken from the center of the image, which can help focus the "subject" of the dream.
- **Saturate Bound**: (`--saturate_bound 1`)
  - Gradually increases the `lower_bound_cutout` over the course of the epochs. This forces the model to focus on fine details early on and larger global structures later.

### Image Processing
- **Affine Augmentations**: (`do_aug=True` in engine)
  - Adds random rotation (±10°) and translation (±10%) to every cutout before passing it to CLIP. This makes the optimization more robust to orientation.
- **Definition Boost**: (Internal toggle in `server.py`)
  - Applies a 20% contrast boost and 50% sharpening to the output frames before they are sent to the client.
- **Augment Both**: (`--aug_both 1`)
  - If optimizing against a target image, this applies the *same* random cutouts to both the generated image and the target image, ensuring spatial correspondence.

### Performance
- **Torch Compile**: (Automatic on CUDA)
  - Uses `torch.compile(inductor)` to fuse kernels for the SIREN model and the CLIP perceptor, significantly reducing latency on NVIDIA GPUs.
