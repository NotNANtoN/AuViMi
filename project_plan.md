# AuViMi Project Plan: Modernizing the Neural Mirror

This document outlines the research and development path for evolving AuViMi into a high-performance, real-time Audio-Visual Mirror using modern ML techniques while preserving its core identity as an **Implicit Neural Representation (INR)** explorer.

## 1. Core Architecture: The Hybrid SIREN-Hash Grid
The original SIREN architecture is beautiful for its "infinite resolution" and mathematical simplicity, but a 16-32 layer MLP is slow to evaluate.

*   **Goal:** Implement a "from-scratch" version of **Multi-Resolution Hash Encodings** (inspired by Instant-NGP).
*   **The Idea:** Instead of feeding raw $(x, y)$ coordinates into a deep MLP, we:
    1.  Map coordinates to several levels of grids.
    2.  Perform hash-based lookups of learnable feature vectors at each level.
    3.  Interpolate these vectors and pass them into a **very shallow** SIREN (e.g., 2-3 layers).
*   **Benefit:** Moves the heavy lifting from expensive matrix multiplications (weights) to cheap memory lookups (the hash grid). This should provide a 10x-20x speedup.

## 2. Optimization Pipeline: Richer Gradients
Currently, we take random cutouts. We can modernize this using techniques from modern GAN and Diffusion training.

### Stochastic Gradient Augmentation
*   Instead of just "cropping," we pass the generated image through a pipeline of random, differentiable transforms: **Rotation, Scaling, Color Jitter, and Perspective Warping**.
*   Since the transforms are differentiable, the CLIP gradients flow through the "warped" view back into the SIREN weights. This forces the SIREN to learn features that look correct from any angle, not just in a static crop.

### DiffAugment (Differentiable Augmentation)
*   **Concept:** Standard augmentations (like those in `torchvision`) can be "leaky" if not handled carefully. DiffAugment ensures the optimization remains stable by applying the same set of transformations to both the "dream" and the "target" (when text-weight < 1.0).
*   This prevents the AI from finding "adversarial patterns" that satisfy CLIP but look like noise to humans.

## 3. Backbones & Representations
*   **Current:** `ViT-B/32` (Classic, reliable).
*   **Future:** **SigLIP** (Sigmoid Language-Image Pre-training). SigLIP replaces the softmax in CLIP with a simpler sigmoid loss, which often results in better performance at the same model size and is more modern.
*   **The "Latent INR" Idea:** Later, we can try using a SIREN to represent the 4-channel latent space of a VAE (like Stable Diffusion's Autoencoder) instead of 3-channel RGB. The SIREN would "dream" in latents, and we'd decode to pixels only for display.

## 4. Hardware Acceleration: Kernel Fusion
Kernel fusion is the "final boss" of optimization.

*   **What it is:** When we run 16 layers of an MLP, the GPU starts a new "job" (kernel) for every layer. The time spent "starting" these jobs can be longer than the actual math.
*   **The Plan:** Fuse the Hash Grid lookup and the shallow SIREN into a **single GPU kernel**.
*   **Difficulty:** High. We can approach this via:
    1.  `torch.compile` (Fixing the RPATH issues we saw).
    2.  `Triton` (Python-based kernel DSL, great for CUDA).
    3.  `Metal Performance Shaders Graph` or custom MSL (Metal Shading Language) for native Mac support.

## 5. Implementation Roadmap
1.  [ ] **INR Upgrade:** Implement a readable Hash Grid encoder in PyTorch.
2.  [ ] **Augmentation Upgrade:** Replace simple cutouts with a `DiffAug` pipeline.
3.  [ ] **Backend Switch:** Explore `SigLIP` via OpenCLIP.
4.  [ ] **Fusion:** Attempt to compile the new shallow hybrid model.
