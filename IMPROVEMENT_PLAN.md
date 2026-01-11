# Deep Daze: Improvement Plan & Analysis

This document outlines the roadmap for AuViMi's Deep Daze implementation, focusing on the next generation of **Interface**, **Performance**, and **Image Quality** enhancements.

---

## 1. Interface & Experience
**Goal**: Transition from a static viewer to a real-time interactive laboratory.

### A. Web-Based Control Dashboard
Now that the engine is running at a stable 2.5 FPS, the primary bottleneck is the inability to tune parameters without restarting the server.
*   **Proposal**: Build a browser-based dashboard to control the live optimization.
*   **Features**:
    *   **Live Sliders**:
        *   `learning_rate`: Increase for faster changes, decrease for stability.
        *   `text_weight`: Shift balance between the webcam feed and the text prompt.
        *   `moment_loss_weight`: Control how strictly the dream matches your room's colors.
        *   `global_view_weight`: Higher values reduce "ghosting" (multiple floating eyes/faces).
    *   **Toggles**:
        *   `use_gabor`: Switch between SIREN and Gabor/WIRE activations on the fly.
        *   `do_aug`: Enable/disable rotation and translation augmentations.
*   **Implementation**: Use FastAPI to serve a simple HTML/Tailwind/JS page that communicates with the optimization loop via a shared state or a configuration queue.

### B. Unified Browser Viewer
*   Stream the JPEG output directly to the dashboard using WebSockets. This eliminates the need for `client.py` and OpenCV entirely, providing a single-window experience.

---

## 2. Performance Optimization
**Goal**: Break the 10 FPS barrier for smooth, low-latency "mirrors."

### A. Fused CUDA Kernels (`tiny-cuda-nn`)
The most significant remaining performance win.
*   **Why**: Standard PyTorch MLP layers launch separate CUDA kernels for every matrix multiply and activation. `tiny-cuda-nn` (TCNN) executes the entire SIREN network in a *single* fused kernel.
*   **Impact**: Potentially 5x-10x speedup for the SIREN generation part of the loop.
*   **Integration**: Replace the `INRNet` class in `deep_daze.py` with a TCNN implementation.

### B. Temporal Consistency (Optical Flow / Latents)
*   **Goal**: Reduce the high-frequency flickering between frames.
*   **Proposal**: Add a "Temporal Loss" that penalizes large changes between the current generated frame and the previous one, perhaps weighted by an optical flow mask from the webcam.

---

## 3. Image Quality & Composition
**Goal**: Move beyond "dreamy textures" to coherent, structured transformations.

### A. Global vs. Patch Balancing
*   **The Issue**: The current "ghosting" effect (many small eyes/faces) happens when CLIP is too focused on small patches (cutouts).
*   **Solution**: Dynamically weight the **Global View** (full-image embedding) against the batch of cutouts. A higher global weight forces the model to respect the overall human silhouette and room structure.

### B. Advanced Moment Matching
*   **Goal**: Better color fidelity.
*   **Proposal**: Instead of just global Mean/Std, implement a **Histogram Matching Loss** or a 3x3 color covariance matrix match to capture more complex relationships between colors (e.g., ensuring "skin tones" stay separate from "background greens").

### C. Prompt Weight Scheduling
*   **Goal**: Prevent the text prompt from "overtaking" the image too quickly.
*   **Proposal**: Automatically decay or oscillate the `text_weight` to let the webcam image periodically "re-anchor" the structure of the dream.

---

## Summary of Next Actions

1.  **Global View Tuning**: Expose the weight of the global view in the loss function to reduce ghosting.
2.  **Web Dashboard**: Implement the FastAPI control panel to allow for real-time iteration.
3.  **TCNN Research**: Investigate the installation of `tiny-cuda-nn` in the current environment for a massive FPS boost.
