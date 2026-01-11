import io
import os

# Set CUDA allocation configuration before any torch calls to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Increase recompile limit for torch.compile to handle dynamic behaviors
try:
    import torch._dynamo

    torch._dynamo.config.recompile_limit = 64
except (ImportError, AttributeError):
    pass

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

from auvimi.engine.deep_daze import Imagine  # noqa: E402
from utils import clean_pid, get_args, kill_old_process  # noqa: E402

# --- Import Logic ---
args = get_args()

# --- Kill Old Processes ---
kill_old_process(create_new=True)

# --- Session Recording Setup ---
SESSION_ROOT = "sessions"
os.makedirs(SESSION_ROOT, exist_ok=True)
current_session_id = time.strftime("%Y%m%d-%H%M%S")
session_dir = os.path.join(SESSION_ROOT, current_session_id)
input_dir = os.path.join(session_dir, "input")
output_dir = os.path.join(session_dir, "output")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Device Setup ---
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
    # Enable TensorFloat32 for better performance on NVIDIA GPUs (Ampere+)
    torch.set_float32_matmul_precision("high")
print(f"Using device: {device}")

# --- Model Initialization ---
print("Initializing local deepdaze engine...")

# We initialize the model globally
model_kwargs = {
    "epochs": args.epochs,
    "image_width": args.size,
    "gradient_accumulate_every": args.gradient_accumulate_every,
    "batch_size": args.batch_size,
    "num_layers": args.num_layers,
    "lr": args.lr,
    "use_gabor": bool(args.use_gabor),
    "gabor_scale": args.gabor_scale,
    "model_name": args.clip_model,
    "aug_both": bool(args.aug_both),
    "open_folder": False,
    "save_progress": False,
    "do_aug": False,
}
model = Imagine(**model_kwargs)

# Move everything to the correct device
model.to(device)
if device == "mps":
    # Force everything to bfloat16 for speed on Mac
    model.to(dtype=torch.bfloat16)
    print("Model moved to MPS in bfloat16.")
elif device == "cuda" and torch.cuda.is_bf16_supported():
    # Use bfloat16 on Ampere+ GPUs for consistency with autocast and better speed
    model.to(dtype=torch.bfloat16)
    print("Model moved to CUDA in bfloat16.")

# Set initial text encoding
text_weight = args.text_weight
text_encoding = None
if args.text:
    print(f"Optimizing on text: {args.text}")
    text_encoding = model.create_text_encoding(args.text)
    text_encoding /= text_encoding.norm(dim=-1, keepdim=True)
    if text_weight == 1.0:
        model.set_clip_encoding(encoding=text_encoding)

# Use modern torch.compile only for CUDA (Stable)
# Moved after initial encoding setup to allow for a proper warmup
if device == "cuda":
    try:
        if hasattr(model, "model"):
            print("Compiling SIREN model with torch.compile (inductor)...")
            model.model = torch.compile(model.model)

            # Explicitly compile the CLIP model for faster encoding steps
            print("Compiling CLIP perceptor...")
            model.perceptor = torch.compile(model.perceptor)

            # Warmup to trigger compilation now instead of during first request
            # This helps avoid a long lag on the first WebSocket message
            print("Warming up compiled model...")
            # Use a dummy encoding if none is set yet
            if text_encoding is not None:
                warmup_encoding = text_encoding
            else:
                # Get the correct output dimension from the CLIP perceptor
                out_dim = 512  # fallback
                if hasattr(model.perceptor, "visual") and hasattr(model.perceptor.visual, "output_dim"):
                    out_dim = model.perceptor.visual.output_dim
                warmup_encoding = torch.randn(1, out_dim, device=device)

            # Ensure it's the right dtype for the model
            if device == "mps":
                warmup_encoding = warmup_encoding.to(dtype=torch.bfloat16)
            elif device == "cuda" and torch.cuda.is_bf16_supported():
                warmup_encoding = warmup_encoding.to(dtype=torch.bfloat16)

            # Dry run doesn't update batch counts
            model.model(warmup_encoding, dry_run=True)
            print("Warmup complete.")
    except Exception as e:
        print(f"Note: Could not use torch.compile: {e}")

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")

    # 0. Send Handshake (Configuration)
    config = {"size": args.size, "gen_backbone": args.gen_backbone, "text": args.text, "text_weight": args.text_weight}
    await websocket.send_json(config)

    img_encoding = 0
    iteration_count = 0

    try:
        while True:
            # 1. Receive Image
            start_time = time.time()
            data = await websocket.receive_bytes()

            # Save Input Frame
            with open(os.path.join(input_dir, f"{iteration_count:05d}.jpg"), "wb") as f:
                f.write(data)

            # Load image directly from bytes (no disk IO)
            img_input = Image.open(io.BytesIO(data)).convert("RGB")

            # 2. Update Encoding
            encode_start = time.time()
            if text_weight < 1.0:
                # Pass PIL image directly
                new_img_encoding = model.create_img_encoding(img_input)

                # Use bfloat16 for MPS
                target_dtype = torch.bfloat16 if device == "mps" else torch.float32
                new_img_encoding = new_img_encoding.to(device=model.device, dtype=target_dtype)

                if isinstance(img_encoding, int):
                    img_encoding = new_img_encoding

                # Detach here to prevent the computational graph from growing indefinitely
                img_encoding = (args.run_avg * img_encoding + (1 - args.run_avg) * new_img_encoding).detach()

                if text_encoding is None:
                    clip_encoding = img_encoding
                else:
                    clip_encoding = img_encoding * (1 - text_weight) + text_encoding * text_weight

                model.set_clip_encoding(img=img_input, encoding=clip_encoding)

            encode_time = time.time() - encode_start

            # 3. Train Step(s)
            train_start = time.time()
            img_tensor = None
            last_avg_timings = {}
            current_loss = 0
            for _ in range(args.opt_steps):
                img_tensor, loss, last_avg_timings = model.train_step(0, iteration_count)
                current_loss = loss.item()
                iteration_count += 1
            train_time = time.time() - train_start

            # 4. Return Result
            if img_tensor is not None:
                # Ensure we don't have NaNs before casting
                img_tensor = torch.nan_to_num(img_tensor, nan=0.5)
                img_np = np.uint8(img_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy() * 255)
                img_pil = Image.fromarray(img_np)

                # OPTIONAL: Definition Boost (Contrast & Sharpness)
                boost_contrast = False
                if boost_contrast:
                    enhancer = ImageEnhance.Contrast(img_pil)
                    img_pil = enhancer.enhance(1.2)  # Boost contrast by 20%
                    enhancer = ImageEnhance.Sharpness(img_pil)
                    img_pil = enhancer.enhance(1.5)  # Sharpen significantly

                # Save Output Frame
                img_pil.save(os.path.join(output_dir, f"{iteration_count:05d}.jpg"), quality=90)

                with io.BytesIO() as buf:
                    img_pil.save(buf, format="JPEG", quality=80)
                    byte_data = buf.getvalue()

                # Send metadata (loss) followed by image data
                await websocket.send_json({"loss": current_loss})
                await websocket.send_bytes(byte_data)

            total_time = time.time() - start_time

            # Detailed Train Breakdown
            t_siren = last_avg_timings.get("siren", 0)
            t_clip = last_avg_timings.get("clip", 0)
            t_cut = last_avg_timings.get("cutouts", 0)
            t_back = last_avg_timings.get("backward", 0)

            print(
                f"Loop: {total_time:.3f}s | Encode: {encode_time:.3f}s | Train: {train_time:.3f}s (SIREN: {t_siren:.3f}s, CLIP: {t_clip:.3f}s, Cut: {t_cut:.3f}s, Back: {t_back:.3f}s) | Loss: {current_loss:.4f}"
            )

    except WebSocketDisconnect:
        print(f"Client disconnected. Encoding videos for session {current_session_id}...")

        # Build MP4s using ffmpeg
        try:
            # Output Video
            out_mp4 = os.path.join(session_dir, "transformed.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "10",
                    "-i",
                    os.path.join(output_dir, "%05d.jpg"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    out_mp4,
                ],
                check=True,
                capture_output=True,
            )

            # Input Video
            in_mp4 = os.path.join(session_dir, "original.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "10",
                    "-i",
                    os.path.join(input_dir, "%05d.jpg"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    in_mp4,
                ],
                check=True,
                capture_output=True,
            )

            print(f"✅ Videos saved to {session_dir}")
        except Exception as e:
            print(f"❌ Failed to encode videos: {e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        await websocket.close()
    finally:
        clean_pid()


if __name__ == "__main__":
    print("Starting server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
