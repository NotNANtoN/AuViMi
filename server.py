import os
import sys
import io
import time
import uvicorn
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from utils import get_args

from auvimi.engine.deep_daze import Imagine

# --- Import Logic ---
args = get_args()

# --- Device Setup ---
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

# --- Model Initialization ---
print(f"Initializing local deepdaze engine...")

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
    "open_folder": False,
    "save_progress": False,
}
model = Imagine(**model_kwargs)

# Move everything to the correct device
model.to(device)
if device == "mps":
    # Force everything to bfloat16 for speed on Mac
    model.to(dtype=torch.bfloat16)
    print("Model moved to MPS in bfloat16.")

# Use modern torch.compile only for CUDA (Stable)
if device == "cuda":
    try:
        if hasattr(model, 'model'):
            print("Compiling SIREN model with torch.compile (inductor)...")
            model.model = torch.compile(model.model)
    except Exception as e:
        print(f"Note: Could not use torch.compile: {e}")

# Set initial text encoding
text_weight = args.text_weight
text_encoding = None
if args.text:
    print(f"Optimizing on text: {args.text}")
    text_encoding = model.create_text_encoding(args.text)
    text_encoding /= text_encoding.norm(dim=-1, keepdim=True)
    if text_weight == 1.0:
        model.set_clip_encoding(encoding=text_encoding)

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")
    
    # 0. Send Handshake (Configuration)
    config = {
        "size": args.size,
        "gen_backbone": args.gen_backbone,
        "text": args.text,
        "text_weight": args.text_weight
    }
    await websocket.send_json(config)
    
    img_encoding = 0
    iteration_count = 0
    
    try:
        while True:
            # 1. Receive Image
            start_time = time.time()
            data = await websocket.receive_bytes()
            
            # Load image directly from bytes (no disk IO)
            img_input = Image.open(io.BytesIO(data)).convert("RGB")
            
            io_read_time = time.time() - start_time
                
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
                
                img_encoding = args.run_avg * img_encoding + (1 - args.run_avg) * new_img_encoding
                
                if text_encoding is None:
                    clip_encoding = img_encoding
                else:
                    clip_encoding = img_encoding * (1 - text_weight) + text_encoding * text_weight

                model.set_clip_encoding(encoding=clip_encoding)
            
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
            write_start = time.time()
            if img_tensor is not None:
                # Ensure we don't have NaNs before casting
                img_tensor = torch.nan_to_num(img_tensor, nan=0.5)
                img_np = np.uint8(img_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy() * 255)
                img_pil = Image.fromarray(img_np)
                
                with io.BytesIO() as buf:
                    img_pil.save(buf, format='JPEG', quality=80)
                    byte_data = buf.getvalue()
                await websocket.send_bytes(byte_data)
            
            io_write_time = time.time() - write_start
            
            total_time = time.time() - start_time
            
            # Detailed Train Breakdown
            t_siren = last_avg_timings.get('siren', 0)
            t_clip = last_avg_timings.get('clip', 0)
            t_cut = last_avg_timings.get('cutouts', 0)
            t_back = last_avg_timings.get('backward', 0)
            
            print(f"Loop: {total_time:.3f}s | Encode: {encode_time:.3f}s | Train: {train_time:.3f}s (SIREN: {t_siren:.3f}s, CLIP: {t_clip:.3f}s, Cut: {t_cut:.3f}s, Back: {t_back:.3f}s) | Loss: {current_loss:.4f}")
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()

if __name__ == "__main__":
    print("Starting server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
