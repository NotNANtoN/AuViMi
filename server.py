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

# Use modern torch.compile if available (Torch 2.0+)
if device != "mps":
    try:
        if hasattr(model, 'model'):
            print("Compiling SIREN model with torch.compile...")
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
            
            temp_path = f"/tmp/auvimi_input_{os.getpid()}.jpg"
            with open(temp_path, "wb") as f:
                f.write(data)
            
            io_read_time = time.time() - start_time
                
            # 2. Update Encoding
            encode_start = time.time()
            if text_weight < 1.0:
                new_img_encoding = model.create_img_encoding(temp_path)
                
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
            for _ in range(args.opt_steps):
                img_tensor, loss = model.train_step(0, iteration_count)
                iteration_count += 1
            train_time = time.time() - train_start
            
            # 4. Return Result
            write_start = time.time()
            if img_tensor is not None:
                img_np = np.uint8(img_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy() * 255)
                img_pil = Image.fromarray(img_np)
                
                with io.BytesIO() as buf:
                    img_pil.save(buf, format='JPEG', quality=80)
                    byte_data = buf.getvalue()
                await websocket.send_bytes(byte_data)
            
            io_write_time = time.time() - write_start
            
            total_time = time.time() - start_time
            print(f"Loop: {total_time:.3f}s | Read: {io_read_time:.3f}s | Encode: {encode_time:.3f}s | Train: {train_time:.3f}s | Write: {io_write_time:.3f}s")
                
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
