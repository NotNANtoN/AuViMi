import cv2
import asyncio
import websockets
import numpy as np
import argparse
import sys
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="Host address of the server")
    parser.add_argument("--port", type=int, default=8000, help="Port of the server")
    return parser.parse_args()

async def stream_video():
    args = get_args()
    uri = f"ws://{args.host}:{args.port}/ws"
    print(f"Connecting to {uri}...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        async with websockets.connect(uri) as websocket:
            # 1. Wait for Handshake (Server Config)
            config_msg = await websocket.recv()
            config = json.loads(config_msg)
            server_size = config.get("size", 256)
            print(f"Connected! Server is running {config.get('gen_backbone')} at {server_size}x{server_size}")
            print("Press 'ESC' to quit.")
            
            while True:
                # 2. Capture Frame
                ret, frame = cap.read()
                if not ret: break

                # 3. Resize to match server's expected size
                frame_resized = cv2.resize(frame, (server_size, server_size))
                
                # 4. Encode & Send
                _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                await websocket.send(buffer.tobytes())

                # 5. Receive & Display
                result_bytes = await websocket.recv()
                
                nparr = np.frombuffer(result_bytes, np.uint8)
                output_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if output_img is not None:
                    # Stack them horizontally
                    comparison = np.hstack((frame_resized, output_img))
                    cv2.imshow(f"AuViMi: Input vs Dream ({server_size}x{server_size})", comparison)

                # 6. Handle Keys
                key = cv2.waitKey(1)
                if key == 27: break

                # 5. Handle Keys
                key = cv2.waitKey(1)
                if key == 27: # ESC
                    break
                    
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(stream_video())
    except KeyboardInterrupt:
        pass

