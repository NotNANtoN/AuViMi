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
    parser.add_argument("--no_graph", action="store_true", help="Disable the loss history graph")
    return parser.parse_args()

def draw_loss_graph(history, width, height):
    # Create black background
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    if len(history) < 2:
        cv2.putText(graph, "Waiting for data...", (10, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return graph

    # Auto-scale the graph
    min_l = min(history)
    max_l = max(history)
    l_range = max_l - min_l if max_l != min_l else 1.0
    
    # Add some padding to the range for better visuals
    min_l -= l_range * 0.1
    max_l += l_range * 0.1
    l_range = max_l - min_l

    # Convert history to pixel coordinates
    points = []
    for i, loss in enumerate(history):
        x = int(i * (width - 1) / (len(history) - 1))
        y = int(height - ((loss - min_l) / l_range * height))
        # Clamp y to graph boundaries
        y = max(0, min(height - 1, y))
        points.append((x, y))

    # Draw lines
    for i in range(len(points) - 1):
        cv2.line(graph, points[i], points[i+1], (0, 255, 0), 2)

    # Draw Current Loss Text
    cv2.putText(graph, f"Loss: {history[-1]:.2f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw Min/Max labels
    cv2.putText(graph, f"Max: {max(history):.1f}", (width - 80, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(graph, f"Min: {min(history):.1f}", (width - 80, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    
    return graph

async def stream_video():
    args = get_args()
    uri = f"ws://{args.host}:{args.port}/ws"
    print(f"Connecting to {uri}...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    loss_history = []
    max_history_len = 200 # Number of points to show in graph

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

                # 5. Receive Metadata & Image
                metadata_raw = await websocket.recv()
                result_bytes = await websocket.recv()
                
                # Parse Loss
                try:
                    metadata = json.loads(metadata_raw)
                    loss_history.append(float(metadata.get("loss", 0)))
                    if len(loss_history) > max_history_len:
                        loss_history.pop(0)
                except Exception as e:
                    print(f"Metadata error: {e}")
                
                nparr = np.frombuffer(result_bytes, np.uint8)
                output_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if output_img is not None:
                    # Stack them horizontally (Input | Dream)
                    comparison = np.hstack((frame_resized, output_img))
                    
                    if not args.no_graph:
                        # Draw and attach graph
                        graph_img = draw_loss_graph(loss_history, server_size, server_size)
                        comparison = np.hstack((comparison, graph_img))
                        
                    cv2.imshow(f"AuViMi: Input | Dream | Loss ({server_size}x{server_size})", comparison)

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

