from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import torch
import base64
from ultralytics import YOLO
from flask_cors import CORS
import os
import gc

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model():
    global model
    if model is None:
        model = YOLO('yolov8n_trained.pt')
    return model

def detect_drones(frame):
    try:
        # Load model on demand
        current_model = load_model()
        
        # Reduce frame size to save memory
        max_size = 640
        height, width = frame.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection with lower memory usage
        with torch.no_grad():
            results = current_model(frame_rgb, conf=0.2)
        
        # Draw boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Drone: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Clean up
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return frame
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get frame data
        data = request.json['frame']
        image_data = base64.b64decode(data.split(',')[1])
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        # Process frame
        processed_frame = detect_drones(frame)
        
        # Encode with lower quality to reduce memory usage
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up
        del frame, processed_frame, buffer
        gc.collect()
        
        return jsonify({"processed_frame": processed_frame_b64})
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
