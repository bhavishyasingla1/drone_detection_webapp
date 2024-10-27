from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import torch
import base64
from ultralytics import YOLO
from flask_cors import CORS
import os
import gc
import logging
import traceback

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model():
    global model
    try:
        if model is None:
            # Force CPU usage to reduce memory consumption
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            torch.set_num_threads(1)  # Limit CPU threads
            
            model_path = 'yolov8n_trained.pt'
            logger.info(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load model with memory optimizations
            model = YOLO(model_path)
            model.to('cpu')  # Force CPU
            torch.set_grad_enabled(False)  # Disable gradient computation
            
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def detect_drones(frame):
    try:
        # Aggressively reduce frame size
        max_size = 416  # Reduced from 640
        height, width = frame.shape[:2]
        scale = max_size / max(height, width)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to RGB with memory optimization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection with minimal memory usage
        results = model(frame_rgb, conf=0.25)  # Increased confidence threshold
        
        # Process results and draw boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Drone: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Clean up
        del results, frame_rgb
        gc.collect()
        
        return frame
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if not request.json or 'frame' not in request.json:
            raise ValueError("No frame data received")
        
        # Decode frame with memory optimization
        image_data = base64.b64decode(request.json['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode frame")
        
        # Process frame
        processed_frame = detect_drones(frame)
        
        # Encode with lower quality to reduce memory usage
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]  # Reduced quality
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up
        del frame, processed_frame, buffer, image_data, nparr
        gc.collect()
        
        return jsonify({"processed_frame": processed_frame_b64, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    try:
        load_model()
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        raise
