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

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model():
    global model
    try:
        if model is None:
            model_path = 'yolov8n_trained.pt'
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Files in directory: {os.listdir('.')}")
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
            model = YOLO(model_path)
            
            # Log model device and properties
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            model.to(device)
            
            logger.info("Model loaded successfully")
            logger.info(f"Model properties: {model}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def detect_drones(frame):
    try:
        logger.info("Starting drone detection")
        logger.info(f"Input frame shape: {frame.shape}")
        
        # Load model on demand
        current_model = load_model()
        
        # Log memory usage
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Reduce frame size to save memory
        max_size = 640
        height, width = frame.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            logger.info(f"Resized frame to: {frame.shape}")
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Log inference start
        logger.info("Starting inference")
        
        # Run detection
        with torch.no_grad():
            results = current_model(frame_rgb, conf=0.2)
            
        # Log detection results
        for result in results:
            num_detections = len(result.boxes)
            logger.info(f"Number of detections: {num_detections}")
            
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                logger.info(f"Detection confidence: {confidence:.2f}")
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Drone: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logger.info("Detection completed successfully")
        
        # Clean up
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return frame
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.route('/')
def index():
    try:
        logger.info("Loading index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return str(e), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        logger.info("Received frame processing request")
        
        # Get frame data
        if not request.json or 'frame' not in request.json:
            raise ValueError("No frame data received")
        
        data = request.json['frame']
        logger.info("Frame data received, decoding...")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data.split(',')[1])
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            raise ValueError("Invalid image data")
        
        # Convert to numpy array
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Image decode error: {str(e)}")
            raise ValueError("Failed to decode image")
        
        if frame is None:
            raise ValueError("Decoded frame is None")
            
        logger.info(f"Successfully decoded frame, shape: {frame.shape}")
        
        # Process frame
        processed_frame = detect_drones(frame)
        
        # Encode processed frame
        logger.info("Encoding processed frame")
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up
        del frame, processed_frame, buffer
        gc.collect()
        
        logger.info("Frame processing completed successfully")
        
        return jsonify({
            "processed_frame": processed_frame_b64,
            "status": "success"
        })
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing frame: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "status": "error",
            "details": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    try:
        logger.info("Starting server...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Available files: {os.listdir('.')}")
        
        # Try to load model at startup to catch any issues early
        load_model()
        
        port = int(os.environ.get('PORT', 10000))
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
