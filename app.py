from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import torch
import base64
from ultralytics import YOLO
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the YOLOv8 model
model = YOLO('yolov8n_trained.pt')

def detect_drones(frame):
    results = model(frame, conf=0.2)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f'Drone: {confidence:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 255, 255), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['frame']
        image_data = base64.b64decode(data.split(',')[1])
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        processed_frame = detect_drones(frame)
        
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"processed_frame": processed_frame_b64})
    except Exception as e:
        print("Error processing frame:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)