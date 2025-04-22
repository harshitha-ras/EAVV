from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
from bodem_explainer import BODEMExplainer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your YOLOv8 model
model_path = "/home/harsh/EAVV/yolo_output/yolov8s_weather_refined/weights/best.pt"
model = YOLO(model_path)

# Initialize BODEM explainer
explainer = BODEMExplainer(model_path, image_size=640, device='cpu')

@app.route('/bodem_explain', methods=['POST'])
def bodem_explain():
    """Generate BODEM explanation for an uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file temporarily
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)
    
    try:
        # Generate BODEM explanation
        image, detections, saliency_maps = explainer.explain(temp_path)
        
        if len(detections) == 0:
            return jsonify({'error': 'No objects detected in image'}), 400
        
        # Create output filename
        output_path = 'static/bodem_explanation.png'
        os.makedirs('static', exist_ok=True)
        
        # Visualize and save explanation
        explainer.visualize_explanation(image, detections, saliency_maps, model.names, output_path)
        
        # Return the path to the generated visualization
        return jsonify({
            'success': True,
            'explanation_path': output_path,
            'num_detections': len(detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/view_bodem', methods=['GET'])
def view_bodem():
    """Render a page to view BODEM explanations"""
    return render_template('bodem_view.html')

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Perform detection
        results = model(filepath)
        
        # Process results
        result_data = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                result_data.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return jsonify({
            'success': True,
            'detections': result_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

