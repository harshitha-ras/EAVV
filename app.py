from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import io
import base64
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Create the Flask application
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Weather Detection application startup')

# Load the YOLOv8 model
model_path = "yolo_output/yolov8s_weather_refined/weights/best.pt"
try:
    model = YOLO(model_path)
    app.logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    print(f"Error loading model: {str(e)}")

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

class BODEMExplainer:
    """
    BODEM (Boundary-based Object Detection Explanation Method) explainer for YOLOv8.
    """
    def __init__(self, model_path, image_size=640, device='cpu'):
        """
        Initialize the BODEM explainer.
        
        Args:
            model_path: Path to YOLOv8 model
            image_size: Input image size for the model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        print(f"Initializing BODEM Explainer with model: {model_path}, device: {device}", flush=True)
        self.model = YOLO(model_path)
        self.image_size = image_size
        self.device = device
        self.model.to(device)
        print(f"Model loaded successfully on {device}", flush=True)
    
    def explain(self, image_path):
        """
        Generate BODEM explanation for an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            image: Original image
            detections: List of detection objects
            saliency_maps: Dictionary mapping detection indices to saliency maps
        """
        try:
            print(f"Explaining image: {image_path}", flush=True)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            print(f"Running inference on image", flush=True)
            results = self.model(image, verbose=False)
            
            # Get detections
            detections = results[0].boxes
            print(f"Found {len(detections)} detections", flush=True)
            
            if len(detections) == 0:
                return image, [], {}
            
            # Generate saliency maps for each detection
            saliency_maps = {}
            for i, detection in enumerate(detections):
                print(f"Generating saliency map for detection {i+1}/{len(detections)}", flush=True)
                saliency_map = self._generate_saliency_map(image, detection)
                saliency_maps[i] = saliency_map
            
            return image, detections, saliency_maps
        except Exception as e:
            print(f"Error in explain method: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None, [], {}
    
    def _generate_saliency_map(self, image, detection):
        """
        Generate saliency map for a detection using BODEM.
        
        Args:
            image: Input image
            detection: Detection object
            
        Returns:
            saliency_map: Saliency map highlighting important regions
        """
        try:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            
            # Create a mask for the detection
            mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            
            # Create a saliency map by applying hierarchical masking
            saliency_map = np.zeros_like(image[:, :, 0], dtype=np.float32)
            
            # Divide the bounding box into a grid
            grid_size = 8
            cell_height = (y2 - y1) // grid_size
            cell_width = (x2 - x1) // grid_size
            
            if cell_height <= 0 or cell_width <= 0:
                # If the box is too small, use the entire box
                saliency_map[y1:y2, x1:x2] = 1.0
                return saliency_map
            
            # Apply hierarchical masking
            for i in range(grid_size):
                for j in range(grid_size):
                    # Calculate cell coordinates
                    cell_x1 = x1 + j * cell_width
                    cell_y1 = y1 + i * cell_height
                    cell_x2 = min(cell_x1 + cell_width, x2)
                    cell_y2 = min(cell_y1 + cell_height, y2)
                    
                    # Create a masked image
                    masked_image = image.copy()
                    masked_image[cell_y1:cell_y2, cell_x1:cell_x2] = 0
                    
                    # Run inference on masked image
                    results = self.model(masked_image, verbose=False)
                    
                    # Check if the detection is still present
                    found = False
                    confidence_diff = 0
                    
                    for new_detection in results[0].boxes:
                        new_x1, new_y1, new_x2, new_y2 = new_detection.xyxy[0].cpu().numpy().astype(int)
                        
                        # Calculate IoU with original detection
                        intersection_x1 = max(x1, new_x1)
                        intersection_y1 = max(y1, new_y1)
                        intersection_x2 = min(x2, new_x2)
                        intersection_y2 = min(y2, new_y2)
                        
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            original_area = (x2 - x1) * (y2 - y1)
                            new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
                            union_area = original_area + new_area - intersection_area
                            iou = intersection_area / union_area
                            
                            if iou > 0.5:
                                found = True
                                # Calculate confidence difference
                                original_conf = detection.conf[0].item()
                                new_conf = new_detection.conf[0].item()
                                confidence_diff = original_conf - new_conf
                                break
                    
                    # Update saliency map based on detection presence
                    if not found:
                        # If detection disappeared, this region is important
                        saliency_map[cell_y1:cell_y2, cell_x1:cell_x2] = 1.0
                    else:
                        # If detection remained but confidence decreased, assign proportional importance
                        saliency_map[cell_y1:cell_y2, cell_x1:cell_x2] = min(1.0, max(0.0, confidence_diff))
            
            # Normalize saliency map
            if np.max(saliency_map) > 0:
                saliency_map = saliency_map / np.max(saliency_map)
            
            return saliency_map
        except Exception as e:
            print(f"Error in _generate_saliency_map: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return np.zeros_like(image[:, :, 0], dtype=np.float32)
    
    def visualize_explanation(self, image, detections, saliency_maps, class_names, output_path):
        """
        Visualize BODEM explanation and save to file.
        
        Args:
            image: Original image
            detections: List of detection objects
            saliency_maps: Dictionary mapping detection indices to saliency maps
            class_names: Dictionary of class names
            output_path: Path to save visualization
        """
        try:
            print(f"Visualizing explanation to: {output_path}", flush=True)
            
            # Create figure
            plt.figure(figsize=(16, 8))
            
            # Plot original image with detections
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original Image with Detections")
            
            # Draw bounding boxes
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(detection.cls[0].item())
                conf = detection.conf[0].item()
                
                class_name = class_names.get(cls_id, f"Class {cls_id}")
                label = f"{class_name} {conf:.2f}"
                
                # Create rectangle patch
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
                plt.text(x1, y1-10, label, color='red', fontsize=10, backgroundcolor='white')
            
            # Plot saliency map
            plt.subplot(1, 2, 2)
            
            # Combine all saliency maps
            combined_saliency = np.zeros_like(image[:, :, 0], dtype=np.float32)
            for saliency_map in saliency_maps.values():
                combined_saliency = np.maximum(combined_saliency, saliency_map)
            
            # Apply colormap to saliency
            saliency_colored = cv2.applyColorMap((combined_saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
            saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay saliency on image
            alpha = 0.7
            overlay = cv2.addWeighted(image, 1-alpha, saliency_colored, alpha, 0)
            
            plt.imshow(overlay)
            plt.title("Saliency Map")
            
            # Save figure
            plt.tight_layout()
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Use non-interactive backend for matplotlib
            plt.savefig(output_path)
            plt.close()
            
            print(f"Explanation saved to: {output_path}", flush=True)
            return True
        except Exception as e:
            print(f"Error in visualize_explanation: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return False

# Initialize BODEM explainer
try:
    explainer = BODEMExplainer(model_path, image_size=640, device='cpu')
    app.logger.info("BODEM explainer initialized successfully")
except Exception as e:
    app.logger.error(f"Error initializing BODEM explainer: {str(e)}")
    print(f"Error initializing BODEM explainer: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """API endpoint for object detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the uploaded file temporarily
        temp_path = 'static/temp_upload.jpg'
        file.save(temp_path)
        
        # Perform detection
        results = model(temp_path)
        
        # Process results
        detections = []
        for i, det in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(float, det.xyxy[0])
            confidence = float(det.conf[0])
            class_id = int(det.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                "id": i,
                "class": class_name,
                "confidence": confidence,
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })
        
        # Save the result image with bounding boxes
        result_path = 'static/detection_result.jpg'
        result_img = results[0].plot()
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'result_image': '/static/detection_result.jpg'
        })
    
    except Exception as e:
        app.logger.error(f"Error in detection: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bodem_explain', methods=['POST'])
def bodem_explain():
    """Generate BODEM explanation for an uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the uploaded file temporarily
        temp_path = 'static/temp_upload.jpg'
        file.save(temp_path)
        
        # Generate BODEM explanation
        image, detections, saliency_maps = explainer.explain(temp_path)
        
        if len(detections) == 0:
            return jsonify({'error': 'No objects detected in image'}), 400
        
        # Create output filename
        output_path = 'static/bodem_explanation.png'
        
        # Visualize and save explanation
        explainer.visualize_explanation(image, detections, saliency_maps, model.names, output_path)
        
        # Return the path to the generated visualization
        return jsonify({
            'success': True,
            'explanation_path': '/static/bodem_explanation.png',
            'num_detections': len(detections)
        })
    
    except Exception as e:
        app.logger.error(f"Error in BODEM explanation: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/templates/<path:filename>')
def serve_template(filename):
    """Serve template files"""
    return send_from_directory('templates', filename)

# Create a simple HTML template for testing
@app.route('/create_test_template')
def create_test_template():
    """Create a simple HTML template for testing"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Detection App</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .upload-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
            }
            .button:hover {
                background-color: #45a049;
            }
            .result-container {
                margin-top: 20px;
                display: none;
            }
            .result-image {
                max-width: 100%;
                border: 1px solid #ddd;
                margin-top: 10px;
            }
            .detection-list {
                margin-top: 20px;
            }
            .error {
                color: red;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Weather Condition Object Detection</h1>
        
        <div class="upload-container">
            <h2>Upload an image for detection</h2>
            <input type="file" id="imageUpload" accept="image/*">
            <button class="button" id="detectButton">Detect Objects</button>
            <div id="error" class="error"></div>
        </div>
        
        <div id="resultContainer" class="result-container">
            <h2>Detection Results</h2>
            <img id="resultImage" class="result-image">
            <div id="detectionList" class="detection-list"></div>
        </div>
        
        <div class="upload-container">
            <h2>Generate BODEM Explanation</h2>
            <input type="file" id="bodemImageUpload" accept="image/*">
            <button class="button" id="explainButton">Generate Explanation</button>
            <div id="bodemError" class="error"></div>
        </div>
        
        <div id="bodemResultContainer" class="result-container">
            <h2>BODEM Explanation</h2>
            <img id="bodemImage" class="result-image">
            <p>The BODEM explanation shows which parts of the image influenced the model's detection decisions. Brighter areas in the saliency map (right side) indicate regions that were more important for the detection.</p>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const imageUpload = document.getElementById('imageUpload');
                const detectButton = document.getElementById('detectButton');
                const resultContainer = document.getElementById('resultContainer');
                const resultImage = document.getElementById('resultImage');
                const detectionList = document.getElementById('detectionList');
                const errorDiv = document.getElementById('error');
                
                const bodemImageUpload = document.getElementById('bodemImageUpload');
                const explainButton = document.getElementById('explainButton');
                const bodemResultContainer = document.getElementById('bodemResultContainer');
                const bodemImage = document.getElementById('bodemImage');
                const bodemErrorDiv = document.getElementById('bodemError');
                
                detectButton.addEventListener('click', function() {
                    if (!imageUpload.files || imageUpload.files.length === 0) {
                        errorDiv.textContent = 'Please select an image first';
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', imageUpload.files[0]);
                    
                    errorDiv.textContent = '';
                    detectButton.disabled = true;
                    detectButton.textContent = 'Detecting...';
                    
                    fetch('/api/detect', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        detectButton.disabled = false;
                        detectButton.textContent = 'Detect Objects';
                        
                        if (data.error) {
                            errorDiv.textContent = data.error;
                            return;
                        }
                        
                        resultImage.src = data.result_image + '?t=' + new Date().getTime();
                        
                        detectionList.innerHTML = '<h3>Detected Objects:</h3>';
                        const ul = document.createElement('ul');
                        
                        data.detections.forEach(det => {
                            const li = document.createElement('li');
                            li.textContent = `${det.class} (Confidence: ${(det.confidence * 100).toFixed(1)}%)`;
                            ul.appendChild(li);
                        });
                        
                        detectionList.appendChild(ul);
                        resultContainer.style.display = 'block';
                    })
                    .catch(error => {
                        detectButton.disabled = false;
                        detectButton.textContent = 'Detect Objects';
                        errorDiv.textContent = 'An error occurred: ' + error.message;
                    });
                });
                
                explainButton.addEventListener('click', function() {
                    if (!bodemImageUpload.files || bodemImageUpload.files.length === 0) {
                        bodemErrorDiv.textContent = 'Please select an image first';
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', bodemImageUpload.files[0]);
                    
                    bodemErrorDiv.textContent = '';
                    explainButton.disabled = true;
                    explainButton.textContent = 'Generating...';
                    
                    fetch('/api/bodem_explain', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        explainButton.disabled = false;
                        explainButton.textContent = 'Generate Explanation';
                        
                        if (data.error) {
                            bodemErrorDiv.textContent = data.error;
                            return;
                        }
                        
                        bodemImage.src = data.explanation_path + '?t=' + new Date().getTime();
                        bodemResultContainer.style.display = 'block';
                    })
                    .catch(error => {
                        explainButton.disabled = false;
                        explainButton.textContent = 'Generate Explanation';
                        bodemErrorDiv.textContent = 'An error occurred: ' + error.message;
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    return "Test template created successfully. Visit the root URL (/) to see it."

if __name__ == '__main__':
    # Create test template on startup
    if not os.path.exists('templates/index.html'):
        os.makedirs('templates', exist_ok=True)
        app.test_client().get('/create_test_template')
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5001, debug=True)
