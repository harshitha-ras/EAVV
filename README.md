# Explainable Object Detection System
[![License: MIT](tps://img.shields.io/badge/Licens

## Executive Summary

The Explainable Object Detection System is an advanced computer vision solution specifically designed to work in adverse weather conditions while providing transparent explanations of its detection decisions. This project combines real-world and synthetic datasets to create a robust detection model that can identify objects in challenging weather scenarios such as fog, rain, snow, and dust.

The system addresses a critical need for autonomous vehicles and surveillance systems to function reliably across all weather conditions, which is essential for safe autonomous driving systems. By incorporating explainability through BODEM (Boundary-based Object Detection Explanation Method), it provides insights into how the model makes decisions in different weather conditions.

## Features
Weather-Robust Detection: Specialized training for fog, rain, snow, and dust conditions

Dual-Dataset Approach: Combines real-world DAWN dataset with synthetic WEDGE dataset

Explainable AI: BODEM visualization to understand model decision-making

Web Interface: Interactive application for uploading and analyzing images

Multiple Model Options: YOLOv8n (lightweight) and YOLOv8s (enhanced accuracy)

## Datasets
DAWN (Detection in Adverse Weather Nature)
1,027 real-world traffic images captured in various adverse weather conditions

Includes fog, snow, rain, and sandstorm scenarios

Emphasizes diverse traffic environments (urban, highway, freeway)

Annotated with object bounding boxes for autonomous driving



WEDGE (Multi-Weather Autonomous Driving)
3,360 synthetic images with simulated weather conditions

Generated using DALL-E 2 with prompts in the format {Objects} on {scenes} when {weather}

Includes 16 extreme weather conditions with 210 images per class

Manually annotated with 16,513 bounding boxes


## Exploratory Data Analysis
Object class distribution
![image](https://github.com/user-attachments/assets/c596e776-6802-4645-a365-7b5721973149)


Aspect Ratio Distribution
![image](https://github.com/user-attachments/assets/9e6cecfb-e246-49f6-9153-ef56307e7235)
- Most objects across weather conditions have aspect ratios between 0-2
- Peak density occurs around aspect ratio of 0.5-1.0
- Long tail distribution extending to aspect ratio of 5
- Different weather conditions show similar aspect ratio patterns

## Model Performance
Our weather-refined YOLOv8s model achieves:

mAP50: 0.496 (49.6%)

mAP50-95: 0.25 (25.0%)

Performance varies significantly across different object classes:


## Detection Performance

| Class       | mAP50 | mAP50-95 | Precision | Recall |
|-------------|-------|----------|-----------|--------|
| person      | 0.650 | 0.255    | 0.758     | 0.582  |
| bicycle     | 0.000 | 0.000    | 1.000     | 0.000  |
| car         | 0.632 | 0.281    | 0.751     | 0.560  |
| motorcycle  | 0.299 | 0.141    | 0.311     | 0.375  |
| bus         | 0.727 | 0.437    | 0.801     | 0.669  | 
| truck       | 0.730 | 0.376    | 0.777     | 0.626  |
| **Overall** | 0.506 | 0.248    | 0.733     | 0.469  |



## Explainability Through BODEM
Fog conditions showed highest mean saliency (0.3711), indicating concentrated feature attention

Dust conditions showed lowest mean saliency (0.3315), suggesting more distributed visual cues

Model relies heavily on silhouettes in fog and edge detection across all conditions

Identified weather-specific adaptations in how the model processes different conditions

## Installation


### Prerequisites


Python 3.8+

Node.js (for web interface)

CUDA-compatible GPU (recommended)

### Setup

git clone https://github.com/harshitha-ras/explainable-object-detection.git
cd explainable-object-detection
python -m venv oenv
source oenv/bin/activate  # On Windows: oenv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd weather-detection-frontend
npm install
npm run build


## Usage
### Data Preparation

Convert XML annotations to YOLO format:
python convert_xml_to_yolo.py

Split dataset into train/val/test:
python data_prep.py


### Training

Train YOLOv8 model with weather refinements (CPU):
python train_yolov8.py --mode train --model n --epochs 50 --batch 2 --img-size 640 --data weather_data.yaml --device cpu --oversample --weather-balance --progressive

Train with GPU (if available):
python train_yolov8.py --mode train --model s --epochs 100 --batch 16 --img-size 640 --data weather_data.yaml --device 0 --oversample --weather-balance --progressive


### Validation and Inference

Validate trained model:
python train_yolov8.py --mode validate --weights yolo_output/yolov8s_weather_refined/weights/best.pt --data weather_data.yaml

Run inference on test images:
python train_yolov8.py --mode predict --weights yolo_output/yolov8s_weather_refined/weights/best.pt --source merged_dataset/test/images --conf 0.25


### Explainable AI Analysis

Generate BODEM explanations:
python apply_bodem.py --model yolo_output/yolov8s_weather_refined/weights/best.pt --data weather_data.yaml --output bodem_explanations --samples 10

Analyze explanations: 
python analyze_bodem.py --dir bodem_explanations


## Web Application
1. Start the Flask application:

python app.py

2. Access the web interface at http://localhost:5001

3. Upload images through the browser interface

4. View detection results with bounding boxes and confidence scores

5. Generate BODEM explanations to understand model decisions



## Project Structure


EAVV/                      # Main package
├── README.md
├── requirements.txt
├── setup.py
├── app.py
├── scripts/
│   ├── make_dataset.py
│   ├── build_features.py
│   ├── model.py
│   ├── convert_to_yolo.py
│   ├── data_prep.py
|   └── format_xmls.py
│   ├── train_yolov8.py
│   ├── apply_bodem.py
│   ├── analyze_bodem.py
│   └── bodem_explainer.py
├── models/
│   └── yolo_output/yolov8s_weather_refined.pt
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── notebooks/
│   └── EDA.ipynb
├── .gitignore
├── weather-detection-frontend/


## Citation
If you use this project in your research, please cite:

@software{ExplainableObjectDetection2025,
  author = {Harshitha, Rasamsetty},
  title = {Explainable Object Detection System},
  url = {https://github.com/harshitha-ras/EAVV},
  year = {2025},
}


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
DAWN dataset from IEEE DataPort

Ultralytics for YOLOv8 implementation

The research community working on object detection in adverse weather conditions