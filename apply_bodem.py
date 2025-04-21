import os
import yaml
import argparse
import sys
import traceback
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2
from bodem_explainer import BODEMExplainer

def load_class_names(yaml_path):
    """Load class names from YAML file."""
    try:
        print(f"Loading class names from {yaml_path}", flush=True)
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        class_names = data.get('names', {})
        print(f"Loaded {len(class_names)} class names", flush=True)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}", flush=True)
        traceback.print_exc()
        return {}

def verify_image_path(image_path):
    """Verify that an image exists and can be opened."""
    if not os.path.exists(image_path):
        print(f"Image does not exist: {image_path}", flush=True)
        return False
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}", flush=True)
            return False
        return True
    except Exception as e:
        print(f"Error verifying image {image_path}: {e}", flush=True)
        return False

def analyze_dataset(model_path, dataset_path, output_dir, class_names, dataset_name, 
                   weather_conditions=None, num_samples=5, device='cpu'):
    """
    Apply BODEM to analyze a dataset.
    
    Args:
        model_path: Path to trained YOLOv8 model
        dataset_path: Path to dataset images
        output_dir: Directory to save explanations
        class_names: Dictionary of class names
        dataset_name: Name of the dataset (DAWN or WEDGE)
        weather_conditions: List of weather conditions to analyze
        num_samples: Number of samples per weather condition
        device: Device to run inference on ('cpu' or 'cuda')
    """
    print(f"\n{'='*80}\nAnalyzing dataset: {dataset_name}\n{'='*80}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Dataset path: {dataset_path}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}", flush=True)
    except Exception as e:
        print(f"Error creating output directory: {e}", flush=True)
        traceback.print_exc()
        return
    
    # Create BODEM explainer
    try:
        print(f"Initializing BODEM explainer with model: {model_path}", flush=True)
        explainer = BODEMExplainer(model_path, image_size=640, device=device)
        print("BODEM explainer initialized successfully", flush=True)
    except Exception as e:
        print(f"Error initializing BODEM explainer: {e}", flush=True)
        traceback.print_exc()
        return
    
    # Test model with a simple inference
    try:
        test_model = YOLO(model_path)
        print("Model loaded successfully for testing", flush=True)
    except Exception as e:
        print(f"Error loading model for testing: {e}", flush=True)
        traceback.print_exc()
        return
    
    # Get all image files
    print(f"Scanning for images in {dataset_path}", flush=True)
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                
                # Extract weather condition from filename or path
                weather = None
                if dataset_name == "DAWN":
                    # For DAWN, weather is in the directory name
                    parts = root.split(os.sep)
                    for part in parts:
                        if part in ["Rain", "Snow", "Sand", "Fog"]:
                            weather = part
                            break
                else:
                    # For WEDGE, weather might be in the filename
                    parts = file.split('_')
                    if len(parts) > 1:
                        for part in parts:
                            if part.lower() in ["rain", "snow", "fog", "dust", "cloudy", "sunny"]:
                                weather = part
                                break
                
                if weather_conditions is None or weather in weather_conditions:
                    # Verify the image can be opened
                    if verify_image_path(image_path):
                        image_files.append((image_path, weather if weather else "Unknown"))
    
    print(f"Found {len(image_files)} valid images", flush=True)
    
    # Group by weather condition
    weather_groups = {}
    for image_path, weather in image_files:
        if weather not in weather_groups:
            weather_groups[weather] = []
        weather_groups[weather].append(image_path)
    
    # Print summary of weather conditions
    print("\nWeather condition summary:", flush=True)
    for weather, images in weather_groups.items():
        print(f"  {weather}: {len(images)} images", flush=True)
    
    # Sample images from each weather condition
    sampled_images = []
    for weather, images in weather_groups.items():
        # Take a sample of images for each weather condition
        samples = images[:num_samples] if len(images) > num_samples else images
        for image_path in samples:
            sampled_images.append((image_path, weather))
    
    print(f"\nSelected {len(sampled_images)} images for analysis", flush=True)
    
    # Apply BODEM to sampled images
    for i, (image_path, weather) in enumerate(sampled_images):
        print(f"\nProcessing image {i+1}/{len(sampled_images)}: {image_path}", flush=True)
        print(f"Weather condition: {weather}", flush=True)
        
        try:
            # Test detection first
            results = test_model(image_path, verbose=False)
            num_detections = len(results[0].boxes)
            print(f"Found {num_detections} detections in image", flush=True)
            
            if num_detections == 0:
                print("Skipping image with no detections", flush=True)
                continue
            
            # Generate explanation
            print("Generating BODEM explanation...", flush=True)
            image, detections, saliency_maps = explainer.explain(image_path)
            
            print(f"Generated explanation with {len(detections)} detections", flush=True)
            
            if len(detections) > 0:
                # Create output filename
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{dataset_name}_{weather}_{base_name.split('.')[0]}_explanation.png")
                
                # Visualize and save explanation
                print(f"Saving explanation to {output_path}", flush=True)
                explainer.visualize_explanation(image, detections, saliency_maps, class_names, output_path)
                
                # Verify the file was created
                if os.path.exists(output_path):
                    print(f"Successfully saved explanation to {output_path}", flush=True)
                else:
                    print(f"Failed to save explanation to {output_path}", flush=True)
            else:
                print("No detections found in BODEM explanation", flush=True)
        except Exception as e:
            print(f"Error processing {image_path}: {e}", flush=True)
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Apply BODEM to analyze object detection datasets")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLOv8 model")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--output", type=str, default="bodem_explanations", help="Output directory for explanations")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per weather condition")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on ('cpu' or 'cuda')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Print script information
    print("\nBODEM Analysis Script", flush=True)
    print(f"Running on Python {sys.version}", flush=True)
    print(f"Arguments: {args}", flush=True)
    
    # Load class names from YAML
    class_names = load_class_names(args.data)
    if not class_names:
        print("Error: Failed to load class names from YAML file", flush=True)
        return
    
    # Create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
        print(f"Created main output directory: {args.output}", flush=True)
    except Exception as e:
        print(f"Error creating main output directory: {e}", flush=True)
        traceback.print_exc()
        return
    
    # Define dataset paths
    try:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        base_path = data_config.get('path', '')
        test_path = os.path.join(base_path, data_config.get('test', 'test/images'))
        print(f"Base path: {base_path}", flush=True)
        print(f"Test path: {test_path}", flush=True)
    except Exception as e:
        print(f"Error loading data config: {e}", flush=True)
        traceback.print_exc()
        return
    
    # Define weather conditions to analyze
    weather_conditions = ["Rain", "Snow", "Sand", "Fog", "Dust", "Unknown"]
    print(f"Weather conditions to analyze: {weather_conditions}", flush=True)
    
    # Analyze DAWN dataset
    dawn_path = os.path.join(test_path, "DAWN")
    if os.path.exists(dawn_path):
        print(f"Found DAWN dataset at {dawn_path}", flush=True)
        dawn_output = os.path.join(args.output, "DAWN")
        analyze_dataset(args.model, dawn_path, dawn_output, class_names, "DAWN", 
                       weather_conditions, args.samples, args.device)
    else:
        print(f"DAWN dataset not found at {dawn_path}", flush=True)
    
    # Analyze WEDGE dataset
    wedge_path = os.path.join(test_path, "WEDGE")
    if os.path.exists(wedge_path):
        print(f"Found WEDGE dataset at {wedge_path}", flush=True)
        wedge_output = os.path.join(args.output, "WEDGE")
        analyze_dataset(args.model, wedge_path, wedge_output, class_names, "WEDGE", 
                       weather_conditions, args.samples, args.device)
    else:
        print(f"WEDGE dataset not found at {wedge_path}", flush=True)
    
    print("\nBODEM analysis complete", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}", flush=True)
        traceback.print_exc()
