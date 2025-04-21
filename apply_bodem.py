import os
import yaml
import argparse
from tqdm import tqdm
from bodem_explainer import BODEMExplainer

def load_class_names(yaml_path):
    """Load class names from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def analyze_dataset(model_path, dataset_path, output_dir, class_names, dataset_name, weather_conditions=None, num_samples=5):
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
    """
    # Create BODEM explainer
    explainer = BODEMExplainer(model_path, image_size=640, device='cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                
                # Extract weather condition from filename
                weather = None
                if dataset_name == "DAWN":
                    # For DAWN, weather is in the directory name
                    parts = root.split('/')
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
                    image_files.append((image_path, weather))
    
    # Group by weather condition
    weather_groups = {}
    for image_path, weather in image_files:
        if weather not in weather_groups:
            weather_groups[weather] = []
        weather_groups[weather].append(image_path)
    
    # Sample images from each weather condition
    sampled_images = []
    for weather, images in weather_groups.items():
        # Take a sample of images for each weather condition
        samples = images[:num_samples] if len(images) > num_samples else images
        for image_path in samples:
            sampled_images.append((image_path, weather))
    
    # Apply BODEM to sampled images
    for image_path, weather in tqdm(sampled_images, desc=f"Analyzing {dataset_name}"):
        try:
            # Generate explanation
            image, detections, saliency_maps = explainer.explain(image_path)
            
            if len(detections) > 0:
                # Create output filename
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{dataset_name}_{weather}_{base_name.split('.')[0]}_explanation.png")
                
                # Visualize and save explanation
                explainer.visualize_explanation(image, detections, saliency_maps, class_names, output_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply BODEM to analyze object detection datasets")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLOv8 model")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--output", type=str, default="bodem_explanations", help="Output directory for explanations")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per weather condition")
    args = parser.parse_args()
    
    # Load class names from YAML
    class_names = load_class_names(args.data)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Define dataset paths
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    base_path = data_config.get('path', '')
    test_path = os.path.join(base_path, data_config.get('test', 'test/images'))
    
    # Define weather conditions to analyze
    weather_conditions = ["Rain", "Snow", "Sand", "Fog", "Dust", "Unknown"]
    
    # Analyze DAWN dataset
    dawn_path = os.path.join(test_path, "DAWN")
    if os.path.exists(dawn_path):
        dawn_output = os.path.join(args.output, "DAWN")
        os.makedirs(dawn_output, exist_ok=True)
        analyze_dataset(args.model, dawn_path, dawn_output, class_names, "DAWN", weather_conditions, args.samples)
    
    # Analyze WEDGE dataset
    wedge_path = os.path.join(test_path, "WEDGE")
    if os.path.exists(wedge_path):
        wedge_output = os.path.join(args.output, "WEDGE")
        os.makedirs(wedge_output, exist_ok=True)
        analyze_dataset(args.model, wedge_path, wedge_output, class_names, "WEDGE", weather_conditions, args.samples)

if __name__ == "__main__":
    main()
