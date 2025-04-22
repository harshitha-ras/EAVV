from ultralytics import YOLO
import os
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
import random
import shutil

def train_yolov8_refined(model_size='n', epochs=100, batch_size=16, img_size=640, 
                         data_yaml='weather_data.yaml', device='0', oversample=True,
                         weather_balanced=True, progressive_learning=True):
    """
    Train YOLOv8 on the DAWN-WEDGE merged dataset with refined training strategies
    
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        data_yaml: Path to the dataset YAML file
        device: Device to train on ('0', '1', 'cpu')
        oversample: Whether to oversample rare classes
        weather_balanced: Whether to balance weather conditions in the dataset
        progressive_learning: Whether to use progressive learning strategy
    """
    # Create output directory
    os.makedirs('yolo_output', exist_ok=True)
    
    # Apply data preparation strategies
    if oversample:
        print("Applying class oversampling for rare classes...")
        data_yaml = create_oversampling_dataset(data_yaml)
    
    if weather_balanced:
        print("Applying weather condition balancing...")
        data_yaml = create_weather_balanced_dataset(data_yaml)
    
    # Load a pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Define training parameters with refined augmentation strategy
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 4,  # Increased from 0 for better CPU utilization
        'project': 'yolo_output',
        'name': f'yolov8{model_size}_weather_refined',
        'patience': 10,                # Early stopping patience
        'save': True,                  # Save checkpoints
        'save_period': 10,             # Save every 10 epochs
        
        # Learning rate refinements
        'lr0': 0.01,                   # Initial learning rate
        'lrf': 0.001,                  # Final learning rate
        'cos_lr': True,                # Use cosine learning rate scheduler
        
        # Enhanced augmentation parameters based on BODEM analysis
        'augment': True,               # Use data augmentation
        'mixup': 0.15,                 # Increased from 0.1 for better generalization
        'mosaic': 1.0,                 # Apply mosaic augmentation
        'degrees': 10.0,               # Rotation augmentation
        'translate': 0.2,              # Increased from 0.1 for better position invariance
        'scale': 0.5,                  # Scale augmentation
        'shear': 2.0,                  # Shear augmentation
        'perspective': 0.0,            # Perspective augmentation
        'flipud': 0.0,                 # Vertical flip augmentation
        'fliplr': 0.5,                 # Horizontal flip augmentation
        
        # Enhanced HSV augmentation for better weather robustness
        'hsv_h': 0.03,                 # Increased from 0.015 for better hue variation
        'hsv_s': 0.6,                  # Decreased from 0.7 to avoid oversaturation
        'hsv_v': 0.5,                  # Increased from 0.4 for better brightness variation
        
        # Warmup parameters
        'warmup_epochs': 3,            # Warmup epochs
        'warmup_momentum': 0.8,        # Warmup momentum
        'warmup_bias_lr': 0.1,         # Warmup bias learning rate
        
        # Loss weights
        'box': 7.5,                    # Box loss weight
        'cls': 0.5,                    # Class loss weight
        'dfl': 1.5,                    # DFL loss weight
        
        # Progressive learning strategy
        'close_mosaic': 10 if progressive_learning else 0,  # Disable mosaic in last 10 epochs
        'verbose': True,               # Print verbose output
    }
    
    # Train the model with refined parameters
    results = model.train(**train_args)
    
    return results, model

def create_weather_balanced_dataset(data_yaml, weather_types=['Rain', 'Snow', 'Fog', 'Dust']):
    """
    Create a dataset with balanced weather conditions through duplication
    
    Args:
        data_yaml: Path to the YAML file
        weather_types: List of weather condition types to balance
    """
    import yaml
    from pathlib import Path
    
    # Load data configuration
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Create balanced directory
    base_dir = Path(data['path'])
    balanced_dir = base_dir / "train_weather_balanced"
    balanced_img_dir = balanced_dir / "images"
    balanced_lbl_dir = balanced_dir / "labels"
    
    os.makedirs(balanced_img_dir, exist_ok=True)
    os.makedirs(balanced_lbl_dir, exist_ok=True)
    
    # Copy all original training data
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    
    # Copy all original files first
    for img_file in os.listdir(train_img_dir):
        shutil.copy2(train_img_dir / img_file, balanced_img_dir / img_file)
        
        # Copy corresponding label file
        lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(train_lbl_dir / lbl_file):
            shutil.copy2(train_lbl_dir / lbl_file, balanced_lbl_dir / lbl_file)
    
    # Count images by weather type
    weather_counts = {weather: 0 for weather in weather_types}
    weather_images = {weather: [] for weather in weather_types}
    
    for img_file in os.listdir(train_img_dir):
        for weather in weather_types:
            if weather.lower() in img_file.lower():
                weather_counts[weather] += 1
                weather_images[weather].append(img_file)
                break
    
    print("Weather condition counts in original dataset:")
    for weather, count in weather_counts.items():
        print(f"  {weather}: {count} images")
    
    # Find the maximum count
    max_count = max(weather_counts.values())
    
    # Duplicate images to balance weather conditions
    for weather, images in weather_images.items():
        if len(images) == 0:
            continue
            
        # How many times to duplicate each image
        duplication_factor = max(1, int(max_count / len(images)))
        remaining = max_count - (len(images) * duplication_factor)
        
        print(f"Duplicating {weather} images {duplication_factor} times plus {remaining} extra")
        
        # Duplicate all images
        for i in range(1, duplication_factor):
            for img_file in images:
                lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                
                new_img_file = f"weather_balance_{weather}_{i}_{img_file}"
                new_lbl_file = f"weather_balance_{weather}_{i}_{lbl_file}"
                
                shutil.copy2(train_img_dir / img_file, balanced_img_dir / new_img_file)
                if os.path.exists(train_lbl_dir / lbl_file):
                    shutil.copy2(train_lbl_dir / lbl_file, balanced_lbl_dir / new_lbl_file)
        
        # Add remaining images
        for i in range(remaining):
            img_file = random.choice(images)
            lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            
            new_img_file = f"weather_balance_{weather}_extra_{i}_{img_file}"
            new_lbl_file = f"weather_balance_{weather}_extra_{i}_{lbl_file}"
            
            shutil.copy2(train_img_dir / img_file, balanced_img_dir / new_img_file)
            if os.path.exists(train_lbl_dir / lbl_file):
                shutil.copy2(train_lbl_dir / lbl_file, balanced_lbl_dir / new_lbl_file)
    
    # Update YAML file
    balanced_yaml = data_yaml.replace('.yaml', '_weather_balanced.yaml')
    data['train'] = "train_weather_balanced/images"
    
    with open(balanced_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Created weather-balanced dataset with {len(os.listdir(balanced_img_dir))} images")
    print(f"Original dataset had {len(os.listdir(train_img_dir))} images")
    
    return balanced_yaml

def create_oversampling_dataset(data_yaml, rare_classes=['bicycle', 'train', 'motorcycle']):
    """
    Create a dataset with oversampling for rare classes
    
    Args:
        data_yaml: Path to the YAML file
        rare_classes: List of rare class names to oversample
    """
    import yaml
    from pathlib import Path
    import random
    import shutil
    
    # Load class names from YAML
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data['names']
    class_ids = {v: k for k, v in class_names.items()}
    
    # Create oversampling directory
    base_dir = Path(data['path'])
    oversample_dir = base_dir / "train_oversample"
    oversample_img_dir = oversample_dir / "images"
    oversample_lbl_dir = oversample_dir / "labels"
    
    os.makedirs(oversample_img_dir, exist_ok=True)
    os.makedirs(oversample_lbl_dir, exist_ok=True)
    
    # Copy all original training data
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    
    for img_file in os.listdir(train_img_dir):
        shutil.copy2(train_img_dir / img_file, oversample_img_dir / img_file)
        
        # Copy corresponding label file
        lbl_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(train_lbl_dir / lbl_file):
            shutil.copy2(train_lbl_dir / lbl_file, oversample_lbl_dir / lbl_file)
    
    # Find images with rare classes
    rare_class_images = []
    for lbl_file in os.listdir(train_lbl_dir):
        with open(train_lbl_dir / lbl_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            class_id = int(line.split()[0])
            class_name = class_names[class_id]
            
            if class_name in rare_classes:
                img_file = lbl_file.replace('.txt', '.jpg')
                if os.path.exists(train_img_dir / img_file):
                    rare_class_images.append((img_file, lbl_file))
                break
    
    # Oversample rare classes (duplicate 5x)
    for i in range(5):  # Duplicate 5 times
        for img_file, lbl_file in rare_class_images:
            new_img_file = f"oversample_{i}_{img_file}"
            new_lbl_file = f"oversample_{i}_{lbl_file}"
            
            shutil.copy2(train_img_dir / img_file, oversample_img_dir / new_img_file)
            shutil.copy2(train_lbl_dir / lbl_file, oversample_lbl_dir / new_lbl_file)
    
    # Update YAML file
    oversample_yaml = data_yaml.replace('.yaml', '_oversample.yaml')
    data['train'] = "train_oversample/images"
    
    with open(oversample_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Created oversampled dataset with {len(os.listdir(oversample_img_dir))} images")
    print(f"Original dataset had {len(os.listdir(train_img_dir))} images")
    print(f"Added {len(rare_class_images) * 5} oversampled rare class images")
    
    return oversample_yaml

def apply_style_transfer_augmentation(data_yaml, output_dir="style_transfer_augmented", 
                                      weather_types=['Dust', 'Snow'], samples_per_weather=50):
    """
    Apply style transfer augmentation to generate additional weather condition images
    
    Args:
        data_yaml: Path to the YAML file
        output_dir: Directory to save augmented images
        weather_types: Weather types to focus on for augmentation
        samples_per_weather: Number of samples to generate per weather type
    """
    # This is a placeholder for the style transfer implementation
    # In a real implementation, you would use a pre-trained style transfer model
    # to generate additional weather condition images
    
    print("Style transfer augmentation would be implemented here")
    print(f"Would generate {samples_per_weather} samples for each of {weather_types}")
    
    # Return the original YAML as this is just a placeholder
    return data_yaml

def validate_model(model_path, data_yaml='weather_data.yaml', img_size=1280):
    """
    Validate a trained YOLOv8 model
    
    Args:
        model_path: Path to the trained model weights
        data_yaml: Path to the dataset YAML file
        img_size: Image size for validation
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate the model
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=16,
        device='0',
        project='yolo_output',
        name='validation',
        plots=True,           # Generate validation plots
        save_json=True,       # Save results to JSON
        save_conf=True,       # Save confidences
        save_txt=True,        # Save results to TXT
        verbose=True,         # Print verbose output
    )
    
    return results

def predict_with_model(model_path, source_path, conf_threshold=0.25, img_size=1280):
    """
    Run inference with a trained YOLOv8 model
    
    Args:
        model_path: Path to the trained model weights
        source_path: Path to the source images or video
        conf_threshold: Confidence threshold for predictions
        img_size: Image size for inference
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source_path,
        conf=conf_threshold,
        imgsz=img_size,
        device='0',
        project='yolo_output',
        name='predictions',
        save=True,            # Save results
        save_txt=True,        # Save results to TXT
        save_conf=True,       # Save confidences
        line_width=2,         # Line width for bounding boxes
        hide_labels=False,    # Show labels
        hide_conf=False,      # Show confidences
        visualize=False,      # Visualize model features
        augment=False,        # TTA (Test Time Augmentation)
        agnostic_nms=False,   # Class-agnostic NMS
        retina_masks=True,    # High-resolution segmentation masks
        max_det=300,          # Maximum detections per image
        vid_stride=1,         # Video frame-rate stride
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DAWN-WEDGE dataset with refined strategies')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'predict'],
                        help='Mode: train, validate, or predict')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--data', type=str, default='weather_data.yaml', help='Path to data YAML')
    parser.add_argument('--weights', type=str, default=None, 
                        help='Path to weights file for validation/prediction')
    parser.add_argument('--source', type=str, default='merged_dataset/test/images', 
                        help='Source path for prediction')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0, 1, cpu)')
    parser.add_argument('--oversample', action='store_true', help='Apply class oversampling')
    parser.add_argument('--weather-balance', action='store_true', help='Apply weather condition balancing')
    parser.add_argument('--progressive', action='store_true', help='Use progressive learning strategy')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        results, model = train_yolov8_refined(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            data_yaml=args.data,
            device=args.device,
            oversample=args.oversample,
            weather_balanced=args.weather_balance,
            progressive_learning=args.progressive
        )
        print(f"Training completed with mAP50: {results.box.map50:.3f}")
        print(f"Best model saved at: yolo_output/yolov8{args.model}_weather_refined/weights/best.pt")
        
    elif args.mode == 'validate':
        if args.weights is None:
            print("Error: Please provide weights file path using --weights")
            exit(1)
        results = validate_model(
            model_path=args.weights,
            data_yaml=args.data,
            img_size=args.img_size
        )
        print(f"Validation mAP50: {results.box.map50:.4f}")
        print(f"Validation mAP50-95: {results.box.map:.4f}")
        
    elif args.mode == 'predict':
        if args.weights is None:
            print("Error: Please provide weights file path using --weights")
            exit(1)
        results = predict_with_model(
            model_path=args.weights,
            source_path=args.source,
            conf_threshold=args.conf,
            img_size=args.img_size
        )
        print(f"Prediction completed. Results saved in yolo_output/predictions/")
