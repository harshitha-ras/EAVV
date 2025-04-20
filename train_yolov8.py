from ultralytics import YOLO
import os
import argparse

def train_yolov8(model_size='n', epochs=50, batch_size=8, img_size=1280, 
                 data_yaml='weather_data.yaml', device='0'):
    """
    Train YOLOv8 on the DAWN-WEDGE merged dataset
    
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        data_yaml: Path to the dataset YAML file
        device: Device to train on ('0', '1', 'cpu')
    """
    # Create output directory
    os.makedirs('yolo_output', exist_ok=True)
    
    # Load a pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='yolo_output',
        name=f'yolov8{model_size}_weather',
        patience=10,                # Early stopping patience
        save=True,                  # Save checkpoints
        save_period=10,             # Save every 10 epochs
        lr0=0.01,                   # Initial learning rate
        lrf=0.001,                  # Final learning rate
        cos_lr=True,                # Use cosine learning rate scheduler
        augment=True,               # Use data augmentation
        mixup=0.1,                  # Apply mixup augmentation
        mosaic=1.0,                 # Apply mosaic augmentation
        degrees=10.0,               # Rotation augmentation
        translate=0.1,              # Translation augmentation
        scale=0.5,                  # Scale augmentation
        shear=2.0,                  # Shear augmentation
        perspective=0.0,            # Perspective augmentation
        flipud=0.0,                 # Vertical flip augmentation
        fliplr=0.5,                 # Horizontal flip augmentation
        hsv_h=0.015,                # HSV hue augmentation
        hsv_s=0.7,                  # HSV saturation augmentation
        hsv_v=0.4,                  # HSV value augmentation
        warmup_epochs=3,            # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        warmup_bias_lr=0.1,         # Warmup bias learning rate
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Class loss weight
        dfl=1.5,                    # DFL loss weight
        close_mosaic=10,            # Disable mosaic augmentation for final epochs
        verbose=True,               # Print verbose output
    )
    
    return results, model

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
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DAWN-WEDGE dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'predict'],
                        help='Mode: train, validate, or predict')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size')
    parser.add_argument('--data', type=str, default='weather_data.yaml', help='Path to data YAML')
    parser.add_argument('--weights', type=str, default=None, 
                        help='Path to weights file for validation/prediction')
    parser.add_argument('--source', type=str, default='merged_dataset/test/images', 
                        help='Source path for prediction')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0, 1, cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        results, model = train_yolov8(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            data_yaml=args.data,
            device=args.device
        )
        print(f"Training completed. Best model saved at: {model.best}")
        
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
