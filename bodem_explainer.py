import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import argparse
from bodem_explainer import BODEMExplainer

class BODEMExplainer:
    def __init__(self, model_path, image_size=640, device='cpu'):
        """
        Initialize the BODEM explainer for object detection models.
        
        Args:
            model_path: Path to the trained YOLOv8 model
            image_size: Input image size for the model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model = YOLO(model_path)
        self.image_size = image_size
        self.device = device
        
    def generate_hierarchical_masks(self, image, detected_object, level=1, max_level=6):
        """
        Generate hierarchical masks for the input image.
        
        Args:
            image: Input image (numpy array)
            detected_object: Bounding box coordinates [x1, y1, x2, y2]
            level: Current hierarchy level
            max_level: Maximum hierarchy level
            
        Returns:
            List of masked images and corresponding masks
        """
        h, w = image.shape[:2]
        
        # Define block size based on level
        if level == 1:
            block_size = (128, 128)
        else:
            # Each level divides the previous level's block size by 2
            prev_block_size = (128 // (2 ** (level - 2)), 128 // (2 ** (level - 2)))
            block_size = (prev_block_size[0] // 2, prev_block_size[1] // 2)
        
        # Resize image to match model input size while maintaining aspect ratio
        scale_factor = min(self.image_size / w, self.image_size / h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Create a canvas with model input size
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized_image
        
        # Adjust bounding box to the resized image
        x1, y1, x2, y2 = detected_object
        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
        
        # Create grid of blocks
        blocks = []
        for y in range(0, self.image_size, block_size[1]):
            for x in range(0, self.image_size, block_size[0]):
                # Check if block overlaps with the object
                if level == 1 or self.block_overlaps_object(x, y, block_size, (x1, y1, x2, y2)):
                    blocks.append((x, y, min(x + block_size[0], self.image_size), min(y + block_size[1], self.image_size)))
        
        # Generate masks and masked images
        masked_images = []
        masks = []
        
        # Number of masks to generate (as per BODEM paper)
        n_masks = 100
        
        for _ in range(n_masks):
            # Create a random binary mask
            mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
            
            # Randomly select blocks to mask
            selected_blocks = np.random.choice(len(blocks), size=np.random.randint(1, len(blocks) + 1), replace=False)
            
            for idx in selected_blocks:
                x1_b, y1_b, x2_b, y2_b = blocks[idx]
                mask[y1_b:y2_b, x1_b:x2_b] = 0
            
            # Apply mask to image
            masked_image = canvas.copy()
            for c in range(3):
                masked_image[:, :, c] = masked_image[:, :, c] * mask
            
            masked_images.append(masked_image)
            masks.append(mask)
        
        return masked_images, masks
    
    def block_overlaps_object(self, x, y, block_size, obj_bbox):
        """Check if a block overlaps with the object bounding box."""
        x1_b, y1_b = x, y
        x2_b, y2_b = x + block_size[0], y + block_size[1]
        x1_o, y1_o, x2_o, y2_o = obj_bbox
        
        # Check for overlap
        return not (x2_b <= x1_o or x1_b >= x2_o or y2_b <= y1_o or y1_b >= y2_o)
    
    def compute_similarity(self, original_obj, masked_obj):
        """Compute IoU similarity between original and masked detections."""
        if masked_obj is None:
            return 0.0
        
        # Extract coordinates
        x1_o, y1_o, x2_o, y2_o = original_obj
        x1_m, y1_m, x2_m, y2_m = masked_obj
        
        # Calculate intersection area
        x1_i = max(x1_o, x1_m)
        y1_i = max(y1_o, y1_m)
        x2_i = min(x2_o, x2_m)
        y2_i = min(y2_o, y2_m)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area_o = (x2_o - x1_o) * (y2_o - y1_o)
        area_m = (x2_m - x1_m) * (y2_m - y1_m)
        union = area_o + area_m - intersection
        
        return intersection / union
    
    def estimate_saliency(self, image, detected_objects, level=1, max_level=6, prev_saliency_map=None):
        """
        Estimate saliency maps for detected objects using hierarchical masking.
        
        Args:
            image: Input image
            detected_objects: List of detected objects (bounding boxes and classes)
            level: Current hierarchy level
            max_level: Maximum hierarchy level
            prev_saliency_map: Saliency map from previous level
            
        Returns:
            Dictionary of saliency maps for each detected object
        """
        saliency_maps = {}
        
        for obj_idx, obj in enumerate(detected_objects):
            bbox, class_id, conf = obj
            
            # Generate masked images
            masked_images, masks = self.generate_hierarchical_masks(image, bbox, level, max_level)
            
            # Initialize saliency map
            if prev_saliency_map is None or obj_idx not in prev_saliency_map:
                saliency_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                saliency_map = prev_saliency_map[obj_idx].copy()
            
            # Run inference on masked images
            for i, masked_image in enumerate(masked_images):
                # Get predictions for masked image
                results = self.model(masked_image, verbose=False)
                
                # Find the corresponding detection (if any)
                max_iou = 0
                matched_obj = None
                
                for det in results[0].boxes.data:
                    det_bbox = det[:4].cpu().numpy()
                    iou = self.compute_similarity(bbox, det_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        matched_obj = det
                
                # Update saliency map based on detection similarity
                importance = 1.0 - max_iou  # Higher importance for regions that affect detection
                
                # Apply importance to mask
                saliency_update = masks[i] * importance
                
                # Update saliency map
                saliency_map += saliency_update
            
            # Normalize saliency map
            if np.max(saliency_map) > 0:
                saliency_map = saliency_map / np.max(saliency_map)
            
            saliency_maps[obj_idx] = saliency_map
        
        return saliency_maps
    
    def explain(self, image_path, max_level=6):
        """
        Generate BODEM explanations for detections in an image.
        
        Args:
            image_path: Path to the input image
            max_level: Maximum hierarchy level for masking
            
        Returns:
            Original image, detections, and saliency maps
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(image, verbose=False)
        
        # Extract detections
        detections = []
        for det in results[0].boxes.data:
            bbox = det[:4].cpu().numpy().astype(int)
            class_id = int(det[5].item())
            conf = det[4].item()
            detections.append((bbox, class_id, conf))
        
        # Initialize saliency maps
        saliency_maps = None
        
        # Hierarchical saliency estimation
        for level in range(1, max_level + 1):
            saliency_maps = self.estimate_saliency(image, detections, level, max_level, saliency_maps)
        
        return image, detections, saliency_maps
    
    def visualize_explanation(self, image, detections, saliency_maps, class_names, output_path=None):
        """
        Visualize BODEM explanations.
        
        Args:
            image: Original image
            detections: List of detected objects
            saliency_maps: Dictionary of saliency maps
            class_names: Dictionary mapping class IDs to names
            output_path: Path to save the visualization
        """
        h, w = image.shape[:2]
        
        # Resize image to match model input size while maintaining aspect ratio
        scale_factor = min(self.image_size / w, self.image_size / h)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Create a canvas with model input size
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized_image
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original image with detections
        axes[0].imshow(canvas)
        axes[0].set_title("Original Image with Detections")
        
        for obj_idx, (bbox, class_id, conf) in enumerate(detections):
            x1, y1, x2, y2 = bbox
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
            
            # Add label
            axes[0].text(x1, y1 - 5, f"{class_name} {conf:.2f}", color='white', backgroundcolor='red', fontsize=10)
        
        # Plot combined saliency map
        combined_saliency = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for obj_idx in saliency_maps:
            combined_saliency += saliency_maps[obj_idx]
        
        if np.max(combined_saliency) > 0:
            combined_saliency = combined_saliency / np.max(combined_saliency)
        
        # Apply colormap to saliency map
        heatmap = cm.jet(combined_saliency)[:, :, :3]
        
        # Overlay heatmap on image
        alpha = 0.7
        overlay = canvas.copy() / 255.0
        for c in range(3):
            overlay[:, :, c] = (1 - alpha) * overlay[:, :, c] + alpha * heatmap[:, :, c] * combined_saliency
        
        axes[1].imshow(overlay)
        axes[1].set_title("BODEM Explanation")
        
        # Remove axes
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


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
