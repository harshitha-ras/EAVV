import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import io
import sys
import contextlib
from tqdm.contrib import DummyTqdmFile

# Create a context manager to redirect stdout/stderr to tqdm.write
@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err
    except Exception as exc:
        raise exc
    finally:
        sys.stdout, sys.stderr = orig_out_err

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
