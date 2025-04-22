import cv2
import numpy as np
import os
import argparse
import time
from collections import defaultdict
import matplotlib.pyplot as plt

'''
Run this file using these commands:
# Process a single image using the combined method
python traditional_object_detection.py --input path/to/image.jpg --method combined

# Process all images in a directory using HOG detection
python traditional_object_detection.py --input merged_dataset/test/images --method hog --output hog_results

# Process images with contour detection
python traditional_object_detection.py --input merged_dataset/test/images --method contour --output contour_results --max-images 50


'''

class TraditionalObjectDetector:
    """
    Traditional object detection using computer vision techniques without deep learning.
    Implements multiple detection methods for comparison.
    """
    def __init__(self, method='combined', min_area=500, visualization=True):
        """
        Initialize the detector with the specified method.
        
        Args:
            method: Detection method ('hog', 'contour', 'feature', or 'combined')
            min_area: Minimum contour area to be considered an object
            visualization: Whether to visualize results
        """
        self.method = method
        self.min_area = min_area
        self.visualization = visualization
        
        # Initialize HOG detector for people
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize Haar cascades for vehicles
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        
        # Initialize feature detector and matcher
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Templates for feature matching
        self.templates = {}
        
    def load_templates(self, template_dir):
        """Load template images for feature matching."""
        if not os.path.exists(template_dir):
            print(f"Template directory {template_dir} not found.")
            return
            
        for class_name in os.listdir(template_dir):
            class_dir = os.path.join(template_dir, class_name)
            if os.path.isdir(class_dir):
                self.templates[class_name] = []
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_file)
                        template = cv2.imread(img_path)
                        if template is not None:
                            # Compute keypoints and descriptors
                            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
                            if descriptors is not None:
                                self.templates[class_name].append({
                                    'image': template,
                                    'keypoints': keypoints,
                                    'descriptors': descriptors
                                })
                print(f"Loaded {len(self.templates[class_name])} templates for class {class_name}")
    
    def detect_objects_hog(self, image):
        """
        Detect people using HOG and vehicles using Haar cascades.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect people using HOG
        people, weights = self.hog.detectMultiScale(
            image, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        for (x, y, w, h), confidence in zip(people, weights):
            detections.append({
                'class': 'person',
                'confidence': float(confidence),
                'bbox': (x, y, x + w, y + h)
            })
        
        # Detect cars using Haar cascade
        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in cars:
            detections.append({
                'class': 'car',
                'confidence': 0.7,  # Haar doesn't provide confidence, using fixed value
                'bbox': (x, y, x + w, y + h)
            })
            
        return detections
    
    def detect_objects_contour(self, image):
        """
        Detect objects using contour detection and shape analysis.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with class and bounding box
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h
            
            # Classify based on shape and aspect ratio
            if 0.8 <= aspect_ratio <= 1.2:
                # Square-ish objects could be people
                class_name = 'person'
            elif aspect_ratio > 1.2 and aspect_ratio < 3.0:
                # Wide objects could be vehicles
                class_name = 'car'
            else:
                # Unknown
                class_name = 'unknown'
                
            detections.append({
                'class': class_name,
                'confidence': 0.5,  # Using fixed confidence for contour method
                'bbox': (x, y, x + w, y + h)
            })
            
        return detections
    
    def detect_objects_feature(self, image):
        """
        Detect objects using feature matching against templates.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        detections = []
        
        if not self.templates:
            return detections
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            return detections
            
        # Match against templates
        for class_name, templates in self.templates.items():
            for template in templates:
                # Match descriptors
                matches = self.feature_matcher.match(descriptors, template['descriptors'])
                
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Keep only good matches
                good_matches = matches[:10] if len(matches) > 10 else matches
                
                if len(good_matches) >= 5:  # Minimum number of good matches
                    # Get matched keypoints
                    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([template['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Calculate homography
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # Get template dimensions
                        h, w = template['image'].shape[:2]
                        
                        # Define corners of template
                        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                        
                        # Transform corners to image
                        dst = cv2.perspectiveTransform(pts, H)
                        
                        # Get bounding box
                        x_min = int(min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
                        y_min = int(min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
                        x_max = int(max(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0]))
                        y_max = int(max(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1]))
                        
                        # Calculate confidence based on number of matches
                        confidence = len(good_matches) / len(matches) if len(matches) > 0 else 0
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (x_min, y_min, x_max, y_max)
                        })
        
        return detections
    
    def detect_objects_combined(self, image):
        """
        Combine multiple detection methods and apply non-maximum suppression.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        # Get detections from each method
        hog_detections = self.detect_objects_hog(image)
        contour_detections = self.detect_objects_contour(image)
        feature_detections = self.detect_objects_feature(image)
        
        # Combine all detections
        all_detections = hog_detections + contour_detections + feature_detections
        
        # Group by class
        class_detections = defaultdict(list)
        for detection in all_detections:
            class_detections[detection['class']].append(detection)
            
        # Apply non-maximum suppression for each class
        final_detections = []
        for class_name, detections in class_detections.items():
            # Convert to format required by NMS
            boxes = np.array([detection['bbox'] for detection in detections])
            if len(boxes) == 0:
                continue
                
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            # Compute areas
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            # Get confidences
            confidences = np.array([detection['confidence'] for detection in detections])
            
            # Sort by confidence
            idxs = np.argsort(confidences)[::-1]
            
            # Initialize list of picked indexes
            pick = []
            
            # Loop until no indices left
            while len(idxs) > 0:
                # Pick the last index and add to list
                last = len(idxs) - 1
                i = idxs[0]
                pick.append(i)
                
                # Find coordinates of intersection
                xx1 = np.maximum(x1[i], x1[idxs[1:]])
                yy1 = np.maximum(y1[i], y1[idxs[1:]])
                xx2 = np.minimum(x2[i], x2[idxs[1:]])
                yy2 = np.minimum(y2[i], y2[idxs[1:]])
                
                # Compute width and height of intersection
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                
                # Compute ratio of overlap
                overlap = (w * h) / areas[idxs[1:]]
                
                # Delete indices with overlap greater than threshold
                idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > 0.5)[0] + 1)))
            
            # Add picked detections to final list
            for i in pick:
                final_detections.append(detections[i])
        
        return final_detections
    
    def detect(self, image_path):
        """
        Detect objects in an image using the specified method.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Original image with bounding boxes and list of detections
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Resize for faster processing
        h, w = image.shape[:2]
        scale = min(1.0, 800 / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect objects using the specified method
        start_time = time.time()
        
        if self.method == 'hog':
            detections = self.detect_objects_hog(image)
        elif self.method == 'contour':
            detections = self.detect_objects_contour(image)
        elif self.method == 'feature':
            detections = self.detect_objects_feature(image)
        else:  # 'combined'
            detections = self.detect_objects_combined(image)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Draw bounding boxes
        result_image = image_rgb.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Define color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green
            elif class_name == 'car':
                color = (255, 0, 0)  # Red
            elif class_name == 'truck':
                color = (0, 0, 255)  # Blue
            elif class_name == 'bus':
                color = (255, 255, 0)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
                
            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add processing time
        cv2.putText(
            result_image, 
            f"Method: {self.method} | Time: {processing_time:.3f}s", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return result_image, detections, processing_time
    
    def process_dataset(self, dataset_dir, output_dir, max_images=None):
        """
        Process all images in a dataset and save results.
        
        Args:
            dataset_dir: Directory containing images
            output_dir: Directory to save results
            max_images: Maximum number of images to process (None for all)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Find all images
        image_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if max_images is not None:
            image_files = image_files[:max_images]
            
        # Process each image
        results = []
        
        for i, image_path in enumerate(image_files):
            try:
                print(f"Processing image {i+1}/{len(image_files)}: {image_path}")
                
                # Detect objects
                result_image, detections, processing_time = self.detect(image_path)
                
                # Save result image
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"{self.method}_{base_name}")
                cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                
                # Save detection results
                results.append({
                    'image_path': image_path,
                    'detections': detections,
                    'processing_time': processing_time
                })
                
                # Visualize if requested
                if self.visualization:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(result_image)
                    plt.title(f"Method: {self.method} | Detections: {len(detections)} | Time: {processing_time:.3f}s")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save summary
        self.save_summary(results, output_dir)
        
        return results
    
    def save_summary(self, results, output_dir):
        """
        Save summary of detection results.
        
        Args:
            results: List of detection results
            output_dir: Directory to save summary
        """
        summary_path = os.path.join(output_dir, f"{self.method}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Traditional Object Detection Summary - Method: {self.method}\n")
            f.write("=" * 80 + "\n\n")
            
            # Calculate statistics
            total_images = len(results)
            total_detections = sum(len(result['detections']) for result in results)
            total_time = sum(result['processing_time'] for result in results)
            avg_time = total_time / total_images if total_images > 0 else 0
            
            # Count detections by class
            class_counts = defaultdict(int)
            for result in results:
                for detection in result['detections']:
                    class_counts[detection['class']] += 1
            
            # Write statistics
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detections per image: {total_detections / total_images:.2f}\n")
            f.write(f"Total processing time: {total_time:.2f} seconds\n")
            f.write(f"Average processing time per image: {avg_time:.3f} seconds\n\n")
            
            f.write("Detections by class:\n")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {class_name}: {count} ({count / total_detections * 100:.1f}%)\n")
            
            f.write("\n")
            f.write("Individual image results:\n")
            
            # Write individual results
            for i, result in enumerate(results):
                f.write(f"\nImage {i+1}: {os.path.basename(result['image_path'])}\n")
                f.write(f"  Processing time: {result['processing_time']:.3f} seconds\n")
                f.write(f"  Detections: {len(result['detections'])}\n")
                
                for j, detection in enumerate(result['detections']):
                    f.write(f"    {j+1}. {detection['class']} (conf: {detection['confidence']:.2f})\n")

def main():
    parser = argparse.ArgumentParser(description='Traditional Object Detection')
    parser.add_argument('--method', default='combined', choices=['hog', 'contour', 'feature', 'combined'],
                        help='Detection method')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', default='traditional_detection_results', help='Path to output directory')
    parser.add_argument('--templates', default='templates', help='Path to template directory for feature matching')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum contour area')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    # Create detector
    detector = TraditionalObjectDetector(
        method=args.method,
        min_area=args.min_area,
        visualization=not args.no_viz
    )
    
    # Load templates for feature matching
    if args.method in ['feature', 'combined']:
        detector.load_templates(args.templates)
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        detector.process_dataset(args.input, args.output, args.max_images)
    else:
        # Process single image
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        try:
            result_image, detections, processing_time = detector.detect(args.input)
            
            # Save result image
            output_path = os.path.join(args.output, f"{args.method}_{os.path.basename(args.input)}")
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # Print detection results
            print(f"Method: {args.method}")
            print(f"Processing time: {processing_time:.3f} seconds")
            print(f"Detections: {len(detections)}")
            
            for i, detection in enumerate(detections):
                print(f"  {i+1}. {detection['class']} (conf: {detection['confidence']:.2f})")
                
            # Visualize if requested
            if not args.no_viz:
                plt.figure(figsize=(12, 8))
                plt.imshow(result_image)
                plt.title(f"Method: {args.method} | Detections: {len(detections)} | Time: {processing_time:.3f}s")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error processing {args.input}: {e}")

if __name__ == "__main__":
    main()
