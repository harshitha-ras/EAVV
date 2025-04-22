import cv2
import numpy as np
import os
from pathlib import Path
import argparse

class TraditionalObjectDetector:
    def __init__(self):
        # Get the directory where the script is located
        script_dir = Path(__file__).resolve().parent
        
        # Define the path to the Haar cascade XML file in the same directory as the script
        cascade_path = script_dir / 'haarcascade_car.xml'
        
        # Initialize the car cascade classifier
        self.car_cascade = cv2.CascadeClassifier(str(cascade_path))
        
        # Check if the cascade loaded successfully
        if self.car_cascade.empty():
            print(f"Error: Could not load car cascade from {cascade_path}")
            print("Please ensure the haarcascade_car.xml file is in the scripts directory")
        else:
            print(f"Successfully loaded car cascade from {cascade_path}")
        
        # Initialize HOG descriptor for people detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize SIFT detector for feature matching
        self.sift = cv2.SIFT_create()

    def process_image(self, image_path, output_path, template_path=None):
        """Process a single image and save the result"""
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return False
        
        try:
            # Detect objects using combined method
            filtered_boxes, all_boxes = self.detect_objects_combined(image, template_path)
            
            # Create a copy of the image to draw on
            output_image = image.copy()
            
            # Draw bounding boxes
            for (x, y, w, h, label, conf) in filtered_boxes:
                # Choose color based on label
                if label == 'person':
                    color = (0, 255, 0)  # Green for people
                elif label == 'car':
                    color = (0, 0, 255)  # Red for cars
                elif label == 'template_match':
                    color = (255, 0, 0)  # Blue for template matches
                else:
                    color = (255, 255, 0)  # Cyan for other objects
                
                # Draw rectangle
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                label_text = f"{label}: {conf:.2f}"
                cv2.putText(output_image, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the output image
            success = cv2.imwrite(str(output_path), output_image)
            if success:
                print(f"Successfully saved detected image to {output_path}")
                return True
            else:
                print(f"Error: Failed to save image to {output_path}")
                return False
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False

    
    def detect_objects_hog(self, image):
        """Detect people using HOG and cars using Haar cascade"""
        # Detect people
        boxes_people, weights = self.hog.detectMultiScale(
            image, 
            winStride=(8, 8),
            padding=(4, 4), 
            scale=1.05
        )
        
        people_boxes = []
        for (x, y, w, h) in boxes_people:
            # Ensure numeric types
            people_boxes.append((int(float(x)), int(float(y)), int(float(w)), int(float(h)), 'person', float(max(weights) if len(weights) > 0 else 0.5)))


        
        # Detect cars using Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        car_boxes = []
        for (x, y, w, h) in cars:
            # Ensure numeric types
            car_boxes.append((int(float(x)), int(float(y)), int(float(w)), int(float(h)), 'car', 0.6))
        
        return people_boxes + car_boxes



    def non_max_suppression(self, boxes, overlap_thresh=0.3):
        """Apply non-maximum suppression to remove overlapping boxes"""
        if len(boxes) == 0:
            return []
        
        # Extract coordinates and labels separately to avoid type conversion issues
        coords = []
        labels = []
        confs = []
        
        for (x, y, w, h, label, conf) in boxes:
            coords.append([float(x), float(y), float(x) + float(w), float(y) + float(h)])
            labels.append(str(label))
            confs.append(float(conf))
        
        # Convert coordinates to numpy array
        coords = np.array(coords, dtype=np.float64)
        confs = np.array(confs, dtype=np.float64)
        
        # Extract coordinates
        x1 = coords[:, 0]
        y1 = coords[:, 1]
        x2 = coords[:, 2]
        y2 = coords[:, 3]
        
        # Calculate area of each box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by confidence (highest first)
        idxs = np.argsort(confs)[::-1]
        
        pick = []
        while len(idxs) > 0:
            # Pick the box with highest confidence
            i = idxs[0]
            pick.append(i)
            
            # Find the intersection with all remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            # Calculate width and height of intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Calculate intersection area
            overlap = (w * h) / area[idxs[1:]]
            
            # Delete overlapping boxes
            to_delete = np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1))
            idxs = np.delete(idxs, to_delete)
        
        # Return selected boxes in original format
        result = []
        for i in pick:
            x, y = coords[i][0], coords[i][1]
            w = coords[i][2] - coords[i][0]
            h = coords[i][3] - coords[i][1]
            result.append((int(x), int(y), int(w), int(h), labels[i], confs[i]))
        
        return result





    def detect_objects_contour(self, image):
        """Detect objects using contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            # Calculate area and filter small contours
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small noise
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Add to boxes with a generic 'object' label
            boxes.append((x, y, w, h, 'object', 0.5))
        
        return boxes
    
    def detect_objects_feature(self, image, template_path=None):
        """Detect objects using feature matching (SIFT)"""
        # If no template is provided, return empty list
        if template_path is None or not os.path.exists(template_path):
            return []
        
        # Load template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return []
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(template, None)
        kp2, des2 = self.sift.detectAndCompute(gray, None)
        
        # If no keypoints found, return empty list
        if des1 is None or des2 is None:
            return []
        
        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # If not enough good matches, return empty list
        if len(good_matches) < 10:
            return []
        
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # If homography is found, get the bounding box
        if H is not None:
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            
            # Get bounding box from transformed points
            x_min = min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0])
            y_min = min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1])
            x_max = max(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0])
            y_max = max(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1])
            
            x = int(float(x_min))
            y = int(float(y_min))
            w = int(float(x_max - x_min))
            h = int(float(y_max - y_min))
            
            # Add to boxes with a 'template_match' label
            return [(x, y, w, h, 'template_match', len(good_matches) / 100)]
        
        return []
    

    
    def detect_objects_combined(self, image, template_path=None):
        """Combine multiple detection methods and apply NMS"""
        # Get detections from each method
        hog_boxes = self.detect_objects_hog(image)
        contour_boxes = self.detect_objects_contour(image)
        feature_boxes = self.detect_objects_feature(image, template_path)
        
        # Combine all detections
        all_boxes = hog_boxes + contour_boxes + feature_boxes
        
        # Apply non-maximum suppression
        filtered_boxes = self.non_max_suppression(all_boxes)
        
        return filtered_boxes, all_boxes
    
    def process_image(self, image_path, output_path, template_path=None):
        """Process a single image and save the result"""
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return False
        
        try:
            # Detect objects using combined method
            filtered_boxes, all_boxes = self.detect_objects_combined(image, template_path)
            
            # Create a copy of the image to draw on
            output_image = image.copy()
            
            # Draw bounding boxes
            for (x, y, w, h, label, conf) in filtered_boxes:
                # Choose color based on label
                if label == 'person':
                    color = (0, 255, 0)  # Green for people
                elif label == 'car':
                    color = (0, 0, 255)  # Red for cars
                elif label == 'template_match':
                    color = (255, 0, 0)  # Blue for template matches
                else:
                    color = (255, 255, 0)  # Cyan for other objects
                
                # Draw rectangle
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                label_text = f"{label}: {conf:.2f}"
                cv2.putText(output_image, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the output image
            success = cv2.imwrite(str(output_path), output_image)
            if success:
                print(f"Successfully saved detected image to {output_path}")
                return True
            else:
                print(f"Error: Failed to save image to {output_path}")
                return False
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Traditional Object Detection')
    parser.add_argument('--image', help='Path to input image (optional)')
    parser.add_argument('--template', help='Path to template image for feature matching (optional)')
    args = parser.parse_args()
    
    # Get project paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define input and output directories
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "outputs" / "traditional_detection_results"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = TraditionalObjectDetector()
    
    # Process specific image if provided, otherwise use test_image.jpg
    if args.image:
        input_image_path = Path(args.image)
        if not input_image_path.exists():
            print(f"Error: Input image {input_image_path} does not exist")
            return
    else:
        input_image_path = input_dir / "test_image.jpg"
        if not input_image_path.exists():
            print(f"Error: Default test image {input_image_path} does not exist")
            return
    
    # Define output path
    output_image_path = output_dir / f"{input_image_path.stem}_detected{input_image_path.suffix}"
    
    # Process the image
    print(f"Processing image: {input_image_path}")
    detector.process_image(input_image_path, output_image_path, args.template)


if __name__ == "__main__":
    main()
