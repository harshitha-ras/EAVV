import cv2
import numpy as np
import argparse

'''
Run this with these commands
# Edge-based detection (default)
python naive_object_detection.py --image your_image.jpg --output result.jpg

# Color-based detection (for detecting red objects)
python naive_object_detection.py --image your_image.jpg --method color --output result.jpg

# Background subtraction (requires a background image)
python naive_object_detection.py --image your_image.jpg --method background --background background.jpg --output result.jpg

'''

def detect_objects_by_color(image_path, color_lower, color_upper, min_area=1000):
    """
    A naive approach to object detection using color thresholding.
    
    Args:
        image_path: Path to the input image
        color_lower: Lower bound of the color range in HSV format
        color_upper: Upper bound of the color range in HSV format
        min_area: Minimum contour area to be considered an object
        
    Returns:
        Original image with bounding boxes drawn around detected objects
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_image, color_lower, color_upper)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    result_image = image.copy()
    object_count = 0
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a label
        object_count += 1
        label = f"Object {object_count}"
        cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image, object_count

def detect_objects_by_edges(image_path, min_area=1000, threshold1=50, threshold2=150):
    """
    A naive approach to object detection using edge detection and contour finding.
    
    Args:
        image_path: Path to the input image
        min_area: Minimum contour area to be considered an object
        threshold1: First threshold for Canny edge detector
        threshold2: Second threshold for Canny edge detector
        
    Returns:
        Original image with bounding boxes drawn around detected objects
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Dilate the edges to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    result_image = image.copy()
    object_count = 0
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a label
        object_count += 1
        label = f"Object {object_count}"
        cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image, object_count

def detect_objects_by_background_subtraction(image_path, background_path, threshold=30, min_area=1000):
    """
    A naive approach to object detection using background subtraction.
    
    Args:
        image_path: Path to the input image
        background_path: Path to the background image
        threshold: Threshold for binary image creation
        min_area: Minimum contour area to be considered an object
        
    Returns:
        Original image with bounding boxes drawn around detected objects
    """
    # Read the images
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)
    
    if image is None or background is None:
        raise ValueError("Could not read image or background")
    
    # Ensure both images have the same size
    if image.shape != background.shape:
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference between image and background
    diff = cv2.absdiff(gray_image, gray_background)
    
    # Apply threshold to get binary image
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.erode(thresholded, kernel, iterations=1)
    thresholded = cv2.dilate(thresholded, kernel, iterations=2)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    result_image = image.copy()
    object_count = 0
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding box
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a label
        object_count += 1
        label = f"Object {object_count}"
        cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image, object_count

def main():
    parser = argparse.ArgumentParser(description='Naive Object Detection')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--method', default='edges', choices=['color', 'edges', 'background'],
                        help='Detection method: color, edges, or background')
    parser.add_argument('--background', help='Path to background image (required for background method)')
    parser.add_argument('--output', default='output.jpg', help='Path to save the output image')
    parser.add_argument('--min-area', type=int, default=1000, help='Minimum contour area')
    args = parser.parse_args()
    
    try:
        if args.method == 'color':
            # Define color range for red objects in HSV
            # Note: You may need to adjust these values for your specific use case
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            result, count = detect_objects_by_color(args.image, lower_red, upper_red, args.min_area)
        
        elif args.method == 'edges':
            result, count = detect_objects_by_edges(args.image, args.min_area)
        
        elif args.method == 'background':
            if args.background is None:
                raise ValueError("Background image path is required for background subtraction method")
            result, count = detect_objects_by_background_subtraction(args.image, args.background, min_area=args.min_area)
        
        # Save the result
        cv2.imwrite(args.output, result)
        print(f"Detected {count} objects. Result saved to {args.output}")
        
        # Display the result (if running in an environment with GUI)
        cv2.imshow('Naive Object Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
