import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Define your classes based on your dataset
# Update this list with all classes in your dataset
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']

def convert_bbox_to_yolo(size, box):
    """
    Convert bounding box from VOC format to YOLO format
    
    Args:
        size: tuple of (width, height) - image dimensions
        box: tuple of (xmin, ymin, xmax, ymax) - VOC format bounding box
        
    Returns:
        tuple of (x_center, y_center, width, height) - normalized YOLO format
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Extract coordinates
    xmin, ymin, xmax, ymax = box
    
    # Convert to YOLO format (center x, center y, width, height)
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Normalize
    x_center = x_center * dw
    y_center = y_center * dh
    w = w * dw
    h = h * dh
    
    return (x_center, y_center, w, h)

def convert_annotation(xml_path, output_path, classes):
    """
    Convert a single XML annotation file to YOLO format
    
    Args:
        xml_path: path to XML file
        output_path: path to save the output YOLO format text file
        classes: list of class names
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size_elem = root.find('size')
        if size_elem is None:
            print(f"Warning: No size element in {xml_path}")
            return False
            
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        
        with open(output_path, 'w') as out_file:
            # Process each object in the XML
            for obj in root.findall('object'):
                # Get class name
                class_name = obj.find('name').text
                
                # Skip if class not in our list
                if class_name not in classes:
                    print(f"Warning: Unknown class {class_name} in {xml_path}")
                    continue
                
                # Get class index
                class_id = classes.index(class_name)
                
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                # Extract coordinates
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                yolo_bbox = convert_bbox_to_yolo((width, height), (xmin, ymin, xmax, ymax))
                
                # Write to file: class_id x_center y_center width height
                out_file.write(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")
                
        return True
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def process_dataset(base_dir, classes):
    """
    Process the entire dataset, converting all XML annotations to YOLO format
    
    Args:
        base_dir: base directory of the dataset
        classes: list of class names
    """
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        # Check if split directory exists
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Skipping.")
            continue
            
        # Create labels directory if it doesn't exist
        labels_dir = os.path.join(split_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get all XML files in annotations directory
        annotations_dir = os.path.join(split_dir, 'annotations')
        xml_files = glob.glob(os.path.join(annotations_dir, '*.xml'))
        
        if not xml_files:
            print(f"Warning: No XML files found in {annotations_dir}")
            continue
            
        print(f"Processing {len(xml_files)} XML files in {split} set...")
        
        # Process each XML file
        success_count = 0
        for xml_file in tqdm(xml_files):
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            
            # Create output path for YOLO format text file
            output_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            # Convert annotation
            if convert_annotation(xml_file, output_path, classes):
                success_count += 1
                
        print(f"Successfully converted {success_count}/{len(xml_files)} annotations for {split} set")

def create_data_yaml(base_dir, classes, output_path='weather_data.yaml'):
    """
    Create YAML configuration file for YOLOv8
    
    Args:
        base_dir: base directory of the dataset
        classes: list of class names
        output_path: path to save the YAML file
    """
    with open(output_path, 'w') as f:
        f.write(f"# YOLOv8 dataset configuration\n")
        f.write(f"path: {os.path.abspath(base_dir)}  # dataset root directory\n")
        f.write(f"train: train/images  # train images relative to 'path'\n")
        f.write(f"val: val/images  # val images relative to 'path'\n")
        f.write(f"test: test/images  # test images relative to 'path'\n\n")
        
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for i, cls in enumerate(classes):
            f.write(f"  {i}: {cls}\n")
            
    print(f"Created YAML configuration file at {output_path}")

def main():
    # Set the base directory to your merged dataset
    base_dir = "/home/harsh/EAVV/merged_dataset"
    
    # Process the dataset
    process_dataset(base_dir, CLASSES)
    
    # Create YAML configuration file
    create_data_yaml(base_dir, CLASSES)
    
    print("Conversion complete! Your dataset is now ready for YOLOv8 training.")
    print("You can now train YOLOv8 with the command:")
    print("python train_yolov8.py --mode train --model n --epochs 50 --batch 2 --img-size 640 --data weather_data.yaml --device cpu")

if __name__ == "__main__":
    main()
