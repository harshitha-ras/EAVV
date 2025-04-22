import xml.etree.ElementTree as ET
import os
import shutil
from tqdm import tqdm

def create_output_directory(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def log_error(file_name, error_message):
    with open('error_log.txt', 'a') as log_file:
        log_file.write(f"Error processing file {file_name}: {error_message}\n")

def process_dataset(dataset_path, output_path, dataset_name):
    """
    Process a dataset (DAWN or WEDGE) by standardizing XMLs and copying images.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_path (str): Path to save standardized files
        dataset_name (str): Name of the dataset ('DAWN' or 'WEDGE')
    """
    xml_files_processed = 0
    img_files_processed = 0
    
    # Create output directories for XMLs and images
    xml_output_path = os.path.join(output_path, "annotations")
    img_output_path = os.path.join(output_path, "images")
    create_output_directory(xml_output_path)
    create_output_directory(img_output_path)
    
    if dataset_name == "DAWN":
        # DAWN has subfolders for weather conditions
        weather_conditions = [d for d in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, d))]
        
        for weather in weather_conditions:
            weather_dir = os.path.join(dataset_path, weather)
            # Look for the PASCAL_VOC subfolder
            pascal_voc_dir = os.path.join(weather_dir, f"{weather}_PASCAL_VOC")
            
            if not os.path.exists(pascal_voc_dir):
                print(f"Warning: {pascal_voc_dir} not found")
                continue
                
            # Process XML files in the PASCAL_VOC subfolder
            xml_files = [f for f in os.listdir(pascal_voc_dir) if f.endswith('.xml')]
            
            for xml_file in tqdm(xml_files, desc=f"Processing {weather} XMLs"):
                try:
                    # Get corresponding image file
                    img_file = xml_file.replace('.xml', '.jpg')
                    xml_path = os.path.join(pascal_voc_dir, xml_file)
                    img_path = os.path.join(pascal_voc_dir, img_file)
                    
                    if not os.path.exists(img_path):
                        # Try looking in the parent weather folder
                        img_path = os.path.join(weather_dir, img_file)
                        if not os.path.exists(img_path):
                            log_error(xml_file, f"Corresponding image file not found: {img_file}")
                            continue
                    
                    # Create standardized XML with weather condition
                    standardize_xml(xml_path, os.path.join(xml_output_path, xml_file), 
                                   dataset_name, weather)
                    
                    # Copy and rename image with dataset and weather prefix
                    new_img_name = f"{dataset_name}_{weather}_{img_file}"
                    shutil.copy2(img_path, os.path.join(img_output_path, new_img_name))
                    
                    xml_files_processed += 1
                    img_files_processed += 1
                    
                except Exception as e:
                    log_error(xml_file, str(e))
    
    elif dataset_name == "WEDGE":
        # WEDGE has all files in one directory
        xml_files = [f for f in os.listdir(dataset_path) if f.endswith('.xml')]
        
        for xml_file in tqdm(xml_files, desc="Processing WEDGE XMLs"):
            try:
                # Get corresponding image file
                img_file = xml_file.replace('.xml', '.jpg')
                xml_path = os.path.join(dataset_path, xml_file)
                img_path = os.path.join(dataset_path, img_file)
                
                if not os.path.exists(img_path):
                    log_error(xml_file, f"Corresponding image file not found: {img_file}")
                    continue
                
                # Create standardized XML (no weather condition for WEDGE)
                standardize_xml(xml_path, os.path.join(xml_output_path, xml_file), 
                               dataset_name, "Unknown")
                
                # Copy and rename image with dataset prefix
                new_img_name = f"{dataset_name}_{img_file}"
                shutil.copy2(img_path, os.path.join(img_output_path, new_img_name))
                
                xml_files_processed += 1
                img_files_processed += 1
                
            except Exception as e:
                log_error(xml_file, str(e))
    
    return xml_files_processed, img_files_processed

def standardize_xml(input_xml_path, output_xml_path, dataset_name, weather_condition):
    """
    Standardizes an XML file and adds dataset and weather metadata.
    
    Args:
        input_xml_path (str): Path to the original XML file
        output_xml_path (str): Path to save the standardized XML
        dataset_name (str): Name of the dataset ('DAWN' or 'WEDGE')
        weather_condition (str): Weather condition of the image
    """
    try:
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        
        # Create a new standardized root element
        standardized_root = ET.Element("annotation")
        
        # Add dataset and weather metadata
        ET.SubElement(standardized_root, "dataset").text = dataset_name
        ET.SubElement(standardized_root, "weather").text = weather_condition
        
        # Add filename
        filename = root.find('filename').text if root.find('filename') is not None else ''
        # Update filename to include dataset and weather
        if dataset_name == "DAWN":
            new_filename = f"{dataset_name}_{weather_condition}_{filename}"
        else:
            new_filename = f"{dataset_name}_{filename}"
        ET.SubElement(standardized_root, "filename").text = new_filename
        
        # Add path
        path = root.find('path').text if root.find('path') is not None else ''
        ET.SubElement(standardized_root, "path").text = path
        
        # Add size information
        size = root.find('size')
        if size is not None:
            size_element = ET.SubElement(standardized_root, "size")
            width = size.find('width').text if size.find('width') is not None else '0'
            height = size.find('height').text if size.find('height') is not None else '0'
            depth = size.find('depth').text if size.find('depth') is not None else '3'
            ET.SubElement(size_element, "width").text = width
            ET.SubElement(size_element, "height").text = height
            ET.SubElement(size_element, "depth").text = depth
        
        # Add objects
        for obj in root.findall('object'):
            obj_element = ET.SubElement(standardized_root, "object")
            
            name = obj.find('name').text if obj.find('name') is not None else 'unknown'
            ET.SubElement(obj_element, "name").text = name
            
            pose = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
            ET.SubElement(obj_element, "pose").text = pose
            
            truncated = obj.find('truncated').text if obj.find('truncated') is not None else '0'
            ET.SubElement(obj_element, "truncated").text = truncated
            
            difficult = obj.find('difficult').text if obj.find('difficult') is not None else '0'
            ET.SubElement(obj_element, "difficult").text = difficult
            
            # Handle bounding box
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                bndbox_element = ET.SubElement(obj_element, "bndbox")
                xmin = bndbox.find('xmin').text if bndbox.find('xmin') is not None else '0'
                ymin = bndbox.find('ymin').text if bndbox.find('ymin') is not None else '0'
                xmax = bndbox.find('xmax').text if bndbox.find('xmax') is not None else '0'
                ymax = bndbox.find('ymax').text if bndbox.find('ymax') is not None else '0'
                ET.SubElement(bndbox_element, "xmin").text = xmin
                ET.SubElement(bndbox_element, "ymin").text = ymin
                ET.SubElement(bndbox_element, "xmax").text = xmax
                ET.SubElement(bndbox_element, "ymax").text = ymax
        
        # Save the standardized XML
        standardized_tree = ET.ElementTree(standardized_root)
        standardized_tree.write(output_xml_path)
        
    except Exception as e:
        log_error(os.path.basename(input_xml_path), str(e))

if __name__ == "__main__":
    current_dir = os.getcwd()
    
    # Define paths for DAWN and WEDGE datasets
    dawn_path = os.path.join(current_dir, "DAWN")
    wedge_path = os.path.join(current_dir, "images")  # Assuming this is where WEDGE is
    
    # Define output directory
    output_path = os.path.join(current_dir, "merged_dataset")
    create_output_directory(output_path)
    
    # Process both datasets
    dawn_xml_count, dawn_img_count = process_dataset(dawn_path, output_path, "DAWN")
    wedge_xml_count, wedge_img_count = process_dataset(wedge_path, output_path, "WEDGE")
    
    print(f"Dataset merging complete:")
    print(f"- DAWN: {dawn_xml_count} XMLs and {dawn_img_count} images processed")
    print(f"- WEDGE: {wedge_xml_count} XMLs and {wedge_img_count} images processed")
    print(f"- Total: {dawn_xml_count + wedge_xml_count} XMLs and {dawn_img_count + wedge_img_count} images")
    print(f"Check 'error_log.txt' for any processing errors.")
