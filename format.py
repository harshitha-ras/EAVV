import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

def create_output_directory(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def log_error(file_name, error_message):
    with open('error_log.txt', 'a') as log_file:
        log_file.write(f"Error processing file {file_name}: {error_message}\n")

def standardize_xml(input_path, output_path):
    """
    Standardizes XML files from DAWN and WEDGE datasets into a uniform format.
    
    Args:
        input_path (str): Path to the directory containing the original XML files.
        output_path (str): Path to save the standardized XML files.
    """
    create_output_directory(output_path)
    
    xml_files = [f for f in os.listdir(input_path) if f.endswith('.xml')]
    
    for file_name in tqdm(xml_files, desc="Processing XML files"):
        file_path = os.path.join(input_path, file_name)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Create a new standardized root element
            standardized_root = ET.Element("annotation")
            
            # Add filename
            filename = root.find('filename').text if root.find('filename') is not None else ''
            ET.SubElement(standardized_root, "filename").text = filename
            
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
            output_file_path = os.path.join(output_path, file_name)
            standardized_tree.write(output_file_path)
        
        except Exception as e:
            log_error(file_name, str(e))

if __name__ == "__main__":
    # Assuming the script is in the DAWN folder
    current_dir = os.getcwd()
    input_directory = os.path.join(current_dir, "combine") #source directory
    output_directory = os.path.join(input_directory, "merged") #destination directory
    standardize_xml(input_directory, output_directory)
    print("XML standardization complete. Check 'error_log.txt' for any processing errors.")
