import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def extract_metadata():
    metadata = []
    
    # Process DAWN dataset
    weather_conditions = ['Rain', 'Snow', 'Sand', 'Fog']
    for weather in weather_conditions:
        pascal_voc_dir = f"DAWN/{weather}/{weather}_PASCAL_VOC"
        if not os.path.exists(pascal_voc_dir):
            continue
            
        xml_files = [f for f in os.listdir(pascal_voc_dir) if f.endswith('.xml')]
        for xml_file in xml_files:
            xml_path = os.path.join(pascal_voc_dir, xml_file)
            img_path = os.path.join(pascal_voc_dir, xml_file.replace('.xml', '.jpg'))
            
            # Extract object classes from XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                objects = root.findall('object')
                classes = [obj.find('name').text for obj in objects if obj.find('name') is not None]
                
                metadata.append({
                    'image_path': img_path,
                    'xml_path': xml_path,
                    'weather': weather,
                    'dataset': 'DAWN',
                    'classes': classes,
                    'filename': xml_file.replace('.xml', '.jpg')
                })
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
    
    # Process WEDGE dataset
    wedge_dir = "images"
    xml_files = [f for f in os.listdir(wedge_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(wedge_dir, xml_file)
        img_path = os.path.join(wedge_dir, xml_file.replace('.xml', '.jpg'))
        
        # Extract weather from filename (assuming format like "rain_xxx.jpg")
        filename = xml_file.replace('.xml', '.jpg')
        weather = "Unknown"
        for w in ['rain', 'snow', 'fog', 'sand', 'dust']:
            if w in filename.lower():
                weather = w.capitalize()
                break
        
        # Extract object classes from XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            classes = [obj.find('name').text for obj in objects if obj.find('name') is not None]
            
            metadata.append({
                'image_path': img_path,
                'xml_path': xml_path,
                'weather': weather,
                'dataset': 'WEDGE',
                'classes': classes,
                'filename': filename
            })
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    return pd.DataFrame(metadata)

def analyze_distribution(metadata_df):
    # Count occurrences of each class
    all_classes = []
    for classes in metadata_df['classes']:
        all_classes.extend(classes)
    
    class_counts = pd.Series(all_classes).value_counts()
    total_images = len(metadata_df)
    
    # Identify rare classes (less than 1% presence)
    rare_classes = class_counts[class_counts < total_images * 0.01].index.tolist()
    
    # Weather distribution
    weather_counts = metadata_df['weather'].value_counts()
    
    print(f"Total images: {total_images}")
    print(f"Weather distribution:\n{weather_counts}")
    print(f"Rare classes (< 1% presence):\n{rare_classes}")
    
    return rare_classes, weather_counts

def create_stratified_splits(metadata_df, rare_classes, test_size=0.15, val_size=0.15):
    """
    Creates train/validation/test splits while handling rare class combinations.
    """
    print("Creating stratification features...")
    
    # Create binary indicators for rare classes
    for rare_class in rare_classes:
        metadata_df[f'has_{rare_class}'] = metadata_df['classes'].apply(
            lambda classes: 1 if rare_class in classes else 0
        )
    
    # Use only weather for stratification (more reliable)
    print("Splitting data with weather stratification...")
    
    # First split: training vs. (validation + test)
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=test_size + val_size,
        random_state=42,
        stratify=metadata_df['weather']
    )
    
    # Second split: validation vs. test
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        random_state=42,
        stratify=temp_df['weather']
    )
    
    # Check if rare classes are represented in all splits
    for rare_class in rare_classes:
        train_has = train_df['classes'].apply(lambda x: rare_class in x).sum()
        val_has = val_df['classes'].apply(lambda x: rare_class in x).sum()
        test_has = test_df['classes'].apply(lambda x: rare_class in x).sum()
        
        print(f"Class '{rare_class}' distribution: Train={train_has}, Val={val_has}, Test={test_has}")
        
        # If any split is missing the rare class, manually move some examples
        if val_has == 0 and train_has > 0:
            # Move one example from train to val
            idx_to_move = train_df[train_df['classes'].apply(lambda x: rare_class in x)].index[0]
            val_df = pd.concat([val_df, train_df.loc[[idx_to_move]]])
            train_df = train_df.drop(idx_to_move)
            print(f"  Moved one '{rare_class}' example from train to validation")
            
        if test_has == 0 and train_has > 0:
            # Move one example from train to test
            idx_to_move = train_df[train_df['classes'].apply(lambda x: rare_class in x)].index[0]
            test_df = pd.concat([test_df, train_df.loc[[idx_to_move]]])
            train_df = train_df.drop(idx_to_move)
            print(f"  Moved one '{rare_class}' example from train to test")
    
    return train_df, val_df, test_df



def verify_splits(train_df, val_df, test_df, rare_classes):
    splits = {
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }
    
    # Check weather distribution
    print("Weather Distribution:")
    for name, df in splits.items():
        print(f"\n{name} Set:")
        print(df['weather'].value_counts(normalize=True) * 100)
    
    # Check rare class distribution
    print("\nRare Class Distribution:")
    for rare_class in rare_classes:
        print(f"\nClass: {rare_class}")
        for name, df in splits.items():
            count = sum(df['classes'].apply(lambda x: rare_class in x))
            percentage = count / len(df) * 100
            print(f"{name}: {count} images ({percentage:.2f}%)")

def create_split_directories(train_df, val_df, test_df):
    # Create directories
    split_dirs = {
        'train': 'merged_dataset/train',
        'val': 'merged_dataset/val',
        'test': 'merged_dataset/test'
    }
    
    for directory in split_dirs.values():
        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'annotations'), exist_ok=True)
    
    # Copy files to appropriate directories
    import shutil
    
    def copy_files(df, split_name):
        for _, row in df.iterrows():
            # Copy image
            img_dest = os.path.join(split_dirs[split_name], 'images', 
                                   f"{row['dataset']}_{row['weather']}_{row['filename']}")
            shutil.copy2(row['image_path'], img_dest)
            
            # Copy annotation
            xml_dest = os.path.join(split_dirs[split_name], 'annotations', 
                                   f"{row['dataset']}_{row['weather']}_{row['filename'].replace('.jpg', '.xml')}")
            shutil.copy2(row['xml_path'], xml_dest)
    
    copy_files(train_df, 'train')
    copy_files(val_df, 'val')
    copy_files(test_df, 'test')
    
    # Print summary
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")

def main():
    # Extract metadata from both datasets
    print("Extracting metadata...")
    metadata_df = extract_metadata()
    
    # Analyze distribution
    print("\nAnalyzing class and weather distribution...")
    rare_classes, weather_counts = analyze_distribution(metadata_df)
    
    # Create stratified splits
    print("\nCreating stratified splits...")
    train_df, val_df, test_df = create_stratified_splits(metadata_df, rare_classes)
    
    # Verify the splits
    print("\nVerifying splits...")
    verify_splits(train_df, val_df, test_df, rare_classes)
    
    # Create the actual split directories
    print("\nCreating split directories and copying files...")
    create_split_directories(train_df, val_df, test_df)
    
    print("\nSplitting complete!")

if __name__ == "__main__":
    main()
