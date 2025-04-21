import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import pandas as pd
import seaborn as sns

def analyze_explanations(explanations_dir):
    """
    Analyze BODEM explanations to compare model behavior across datasets and weather conditions.
    
    Args:
        explanations_dir: Directory containing BODEM explanations
    """
    # Create dictionaries to store analysis results
    weather_performance = defaultdict(lambda: defaultdict(list))
    dataset_performance = defaultdict(list)
    class_performance = defaultdict(lambda: defaultdict(list))
    
    # Analyze explanation images
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            if file.endswith("_explanation.png"):
                # Extract metadata from filename
                parts = file.split('_')
                dataset = parts[0]
                weather = parts[1]
                
                # Load explanation image
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                
                # Extract saliency information (right half of the image is the saliency map)
                h, w = img.shape[:2]
                saliency_img = img[:, w//2:, :]
                
                # Calculate saliency statistics
                saliency_gray = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2GRAY)
                saliency_mean = np.mean(saliency_gray) / 255.0
                saliency_std = np.std(saliency_gray) / 255.0
                saliency_max = np.max(saliency_gray) / 255.0
                
                # Store results by weather condition
                weather_performance[weather][dataset].append({
                    'mean_saliency': saliency_mean,
                    'std_saliency': saliency_std,
                    'max_saliency': saliency_max,
                    'file': file
                })
                
                # Store results by dataset
                dataset_performance[dataset].append({
                    'weather': weather,
                    'mean_saliency': saliency_mean,
                    'std_saliency': saliency_std,
                    'max_saliency': saliency_max,
                    'file': file
                })
    
    # Create visualizations for weather comparison
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    weather_data = []
    for weather in weather_performance:
        for dataset in weather_performance[weather]:
            for entry in weather_performance[weather][dataset]:
                weather_data.append({
                    'Weather': weather,
                    'Dataset': dataset,
                    'Mean Saliency': entry['mean_saliency']
                })
    
    weather_df = pd.DataFrame(weather_data)
    
    # Create grouped bar plot
    sns.barplot(x='Weather', y='Mean Saliency', hue='Dataset', data=weather_df)
    plt.title('Mean Saliency by Weather Condition and Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(explanations_dir, 'weather_comparison.png'))
    
    # Create dataset comparison visualization
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    dataset_data = []
    for dataset in dataset_performance:
        mean_saliency = np.mean([entry['mean_saliency'] for entry in dataset_performance[dataset]])
        std_saliency = np.std([entry['mean_saliency'] for entry in dataset_performance[dataset]])
        dataset_data.append({
            'Dataset': dataset,
            'Mean Saliency': mean_saliency,
            'Std Saliency': std_saliency
        })
    
    dataset_df = pd.DataFrame(dataset_data)
    
    # Create bar plot with error bars
    sns.barplot(x='Dataset', y='Mean Saliency', data=dataset_df)
    plt.errorbar(
        x=range(len(dataset_df)),
        y=dataset_df['Mean Saliency'],
        yerr=dataset_df['Std Saliency'],
        fmt='none',
        color='black',
        capsize=5
    )
    plt.title('Mean Saliency by Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(explanations_dir, 'dataset_comparison.png'))
    
    # Generate summary report
    with open(os.path.join(explanations_dir, 'analysis_report.txt'), 'w') as f:
        f.write("# BODEM Analysis Report\n\n")
        
        f.write("## Dataset Comparison\n\n")
        for dataset in sorted(dataset_performance.keys()):
            mean_saliency = np.mean([entry['mean_saliency'] for entry in dataset_performance[dataset]])
            std_saliency = np.std([entry['mean_saliency'] for entry in dataset_performance[dataset]])
            f.write(f"{dataset}: Mean Saliency = {mean_saliency:.4f} ± {std_saliency:.4f}\n")
        
        f.write("\n## Weather Condition Comparison\n\n")
        for weather in sorted(weather_performance.keys()):
            f.write(f"\n### {weather}\n\n")
            for dataset in sorted(weather_performance[weather].keys()):
                mean_saliency = np.mean([entry['mean_saliency'] for entry in weather_performance[weather][dataset]])
                std_saliency = np.std([entry['mean_saliency'] for entry in weather_performance[weather][dataset]])
                f.write(f"{dataset}: Mean Saliency = {mean_saliency:.4f} ± {std_saliency:.4f}\n")
    
    print(f"Analysis complete. Results saved to {explanations_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze BODEM explanations")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing BODEM explanations")
    args = parser.parse_args()
    
    analyze_explanations(args.dir)
