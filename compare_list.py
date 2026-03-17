from pathlib import Path
import matplotlib.pyplot as plt

def read_predictions(filepath):
    """
    Read a predictions file in format: image_path mean_value
    Returns a dictionary {image_name: mean_value}
    """
    predictions = {}
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        parts = line.rsplit(' ', 1) 
        if len(parts) == 2:
            image_path, mean_str = parts
            image_name = Path(image_path).name
            mean_value = float(mean_str)
            predictions[image_name] = mean_value

    return predictions

def compare_predictions(file1, file2, actual_values):
    """
    Compare two prediction files with actual values
    Returns:
        corrected: list of images where the prediction is better in file2 compared to file1
        corrupted: list of images where the prediction is degraded in file2 compared to file1
        unchanged: list of images where the prediction is the same
    """
    pred1 = read_predictions(file1)
    pred2 = read_predictions(file2)
    data = read_predictions(actual_values)
    
    common_images = set(pred1.keys()) & set(pred2.keys()) & set(data.keys())
    
    corrected = []
    corrupted = []
    unchanged = []
    
    for image_name in common_images:
        value = data[image_name]
        diff1 = abs(value - pred1[image_name])
        diff2 = abs(value - pred2[image_name])
        
        
        if diff2 < diff1:
            corrected.append(image_name)
        elif diff2 > diff1:
            corrupted.append(image_name)
        else:
            unchanged.append(image_name)
    
    return corrected, corrupted, unchanged

def plot_corrected_corrupted(file1, file2, actual_values):
    # Compare predictions
    corrected, corrupted, unchanged = compare_predictions(file1, file2, actual_values)
    
    corrected_count = len(corrected)
    corrupted_count = len(corrupted)

    categories = ['corrected', 'corrupted']
    values = [corrected_count, corrupted_count]
    colors = ['#2ecc71', '#e74c3c']

    plt.figure(figsize=(6, 5))
    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Number of images')
    plt.title('Corrected vs Corrupted')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
    #plt.savefig("corrected_vs_corrupted2.png")

    return corrected, corrupted, unchanged

#plot_corrected_corrupted("test_results/image_mean_non_tta.txt", "test_results/image_mean_tta_weighted.txt", "test_results/apparents_ages.txt")