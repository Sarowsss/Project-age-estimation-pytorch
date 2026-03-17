from pathlib import Path
import numpy as np
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


def bucket(delta):
    if delta < 0.5:
        return "negligible"   # < 0.5 year
    elif delta < 3:
        return "moderate"     # 0.5–3 years
    else:
        return "severe"       # > 3 years


def compare_predictions(file1, file2, actual_values):
    """
    Compare two prediction files with actual values.
    Returns:
        corrected: list of (image_name, delta) where file2 is better
        corrupted: list of (image_name, delta) where file2 is worse
        unchanged: list of image_name where predictions are equal
    Also prints a magnitude-aware summary.
    """
    pred1 = read_predictions(file1)
    pred2 = read_predictions(file2)
    data  = read_predictions(actual_values)

    common_images = set(pred1.keys()) & set(pred2.keys()) & set(data.keys())

    corrected = []
    corrupted = []
    unchanged = []

    for image_name in common_images:
        value = data[image_name]
        diff1 = abs(value - pred1[image_name])
        diff2 = abs(value - pred2[image_name])
        delta = diff2 - diff1  # negative = improvement, positive = degradation

        if delta < 0:
            corrected.append((image_name, abs(delta)))
        elif delta > 0:
            corrupted.append((image_name, abs(delta)))
        else:
            unchanged.append(image_name)

    # summary stats
    correction_gains  = [d for _, d in corrected]
    corruption_losses = [d for _, d in corrupted]

    print(f"Corrected : {len(corrected):>5} images | avg gain : {np.mean(correction_gains):.3f} yrs | max : {np.max(correction_gains):.3f} yrs" if corrected else "Corrected :     0 images")
    print(f"Corrupted : {len(corrupted):>5} images | avg loss : {np.mean(corruption_losses):.3f} yrs | max : {np.max(corruption_losses):.3f} yrs" if corrupted else "Corrupted :     0 images")
    print(f"Unchanged : {len(unchanged):>5} images")

    total_gain = sum(correction_gains)
    total_loss = sum(corruption_losses)
    print(f"Net improvement score : {total_gain - total_loss:.3f}  (positive = file2 better overall)")

    # severity buckets
    buckets = ["negligible", "moderate", "severe"]
    from collections import defaultdict
    corr_buckets = defaultdict(int)
    corp_buckets = defaultdict(int)
    for _, d in corrected:
        corr_buckets[bucket(d)] += 1
    for _, d in corrupted:
        corp_buckets[bucket(d)] += 1

    print("\nCorrections by severity  (<0.5yr / 0.5-3yr / >3yr):",
          f"negligible={corr_buckets['negligible']}",
          f"moderate={corr_buckets['moderate']}",
          f"severe={corr_buckets['severe']}")
    print("Corruptions by severity  (<0.5yr / 0.5-3yr / >3yr):",
          f"negligible={corp_buckets['negligible']}",
          f"moderate={corp_buckets['moderate']}",
          f"severe={corp_buckets['severe']}")

    return corrected, corrupted, unchanged


def plot_corrected_corrupted(file1, file2, actual_values):
    corrected, corrupted, unchanged = compare_predictions(file1, file2, actual_values)

    correction_gains  = [d for _, d in corrected]
    corruption_losses = [d for _, d in corrupted]

    buckets = ["negligible", "moderate", "severe"]
    bucket_labels = ["<0.5 yr", "0.5–3 yrs", ">3 yrs"]
    from collections import defaultdict
    corr_buckets = defaultdict(int)
    corp_buckets = defaultdict(int)
    for _, d in corrected:
        corr_buckets[bucket(d)] += 1
    for _, d in corrupted:
        corp_buckets[bucket(d)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left plot: count bar chart with avg error annotation ---
    ax1 = axes[0]
    categories = ['Corrected', 'Corrupted']
    counts     = [len(corrected), len(corrupted)]
    avgs       = [
        np.mean(correction_gains)  if correction_gains  else 0,
        np.mean(corruption_losses) if corruption_losses else 0
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
    for bar, avg in zip(bars, avgs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{int(height)}\navg Δ {avg:.2f} yrs',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Number of images')
    ax1.set_title('Corrected vs Corrupted')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- right plot: stacked severity bar chart ---
    ax2 = axes[1]
    severity_colors = {'negligible': '#a8d5a2', 'moderate': '#f0c040', 'severe': '#e05c5c'}
    corr_vals = [corr_buckets[b] for b in buckets]
    corp_vals = [corp_buckets[b] for b in buckets]
    x = np.arange(2)
    bottom_corr = 0
    bottom_corp = 0
    for i, b in enumerate(buckets):
        ax2.bar(0, corr_vals[i], bottom=bottom_corr, color=severity_colors[b],
                edgecolor='black', linewidth=0.8, label=bucket_labels[i] if i == 0 else "")
        ax2.bar(1, corp_vals[i], bottom=bottom_corp, color=severity_colors[b],
                edgecolor='black', linewidth=0.8)
        bottom_corr += corr_vals[i]
        bottom_corp += corp_vals[i]
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Corrected', 'Corrupted'])
    ax2.set_ylabel('Number of images')
    ax2.set_title('Severity breakdown')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # manual legend for severity
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=severity_colors[b], edgecolor='black', label=f'{bucket_labels[i]} error change')
                       for i, b in enumerate(buckets)]
    ax2.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    plt.show()
    # plt.savefig("corrected_vs_corrupted.png")

    return


#plot_corrected_corrupted("test_results/image_mean_non_tta.txt", "test_results/image_mean_tta_weighted.txt", "test_results/apparents_ages.txt") 