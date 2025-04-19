import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# results = "base_comparison"
results = "."
sig_wm_val_pass_base = os.path.join(results, "sig_wm_val_pass_base_inference_results.csv")
sig_wm_no_pass_base = os.path.join(results, "sig_wm_no_pass_base_inference_results.csv")
sig_wm_forged_pass_base = os.path.join(results, "sig_wm_forged_pass_base_inference_results.csv")

sig_wm_valid_pass_pruned = os.path.join(results, "sig_wm_valid_pass_pruned_inference_results.csv")
sig_wm_no_pass_pruned = os.path.join(results, "sig_wm_no_pass_pruned_inference_results.csv")
sig_wm_forged_pass_pruned = os.path.join(results, "sig_wm_forged_pass_pruned_inference_results.csv")

sig_wm_valid_pass_distilled = os.path.join(results, "sig_wm_valid_pass_distilled_inference_results.csv")
sig_wm_no_pass_distilled = os.path.join(results, "sig_wm_no_pass_distilled_inference_results.csv")
sig_wm_forged_pass_distilled = os.path.join(results, "sig_wm_forged_pass_distilled_inference_results.csv")

fig = os.path.join(results, "sig_enc_f1_scores_comparison.png")


def read_csv(file_path):
    """
    Read a CSV file and return the DataFrame with correct column names.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure consistent column naming
        df.columns = df.columns.str.strip()  # Remove any accidental whitespace
        
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")
    return None


df_base_pass = read_csv(sig_wm_val_pass_base)
df_base_no_pass = read_csv(sig_wm_no_pass_base)
df_base_forged_pass = read_csv(sig_wm_forged_pass_base)

df_pruned_pass = read_csv(sig_wm_valid_pass_pruned)
df_pruned_no_pass = read_csv(sig_wm_no_pass_pruned)
df_pruned_forged_pass = read_csv(sig_wm_forged_pass_pruned)

df_dist_pass = read_csv(sig_wm_valid_pass_distilled)
df_dist_no_pass = read_csv(sig_wm_no_pass_distilled)
df_dist_forged_pass = read_csv(sig_wm_forged_pass_distilled)

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)


def plot_watermark_accuracy():
    """
    Plot a bar chart comparing watermark detection accuracy for pruned and baseline models
    using the weight perturbation method.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    plt.figure(figsize=(10, 6))
    
    # Define labels and accuracy values
    labels = ['Pruned Model', 'Baseline Model']
    accuracy_values = [70.15, 100.00]
    
    # Plot bar chart
    plt.bar(labels, accuracy_values, color=['green', 'blue'])
    
    # Add labels and title
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Watermark Detection\nAccuracy (%)', fontweight='bold')
    plt.title('Watermark Detection Accuracy\n(Weight Perturbation Method)', fontsize=20, pad=20)
    
    # Display values on bars
    for i, v in enumerate(accuracy_values):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 110)  # Set y-axis limit
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig("watermark_accuracy.png")
    plt.show()

def plot_f1_scores(class_name):
    """
    Plot F1 scores for Baseline, Pruned, and Distilled models with respect to
    Valid, No, and Forged Passports. Optimized for IEEE format.
    """

    # Extract F1 scores for the specified class
    f1_valid_pass = [
        df_base_pass[df_base_pass['Class'] == class_name]['F1 Score'].values[0],
        df_pruned_pass[df_pruned_pass['Class'] == class_name]['F1 Score'].values[0],
        df_dist_pass[df_dist_pass['Class'] == class_name]['F1 Score'].values[0]
    ]

    f1_no_pass = [
        df_base_no_pass[df_base_no_pass['Class'] == class_name]['F1 Score'].values[0],
        df_pruned_no_pass[df_pruned_no_pass['Class'] == class_name]['F1 Score'].values[0],
        df_dist_no_pass[df_dist_no_pass['Class'] == class_name]['F1 Score'].values[0]
    ]

    f1_forged_pass = [
        df_base_forged_pass[df_base_forged_pass['Class'] == class_name]['F1 Score'].values[0],
        df_pruned_forged_pass[df_pruned_forged_pass['Class'] == class_name]['F1 Score'].values[0],
        df_dist_forged_pass[df_dist_forged_pass['Class'] == class_name]['F1 Score'].values[0]
    ]

    # Set plot configuration for IEEE format
    plt.figure(dpi=300)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Model types as x-axis categories
    models = ['Baseline', 'Pruned', 'Distilled']
    x = np.arange(len(models))  # [0, 1, 2]
    width = 0.25

    # Plot each category as a separate bar group
    plt.bar(x - width, f1_valid_pass, width, label='Valid Passport', color='#3CAE63')
    plt.bar(x,         f1_no_pass, width, label='No Passport', color='#FFA896')
    plt.bar(x + width, f1_forged_pass, width, label='Forged Passport', color='#CD1C18')

    # Plot decorations
    plt.ylabel('F1 Score', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Passport-based Watermarking Performance', fontsize=14, pad=5)
    plt.xticks(x, models)
    plt.ylim(0, max(max(f1_valid_pass), max(f1_no_pass), max(f1_forged_pass)) * 1.1)
    plt.grid(axis='y', alpha=0.3)

    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout()
    plt.savefig(f"sig_enc_f1_scores.png", format='png', bbox_inches='tight')
    plt.close()

# def plot_f1_scores(class_name):
#     """
#     Plot F1 scores for Baseline, Pruned, and Distilled models with respect to
#     Valid, No, and Forged Passports. Optimized for IEEE format with annotations and highlights.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Extract F1 scores for the specified class
#     f1_valid_pass = [
#         df_base_pass[df_base_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_pruned_pass[df_pruned_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_dist_pass[df_dist_pass['Class'] == class_name]['F1 Score'].values[0]
#     ]

#     f1_no_pass = [
#         df_base_no_pass[df_base_no_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_pruned_no_pass[df_pruned_no_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_dist_no_pass[df_dist_no_pass['Class'] == class_name]['F1 Score'].values[0]
#     ]

#     f1_forged_pass = [
#         df_base_forged_pass[df_base_forged_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_pruned_forged_pass[df_pruned_forged_pass['Class'] == class_name]['F1 Score'].values[0],
#         df_dist_forged_pass[df_dist_forged_pass['Class'] == class_name]['F1 Score'].values[0]
#     ]

#     plt.figure(dpi=300)
#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.labelsize': 12,
#         'axes.titlesize': 12,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 12
#     })

#     models = ['Baseline', 'Pruned', 'Distilled']
#     x = np.arange(len(models))  # [0, 1, 2]
#     width = 0.25

#     # Bars
#     bars_valid = plt.bar(x - width, f1_valid_pass, width, label='Valid Passport', color='#CD1C18')
#     bars_no = plt.bar(x, f1_no_pass, width, label='No Passport', color='#FFA896')
#     bars_forged = plt.bar(x + width, f1_forged_pass, width, label='Forged Passport', color='#9B1313')

#     # Labels and title
#     plt.ylabel('F1 Score', fontweight='bold')
#     plt.xlabel('Model', fontweight='bold')
#     plt.title('Passport-based Watermarking Performance', fontsize=14, pad=5)
#     plt.xticks(x, models)
#     plt.ylim(0, max(max(f1_valid_pass), max(f1_no_pass), max(f1_forged_pass)) * 1.1)
#     plt.grid(axis='y', alpha=0.3)
#     plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

#     # Highlight region over No/Forged Passport (group highlighting)
#     y_max = plt.ylim()[1]
#     for i in x:
#         plt.gca().add_patch(plt.Rectangle(
#             (i - width / 2, 0), width * 2, y_max, color='gray', alpha=0.1, zorder=0
#         ))

#     # Annotation above No and Forged Passport bars
#     for i in range(len(models)):
#         # No Passport
#         plt.text(x[i], f1_no_pass[i] + 0.01, "↓ Perf. Drop", ha='center', va='bottom', fontsize=10, color='#B03A2E')
#         # Forged Passport
#         plt.text(x[i] + width, f1_forged_pass[i] + 0.01, "↓ Perf. Drop", ha='center', va='bottom', fontsize=10, color='#641E16')

#     # Final layout and save
#     plt.tight_layout()
#     plt.savefig(f"sig_enc_f1_scores.png", format='png', bbox_inches='tight')
#     plt.close()


def main():
    plot_f1_scores("Total")


if __name__ == "__main__":
    main()
