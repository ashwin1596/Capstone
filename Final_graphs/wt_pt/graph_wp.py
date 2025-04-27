import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

results = "."
gen_inf_res_wt_pt_wm = os.path.join(results, "bd_inference_wt_pt_wm_results.csv")
gen_inf_res_wt_pt_wm_distilled = os.path.join(results, "bd_inference_wt_pt_wm_results_distilled.csv")
gen_inf_res_wt_pt_wm_pruned = os.path.join(results, "bd_inference_wt_pt_wm_results_pruned.csv")
fig = os.path.join(results, "wp_f1_scores_comparison.png")

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)

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


def plot_watermark_accuracy():
    """
    Plot a bar chart comparing watermark detection accuracy for pruned and baseline models
    using the weight perturbation method.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        # 'font.size': 12,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        # 'legend.fontsize': 12
    })

    # Define labels and accuracy values
    labels = ['Baseline', 'Pruned']
    accuracy_values = [100.00, 70.15]
    
    # Plot bar chart
    plt.bar(labels, accuracy_values, color=['#FFDE21', '#E0BC00'])
    
    # Add labels and title
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Weight Pt. Scheme -\nWatermark Detection Accuracy', pad=5)
    
    # Display values on bars
    # for i, v in enumerate(accuracy_values):
        # plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')
    
    plt.ylim(0, 110)  # Set y-axis limit
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig("wm_watermark_detection_acc.png")
    plt.close()

def plot_f1_scores_barchart_general(df_base, df_distilled, df_pruned, class_name):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models for a specific class.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        # 'font.size': 12,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        # 'legend.fontsize': 12
    })
    # Extract F1 scores for the specified class
    f1_base = df_base[df_base['Class'] == class_name]['F1 Score'].values[0]
    f1_distilled = df_distilled[df_distilled['Class'] == class_name]['F1 Score'].values[0]
    f1_pruned = df_pruned[df_pruned['Class'] == class_name]['F1 Score'].values[0]

    # Bar plot for F1 scores
    labels = ['Baseline', 'Distilled', 'Pruned']
    f1_scores = [f1_base, f1_distilled, f1_pruned]
    bars = plt.bar(labels, f1_scores, color=['#FFDE21', '#FFEA99', '#E0BC00'])
    # bars = plt.bar(labels, f1_scores, color=['#FFDE21', '#FFEA99', '#E0BC00'])
      
    # # Add values above each bar
    # for bar, score in zip(bars, f1_scores):
    #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
    #              f"{score:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Graph details
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title('Weight Pt. Scheme F1 Score\n(CIFAR-10 Mean)', pad=5)
    plt.ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig)
    plt.close()  # Close the plot to avoid display issues in some environments


def main():
    # Read CSV files
    df_wt_pt_wm = read_csv(gen_inf_res_wt_pt_wm)
    df_wt_pt_wm_distilled = read_csv(gen_inf_res_wt_pt_wm_distilled)
    df_wt_pt_wm_pruned = read_csv(gen_inf_res_wt_pt_wm_pruned)

    # Ensure all files were read properly before plotting
    if df_wt_pt_wm is not None and df_wt_pt_wm_distilled is not None and df_wt_pt_wm_pruned is not None:
        plot_f1_scores_barchart_general(df_wt_pt_wm, df_wt_pt_wm_distilled, df_wt_pt_wm_pruned, "Total")
    else:
        print("One or more CSV files could not be read. Exiting.")

    # Call the function to generate the plot
    plot_watermark_accuracy()

if __name__ == "__main__":
    main()
