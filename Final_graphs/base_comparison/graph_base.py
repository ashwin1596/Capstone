import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# results = "base_comparison"
results = "."
gen_inf_res_base = os.path.join(results, "base_inference_results.csv")
gen_inf_res_wt_pt_wm = os.path.join(results, "inference_wt_pt_wm_results.csv")
gen_inf_res_multicolor = os.path.join(results, "bd_inference_general_results.csv")
gen_inf_res_sig_enc = os.path.join(results, "sig_wm_val_pass_base_inference_results.csv")
fig = os.path.join(results, "baseline_f1_scores_comparison.png")

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

def plot_f1_scores_barchart_general(df_base, df_wt_pt_wm, df_bd, df_sig_enc, class_name):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models for a specific class.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        # 'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        # 'legend.fontsize': 12
    })

    # Extract F1 scores for the specified class
    f1_base = df_base[df_base['Class'] == class_name]['F1 Score'].values[0]
    f1_wt_pt_wm = df_wt_pt_wm[df_wt_pt_wm['Class'] == class_name]['F1 Score'].values[0]
    f1_bd = df_bd[df_bd['Class'] == class_name]['F1 Score'].values[0]
    f1_sig_enc = df_sig_enc[df_sig_enc['Class'] == class_name]['F1 Score'].values[0]

    # Bar plot for F1 scores
    labels = ['Baseline\nModel', 'Backdoor\nEmbedded', 'Weight\nPerturbed', 'Passport\nEmbedded']
    f1_scores = [f1_base, f1_bd, f1_wt_pt_wm, f1_sig_enc]
    # plt.bar(labels, f1_scores, color=['#1984c5', '#e2e2e2', '#a7d5ed', '#c23728'])
    plt.bar(labels, f1_scores, color=['#1A4A96', '#2D68C4', '#143A78', '#69A3E1'])

    # Graph details
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title(f'F1 Score Comparison:\nBaseline vs. Watermarked Models', pad=10)
    plt.ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig)
    plt.show()

# def plot_f1_scores(df_base, df_wt_pt_wm, df_multicolor, df_sig_enc, fig_path="f1_scores_comparison.pdf"):
#     """
#     Plot F1 scores for Baseline, Distilled, and Pruned models with 10 classes.
#     Optimized for IEEE paper format with angled axis labels instead of value annotations.
#     """
#     # Set the non-interactive backend to avoid display issues
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Create figure with high resolution
#     # plt.figure(figsize=(10, 9), dpi=300)
#     plt.figure(dpi=300)
    
#     # Extract data
#     classes = df_base['Class'].to_numpy()
#     f1_base = df_base['F1 Score'].to_numpy()
#     f1_wt_pt_wm = df_wt_pt_wm['F1 Score'].to_numpy()
#     f1_multicolor = df_multicolor['F1 Score'].to_numpy()
#     f1_sig_enc = df_sig_enc['F1 Score'].to_numpy()
    
#     # Set larger font sizes for IEEE format
#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.labelsize': 12,
#         'axes.titlesize': 12,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 12
#     })

#     # Create positions for grouped bars with added space between groups
#     group_gap = 0.5  # Adjust this value to increase or decrease the gap
#     x = np.arange(len(classes)) * (1 + group_gap)  # Add space between groups
#     width = 0.25  # width of bars

#     # Plot grouped bars with distinct patterns for better differentiation in print
#     plt.bar(x - 1.5 * width, f1_base, width, label='Baseline', color='#CD1C18')
#     plt.bar(x - 0.5 * width, f1_wt_pt_wm, width, label='Weight-Pt', color='#FFA896') 
#     plt.bar(x + 0.5 * width, f1_multicolor, width, label='Backdoor', color='#9B1313')
#     plt.bar(x + 1.5 * width, f1_sig_enc, width, label='Passport', color='#38000A')
   
#     # Add grid, labels and title
#     plt.grid(axis='y', alpha=0.3)
#     plt.ylabel('F1 Score', fontweight='bold')
#     plt.xlabel('Class', fontweight='bold')
#     plt.title('F1 Score Comparison Across Model Variants', fontsize=14, pad=5)
    
#     # Set y-axis to start from 0 and have a reasonable upper limit
#     plt.ylim(0, max(max(f1_base), max(f1_wt_pt_wm), max(f1_multicolor), max(f1_sig_enc)) * 1.1)
    
#     # Set x-ticks in the middle of grouped bars with 45-degree rotation
#     plt.xticks(x, classes, rotation=45, ha='right')
    
#     # Customize legend with clear background
#     plt.legend(frameon=False, facecolor='white', edgecolor='black', 
#                loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4)
    
#     # Make sure everything fits properly
#     plt.tight_layout()
    
#     # Save with high quality
#     plt.savefig(fig_path, format='png', bbox_inches='tight')

def main():
    # Read CSV files
    df_base = read_csv(gen_inf_res_base)
    df_wt_pt_wm = read_csv(gen_inf_res_wt_pt_wm)
    df_bd = read_csv(gen_inf_res_multicolor)
    df_sig_enc = read_csv(gen_inf_res_sig_enc)

    plot_f1_scores_barchart_general(df_base, df_wt_pt_wm, df_bd, df_sig_enc, class_name="Total")

    # # Ensure all files were read properly before plotting
    # if df_base is not None and df_wt_pt_wm is not None and df_multicolor is not None and df_sig_enc is not None:
    #     plot_f1_scores(df_base, df_wt_pt_wm, df_multicolor, df_sig_enc, fig_path=fig)
    # else:
    #     print("One or more CSV files could not be read. Exiting.")


if __name__ == "__main__":
    main()
