import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# results = "base_comparison"
# gen_inf_res_base = os.path.join(results, "bd_inference_general_results.csv")
# gen_inf_res_wt_pt_wm = os.path.join(results, "bd_inference_wt_pt_wm_results.csv")
# gen_inf_res_multicolor = os.path.join(results, "bd_inference_general_results_multicolor.csv")
# # gen_inf_res_multicolor = os.path.join(results, "bd_inference_general_results_multicolor.csv")
# fig = os.path.join(results, "baseline_f1_scores_comparison.png")

results = "weight_pt_watermark"
gen_inf_res_base = os.path.join(results, "bd_inference_general_results.csv")
gen_inf_res_wt_pt_wm = os.path.join(results, "bd_inference_wt_pt_wm_results.csv")
gen_inf_res_wt_pt_wm_distilled = os.path.join(results, "bd_inference_wt_pt_wm_results_distilled.csv")
gen_inf_res_wt_pt_wm_pruned = os.path.join(results, "bd_inference_wt_pt_wm_results_pruned.csv")
fig = os.path.join(results, "f1_scores_comparison.png")

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)

# # File paths
# gen_inf_res_base = "bd_inference_general_results.csv"         # Baseline
# gen_inf_res_3colors = "bd_inference_general_results_3colors.csv"  # Distilled
# gen_inf_res_multicolor = "bd_inference_general_results_multicolor.csv"  # Pruned

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


def plot_f1_scores(df_base, df_wt_pt_wm, df_multicolor, fig_path="f1_scores_comparison.pdf"):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models with 10 classes.
    Optimized for IEEE paper format with angled axis labels instead of value annotations.
    """
    # Set the non-interactive backend to avoid display issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with high resolution
    plt.figure(figsize=(10, 9), dpi=300)
    
    # Extract data
    classes = df_base['Class'].to_numpy()
    f1_base = df_base['F1 Score'].to_numpy()
    f1_wt_pt_wm = df_wt_pt_wm['F1 Score'].to_numpy()
    f1_multicolor = df_multicolor['F1 Score'].to_numpy()
    
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })
    
    # Create positions for grouped bars
    x = np.arange(len(classes))
    width = 0.25  # width of bars

    # Plot grouped bars with distinct patterns for better differentiation in print
    plt.bar(x - width, f1_base, width, label='Baseline Watermarked', color='#1f77b4', 
            edgecolor='black', linewidth=1, hatch='')
    plt.bar(x, f1_wt_pt_wm, width, label='Distilled', color='#ff7f0e', 
            edgecolor='black', linewidth=1, hatch='/')
    plt.bar(x + width, f1_multicolor, width, label='Pruned', color='#2ca02c', 
            edgecolor='black', linewidth=1, hatch='\\')

    # # Plot grouped bars with distinct patterns for better differentiation in print
    # plt.bar(x - width, f1_base, width, label='Baseline', color='#1f77b4', 
    #         edgecolor='black', linewidth=1, hatch='')
    # plt.bar(x, f1_wt_pt_wm, width, label='Weight Perturbation', color='#ff7f0e', 
    #         edgecolor='black', linewidth=1, hatch='/')
    # plt.bar(x + width, f1_multicolor, width, label='Backdoor', color='#2ca02c', 
    #         edgecolor='black', linewidth=1, hatch='\\')
    
    # Add grid, labels and title
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('F1 Score', fontweight='bold')
    plt.xlabel('Class', fontweight='bold')
    plt.title('Weight Perturbation F1 Score Comparison:\nBaseline Watermarked vs Distilled vs Pruned', fontsize=20, pad=20)
    # plt.title('F1 Score Comparison Across Model Variants', fontsize=20, pad=20)
    
    # Set y-axis to start from 0 and have a reasonable upper limit
    plt.ylim(0, max(max(f1_base), max(f1_wt_pt_wm), max(f1_multicolor)) * 1.1)
    
    # Set x-ticks in the middle of grouped bars with 45-degree rotation
    plt.xticks(x, classes, rotation=45, ha='right')
    
    # Customize legend with clear background
    plt.legend(frameon=True, facecolor='white', edgecolor='black', 
               loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    
    # Make sure everything fits properly
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(fig_path, format='png', bbox_inches='tight')


# def plot_f1_scores(df_base, df_wt_pt_wm, df_multicolor, fig_path):
#     """
#     Plot F1 scores for Baseline, Distilled, and Pruned models.
#     """
#     plt.figure(figsize=(10, 9))

#     # Set larger font sizes for IEEE format
#     plt.rcParams.update({
#         'font.size': 24,
#         'axes.labelsize': 24,
#         'axes.titlesize': 24,
#         'xtick.labelsize': 18,
#         'ytick.labelsize': 18,
#         'legend.fontsize': 20
#     })

#     # Extract and convert to 1D NumPy arrays
#     classes = df_base['Class'].to_numpy()
#     f1_base = df_base['F1 Score'].to_numpy()
#     f1_wt_pt_wm = df_wt_pt_wm['F1 Score'].to_numpy()
#     f1_multicolor = df_multicolor['F1 Score'].to_numpy()

#     # Plot F1 scores from each model
#     plt.plot(classes, f1_base, marker='o', linestyle='-', label='Baseline (F1)')
#     plt.plot(classes, f1_wt_pt_wm, marker='s', linestyle='--', label='Wt-Pt_WM (F1)')
#     plt.plot(classes, f1_multicolor, marker='^', linestyle='-.', label='Bd_WM (F1)')

#     # Graph details
#     plt.xlabel('Class', fontweight='bold')
#     plt.ylabel('F1 Score', fontweight='bold')
#     plt.title('F1 Score Comparison:\nBaseline vs Weight_Perturbed_WM vs Backdoor_WM', fontsize=20, pad=20)
#     plt.xticks(rotation=45)
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(fig_path)
#     plt.show()

# def main():
#     # Read CSV files
#     df_wt_pt_wm = read_csv(gen_inf_res_wt_pt_wm)
#     df_wt_pt_wm_distilled = read_csv(gen_inf_res_wt_pt_wm_distilled)
#     df_wt_pt_wm_pruned = read_csv(gen_inf_res_wt_pt_wm_pruned)

#     # Ensure all files were read properly before plotting
#     if df_wt_pt_wm is not None and df_wt_pt_wm_distilled is not None and df_wt_pt_wm_pruned is not None:
#         plot_f1_scores(df_wt_pt_wm, df_wt_pt_wm_distilled, df_wt_pt_wm_pruned, fig_path=fig)
#     else:
#         print("One or more CSV files could not be read. Exiting.")

# if __name__ == "__main__":
#     main()


def main():
    # # Read CSV files
    # df_base = read_csv(gen_inf_res_base)
    # df_wt_pt_wm = read_csv(gen_inf_res_wt_pt_wm)
    # df_multicolor = read_csv(gen_inf_res_multicolor)

    # # Ensure all files were read properly before plotting
    # if df_base is not None and df_wt_pt_wm is not None and df_multicolor is not None:
    #     plot_f1_scores(df_base, df_wt_pt_wm, df_multicolor, fig_path=fig)
    # else:
    #     print("One or more CSV files could not be read. Exiting.")

    # Call the function to generate the plot
    plot_watermark_accuracy()


if __name__ == "__main__":
    main()
