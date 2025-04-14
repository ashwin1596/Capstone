import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

# results = "3color"
# gen_inf_res_base = os.path.join(results, "bd_inference_general_results_3colors.csv")
# gen_inf_res_distilled = os.path.join(results, "bd_inference_general_results_3colors_distilled.csv")
# gen_inf_res_pruned = os.path.join(results, "bd_inference_general_results_3colors_pruned.csv")
# fig = os.path.join(results, "f1_scores_comparison_3color_general.png")

# results = "multicolor"
# gen_inf_res_base = os.path.join(results, "bd_inference_general_results_multicolor.csv")
# gen_inf_res_distilled = os.path.join(results, "bd_inference_general_results_multicolor_distilled.csv")
# gen_inf_res_pruned = os.path.join(results, "bd_inference_general_results_multicolor_pruned.csv")
# fig = os.path.join(results, "f1_scores_comparison_multicolor_general.png")

# results = "3color"
# trig_inf_res_base = os.path.join(results, "bd_inference_trigger_results_3color_base.csv")
# trig_inf_res_distilled = os.path.join(results, "bd_inference_trigger_results_3color_distilled.csv")
# trig_inf_res_pruned = os.path.join(results, "bd_inference_trigger_results_3color_pruned.csv")
# fig = os.path.join(results, "f1_scores_comparison_3color_triggered.png")

results = "multicolor"
trig_inf_res_base = os.path.join(results, "bd_inference_trigger_results_multicolor_base.csv")
trig_inf_res_distilled = os.path.join(results, "bd_inference_trigger_results_multicolor_distilled.csv")
trig_inf_res_pruned = os.path.join(results, "bd_inference_trigger_results_multicolor_pruned.csv")
fig = os.path.join(results, "f1_scores_comparison_multicolor_triggered.png")

# Ensure checkpoint directory exists
os.makedirs(results, exist_ok=True)

# # File paths
# gen_inf_res_base = "bd_inference_general_results.csv"         # Baseline
# gen_inf_res_distilled = "bd_inference_general_results_3colors.csv"  # Distilled
# gen_inf_res_pruned = "bd_inference_general_results_multicolor.csv"  # Pruned

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
    
    # Add grid, labels and title
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('F1 Score', fontweight='bold')
    plt.xlabel('Class', fontweight='bold')
    plt.title('Backdoor WM F1 Score Comparison:\nBaseline Watermarked vs Distilled vs Pruned', fontsize=20, pad=20)
    
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

def plot_f1_scores_barchart(df_base, df_distilled, df_pruned, class_name):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models for a specific class.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20
    })

    plt.figure(figsize=(10, 6))

    # Extract F1 scores for the specified class
    f1_base = df_base[df_base['Class'] == class_name]['F1 Score'].values[0]
    f1_distilled = df_distilled[df_distilled['Class'] == class_name]['F1 Score'].values[0]
    f1_pruned = df_pruned[df_pruned['Class'] == class_name]['F1 Score'].values[0]

    # Bar plot for F1 scores
    labels = ['Baseline', 'Distilled', 'Pruned']
    f1_scores = [f1_base, f1_distilled, f1_pruned]
    plt.bar(labels, f1_scores, color=['blue', 'orange', 'green'])

    # Graph details
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title(f'Backdoor WM F1 Score Comparison for trigger enabled class: {class_name}', fontsize=20, pad=20)
    plt.ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig)
    plt.show()

def main():
    # # Read CSV files
    # df_base = read_csv(gen_inf_res_base)
    # df_distilled = read_csv(gen_inf_res_distilled)
    # df_pruned = read_csv(gen_inf_res_pruned)

    df_base = read_csv(trig_inf_res_base)
    df_distilled = read_csv(trig_inf_res_distilled)
    df_pruned = read_csv(trig_inf_res_pruned)

    # Ensure all files were read properly before plotting
    if df_base is not None and df_distilled is not None and df_pruned is not None:
        plot_f1_scores_barchart(df_base, df_distilled, df_pruned, "7")
    else:
        print("One or more CSV files could not be read. Exiting.")

    # # Ensure all files were read properly before plotting
    # if df_base is not None and df_distilled is not None and df_pruned is not None:
    #     plot_f1_scores(df_base, df_distilled, df_pruned, fig_path=fig)  
    # else:
    #     print("One or more CSV files could not be read. Exiting.")

if __name__ == "__main__":
    main()
