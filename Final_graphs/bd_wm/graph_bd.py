import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-GUI backend
import os
import sys

results = "."
gen_inf_res_base = os.path.join(results, "bd_inference_general_results_multicolor.csv")
gen_inf_res_distilled = os.path.join(results, "bd_inference_general_results_multicolor_distilled.csv")
gen_inf_res_pruned = os.path.join(results, "bd_inference_general_results_multicolor_pruned.csv")
fig_general = os.path.join(results, "bd_f1_scores_comparison.png")

results = "."
trig_inf_res_base = os.path.join(results, "bd_inference_trigger_results_multicolor_base.csv")
trig_inf_res_distilled = os.path.join(results, "bd_inference_trigger_results_multicolor_distilled.csv")
trig_inf_res_pruned = os.path.join(results, "bd_inference_trigger_results_multicolor_pruned.csv")
fig_trigger = os.path.join(results, "bd_watermark_detection.png")

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

def plot_f1_scores_barchart_general(df_base, df_distilled, df_pruned, class_name, fig):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models for a specific class.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Extract F1 scores for the specified class
    f1_base = df_base[df_base['Class'] == class_name]['F1 Score'].values[0]
    f1_distilled = df_distilled[df_distilled['Class'] == class_name]['F1 Score'].values[0]
    f1_pruned = df_pruned[df_pruned['Class'] == class_name]['F1 Score'].values[0]

    # Bar plot for F1 scores
    labels = ['Baseline', 'Distilled', 'Pruned']
    f1_scores = [f1_base, f1_distilled, f1_pruned]
    plt.bar(labels, f1_scores, color=['#FFDE21', '#FFEA99', '#E0BC00'])

    # Graph details
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title(f'Backdoor Watermarking Performance — F1 Score\n (Mean over CIFAR-10 Classes)', fontsize=14, pad=5)
    plt.ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig)
    plt.close()  # Close the plot to avoid display issues in some environments

def plot_f1_scores_barchart_trigger(df_base, df_distilled, df_pruned, class_name, fig):
    """
    Plot F1 scores for Baseline, Distilled, and Pruned models for a specific class.
    """
    # Set larger font sizes for IEEE format
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Extract F1 scores for the specified class
    f1_base = df_base[df_base['Class'] == class_name]['F1 Score'].values[0]
    f1_distilled = df_distilled[df_distilled['Class'] == class_name]['F1 Score'].values[0]
    f1_pruned = df_pruned[df_pruned['Class'] == class_name]['F1 Score'].values[0]

    # Bar plot for F1 scores
    labels = ['Baseline', 'Distilled', 'Pruned']
    f1_scores = [f1_base, f1_distilled, f1_pruned]
    plt.bar(labels, f1_scores, color=['#FFDE21', '#FFEA99', '#E0BC00'])

    # Graph details
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title(f'Backdoor Watermark Detection Rate — F1 Score \n(for trigger enabled class: {class_name})', fontsize=14, pad=5)
    plt.ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig)
    plt.close()  # Close the plot to avoid display issues in some environments

def main():
    # Read CSV files
    gen_df_base = read_csv(gen_inf_res_base)
    gen_df_distilled = read_csv(gen_inf_res_distilled)
    gen_df_pruned = read_csv(gen_inf_res_pruned)

    trig_df_base = read_csv(trig_inf_res_base)
    trig_df_distilled = read_csv(trig_inf_res_distilled)
    trig_df_pruned = read_csv(trig_inf_res_pruned)

    # Ensure all files were read properly before plotting
    if trig_df_base is not None and trig_df_distilled is not None and trig_df_pruned is not None:
        plot_f1_scores_barchart_trigger(trig_df_base, trig_df_distilled, trig_df_pruned, "7", fig_trigger)
    else:
        print("One or more CSV files could not be read. Exiting.")

    # Ensure all files were read properly before plotting
    if gen_df_base is not None and gen_df_distilled is not None and gen_df_pruned is not None:
        plot_f1_scores_barchart_general(gen_df_base, gen_df_distilled, gen_df_pruned, "Total", fig_general)  
    else:
        print("One or more CSV files could not be read. Exiting.")

if __name__ == "__main__":
    main()
