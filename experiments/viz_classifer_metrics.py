"""
This script visualizes the cross-validated testing scores for each of machine learning classifiers.

**Input** 
-test_CV.csv 

**Output** 
-fig6c_CVtestscores.svg'

Daniel Yanes |  University of Nottingham
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CV_test_scores = pd.read_csv('results/ML_classifiers/test_CV.csv', index_col=0)


def plot_grouped_bars(df, filename=None):
    """
    Plots a grouped bar chart where each group represents a model and each bar within the group represents a different metric.
    Includes error bars based on the provided DataFrame with a repeating pattern of (Metric, SD), showing both upper and lower error bars.

    Parameters:
    df (pd.DataFrame): DataFrame containing mean metric scores and standard deviations for each model.
    filename (str): Optional. Filename to save the plot.
    """
    
    # Define figure and axis (wider for consistency with side-by-side figures)
    fig, ax = plt.subplots(figsize=(12, 6))  
    
    # Define the number of models and metrics
    num_models = len(df.index)
    num_metrics = len(df.columns) // 2  # Since columns alternate between metrics and SDs
    
    # Define the positions of the bars
    bar_width = 0.2
    indices = np.arange(num_models)
    
    blue_shades = ["#ADD8E6", "#87CEEB", "#4682B4"]
    
    
    # Loop through each metric and plot the bars with error bars
    for i in range(num_metrics):
        metric = df.columns[i * 2]  # Get metric name
        sd_metric = df.columns[i * 2 + 1]  # Get corresponding standard deviation
        
        means = df[metric]
        errors = df[sd_metric]
        
        # Clean up the metric name for the legend
        clean_metric_name = metric.replace('mean_', '').replace('_metric', '').replace('_', ' ').capitalize()
        
        # Plot bars with consistent aesthetics
        ax.bar(indices + i * bar_width, means, bar_width, label=clean_metric_name, 
               color=blue_shades[i % len(blue_shades)], edgecolor='black', 
               yerr=errors, capsize=5, error_kw=dict(lw=1.5, capthick=1.2), alpha = 0.8)
    
    # Labels and titles with consistent font styling
    ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    
    # Set x-axis labels and ticks
    ax.set_xticks(indices + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(df.index, rotation=40, fontsize=16, fontweight='bold', ha = 'right')
    
    # Adjust y-axis limits and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.yticks(fontsize=14)
    
    # Improve grid aesthetics
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add a legend
    ax.legend(title='Metrics', fontsize=9, title_fontsize=11, loc="upper right")
    
    # Optimize layout
    plt.tight_layout()
    
    # Save the plot if a filename is provided
    if filename:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'
        plt.savefig(filename, dpi=1200, bbox_inches='tight')  # Matching previous dpi
    else:
        plt.show()

# plot cv testing scores 
plot_grouped_bars(CV_test_scores, 'figures/fig6c_CVtestscores.svg')


