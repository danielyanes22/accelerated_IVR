"""
Script for generating figure 3c - Simulated weibull drug release profile for each cluster assigned based on PCA-KMC (k = 3).

**Inputs**
- weibull_params.csv: A CSV file containing the filtered Weibull parameters for each file.
- 3_PCA_KMC.csv: A CSV file containing the file/ IVR ID, Weibull parameters, principal components, and cluster assignments.

**Figures**
- fig_3c_drugrelease.svg: Simulated drug release profiles for each cluster based on the Weibull parameters.
- fig_3a_drug_release_preclustering.svg: Simulated drug release profiles for each file based on the Weibull parameters.

Daniel Yanes | 13/02/2025 | University of Nottingham
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from src.model_fitting import weibull
pd.set_option('display.float_format', lambda x: '%.3f' % x)

weibull_df = pd.read_csv(f'data/clean/weibull_params.csv')
weibull_clusts = pd.read_csv(f'results/clustering/3_PCA_KMC.csv')


#add +1 to every value in weibull_clusts['cluster'] to avoid 0 indexing
weibull_clusts['cluster'] = weibull_clusts['cluster'] + 1
params = weibull_df[['ID', 'alpha', 'beta']]


colors = ['red', 'blue', 'green']
cluster_dict = {}
num_clusters = 3

for cluster_num in range(0, num_clusters + 1):
    cluster_df = weibull_clusts[weibull_clusts['cluster'] == cluster_num]
    cluster_dict[f'clust_{cluster_num}'] = cluster_df  #cluster_dict['clust_1']
    
summary_dict = {}

for cluster_name, cluster_df in cluster_dict.items():
    mean_vals = cluster_df[['alpha', 'beta']].mean()
    min_vals = cluster_df[['alpha', 'beta']].min()
    max_vals = cluster_df[['alpha', 'beta']].max()
    
    summary_dict[cluster_name] = pd.concat([mean_vals, min_vals, max_vals], axis=1)
    summary_dict[cluster_name].columns = ['mean', 'min', 'max']


clusters = ['clust_1', 'clust_2', 'clust_3']
t = np.linspace(0, 24,500) 

summary_dfs = []

for cluster in clusters:
    # Extract parameters for min, mean, and max from summary_dict
    params = summary_dict[cluster].loc[['alpha', 'beta'], ['min', 'mean', 'max']].values.T
    
    weibull_release = np.array([weibull(t, *p) for p in params]) # Calculate Weibull release for min, mean, and max parameters
    
    clust_df = pd.DataFrame({
        't': t,
        'min': weibull_release[0],
        'mean': weibull_release[1],
        'max': weibull_release[2]
    })
    
    summary_dfs.append(clust_df)  #extract using summarys_dfs[0] for cluster_1 

cluster_dfs = {}
num_clusters = len(clusters)

for i in range(1, num_clusters+1):
    cluster_name = f'clust_{i}'
    cluster_dict_i = cluster_dict[cluster_name]
    
    cluster_df = pd.DataFrame({'t': t})  
    
    for file in cluster_dict_i['file']:
        alpha = cluster_dict_i.loc[cluster_dict_i['file'] == file, 'alpha'].values[0]
        beta = cluster_dict_i.loc[cluster_dict_i['file'] == file, 'beta'].values[0]
        
        dr = weibull(t, alpha, beta)
        
        cluster_df[file] = dr
    
    cluster_dfs[cluster_name] = cluster_df #cluster_dfs['clust_4'] example indexing 

#join summary_dfs[x] and cluster_dfs['clust_x']
predicted_dr = {}

for i, cluster_name in enumerate(clusters):
    concat_df = pd.concat([summary_dfs[i], cluster_dfs[cluster_name]], axis=1) # Concatenate summary DataFrame with cluster DataFrame horizontally
    predicted_dr[cluster_name] = concat_df
    predicted_dr[cluster_name] = predicted_dr[cluster_name].loc[:,~predicted_dr[cluster_name].columns.duplicated()] #predicted_dr['clust_x'] example indexing
    
def plot_clust_in_grid(line_colors):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Loop through clusters and corresponding line colors
    for clust_num, line_color, ax in zip([3, 1, 2], line_colors, axes.flatten()):
        clust_df = predicted_dr[f"clust_{clust_num}"].iloc[:, 4:]
        #summary_df = predicted_dr[f"clust_{clust_num}"].iloc[:, :4]  # min, mean, max
        
        # Plot each individual Weibull release curve with a lighter color
        for col in clust_df.columns:
            ax.plot(t, clust_df[col], color=line_color, alpha=0.2)
                 
        ax.set_xlabel('Time / Hours', fontsize=18, fontweight='bold')
                        
        #set the title of each plot to medium, fast and slow 
        if clust_num == 1:
            ax.set_title('Medium', fontsize=14, fontweight='bold')
            ax.set_yticklabels([])
        elif clust_num == 2:
            ax.set_title('Fast', fontsize=14, fontweight='bold')
            ax.set_yticklabels([])
        else:
            ax.set_title('Slow', fontsize=14, fontweight='bold')
            ax.set_ylabel('Simulated drug release / % ', fontsize=18, fontweight='bold')
        
        #ax.set_title(f"{clust_num}", fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.5)
    
    
    plt.tight_layout()

    return fig 


line_colors = ['red', 'blue', 'green']

figure = plot_clust_in_grid(line_colors)
figure.savefig('figures/fig_3c_drugrelease.svg', dpi = 1200, bbox_inches = 'tight')


fig, ax = plt.subplots()

params = weibull_df[['ID', 'alpha', 'beta']]

for index, row in params.iterrows():
    alpha = row['alpha']
    beta = row['beta']
    dr = weibull(t, alpha, beta)
    ax.plot(t, dr)

ax.set_xlabel('Time / Hours', weight = 'bold', fontsize = 18)
ax.set_ylabel('Simulated drug release / %', weight = 'bold', fontsize = 18)


fig.savefig('figures/fig3a_drug_release_preclustering.svg', dpi = 1200)