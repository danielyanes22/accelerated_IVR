""" 
Performing PCA and K-means clustering on the Weibull parameters to identify distinct release profiles. WSS and silhouette coeffcient are all calculated 
and visualised here for varying values number of clusters (k). Scree plots are also examined.

**Inputs**
- weibull_params.csv: A CSV file containing the filtered Weibull parameters for each file.

**Outputs**
- PCA_KMC.csv: A CSV file containing the file/ IVR ID, Weibull parameters, principal components, and cluster assignments.

**Figures**
- fig_S5_screeplot.svg: Scree plot showing the proportion of variance explained by each principal component.
- fig_S6PCA_KMC.svg: Scatter plots of the first two principal components with K-means cluster assignments for k = 2 to 7.
- fig_S7_WSS.svg: Line plot showing the within-cluster sum of squares (WSS) for k = 1 to 10.
- fig_S8_silhouette_coefficient.svg: Silhouette plots showing the silhouette coefficient for k = 2 to 7.
- fig_3b_PCAKMC.svg: Scatter plot of the first two principal components with K-means cluster assignments for k = 3.

Daniel Yanes | University of Nottingham

"""


import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.colors import ListedColormap, Normalize
from src.model_fitting import weibull

experiment_folder = 'clustering'


weibull_df = pd.read_csv(f'data/clean/weibull_params.csv')
params = weibull_df[['ID', 'alpha', 'beta']]


time = np.linspace(0, 24, 500)
# Initialize DataFrame with time as the first column
weibull_release_df = pd.DataFrame({'t': time})

file_ids = []

# Create Weibull release profiles for each file and add as columns to the DataFrame
for index, row in params.iterrows():
    a = row['alpha']
    b = row['beta']
    weibull_release = weibull(time, a, b)  
    weibull_release_df[row['ID']] = weibull_release 
    file_ids.append(row['ID'])


# Transpose the DataFrame and set column names to the time points
weibull_release_df = weibull_release_df.T
weibull_release_df.columns = time

# Drop the first row (time array) before performing PCA
weibull_release_df = weibull_release_df.iloc[1:]

# Standardize the release profiles
sim_release = StandardScaler().fit_transform(weibull_release_df)

# Perform PCA
pca = PCA(n_components=10, random_state=1884)
PCs = pca.fit_transform(sim_release)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=PCs, columns=[f'PC{i+1}' for i in range(PCs.shape[1])])

# Retain only the first two principal components
pca_df = pca_df[['PC1', 'PC2']]
pca_df['ID'] = file_ids


#join pca_df with params on ID 
params_pca = pd.merge(params, pca_df, on = 'ID')

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.savefig(f'figures/SI/fig_S5_screeplot.svg', dpi = 1200, bbox_inches = 'tight')


# Create subplots for each value of k
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
k_values = range(2, 8)

# Iterate through k values and plot the clusters
for k, ax in zip(k_values, axes.flatten()):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=1884)
    kmeans.fit(params_pca[['PC1', 'PC2']])
    
    # Get cluster assignments and cluster centers
    labels = kmeans.labels_
    
    # Plot data points with cluster assignments
    scatter = ax.scatter(params_pca['PC1'], params_pca['PC2'], c=labels, cmap='viridis', s=50, alpha=0.5)
    ax.set_title(f'k = {k}', fontsize=15, fontweight='bold')
    # Remove x axis from top row of plots
    if k in [2, 3, 4]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    
    else:
        ax.set_xlabel('PC1', fontsize=15, fontweight='bold')
    
    # Remove y axis from right column of plots
    if k in [2, 5]:
        ax.set_ylabel('PC2', fontsize = 15, fontweight='bold')
    
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])
    
plt.tight_layout()
fig.savefig(f'figures/SI/fig_S6PCA_KMC.svg', dpi = 1200, bbox_inches = 'tight')

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k, random_state= 1884).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points.iloc[i, 0] - curr_center[0]) ** 2 + (points.iloc[i, 1] - curr_center[1]) ** 2 
        sse.append(curr_sse)
    return sse


wss = calculate_WSS(params_pca[['PC1', 'PC2']], 10)

#plot WSS versus k 
fig = plt.figure()
plt.scatter(range(1, 11), wss)
plt.plot(range(1, 11), wss, marker = 'o', linestyle = '--', color = 'black', linewidth = 2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster sum of squares (WSS)')
plt.savefig('figures/SI/figs7_WSS.svg', dpi = 1200, bbox_inches = 'tight')
plt.close()

# Create subplots for each value of k
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Iterate through k values and plot silhouette plots
for k, ax in zip(k_values, axes.flatten()):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=1884)
    kmeans.fit(params_pca[['PC1', 'PC2']])
    cluster_labels = kmeans.labels_
    
    # Compute silhouette scores
    silhouette_avg = silhouette_score(params_pca[['PC1', 'PC2']], cluster_labels, metric = 'euclidean')
    sample_silhouette_values = silhouette_samples(params_pca[['PC1', 'PC2']], cluster_labels)
    
    y_lower = 10
    for i in range(k):
        # Aggregate silhouette scores for samples belonging to cluster i and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        #ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  

    ax.set_title(f'k = {k}', fontsize=15, fontweight='bold')
    #annotate with average silhouette score
    ax.text(0.5, 0.5, f'{silhouette_avg:.2f}', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(params_pca[['PC1', 'PC2']]) + (k + 1) * 10])

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])

        # Remove x axis from top row of plots
    if k in [2, 3, 4]:
        ax.set_xticklabels([])
        ax.set_xticks([])
    
    else:
        ax.set_xlabel('Silhouette coeffecient', fontsize=15, fontweight='bold')
    
    # Remove y axis from right column of plots
    if k in [2, 5]:
        ax.set_ylabel('Cluster number (k)', fontsize = 15, fontweight='bold')
    
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])

plt.savefig(f'figures/SI/figS8_silhouette_coefficient.svg', dpi = 1200, bbox_inches = 'tight')
plt.close()

#plot the PCA plot with k = 3
fig, ax = plt.subplots()
kmeans = KMeans(n_clusters=3, random_state=1884)
kmeans.fit(params_pca[['PC1', 'PC2']])
labels = kmeans.labels_
params_pca['cluster'] = labels

# Map cluster labels kinetic class
cluster_mapping = {0: 'Medium', 1: 'Fast', 2: 'Slow'}
params_pca['cluster_name'] = params_pca['cluster'].map(cluster_mapping)

custom_cmap = ListedColormap(['green', 'blue', 'red'])  # Match colors: Fast, Medium, Slow
norm = Normalize(vmin=0, vmax=2)

# Re-map cluster values to match the desired color order
params_pca['color_order'] = params_pca['cluster'].map({0: 1, 1: 0, 2: 2})  # Fast -> 0, Medium -> 1, Slow -> 2

# Plot data points with cluster assignments
scatter = ax.scatter(
    params_pca['PC1'], 
    params_pca['PC2'], 
    c=params_pca['color_order'],  # Re-mapped cluster values for correct color ordering
    cmap=custom_cmap, 
    s=50, 
    alpha=0.5,
    norm=norm
)

# Add axis labels
ax.set_xlabel('PC1', fontsize=18, fontweight='bold')
ax.set_ylabel('PC2', fontsize=18, fontweight='bold')

# Add a color bar with custom ticks and labels
cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Fast', 'Medium', 'Slow'])  # Custom labels with "Slow" at the top
cbar.ax.set_title('Cluster', fontsize=14, fontweight='bold', pad=10)
fig.savefig(f'figures/fig_3b_PCAKMC.svg', dpi = 1200, bbox_inches = 'tight')
plt.close()

#add cluster label for k = 3 to the original dataframe
kmeans = KMeans(n_clusters=3, random_state=1884)
kmeans.fit(params_pca[['PC1', 'PC2']])
params_pca['cluster'] = kmeans.labels_ 
params_pca = params_pca.rename(columns = {'ID': 'file'})

params_pca.to_csv(f'results/{experiment_folder}/3_PCA_KMC.csv', index = False)
print("Done!")
