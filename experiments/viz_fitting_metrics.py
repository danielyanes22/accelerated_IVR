"""
This script generates boxplots for the fitting metrics of each kinetic model: RRMSE, AIC, and F2. In addition to EPCD plot of the absolute error.

Daniel Yanes |  University of Nottingham 
"""

import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

experiment_folder = 'fitting'

MAE_df = pd.read_csv(f'results/{experiment_folder}/MAE_df.csv')
AE_df = pd.read_csv(f'results/{experiment_folder}/AE_df.csv')
AIC_df = pd.read_csv(f'results/{experiment_folder}/aic_df.csv')
RRMSE_df = pd.read_csv(f'results/{experiment_folder}/RRMSE_df.csv')
F2_df = pd.read_csv(f'results/{experiment_folder}/F2_df.csv')
results_df = pd.read_csv(f'results/{experiment_folder}/results_df.csv')
exp_df = pd.read_csv(f'results/{experiment_folder}/drug_release_exp.csv')
sim_df = pd.read_csv(f'results/{experiment_folder}/sim_df.csv')


def remove_string(string):
    return string.replace('Absolute_Error', '')

def remove_string_AIC(string):
    return string.replace(' AIC', '')

#### ABSOLUTE ERROR BOXPLOT ####
AE_df.columns = AE_df.columns.map(remove_string)
AE_df = AE_df.rename(columns = {'Zero_order': 'Zero order', 'First_order': 'First order', 'Korsmeyer_Peppas': 'Korsmeyer Peppas'})
AE_df_1 = AE_df.iloc[:, 2:]
MAE_df.columns = MAE_df.columns.map(remove_string)
mean_AE_order = AE_df_1.mean().sort_values(ascending=False).index
lis_AE_order = mean_AE_order.tolist()

#CDF plot
f, boxplot = plt.subplots(figsize = (12, 6))
cdf_plot = sns.ecdfplot(data=AE_df_1, color='black', linewidth=1.2, log_scale=True)
boxplot.set_xlabel("Absolute error", fontsize=14, color = 'black', 
                   weight="bold")
boxplot.set_ylabel("Cumultative probability", fontsize=14, color = 'black', 
                   weight="bold")
plt.savefig(f'figures/SI/Fig_S2_CPD.svg', dpi=1200, bbox_inches='tight')


### RRMSE, AIC, F2 boxplot ###

RRMSE_df_1 = RRMSE_df.iloc[:, 2:]
mean_RRMSE_order = RRMSE_df_1.mean().sort_values(ascending=False).index
lis_RRMSE_order = mean_RRMSE_order.tolist()


# Define the color palette
palette = sns.color_palette("tab10")

# Create a combined figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# MAE Boxplot
sns.set_style("white")
sns.boxplot(palette=palette, data=RRMSE_df_1, saturation=1, order=lis_RRMSE_order ,
            boxprops=dict(linewidth=1.0, edgecolor='black', alpha=0.8),
            whiskerprops=dict(linewidth=1.0, color='black'),
            capprops=dict(linewidth=1.0, color='black'),
            flierprops=dict(marker="d", markerfacecolor="black", markeredgecolor="black", markersize=1, alpha=0.2),
            medianprops=dict(color="black", linewidth=1.0, linestyle='--'), showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0, markeredgecolor="black", markersize=3, linewidth=0.05, zorder=10),
            orient='h', ax=ax1)
sns.stripplot(data=RRMSE_df_1, marker="o", edgecolor='white', order=lis_RRMSE_order,
              alpha=0.25, size=8, linewidth=0.3, color='black', jitter=True, zorder=0, orient='h', ax=ax1)

for i, col_name in enumerate(lis_RRMSE_order):
    mean_val = RRMSE_df_1[col_name].mean()
    ax1.text(x=mean_val, y=i-0.54, s=f"{mean_val:.2f}", size=18, color='black', ha='center', va='top', weight='bold')

ax1.set_xlabel("RRMSE", fontsize=18, color='black', weight="bold")
ax1.tick_params(colors='black', which='both')
ax1.axes.yaxis.label.set_color('black')
ax1.axes.xaxis.label.set_color('black')
ax1.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1))
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')
ax1.spines['right'].set_color('black')
ax1.spines['top'].set_color('black')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, color='black', fontsize=16)
ax1.tick_params(bottom=True, left=True)
ax1.set_yticks(np.arange(len(lis_RRMSE_order)))
ax1.set_yticklabels(lis_RRMSE_order, fontsize=18, weight="bold", rotation=45)

# AIC Boxplot
AIC_df_1 = AIC_df.iloc[:, 2:]
mean_aic_order = AIC_df_1.mean().sort_values(ascending=False).index
lis_AIC_order = mean_aic_order.tolist()

sns.boxplot(palette=palette, data=AIC_df_1, saturation=1, order=lis_AIC_order,
            boxprops=dict(linewidth=1.0, edgecolor='black', alpha=0.8),
            whiskerprops=dict(linewidth=1.0, color='black'),
            capprops=dict(linewidth=1.0, color='black'),
            flierprops=dict(marker="d", markerfacecolor="black", markeredgecolor="black", markersize=1, alpha=0.2),
            medianprops=dict(color="black", linewidth=1.0, linestyle='--'), showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0, markeredgecolor="black", markersize=3, linewidth=0.05, zorder=10),
            orient='h', ax=ax2)
sns.stripplot(data=AIC_df_1, marker="o", edgecolor='white', order=lis_AIC_order,
              alpha=0.25, size=8, linewidth=0.3, color='black', jitter=True, zorder=0, orient='h', ax=ax2)

for i, col_name in enumerate(lis_AIC_order):
    mean_val = AIC_df_1[col_name].mean()
    ax2.text(x=mean_val, y=i-0.54, s=f"{mean_val:.2f}", size=18, color='black', ha='center', va='top', weight='bold')

ax2.set_xlabel("AIC", fontsize=18, color='black', weight="bold")
ax2.tick_params(colors='black', which='both')
ax2.axes.yaxis.label.set_color('black')
ax2.axes.xaxis.label.set_color('black')
ax2.set(xlim=(-120, 250), xticks=np.arange(-120, 250, 20))
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_color('black')
ax2.spines['right'].set_color('black')
ax2.spines['top'].set_color('black')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, color='black', fontsize=16)
ax2.tick_params(bottom=True, left=True)
ax2.set_yticks(np.arange(len(lis_AIC_order)))
ax2.set_yticklabels([''] * len(lis_AIC_order))


plt.tight_layout()
plt.savefig(f'figures/SI/Fig_S1_RRMSE_AIC.svg', dpi=1200, bbox_inches='tight')



# Ensure axes are defined
fig, ax = plt.subplots(figsize=(6, 4))

# Order columns by mean values
F2_df_1 = F2_df.iloc[:, 2:]
mean_F2_order = F2_df_1.mean().sort_values(ascending=True).index
lis_F2_order = mean_F2_order.tolist()

# Boxplot with ordered data
sns.boxplot(palette=palette, data=F2_df_1, saturation=1, order=lis_F2_order,
            boxprops=dict(linewidth=1.0, edgecolor='black', alpha=0.8),
            whiskerprops=dict(linewidth=1.0, color='black'),
            capprops=dict(linewidth=1.0, color='black'),
            flierprops=dict(marker="d", markerfacecolor="black", markeredgecolor="black", markersize=1, alpha=0.2),
            medianprops=dict(color="black", linewidth=1.0, linestyle='--'), showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0, markeredgecolor="black", markersize=3, linewidth=0.05, zorder=10),
            orient='h', ax=ax)

# Overlay stripplot
sns.stripplot(data=F2_df_1, marker="o", edgecolor='white', order=lis_F2_order,
              alpha=0.25, size=6, linewidth=0.3, color='black', jitter=True, zorder=0, orient='h', ax=ax)

# Annotate mean values
for i, col_name in enumerate(lis_F2_order):
    mean_val = F2_df_1[col_name].mean()
    ax.text(x=mean_val, y=i - 0.6, s=f"{mean_val:.2f}", size=8, color='black', ha='center', va='top', weight='bold')

# Axis settings
ax.set_xlabel("f2", fontsize=14, color='black', weight="bold")
ax.tick_params(colors='black', which='both')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 105, 5))
ax.tick_params(bottom=True, left=True)
ax.set_yticks(np.arange(len(lis_F2_order)))
ax.set_yticklabels(lis_F2_order, fontsize=12, weight="bold", rotation=45)
ax.axvline(x=50, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
ax.spines.values()
plt.tight_layout()
plt.savefig(f'figures/fig_2_F2_boxplot.svg', dpi=1200, bbox_inches='tight')

# Sort the DataFrame by 'Weibull' f2 score
sorted_f2_df = F2_df.sort_values(by='Weibull').reset_index(drop=True)

# Dynamically determine the number of profiles to plot (half the total)
n_profiles = len(sorted_f2_df) // 2
subset_df = sorted_f2_df.tail(n_profiles)   ### swap head / tail for the first / later half of available data respectively

# Dynamically calculate grid size for subplots
ncols = 12  # Set the number of columns (adjustable)
nrows = (n_profiles // ncols) + int(n_profiles % ncols > 0)  # Calculate rows dynamically

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex = False, sharey= False, squeeze = True, figsize=(ncols * 1.5, nrows * 2.5))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Plot each selected profile
for i, (idx, row) in enumerate(subset_df.iterrows()):
    file_name = row['File Name']
    weibull_f2 = row['Weibull']

    # Extract experimental and simulated data
    exp_data = exp_df[exp_df['file_name'] == file_name]
    sim_data = sim_df[sim_df['file_name'] == file_name]

    # Plot experimental and simulated data
    axes[i].scatter(exp_data['time (Hrs)'], exp_data['release_percent'], label='Experimental', marker='o', s = 5)
    axes[i].plot(sim_data['time'], sim_data['weibull predicted release %'], label='Simulated', linestyle='--', color='red')
    
    # Annotate with weibull f2 score
    axes[i].text(
        0.95, 0.2, f"{weibull_f2:.2f}",
        transform=axes[i].transAxes, fontsize=7, verticalalignment='bottom',
        horizontalalignment='right', color='red')
    

    # Add the file number as a title at the top-left corner of the subplot
    file_number = int(file_name.split('/')[-1].split('..')[0])
    axes[i].set_title(f"{file_number}", size = 6, pad = 3)    
    axes[i].tick_params(axis='both', which='major', labelsize=5)
    
    # Label x-axis with min and max values (rounded to integers)
    axes[i].set_xlim([exp_data['time (Hrs)'].min(), exp_data['time (Hrs)'].max()])
    axes[i].set_xticks([int(exp_data['time (Hrs)'].min()), int(exp_data['time (Hrs)'].max())])
    

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Add global x and y labels
fig.text(0.5, 0.04, 'Time (Hrs)', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Drug Release %', va='center', rotation='vertical', fontsize=14)

# Adjust layout for better spacing
plt.subplots_adjust(hspace=0.9, wspace=0.4)

# Add a single legend outside the subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=10)

# Show the plot
plt.savefig(f'figures/SI/fig_s4_f2_tail.svg', dpi=1200, bbox_inches='tight')