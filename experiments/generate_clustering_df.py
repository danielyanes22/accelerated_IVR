"""
Script for processing Weibull model parameters and filtering based on F2 values, biphasic profiles and time units.

This script performs the following steps:
1. **Load Data**: Reads parameter data (`param_df.pkl`), F2 values (`F2_df.csv`), and time units (`time_units.csv`).
2. **Filter Weibull Models**: Extracts only the rows corresponding to the Weibull model.
3. **Apply Filters**: 
   - Keeps only rows where `F2 > 50`.
   - Removes specific bi-phasic drug release files.
   - Filters only rows originally done in hours.
4. **Save Processed Data**: Exports the cleaned dataset to `data/clean/weibull_params.csv` for .

**Outputs:**
- `data/unprocessed/weibullparams_f2.csv` → Intermediate merged data.
- `data/clean/weibull_params.csv` → Final processed data for drug release clustering.


Daniel Yanes | 13/02/2025 | University of Nottingham
"""

import pandas as pd 

experiment_folder = 'fitting'

#import datasets
param_df = pd.read_pickle(f'results/{experiment_folder}/param_df.pkl')
f2_df = pd.read_csv(f'results/{experiment_folder}/F2_df.csv')
time_units = pd.read_csv(f'data/time_units.csv')


#Select weibull model parameters and combine with f2 score 
weibull_params = param_df[param_df['Model'] == 'weibull']

print('Number of files left after quality appraisal:', weibull_params.shape[0])
f2_df.rename(columns={'File Name': 'File_Name', 'Weibull': 'F2'}, inplace=True)
f2_df = f2_df[['File_Name', 'F2']]
weibull_params = pd.merge(weibull_params, f2_df, on='File_Name', how='inner')
weibull_params.to_csv(f'data/unprocessed/weibullparams_f2.csv', index=False)



#filter plots where f2 > 50 and remove bi-phasic plots
filtered_weibull = weibull_params[weibull_params['F2'] > 50]

folder = 'data/drug_release/'

bi_phasic_files = [
    'data/drug_release/238..csv',
    'data/drug_release/234..csv',
    'data/drug_release/235..csv'
]
filtered_weibull = filtered_weibull[~filtered_weibull['File_Name'].isin(bi_phasic_files)]

print('Number of files left after removing f2 < 50 and biphasic:', filtered_weibull.shape[0])

#Extract weibull parameters (alpha, beta) for subsequent plotting / analysis 
w_params = filtered_weibull[['Optimized_Parameters']]
w_rows = w_params.iloc[0:,0]
files = filtered_weibull['File_Name']
alpha_l = []
beta_l = []

for row_i in range(len(w_rows)):
    row = w_rows.values[row_i]
    alpha_l.append(row[0])
    beta_l.append(row[1])

weibull_df = pd.DataFrame({
    'File_Name': files,
    'alpha': alpha_l,
        'beta': beta_l})

weibull_df['File_Name'] = weibull_df['File_Name'].str.replace('..csv', '')
weibull_df['File_Name'] = weibull_df['File_Name'].str.replace('data/drug_release/', '')
weibull_df.rename(columns={'File_Name': 'ID'}, inplace=True)


weibull_df['ID'] = weibull_df['ID'].astype(str)
time_units['ID'] = time_units['ID'].astype(str)
weibull_df_1 = pd.merge(weibull_df, time_units, on='ID', how='inner')

#Filter only the rows where time units are in hours 
weibull_df_hours = weibull_df_1[weibull_df_1['Time_units'] == 'hours']

print('Number of files left after selecting hours:', weibull_df_hours.shape[0])

weibull_df_hours.to_csv(f'data/clean/weibull_params.csv', index=False)