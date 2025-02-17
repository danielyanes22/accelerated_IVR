"""

Script to perform parameter estimation of each model using the experimental IVR data obtained from the literature. 
The script uses the experimental IVR data to fit 6 different models and optimise the parameters for each model.
The aim of the analysis is to determine the most versatile model to describe the drug release profiles in the dataset with sufficient accuraacy.

Daniel Yanes | 12/02/2025 | University of Nottingham

"""

import numpy as np 
import scipy as sp
import pandas as pd
from src.model_fitting import (file_names, models, 
                               model_names, lhs_samples, 
                               weibull, get_model_params)

np.set_printoptions(suppress=True)

experiment_folder = 'fitting'

#each file references IVR.ID in db, use this df to compare experimental data to simulated data 

time_units = pd.read_csv('data/time_units.csv')
param_df = pd.DataFrame(columns=['File_Name', 'Model', 'Optimized_Parameters', 'MAE'])
exp_dfs = []

for file_name in file_names:
    drug_release_plot = np.genfromtxt(file_name, delimiter=',')
        
    file_ID = int(file_name.split('/')[-1].split('..')[0])
        
    units = time_units[time_units['ID'] == file_ID]
    units = units['Time_units'].values[0]
    
    time = drug_release_plot[:, 0]
    time[time < 0] = 0
    
    release_percent = drug_release_plot[:, 1]
    
    if units == 'mins':
        time = time/60
    
    elif units == 'seconds':
        time = time/3600

    else:
        time = time
    
    exp_df = pd.DataFrame({'time (Hrs)': time, 'release_percent': release_percent, 'file_name': file_name})
    exp_dfs.append(exp_df)

    for model, params in zip(models, lhs_samples):
        optim_params = None
        low_mae = float('inf')
        
        for i in range(len(params)):

            if model.__name__ == weibull.__name__:
                try:
                    popt, _ = sp.optimize.curve_fit (model, time, release_percent, params[i], bounds = ((1e-8, 1e-8), (300, 5)))
                    simulated = model(time, *popt)
                    mae = np.mean(np.abs(release_percent - simulated))
                    if mae < low_mae:
                        low_mae = mae
                        optim_params = popt
                except RuntimeError:
                    popt = np.nan
                    mae = np.nan
            else:
                try:
                    popt, _ = sp.optimize.curve_fit (model, time, release_percent, params[i])
                    simulated = model(time, *popt)
                    mae = np.mean(np.abs(release_percent - simulated))
                    if mae < low_mae:
                        low_mae = mae
                        optim_params = popt
                except RuntimeError:
                    popt = np.nan
                    mae = np.nan
        # Create a DataFrame row for this file's results with the current model
        fit = pd.DataFrame({
            'File_Name': [file_name],
            'Model': [model.__name__],
            'Optimized_Parameters': [optim_params],
            'MAE': [low_mae]
        })
        # Append this row to the overall DataFrame
        param_df = pd.concat([param_df, fit], ignore_index=True)

param_df.to_csv(f'results/{experiment_folder}/fit_all.csv', index=False)

print('optimisation complete!')

drug_release_exp = pd.concat(exp_dfs, ignore_index = True)   #each file references IVR.ID in db, use this df to compare experimental data to simulated data 

drug_release_exp.to_csv(f'results/{experiment_folder}/drug_release_exp.csv', index = False)

###### SELECT OPTIMIUM SET OF INITIAL PARAMETERS FOR EACH MODEL BASED ON LOWEST MAE ######

param_df.to_pickle(f'results/{experiment_folder}/param_df.pkl')

#simulate dr across multiple time points and experimental time points based on fitted parameters


params = ['K_0', 'K_1', 'K_h', 'kKP', 'n', 'a', 'b', 'T_i', 'a_reciprocal', 'k_reciprocal']
sim_df_list = []
sim_exp_df_list = []
files_exp = []
time_exp = []
model_exp = []
files_sim = []
time_sim = []
model_sim = []

#get the time points for each file
for file_name in file_names:
    drug_release_plot = np.genfromtxt(file_name, delimiter = ',')
    
    x_exp_time = drug_release_plot[:, 0] #experimental time data
    x_exp_time[x_exp_time < 0] = 0
    
    file_ID = int(file_name.split('/')[-1].split('..')[0])
    
    units = time_units[time_units['ID'] == file_ID]
    units = units['Time_units'].values[0]
  
    release_percent = drug_release_plot[:, 1] #experimental release data
    
    if units == 'mins':
        x_exp_time = x_exp_time/60
    
    elif units == 'seconds':
        x_exp_time = x_exp_time/3600

    else:
        x_exp_time = x_exp_time
    
    x_model = np.linspace(0, max(x_exp_time),200)  #time data points to use for simulated data across same time range as experimental values
    
    for model_s, model_f in zip(model_names, models):
        model_s = model_s[0].lower() + model_s[1:]
        optim_params_df = get_model_params(file_name, model_s, param_df)
        optim_param_array = optim_params_df['Optimized_Parameters'].values[0]

        #df_sim, df_model_exp - simulated data at a range of time points, predictions at experimental time points
        try:
            if len(optim_param_array) == 1:
                y_model = model_f(x_model, optim_param_array[0])
                y_model_exp = model_f(x_exp_time, optim_param_array[0])
                df_sim = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_model,
                    f"{model_s} predicted release %": y_model})
                df_model_exp = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_exp_time,
                    f"{model_s} predicted release %": y_model_exp})
                sim_df_list.append(df_sim), sim_exp_df_list.append(df_model_exp)
            
            
            elif len(optim_param_array) == 2:
                y_model = model_f(x_model, optim_param_array[0], optim_param_array[1])
                y_model_exp = model_f(x_exp_time, optim_param_array[0], optim_param_array[1])
                df_sim = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_model,
                    f"{model_s} predicted release %": y_model})
                df_model_exp = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_exp_time,
                    f"{model_s} predicted release %": y_model_exp})
                sim_df_list.append(df_sim), sim_exp_df_list.append(df_model_exp)
            
            
            else:
                y_model = model_f(x_model, optim_param_array[0], optim_param_array[1], optim_param_array[2])
                y_model_exp = model_f(x_exp_time, optim_param_array[0], optim_param_array[1], optim_param_array[2])
                df_sim = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_model,
                    f"{model_s} predicted release %": y_model})
                df_model_exp = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_exp_time,
                    f"{model_s} predicted release %": y_model_exp})
                sim_df_list.append(df_sim), sim_exp_df_list.append(df_model_exp)
        
        #handle error if recieving NaN values for optimised parameters 
        except TypeError:
            df_sim = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_model,
                    f"{model_s} predicted release %": np.nan})
            df_model_exp = pd.DataFrame({
                    'file_name': file_name,
                    'time': x_exp_time,
                    f"{model_s} predicted release %": np.nan})
            sim_df_list.append(df_sim), sim_exp_df_list.append(df_model_exp)
            continue

###process the list of dataframes into singular dataframe
num_batches = int(round(len(sim_exp_df_list)/len(model_names))) #number of dataframes to generate

sim_exp_dfs = []
sim_dfs = []
i = 0 - len(model_names)
j = 0
for df in range(num_batches):
    i += len(model_names) 
    j += len(model_names)
    #join list of dataframes horizontally for every nth column
    df_x = pd.concat(sim_exp_df_list[i:j], axis = 1) 
    df_j = pd.concat(sim_df_list[i:j], axis = 1) 
    df_x = df_x.loc[:,~df_x.columns.duplicated()]
    df_j = df_j.loc[:,~df_j.columns.duplicated()]
    sim_exp_dfs.append(df_x)
    sim_dfs.append(df_j)
sim_exp_df = pd.concat(sim_exp_dfs, axis = 0)
sim_df = pd.concat(sim_dfs, axis = 0)



###Check merging of dataframes was successful 
#check whether the length of drug_release_exp and sim_exp_df rows are equal

if drug_release_exp.shape[:1] == sim_exp_df.shape[:1]:
    
    print('reindexing sim_exp_df')
    sim_exp_df.reset_index(drop = True, inplace = True)

    if (drug_release_exp.index == sim_exp_df.index).all() == True:  #"ValueError("Lengths must match to compare") if not equal lengths
        
        #merge the experimental data with the simulated data based on index - assumes order of files and time points is the same for each file
        results_df = sim_exp_df.merge(drug_release_exp, left_index = True, right_index = True)
        print('merged sim_exp_df and drug_release_exp')
        results_df.rename(columns = {'zero_order predicted release %': 'Zero_order'}, inplace = True) 
        results_df.rename(columns = {'first_order predicted release %': 'First_order'}, inplace = True)
        results_df.rename(columns = {'higuchi predicted release %': 'Higuchi'}, inplace = True)
        results_df.rename(columns = {'korsmeyer_Peppas predicted release %': 'Korsmeyer_Peppas'}, inplace = True)
        results_df.rename(columns = {'weibull predicted release %': 'Weibull'}, inplace = True)
        results_df.rename(columns = {'reciprocal predicted release %': 'Reciprocal'}, inplace = True)
        results_df.rename(columns = {'release_percent': 'Experimental release %'}, inplace = True)

        
        if results_df['time'].equals(results_df['time (Hrs)']) == True:
            results_df.drop(columns = 'time', inplace = True)
        
        else:
            print('Experimental time points and simulated time points do not match')
        
        #check that the file_name_x and file_name_y columns are equal in each row
        if results_df['file_name_x'].equals(results_df['file_name_y']) == True:
            results_df.drop(columns = 'file_name_y', inplace = True)
            results_df.rename(columns = {'file_name_x': 'file_name'}, inplace = True)
            print('merging and checking dataframes complete!')
        
        else:
            print('File names of experimental data and simulated data do not match')
    
    else:
        print('Index of experimental data and simulated data do not match')

else:
    print('Checking for missing rows!')
    merged_df = pd.merge(sim_exp_df, drug_release_exp, on='file_name', how='outer', indicator=True)
    # Rows that are only in drug_release but not in sim_exp_df
    missing_in_sim_exp_df = merged_df[merged_df['_merge'] == 'right_only']
    print("Rows missing in sim_exp_df:")
    print(missing_in_sim_exp_df)


results_df.to_csv(f'results/{experiment_folder}/results_df.csv', index = False)
sim_df.to_csv(f'results/{experiment_folder}/sim_df.csv', index = False)

print('complete')
