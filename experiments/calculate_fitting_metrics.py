"""

Script to calculate the fitting metrics (AE, MAE, AIC, F2, RRMSE) for each model and save the results in a csv file in 'results/fitting' folder.

Daniel Yanes | University of Nottingham

"""

from src.model_fitting import (model_names, calculate_f2, 
                               calculate_rrmse, calculate_aic, 
                               calculate_absolute_error)
import pandas as pd

experiment_folder = 'fitting'

results_df = pd.read_csv(f'results/{experiment_folder}/results_df.csv')

# Group by 'File Name'
grouped = results_df.groupby('file_name')
# Store the AIC, F2  results in a list
F2_results = []
aic_results = []
rrmse = []

# Iterate over each group (each file)
for file_name, group in grouped:
    y_exp = group['Experimental release %']
    y_pred_zero = group['Zero_order']
    y_pred_first = group['First_order']
    y_pred_higuchi = group['Higuchi']
    y_pred_KP = group['Korsmeyer_Peppas']
    y_pred_weibull = group['Weibull']
    y_pred_reciprocal = group['Reciprocal']
    
    
    # Calculate AIC for each model
    aic_zero = calculate_aic(y_exp, y_pred_zero, 1)
    aic_first =calculate_aic(y_exp, y_pred_first, 1)
    aic_higuchi = calculate_aic(y_exp, y_pred_higuchi, 1)
    aic_KP = calculate_aic(y_exp, y_pred_KP, 2)
    aic_weibull = calculate_aic(y_exp, y_pred_weibull, 3)
    aic_reciprocal = calculate_aic(y_exp, y_pred_reciprocal, 1)
    
    #Calculate F2 for each model
    f2_zero = calculate_f2(y_exp, y_pred_zero)
    f2_first = calculate_f2(y_exp, y_pred_first)
    f2_higuchi = calculate_f2(y_exp, y_pred_higuchi)
    f2_KP = calculate_f2(y_exp, y_pred_KP)
    f2_weibull = calculate_f2(y_exp, y_pred_weibull)
    f2_reciprocal = calculate_f2(y_exp, y_pred_reciprocal)
    
    # Calculate RRMSE for each model
    rrmse_zero = calculate_rrmse(y_exp, y_pred_zero)
    rrmse_first = calculate_rrmse(y_exp, y_pred_first)
    rrmse_higuchi = calculate_rrmse(y_exp, y_pred_higuchi)
    rrmse_KP = calculate_rrmse(y_exp, y_pred_KP)
    rrmse_weibull = calculate_rrmse(y_exp, y_pred_weibull)
    rrmse_reciprocal = calculate_rrmse(y_exp, y_pred_reciprocal)
    
    # Append results to the list
    aic_results.append({
        'File Name': file_name,
        'Zero Order': aic_zero,
        'First Order': aic_first,
        'Higuchi': aic_higuchi,
        'Korsmeyer-Peppas': aic_KP,
        'Weibull': aic_weibull,
        'Reciprocal': aic_reciprocal})

    F2_results.append({
        'File Name': file_name,
        'Zero Order': f2_zero,
        'First Order': f2_first,
        'Higuchi': f2_higuchi,
        'Korsmeyer-Peppas': f2_KP,
        'Weibull': f2_weibull,
        'Reciprocal': f2_reciprocal})
    
    rrmse.append({
        'File Name': file_name,
        'Zero Order': rrmse_zero,
        'First Order': rrmse_first,
        'Higuchi': rrmse_higuchi,
        'Korsmeyer-Peppas': rrmse_KP,
        'Weibull': rrmse_weibull,
        'Reciprocal': rrmse_reciprocal})
    
# Convert the results to a DataFrame
aic_df = pd.DataFrame(aic_results)
F2_df = pd.DataFrame(F2_results)
RRMSE_df = pd.DataFrame(rrmse)

F2_df.to_csv(f'results/{experiment_folder}/F2_df.csv')
aic_df.to_csv(f'results/{experiment_folder}/aic_df.csv')
RRMSE_df.to_csv(f'results/{experiment_folder}/RRMSE_df.csv')


zero_order = calculate_absolute_error(model_names[0], results_df)
first_order = calculate_absolute_error(model_names[1], results_df)
higuchi = calculate_absolute_error(model_names[2], results_df)
kp = calculate_absolute_error(model_names[3], results_df)
weibull = calculate_absolute_error(model_names[4], results_df)
reciprocal = calculate_absolute_error(model_names[5], results_df)

AE_df = results_df[['file_name','Zero_orderAbsolute_Error','First_orderAbsolute_Error',
                    'HiguchiAbsolute_Error','Korsmeyer_PeppasAbsolute_Error',
                    'WeibullAbsolute_Error','ReciprocalAbsolute_Error']]

MAE_df = AE_df.groupby('file_name')[['Zero_orderAbsolute_Error','First_orderAbsolute_Error',
                    'HiguchiAbsolute_Error','Korsmeyer_PeppasAbsolute_Error',
                    'WeibullAbsolute_Error','ReciprocalAbsolute_Error']].mean()


AE_df.to_csv(f'results/{experiment_folder}/AE_df.csv')
MAE_df.to_csv(f'results/{experiment_folder}/MAE_df.csv')

print(f'Calculated metrics saved to results/{experiment_folder}/')