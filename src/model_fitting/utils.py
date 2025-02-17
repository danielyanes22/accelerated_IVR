import numpy as np 
import scipy as sp 
import pandas as pd
import os


#kinetic models below for fitting to drug_release data
def zero_order (t, ko):
   return ko*t

def first_order (t, k1):
    return 100* (1-np.exp(-k1*t))

def higuchi (t, kh):
    return kh*(t**0.5)

def korsmeyer_Peppas (t, kKP, n):
    return kKP*(t**n)

def weibull (t, a, b):   
    return 100 * (1 - np.exp(-((t)**b)/a)) 

def reciprocal (t, k, a):
    return (k*t)/(1 + a*t)


def get_model_params(file_name: str, model_name: str, df_name) -> np.ndarray:
    """
    Retrieve optimized parameters for a given model from a DataFrame.
    
    Parameters:
    - file_name: Name of the file.
    - model_name: Name of the release model.
    - df_name: DataFrame containing parameter data.
    
    Returns:
    - DataFrame slice with the relevant fitted parameters.
    """
    return df_name[(df_name['File_Name'] == file_name) & (df_name['Model'] == model_name)]


#Generate list of file names that pass quality appraisal process 
quality_report_initial = pd.read_csv('data/quality_reporting.csv')
good_quality = quality_report_initial[(quality_report_initial['Unnamed: 3'] == 'Good') | (quality_report_initial['Unnamed: 3'] == 'Medium')]
final_quality = good_quality[(good_quality ['Performance bias '] == 'Yes') & (good_quality['Unnamed: 5'] == 'Yes')]
file_list = final_quality['Unnamed: 1'].tolist()
file_names_quality = [str(f) + '..csv' for f in file_list]
folder_path = 'data/drug_release'
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('..csv')]
# Get the list of files searched for using the 'targeted string search method'. These files are not in the quality report as 
# they already pass the quality searching process
index = file_names.index(file_names_quality[-1])
last_file = file_names[index]
last_file_number = int(last_file.split('..')[0])
new_files = sorted(
    [f for f in file_names if int(f.split('..')[0]) > last_file_number],
    key=lambda x: int(x.split('..')[0]))

#combine the two lists of file names
file_names = file_names_quality + new_files
file_names = [os.path.normpath(os.path.join(folder_path, f)) for f in file_names]

#final list of file names that pass quality appraisal process
file_names = [s.replace('\\', '/') for s in file_names]

#store model functions and names in list for iteration when fitting models to data
models = [zero_order, first_order, higuchi, korsmeyer_Peppas, weibull, reciprocal] #model functions 
model_names = ['Zero_order', 'First_order', 'Higuchi', 'Korsmeyer_Peppas', 'Weibull', 'Reciprocal']  