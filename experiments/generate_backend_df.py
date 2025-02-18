"""

Script to query the database and generate the unprocessed backend dataframe. Users specify desired features, and the script extracts, joins
IVR, CQA, API data and calculates weighted lipid properties and saves the dataframe as a csv file in 'data/unprocessed' folder.

Daniel Yanes |  University of Nottingham

"""

import pandas as pd
from src.database import (conn, create_combined_df, formulation_ID_list, 
                          get_formulation_data, get_component_properties, 
                          merge_weighted_properties, calculate_weighted_Tm_Mw)

### Run from root directory: 'py -m experiments.generate_backend_df' ###

# ---- Step 1: Retrieve API information and Join with IVR Table ----
api_query = f"""SELECT 
                IVR.ID,
                IVR.formulation_ID, 
                formulation.API_ID
            FROM 
                IVR 
            JOIN 
                formulation 
            ON 
                formulation.ID = IVR.formulation_ID;"""

API_df = pd.read_sql(api_query, conn)
API_df = pd.DataFrame(API_df)
API_df = API_df.rename(columns = {'ID': 'IVR_ID'})

# Retrieve API names
API_name_query = f"""SELECT 
                ID,
                API_name
            FROM
                API_name;"""

#Merge API information into a single dataframe
API_df_name = pd.read_sql(API_name_query, conn)
API_df_name = API_df_name.rename(columns = {'ID': 'API_ID'})
API_df_name = pd.merge(API_df, API_df_name, on = 'API_ID')
API_df_name = API_df_name.drop(columns = ['formulation_ID'])

print('Prepared API dataframe!')

# ---- Step 2: Calculate weighted properties (Mw, Tm) for each formulation ----

weighted_df = pd.DataFrame(columns=['formulation_ID', 'weighted_Mw', 'weighted_Tm'])
n_formulations = len(formulation_ID_list)

#Process each formulation to compute weighted properties 
for formulation in formulation_ID_list:
    formulation_data = get_formulation_data(formulation, conn)
    sample_components = get_component_properties(formulation_data.component_IDs, conn)
    merged_df = merge_weighted_properties(formulation_data.molar_ratios, sample_components)
    weighted_properties_X = calculate_weighted_Tm_Mw(merged_df, formulation_data.total_mole_fraction)

    weighted_df = weighted_df._append({
    'formulation_ID': formulation,
    'weighted_Mw': weighted_properties_X.weighted_Mw,
    'weighted_Tm': weighted_properties_X.weighted_Tm
    }, ignore_index=True)

# ---- Step 3: Update Experimental Weighted Tm Values ----

# Retrieve experimental weighted Tm values
query_Tm = f"""SELECT 
        formulation_ID, 
        weighted_Tm
    FROM 
        formulation_CPPs_CQAs"""
exp_Tm = pd.read_sql(query_Tm, conn)

# Filter non-null experimental Tm values and update weighted_df with available experimental Tm values 
exp_IDs = exp_Tm['weighted_Tm'].notna()
exp_Tm = exp_Tm[exp_IDs]
weighted_df = weighted_df.set_index('formulation_ID')
exp_Tm = exp_Tm.set_index('formulation_ID')
weighted_df.update(exp_Tm)
weighted_df = weighted_df.reset_index()

print('Prepared weighted properties dataframe!')

# ---- Step 4: Generate unprocessed backend dataframe ----

# Define IVR features to be included in the dataframe
features_IVR_all = f"""IVR.ID, release_method, media_pH, media_temp_oC""" #input as string
features_CQAs_all = ['drug_loading','structure_type', 'Z_average_nm', 'PDI', 'zeta_potential'] #input as list 

df_all = create_combined_df(features_IVR_all, features_CQAs_all, conn).rename(columns = {'ID': 'IVR_ID'})

#Merge API information and weighted properties into a single dataframe
df_all_combined = pd.merge(df_all, API_df_name, on = 'IVR_ID')
df_all_combined = pd.merge(df_all_combined, weighted_df, on = 'formulation_ID')
df_all_combined.to_csv('data/unprocessed/backend_data.csv')

#Export the time units of each IVR profile (IVR_ID) to a csv file 
time_query = """
            SELECT 
                ID, Time_units
            FROM 
                IVR
     """

time_units = pd.read_sql(time_query, conn)
time_units.to_csv('data/time_units.csv', index=False)

