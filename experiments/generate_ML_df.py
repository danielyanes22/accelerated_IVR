"""
Script to join drug release profile kinetic class assigned by PCA-KMC to feature inputs
and preprocess the data for ML modeling.

Outputs:
- df_7_features_clean: Processed dataset with 7 features (excludes 'release_method' and 'structure_type').
- df_9_features_clean: Processed dataset with 9 features (includes 'release_method' and 'structure_type').
"""

import pandas as pd

# Load drug release profile assignment and back end data 
df_clust = pd.read_csv("results/clustering/3_PCA_KMC.csv")
df_backend = pd.read_csv("data/unprocessed/backend_data.csv")

# Standardise ID column names suitable for joining - ID corresponds to IVR_ID = file name of the digitised drug release profile
df_clust.rename(columns={"file": "ID"}, inplace=True)
df_backend.rename(columns={"IVR_ID": "ID"}, inplace=True)

df_clust["ID"] = df_clust["ID"].astype(int)
df_backend.drop(columns=["Unnamed: 0"], inplace=True)

# Merge drug release assignment to backend data
df_join = df_clust.join(df_backend.set_index("ID"), on="ID")

print("Shape after joining:", df_join.shape)

def move_columns_to_end(df, columns):
    return df[[col for col in df.columns if col not in columns] + columns]
df_join = move_columns_to_end(df_join, ["alpha", "beta", "PC1", "PC2", "cluster"])

# Drop columns that are not useful for ML modeling and remove rows with missing values
drop_cols = [
    "cluster_name", "color_order", "formulation_ID", "PC1", "PC2", "ID",
    "alpha", "beta", "API_ID", "zeta_potential", "PDI"
]
df_modelling = df_join.drop(columns=drop_cols, axis=1)
df_clean = df_modelling.dropna()
print("Shape after dropping NaNs:", df_clean.shape)


#get number distribution of unique values in categorical columns
print("Number of unique values in 'structure_type':", df_clean["release_method"].value_counts())
print("Number of unique values in 'structure_type':", df_clean["structure_type"].value_counts())
print('Number of unique values in "API_name":', df_clean["API_name"].value_counts())

#Ensure consistent label encoding despite how data is loaded / pandas version used 
release_methods = ['Float A Lyzer', 'Dialysis', 'Dispersion releaser', 'Dowex 50 cation exchange', 'ModifiedUSP-4', 'Sample and Separate']
structure_types = ['LUV', 'MLV', 'SUV']
API_names = ['BAY 87-2243', 'amoxicillin', 'amphotericin B', 'atenolol', 'bupivacaine', 'doxorubicin', 'methotrexate', 'propranolol', 'sulforhodamine B', 'temoporfin', 'tetrodotoxin']

# Create 7-feature dataset (exclude 'release_method' and 'structure_type') and encode categorical variables for both datasets
df_7_features = df_clean.drop(columns=["release_method", "structure_type"])

#label encode categorical features 


df_clean['release_method'] = df_clean['release_method'].astype('category').cat.set_categories(
    release_methods, ordered=True).cat.codes

df_clean['structure_type'] = df_clean['structure_type'].astype('category').cat.set_categories(
    structure_types, ordered=True).cat.codes

df_clean['API_name'] = df_clean['API_name'].astype('category').cat.set_categories(
    API_names, ordered=True).cat.codes

df_clean.rename(columns={"API_name": "API_type"}, inplace=True)

df_7_features["API_name"] = df_7_features["API_name"].astype("category").cat.codes
df_7_features.rename(columns={"API_name": "API_type"}, inplace=True)

# Adjust cluster labels to avoid zero indexing and remove duplicate rows
df_clean["cluster"] += 1
df_7_features["cluster"] += 1

df_7_features_clean = df_7_features.drop_duplicates(subset=df_7_features.columns[:-1])
df_9_features_clean = df_clean.drop_duplicates(subset=df_clean.columns[:-1])

print("Final shape of 7-feature dataset:", df_7_features_clean.shape)
print("Final shape of 9-feature dataset:", df_9_features_clean.shape)

df_7_features_clean.to_csv("data/clean/ML_7_features_df.csv", index=False)
df_9_features_clean.to_csv("data/clean/ML_9_features_df.csv", index=False)