import sqlite3
import pandas as pd

def create_combined_df(features_IVR: str, features_CQAs: list, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Combines the CQAs table with the IVR table based on user selected features.
    
    Parameters:
    - features_IVR (str): Comma-separated feature names from the IVR table.
      Example: "release_method, media_pH, media_temp_oC"
      
    - features_CQAs (list): List of feature names from the CQAs table.
      Example: ['drug_loading', 'structure_type', 'Z_average_nm']
      
    - conn (sqlite3.Connection): SQLite database connection object.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the selected features along with the formulation ID.
    """

    # Format CQAs feature names to match database column names
    formatted_CQAs = [f'formulation_CPPs_CQAs.{feature}' for feature in features_CQAs]
    
    # Convert the list of CQAs into a comma-separated string
    CQA_string = ", ".join(formatted_CQAs)
    
    # Combine IVR features and formatted CQA features into a single query string
    selected_columns = f"{features_IVR}, {CQA_string}"

    # Construct SQL query to join IVR and CQAs tables based on formulation_ID
    query = f"""
        SELECT 
            IVR.formulation_ID, 
            {selected_columns}
        FROM 
            IVR
        JOIN 
            formulation_CPPs_CQAs
        ON 
            formulation_CPPs_CQAs.formulation_ID = IVR.formulation_ID;
    """

    try:
        return pd.read_sql(query, conn)
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}")
        return None






    