import pandas as pd
import sqlite3
from collections import namedtuple

def get_formulation_data(formulation_ID: int, conn: sqlite3.Connection) -> namedtuple:
    """
    Retrieves the molar ratio and component details for a given formulation.

    Parameters:
    - formulation_ID (int): The ID of the formulation to retrieve data for.
    - conn (sqlite3.Connection): SQLite database connection object.

    Returns:
    - namedtuple: A named tuple containing:
        - molar_ratios (pd.DataFrame): DataFrame with component_ID and mole fractions.
        - total_mole_fraction (float): Sum of mole fractions.
        - component_IDs (list): List of component IDs in the formulation.
    """

    # Query formulation composition to get molar ratios of components
    query = """
        SELECT 
            component_ID, 
            molar_ratio
        FROM 
            formulation_composition
        WHERE 
            formulation_ID = ?
    """
    
    molar_ratios = pd.read_sql(query, conn, params=(formulation_ID,))
    
    if molar_ratios.empty:
        print(f"No data found for formulation_ID {formulation_ID}")
        return None

    molar_ratios['molar_ratio'] /= 100
    total_mole_fraction = molar_ratios['molar_ratio'].sum()
    
    component_IDs = molar_ratios['component_ID'].tolist()
    FormulationData = namedtuple("FormulationData", ["molar_ratios", "total_mole_fraction", "component_IDs"])
    
    return FormulationData(molar_ratios, total_mole_fraction, component_IDs)


def get_component_properties(component_IDs: list, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Retrieves component properties for a given list of component IDs.

    Parameters:
    - component_IDs (list): List of component IDs to retrieve properties for.
    - conn (sqlite3.Connection): SQLite database connection object.

    Returns:
    - pd.DataFrame: A DataFrame containing the component properties.
    """

    if not component_IDs:
        print("Warning: Empty component_IDs list provided.")
        return pd.DataFrame()  # Return empty DataFrame if no IDs are given

    # Create placeholders for SQL query based on the number of component_IDs
    placeholders = ', '.join(['?' for _ in component_IDs])

    # SQL query to fetch properties for the given component IDs
    query = f"""
        SELECT 
            *
        FROM 
            component_properties
        WHERE 
            component_ID IN ({placeholders})
    """

    try:
        # Execute query and store results as a pandas DataFrame
        component_properties = pd.read_sql(query, conn, params=component_IDs)
        return component_properties
    except sqlite3.OperationalError as e:
        print(f"SQL Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of an error


def merge_weighted_properties(molar_ratio_df: pd.DataFrame, component_df: pd.DataFrame,) -> pd.DataFrame:
    """
    Merges component properties with molar ratios and calculates a new column 'weighted_property'.

    Parameters:
    - component_df (pd.DataFrame): DataFrame containing component properties (must include 'component_ID' and 'property_value').
    - molar_ratio_df (pd.DataFrame): DataFrame containing molar ratios (must include 'component_ID' and 'molar_ratio').

    Returns:
    - pd.DataFrame: Merged DataFrame with an additional 'weighted_property' column.
    """
    # Merge component properties with molar ratios
    merged_df = pd.merge(molar_ratio_df, component_df, on='component_ID')

    # Calculate weighted property (molar_ratio * property_value)
    try:
        merged_df['weighted_property'] = merged_df['molar_ratio'] * merged_df['property_value']
    except TypeError:
        merged_df['weighted_property'] = 'ERROR!'

    return merged_df

def calculate_weighted_Tm_Mw(weighted_df: pd.DataFrame, total_mole_fraction: float) -> namedtuple:
    """
    Calculates weighted Tm (Phase transition temperature) and Mw (molecular weight).

    Parameters:
    - weighted_df (pd.DataFrame): DataFrame containing 'property_name' and 'weighted_property' columns.
    - total_mole_fraction (float): The sum of molar fractions.

    Returns:
    - namedtuple: A named tuple containing weighted Tm and Mw of each formulation
    """
    # Filter rows containing Tm and Mw properties
    weighted_Tm_df = weighted_df[weighted_df['property_name'] == 'Tm']
    weighted_Mw_df = weighted_df[weighted_df['property_name'] == 'Mw']

    try:
        weighted_Tm = weighted_Tm_df['weighted_property'].sum() / total_mole_fraction
        weighted_Mw = weighted_Mw_df['weighted_property'].sum() / total_mole_fraction

        weighted_Tm = f"{weighted_Tm:.1f}"
        weighted_Mw = f"{weighted_Mw:.1f}"
        
        Desc = namedtuple ("Desc", ["weighted_Tm", "weighted_Mw"])
        return Desc(weighted_Tm, weighted_Mw)
        
    except:
        weighted_Mw = 0 
        weighted_Tm = 0 
        Desc = namedtuple ("Desc", ["weighted_Tm", "weighted_Mw"])
        return Desc(weighted_Tm, weighted_Mw)