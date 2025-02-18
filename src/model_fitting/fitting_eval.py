from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

np.set_printoptions(precision = 3, suppress=True)

def calculate_absolute_error(mod_name: str, fitted_parameter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the absolute error between the experimental release percentage 
    and the model's predicted values.

    Parameters:
    mod_name (str): The name of the model whose predictions are being evaluated.
    fitted_parameter_df (pd.DataFrame): A DataFrame containing experimental release 
                                        percentages and model predictions.

    Returns:
    pd.DataFrame: The input DataFrame with an additional column for absolute error,
                  named as "{mod_name}Absolute_Error".
    """
    fitted_parameter_df[f"{mod_name}Absolute_Error"] = abs(
        fitted_parameter_df['Experimental release %'] - fitted_parameter_df[f"{mod_name}"]
    )
    return fitted_parameter_df


def calculate_aic(y_exp: np.ndarray, y_pred: np.ndarray, params: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) score for a given model.

    Reference: "https://github.com/hadinh1306/regscore-py/tree/master/RegscorePy"

    Parameters:
    y_exp (np.ndarray): Experimental drug release percentage values.
    y_pred (np.ndarray): Predicted drug release percentage values.
    params (int): Number of parameters used in the model.

    Returns:
    float: AIC score of the model, rounded to three decimal places. 
           Returns NaN if y_pred contains NaN values or params is NaN.
    """

    # Ensure inputs are numpy arrays
    y_exp = np.asarray(y_exp)
    y_pred = np.asarray(y_pred)

    # Check for NaN values
    if np.isnan(y_pred).any() or np.isnan(params):
        return np.nan

    # Calculate residuals
    residual = np.subtract(y_pred, y_exp)

    # Sum of Squared Errors (SSE)
    SSE = np.sum(np.power(residual, 2))

    # Number of observations
    n = len(y_exp)

    # Calculate AIC score
    aic_score = n * np.log(SSE / n) + 2 * params

    return round(aic_score, 3)

def calculate_f2(y_exp: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the similarity factor (f2) between two dissolution profiles.

    The f2 metric is commonly used in pharmaceutical sciences to compare drug release profiles.

    Parameters:
    y_exp (np.ndarray): Experimental/reference drug release percentage values.
    y_pred (np.ndarray): Predicted drug release percentage values.

    Returns:
    float: The f2 similarity factor.

    Raises:
    ValueError: If the lengths of y_exp and y_pred do not match.
    """
    
    if len(y_exp) != len(y_pred):
        raise ValueError('The length of the reference and prediction arrays must be equal')

    n = len(y_exp)

    # Sum of squared differences
    sum_squared_diff = np.sum((np.array(y_exp) - np.array(y_pred)) ** 2)

    # Calculate f2 using the formula
    f2_value = 50 * np.log10(100 / np.sqrt(1 + (sum_squared_diff / n))) 

    return f2_value


def calculate_rrmse(y_exp: np.ndarray, y_pred: np.ndarray) -> float:
    return root_mean_squared_error(y_exp,y_pred) / np.mean(y_exp) 