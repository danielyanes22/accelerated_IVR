from sklearn.metrics import root_mean_squared_error
import numpy as np

np.set_printoptions(precision = 3, suppress=True)

def calculate_absolute_error(mod_name, fitted_parameter_df):
    fitted_parameter_df[f"{mod_name}Absolute_Error"] = abs(fitted_parameter_df['Experimental release %'] - fitted_parameter_df[f"{mod_name}"])
    return fitted_parameter_df

def calculate_aic(y_exp, y_pred, params):
    """Return AIC value for a given model and file.
    #REF "https://github.com/hadinh1306/regscore-py/tree/master/RegscorePy"
    Input:
        y_exp: experimental drug release % values
        y_pred: predicted drug release % values
        param (int): number of parameters used in the model
    
    Output: 
        aic_score: int or float AIC score of the model 
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


def calculate_f2(y_exp, y_pred):
    if len (y_exp) != len(y_pred):
        raise ValueError('The length of the reference and prediction arrays must be equal')
    n = len(y_exp)
    # Sum of squared differences
    sum_squared_diff = np.sum((np.array(y_exp) - np.array(y_pred)) ** 2)
    # Calculate f2 using the formula
    f2_value = 50 * np.log10(100 / np.sqrt(1 + (sum_squared_diff / n))) 
    return f2_value

def calculate_rrmse(y_exp, y_pred):
    return root_mean_squared_error(y_exp,y_pred) / np.mean(y_exp) 