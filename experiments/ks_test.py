import pandas as pd 
from scipy.stats import ks_2samp

""" 
ks test to compare aboslute error distributions of Weibull and reciprocal fit.

Daniel Yanes | 13/02/2025 | University of Nottingham
"""

AE_df = pd.read_csv('results/fitting/AE_df.csv')

Weibull_AE = AE_df['WeibullAbsolute_Error'].to_numpy()
reciprocal_AE = AE_df['ReciprocalAbsolute_Error'].to_numpy()

ks_stat, p_value = ks_2samp(reciprocal_AE, Weibull_AE, alternative='less')

ks_results = pd.DataFrame({
    'KS Statistic': [ks_stat],
    'P-Value': [p_value]
})

ks_results.to_csv('results/fitting/ks_results.csv')


