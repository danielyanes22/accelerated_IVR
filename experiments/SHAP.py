"""
This script computes SHAP (SHapley Additive exPlanations) values for the trained XGBoost classifier to explain feature importance for different response classes. 
It generates beeswarm plots for the "Slow" and "Fast" classes, highlighting the impact of each feature on the model's predictions.

**Inputs**  
- `X`, `y` from `training_scores.py`  

**Outputs**  
- SHAP values for feature importance in XGBoost classification  
- `figures/fig5_SHAP.svg`

Daniel Yanes | University of Nottingham
"""

# libraries 
import numpy as np
import matplotlib.pyplot as plt

#Classifiers
import xgboost as xgb

#shap library 
import shap

#get data 
from training_scores import X, y

xgb_clf = xgb.XGBClassifier(random_state=1884, use_label_encoder=False, eval_metric='mlogloss')
clf = xgb_clf.fit(X, y)

y_pred = clf.predict(X)
y = np.asarray(y, dtype=int)

class_names = ['Medium', 'Fast', 'Slow']
explainer = shap.TreeExplainer(clf)

features = list(X.columns)

shap_values = explainer.shap_values(X)


# Create a 2x2 subplot grid
fig = plt.figure(figsize=(15, 10))

# Function to adjust font size of feature names in SHAP summary plots
def adjust_font_size(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size) 

# Create the first summary plot
ax1 = fig.add_subplot(2, 2, 1)  # Specify the position of the subplot
shap.summary_plot(shap_values[2], X, feature_names = features, show = False)  #slow
adjust_font_size(ax1, 10)


# Create the second summary plot
ax2 = fig.add_subplot(2, 2, 2) # Specify the position of the subplot
shap.summary_plot(shap_values[1] , X, feature_names = features, show = False) #fast
adjust_font_size(ax2, 10)


# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('figures/fig5_SHAP.svg', dpi = 1200, bbox_inches = 'tight')