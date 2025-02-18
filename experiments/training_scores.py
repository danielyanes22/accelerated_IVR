"""
Calculating training scores for all classifiers trained on 9-feature dataset. 

**Inputs** 
-ML_9_features_df.csv

**Output**
-training_scores.csv
-fig4_radarplot.svg

Daniel Yanes | University of Nottingham
"""

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import matplotlib.pyplot as plt

#metrics
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score

ML_df = pd.read_csv('data/clean/ML_9_features_df.csv')


#Split features and target variable, scale and encode target output (kinetic class)
X = ML_df.drop(columns=['cluster'])
y = ML_df['cluster']


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
le = LabelEncoder()
y = le.fit_transform(y)

unique, counts = np.unique(y, return_counts=True)
print("Cluster distribution:\n", np.asarray((unique, counts)).T)

# Define classifiers
classifiers = [
    DecisionTreeClassifier(random_state=1884), 
    SVC(random_state=1884, probability=True),  # SVC with probability=True to get predict_proba
    GaussianNB(), 
    KNeighborsClassifier(), 
    LogisticRegression(random_state=1884), 
    RandomForestClassifier(random_state=1884),
    xgb.XGBClassifier(random_state=1884, use_label_encoder=False, eval_metric='mlogloss')
]

classifier_names = [
    'DecisionTreeClassifier', 'SVC', 'GaussianNB', 'KNeighborsClassifier', 
    'LogisticRegression', 'RandomForestClassifier', 'XGBClassifier'
]

unique, counts = np.unique(y, return_counts=True)
classes = np.asarray((unique)).T

classes = classes.tolist()


# Initialize a dictionary to hold the results
results = {name: {} for name in classifier_names}


# Function to compute metrics
def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='micro')
    return metrics

# Train and evaluate each classifier
num_classes = len(np.unique(y))
for name, clf in zip(classifier_names, classifiers):
    
    if name == 'XGBClassifier':
        clf.set_params(num_class=num_classes)
    
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X) if hasattr(clf, "predict_proba") else clf.decision_function(X)
    if y_prob.ndim == 1:
        y_prob = np.vstack([1-y_prob, y_prob]).T  # For binary classifiers with decision_function
    results[name] = compute_metrics(y, y_pred)

# Export results to csv 
results_df = pd.DataFrame(results).T  # Transpose for better readability
results_df = results_df.sort_values(by='Balanced Accuracy', ascending=False)

def model_eval(df, filename=None):
    # Number of variables we're plotting
    num_vars = len(df.columns)

    # Compute angle for each axis and rotate by 90 degrees
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles = [angle + np.pi/2 for angle in angles]  # Rotate by 90 degrees
    angles += angles[:1]

    # Initialize radar plot
    #fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(18, 8))
    #gs = GridSpec(1, 1, width_ratios=[1, 1], wspace=0.20)  # Adjust width ratios as needed
    
    # Radar plot
    ax_radar = fig.add_subplot( polar=True)
    
    max_mean = df.mean(axis=1).max()
    best_model_indices = df[df.mean(axis=1) == max_mean].index.tolist()
    
    for idx, row in df.iterrows():
        data = row.tolist()
        data += data[:1]
        
        # Highlight the best model
        if idx in best_model_indices:
            ax_radar.plot(angles, data, linewidth=2, linestyle='solid', label=idx, color='red')
        else:
            ax_radar.plot(angles, data, linewidth=1, linestyle='dashed', label=idx)
            ax_radar.fill(angles, data, alpha=0.25)
    
    # Labels for each variable
    ax_radar.set_yticklabels([])
    ax_radar.set_xticks(angles[:-1])
    
    
    xticks = ax_radar.set_xticklabels(df.columns, size=12, fontweight='bold')
    

    for angle in angles[:-1]:
        ax_radar.text(angle, 1.0, '1.0', horizontalalignment='center', size=9, color='black', weight='bold')
        ax_radar.text(angle, 0.8, '0.8', horizontalalignment='center', size=9, color='black', weight='bold')
        ax_radar.text(angle, 0.7, '0.7', horizontalalignment='center', size=9, color='black', weight='bold')
        ax_radar.text(angle, 0.6, '0.6', horizontalalignment='center', size=9, color='black', weight='bold')

    # Add a legend
    ax_radar.legend(loc='center left',bbox_to_anchor=(-0.5, 0.5), fontsize=12)

    
    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    results_df.to_csv('results/ML_classifiers/training_scores.csv')
    # Plot the radar chart
    fig = model_eval(results_df, 'figures/fig4_radarplot.svg')

