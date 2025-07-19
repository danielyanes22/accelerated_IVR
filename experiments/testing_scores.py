"""
This script trains and evaluates multiple classifiers (Decision Tree, SVM, Naïve Bayes, KNN, Logistic Regression, Random Forest, and XGBoost) on a preprocessed dataset 
with 7 features using 5-fold stratified cross-validation. Performance is assessed using balanced accuracy, MCC, and F1-score. The trained models are saved as `.pkl` files, 
and evaluation results are stored in CSV format.

**Inputs**  
- `ML_7_features_df.csv` (cleaned dataset with 7 features)  

**Outputs**  
- Trained machine learning models stored as pkl files in `models/`  
- Cross-validation results stored in `results/ML_classifiers/`  
  - `train_CV.csv` (training metrics)  
  - `test_CV.csv` (test metrics)  

Daniel Yanes | University of Nottingham
"""


# libraries 
import pandas as pd
import numpy as np
import pickle
import itertools

#Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder
#metrics
from sklearn.metrics import balanced_accuracy_score,  matthews_corrcoef, f1_score

#cv
from sklearn.model_selection import StratifiedKFold

#stats
from scipy.stats import wilcoxon


ML_df = pd.read_csv('data/clean/ML_7_features_df.csv')

#Split features and target variable, scale and encode target output (kinetic class)
X = ML_df.drop(columns=['cluster'])
y = ML_df['cluster']


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
le = LabelEncoder()
y = le.fit_transform(y)


# Define classifiers
classifiers = [
    DecisionTreeClassifier(random_state=1884), 
    SVC(random_state=1884, probability=True),  # SVC with probability=True to get predict_proba
    GaussianNB(), 
    KNeighborsClassifier(n_neighbors=3), 
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

cv = StratifiedKFold(n_splits=5, random_state=15, shuffle=True)

# Train and evaluate each classifier
num_classes = len(np.unique(y))

# Initialize dictionaries to store the results
results_train = {}
results_test = {}

results_test_per_fold = {}

fold_scores = {
    'balaccuracy': {},
    'f1': {},
    'mcc': {}
}

# Initialize lists inside the loop to avoid accumulation across classifiers
for name, clf in zip(classifier_names, classifiers):
    if name == 'XGBClassifier':
        clf.set_params(num_class=num_classes)
    
    # Store training and test metrics
    train_f1, train_balaccuracy, train_mcc = [], [], []
    test_f1, test_balaccuracy, test_mcc = [], [], []
        
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        
        # Calculate training predictions
        y_pred_train = clf.predict(X_train)
        
        train_f1.append(f1_score(y_train, y_pred_train, average='micro'))
        train_balaccuracy.append(balanced_accuracy_score(y_train, y_pred_train))
        train_mcc.append(matthews_corrcoef(y_train, y_pred_train))

        # Calculate test predictions
        y_pred_test = clf.predict(X_test)
        
        test_f1.append(f1_score(y_test, y_pred_test, average='micro'))
        test_balaccuracy.append(balanced_accuracy_score(y_test, y_pred_test))
        test_mcc.append(matthews_corrcoef(y_test, y_pred_test))
    
    # Convert lists to numpy arrays to calculate mean and std
    train_f1, train_balaccuracy, train_mcc = map(np.array, [train_f1, train_balaccuracy, train_mcc])
    test_f1, test_balaccuracy, test_mcc = map(np.array, [test_f1, test_balaccuracy, test_mcc])
        
    # Store per-fold test metrics for Wilcoxon test and exporting
    results_test_per_fold[name] = {
    'balaccuracy': test_balaccuracy,
    'f1': test_f1,
    'mcc': test_mcc
    }
    
    # Calculate mean/std
    results_train[name] = {
        'mean_f1': np.mean(train_f1), 'sd_f1': np.std(train_f1),
        'mean_balaccuracy': np.mean(train_balaccuracy), 'sd_balaccuracy': np.std(train_balaccuracy),
        'mean_mcc': np.mean(train_mcc), 'sd_mcc': np.std(train_mcc)
    }

    results_test[name] = {
        'mean_f1': np.mean(test_f1), 'sd_f1': np.std(test_f1),
        'mean_balaccuracy': np.mean(test_balaccuracy), 'sd_balaccuracy': np.std(test_balaccuracy),
        'mean_mcc': np.mean(test_mcc), 'sd_mcc': np.std(test_mcc)
    }
    

    model_filename = f"models/{name}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)


# Convert the average / SD results dictionary to DataFrames
results_train_df = pd.DataFrame(results_train).T
results_test_df = pd.DataFrame(results_test).T

metrics = ['balaccuracy', 'f1', 'mcc']
all_stats = []
for metric in metrics:
    print(f"\n--- {metric.upper()} ---")
    model_scores = {model: results_test_per_fold[model][metric] for model in results_test_per_fold}
    model_names = list(model_scores.keys())
    comparisons = list(itertools.combinations(model_names, 2))

    stats = []
    for m1, m2 in comparisons:
        s1, s2 = model_scores[m1], model_scores[m2]
        try:
            stat, p = wilcoxon(s1, s2, zero_method='pratt', alternative='two-sided')
            stats.append({'Model 1': m1, 'Model 2': m2, 'Statistic': stat, 'p-value': p})
        except ValueError:
            stats.append({'Model 1': m1, 'Model 2': m2, 'Statistic': np.nan, 'p-value': 1.0})  # fallback

        all_stats.append({
            'Metric': metric.upper(),
            'Model 1': m1,
            'Model 2': m2,
            'Statistic': stat,
            'p-value': p
        })
        
df_stats = pd.DataFrame(all_stats)

# Unpack results_test_per_fold into a DataFrame for per-fold results
fold_results = []
n_folds = len(next(iter(results_test_per_fold.values()))['balaccuracy'])

for model, metrics_dict in results_test_per_fold.items():
    for fold in range(n_folds):
        fold_results.append({
            'Model': model,
            'Fold': fold + 1,
            'Balanced Accuracy': metrics_dict['balaccuracy'][fold],
            'F1 Score': metrics_dict['f1'][fold],
            'MCC': metrics_dict['mcc'][fold]
        })

# Convert to DataFrame
df_fold_results = pd.DataFrame(fold_results)

if __name__ == '__main__':
    results_train_df.to_csv('results/ML_classifiers/train_CV.csv')
    results_test_df.to_csv('results/ML_classifiers/test_CV.csv')
    df_fold_results.to_csv("results/ML_classifiers/model_per_fold_results.csv", index=False)
    df_stats.to_csv("results/ML_classifiers/wilcoxon_test_summary.csv", index=False)

