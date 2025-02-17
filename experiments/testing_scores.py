# libraries 
import pandas as pd
import numpy as np

#Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

#metrics
from sklearn.metrics import balanced_accuracy_score,  matthews_corrcoef, f1_score

#cv
from sklearn.model_selection import StratifiedKFold


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


# Function to compute metrics
def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='micro')
    return metrics




cv = StratifiedKFold(n_splits=5, random_state=15, shuffle=True)

# Train and evaluate each classifier
num_classes = len(np.unique(y))

# Initialize dictionaries to store the results
results_train = {}
results_test = {}

# Initialize lists inside the loop to avoid accumulation across classifiers
for name, clf in zip(classifier_names, classifiers):
    if name == 'XGBClassifier':
        clf.set_params(num_class=num_classes)
    
    # Store training metrics
    train_f1 = []
    train_balaccuracy = []
    train_mcc = []

    # Store test metrics
    test_f1 = []
    test_balaccuracy = []
    test_mcc = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        print(X_train.shape, X_test.shape)
        
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
    train_f1 = np.array(train_f1)
    train_balaccuracy = np.array(train_balaccuracy)
    train_mcc = np.array(train_mcc)
    
    test_f1 = np.array(test_f1)
    test_balaccuracy = np.array(test_balaccuracy)
    test_mcc = np.array(test_mcc)
    
    # Calculate mean and standard deviation for training metrics
    mean_train_f1 = np.mean(train_f1)
    mean_train_balaccuracy = np.mean(train_balaccuracy)
    mean_train_mcc = np.mean(train_mcc)
    
    sd_train_f1 = np.std(train_f1)
    sd_train_balaccuracy = np.std(train_balaccuracy)
    sd_train_mcc = np.std(train_mcc)
    
    # Calculate mean and standard deviation for test metrics
    mean_test_f1 = np.mean(test_f1)
    mean_test_balaccuracy = np.mean(test_balaccuracy)
    mean_test_mcc = np.mean(test_mcc)
    
    sd_test_f1 = np.std(test_f1)
    sd_test_balaccuracy = np.std(test_balaccuracy)
    sd_test_mcc = np.std(test_mcc)
    
    # Store the results in the dictionaries
    results_train[name] = {
        'mean_f1': mean_train_f1, 'sd_f1': sd_train_f1, 
        'mean_balaccuracy': mean_train_balaccuracy, 'sd_balaccuracy': sd_train_balaccuracy, 
        'mean_mcc': mean_train_mcc, 'sd_mcc': sd_train_mcc
    }
    
    results_test[name] = {
        'mean_f1': mean_test_f1, 'sd_f1': sd_test_f1, 
        'mean_balaccuracy': mean_test_balaccuracy, 'sd_balaccuracy': sd_test_balaccuracy, 
        'mean_mcc': mean_test_mcc, 'sd_mcc': sd_test_mcc
    }

# Convert the results dictionary to DataFrames
results_train_df = pd.DataFrame(results_train).T
results_test_df = pd.DataFrame(results_test).T

if __name__ == '__main__':
    results_train_df.to_csv('results/ML_classifiers/train_CV.csv')
    results_test_df.to_csv('results/ML_classifiers/test_CV.csv')



