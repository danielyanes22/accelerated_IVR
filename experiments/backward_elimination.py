from training_scores import X, y 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Define minimum number of features to consider
min_features_to_select = 1

# Initialize the classifier
clf = xgb.XGBClassifier(random_state=1884, use_label_encoder=False, eval_metric='mlogloss')

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, random_state=15, shuffle=True)

# Initialize RFECV
rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="balanced_accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2
)

# Fit RFECV
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")

# Convert RFECV results to DataFrame for easier analysis
cv_results = pd.DataFrame(rfecv.cv_results_)

cv_results.index += 1

# summarize all features
for i in range(X.shape[1]):
 print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))

# Plotting
plt.figure(figsize=(6, 4.5))
plt.plot(cv_results["mean_test_score"], marker='o', linestyle='-', color='deepskyblue')
plt.errorbar(
    x=cv_results["mean_test_score"].index,
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
    fmt='o',
    color='deepskyblue',
    capsize=5)
plt.xlabel("Number of Features Selected",  fontsize=16, fontweight='bold')
plt.ylabel("Balanced accuracy score", fontsize=16, fontweight='bold')

#increase size of axis labels and ticks 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
# Set y-axis ticks to increments of 0.1
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig('figures/fig6A_backwardelimination.svg', dpi=1200, bbox_inches='tight')