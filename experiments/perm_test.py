from testing_scores import X, y


# libraries 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, permutation_test_score

import xgboost as xgb

cv_test_scores = pd.read_csv('results/ML_classifiers/test_CV.csv')

#get the column called mean_balaccuracy
bal_accuracy_test = cv_test_scores['mean_balaccuracy'].values

#get the index called 'XGBClassifier in bal_accuracy_test
XGB_test = bal_accuracy_test[6]

clf = xgb.XGBClassifier(random_state=1884, use_label_encoder=False, eval_metric='mlogloss')
k = 5
kf = StratifiedKFold(n_splits=k, random_state=15, shuffle=True)


score, perm_scores, pvalue = permutation_test_score(clf, X, y, scoring="balanced_accuracy", cv=k, n_permutations=1000, random_state= 15)

print("p-value", pvalue)

# Create figure with matching size
plt.figure(figsize=(4.5, 4.5))

# Histogram of permutation scores
plt.hist(
    perm_scores, bins=10, density=True, alpha=0.75, 
    color='deepskyblue', edgecolor="black")

# Add vertical line for observed balanced accuracy
plt.axvline(XGB_test, ls="--", color="black", linewidth=2)

# Compute text position
ymin, ymax = plt.ylim()
text_x = 0.3 * max(perm_scores)  # Position X for text
text_y = ymax * 1.05  # Position Y slightly above max

# Score label text
score_label = (
    f"Mean 5-fold CV Balanced Accuracy: {XGB_test:.2f}\n"
    f"(p-value: {pvalue:.3f})"
)

# Add text annotation
plt.text(text_x, text_y, score_label, fontsize=12, color="black")

# Labels with consistent font sizes
plt.xlabel("Balanced Accuracy Score", fontsize=16, fontweight='bold')
plt.ylabel("Probability Density", fontsize=16, fontweight='bold')

# Increase size of axis labels and ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Improve grid aesthetics
plt.grid(True, linestyle="--", alpha=0.7)

# Optimize layout and save
plt.tight_layout()
plt.savefig('figures/fig6B_permtest.svg', dpi=1200, bbox_inches="tight")

print('done!')
