"""
This script performs a permutation test to assess the statistical significance of the XGBoost classifier's balanced accuracy trained on 7 features. 
It compares the observed test score to a distribution of scores obtained by randomly permuting the labels. The resulting p-value indicates whether the model's 
performance is significantly better than random chance. A histogram of permutation scores is generated, with a dashed line marking the observed balanced accuracy.

**Inputs**  
- `X`, `y` from `testing_scores.py`  
- `results/ML_classifiers/test_CV.csv`  

**Outputs**  
- Permutation test results including p-value  
- Statistical comparison of observed balanced accuracy   
- `figures/fig6B_permtest.svg`  

Daniel Yanes | University of Nottingham
"""


from testing_scores import X, y


# libraries 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

import xgboost as xgb


clf = xgb.XGBClassifier(random_state=1884, use_label_encoder=False, eval_metric='mlogloss')
k = 5
kf = StratifiedKFold(n_splits=k, random_state=15, shuffle=True)


#range of permutation counts
n_perms_list = [10, 25, 50, 100, 200 ,500, 750, 1000]
results = []

for n_perms in n_perms_list:
    if n_perms != 1000:
        print(f"Running test with {n_perms} permutations...")
        score, perm_scores, pval = permutation_test_score(
            clf, X, y,
            scoring="balanced_accuracy",
            cv=k,
            n_permutations=n_perms,
            random_state=15
        )
        results.append({
            "n_permutations": n_perms,
            "p_value": pval
        })
    else:
        print(f"Running final test with {n_perms} permutations...")
        score, perm_scores, pval = permutation_test_score(
            clf, X, y,
            scoring="balanced_accuracy",
            cv=kf,
            n_permutations=n_perms,
            random_state=15
        )
        results.append({
            "n_permutations": n_perms,
            "p_value": pval
        })

        # Create dataframe of permutation scores
        df_perms = pd.DataFrame({
            'permutation_score': perm_scores,
            'type': 'permutation'  # sets all rows to 'permutation'
        })

        # Add the observed score as its own row
        df_observed = pd.DataFrame({
            'permutation_score': [score],
            'type': 'observed'
        })

        # Combine into a single dataframe
        df_1000 = pd.concat([df_perms, df_observed], ignore_index=True)

        # Save
        df_1000.to_csv("results/ML_classifiers/permutation_scores_1000.csv", index=False)



#store stability df
df_stability = pd.DataFrame(results)
df_stability.to_csv("results/ML_classifiers/p_values.csv", index=False)

# Plot p-value convergence
plt.figure(figsize=(6, 4.5))
plt.plot(df_stability["n_permutations"], df_stability["p_value"], marker='o', color='deepskyblue')
plt.axhline(y=df_stability['p_value'].iloc[-1], ls='--', color='black', label=f'Final p = {df_stability["p_value"].iloc[-1]:.3f}')
plt.xlabel("Number of Permutations", fontsize=14)
plt.ylabel("Empirical p-value", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.legend()
plt.savefig("figures/pvalue_stability_single_run.svg", dpi=1200, bbox_inches="tight")

print('done plotting p-value stability!')


# Separate permutation scores and observed score
perm_scores = df_1000[df_1000["type"] == "permutation"]["permutation_score"].values
observed_score = df_1000[df_1000["type"] == "observed"]["permutation_score"].values[0]

# Recalculate empirical p-value
pval = (sum(perm_scores >= observed_score) + 1) / (len(perm_scores) + 1)

# Create figure
plt.figure(figsize=(4.5, 4.5))

# Histogram
plt.hist(perm_scores, bins=10, density=True, alpha=0.75, 
         color='deepskyblue', edgecolor="black")

# Observed score line
plt.axvline(observed_score, ls="--", color="black", linewidth=2)

# Text annotation
ymin, ymax = plt.ylim()
text_x = 0.3 * max(perm_scores)
text_y = ymax * 1.05
score_label = (
    f"Mean 5-fold CV Balanced Accuracy: {observed_score:.2f}\n"
    f"(p-value: {pval:.3f})"
)
plt.text(text_x, text_y, score_label, fontsize=12, color="black")

# Labels
plt.xlabel("Balanced Accuracy Score", fontsize=16, fontweight='bold')
plt.ylabel("Probability Density", fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Final touches
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig('figures/fig6B_permtest.svg', dpi=1200, bbox_inches="tight")
plt.show()

print('done plotting histogram!')

