"""

Order to run scripts in the experiments folder.

Any script requiring an import from src, required py -m experiments.{relevant_Pythonfile.py}

1.generate_backend_df.py:
Py -m experiments.generate_backend_df
**Output**
-	Time_units.csv
-	Backend_data.csv

2. parameter_estimation.py
All time units converted into hours prior to model fitting 
**Input**
-	Time_units.csv
**Output**
-	Param_df.pkl
-	Results_df.csv
-	Sim_df.csv
-	Drug_release_exp.csv

3. calculate_fitting_metrics.py
**Input**
-	results_df.csv

**Output**
-	f2_df
-	aic_df
-	RRMSE_DF
-	AE_df
-	MAE_df

4. viz_fitting_metrics.py
**Input**
-	MAE_df.csv
-	AE_df.csv
-	AIC_df.csv
-	RRMSE_df.csv
-	F2_df.csv
-	results_df.csv
-	exp_df.csv
-	sim_df.csv
**Output**
-	fig_2_f2_boxplot.svg
-	fig_S1_RRMSE_AIC.svg
-	fig_S2_CPD.svg
-	fig_S3_f2_head.svg
-	fig_s4_f2_tail.svg

5. ks-test.py
**Input** 
-	AE_df.csv
**Output**
-	Ks_results.csv

6. generate_clustering_df.py
**Input**
-	param_df.pkl 
-	f2_df.csv
-	time_units.csv
**Output**
-	weibullparams_f2.csv
-	weibull_params.csv

7. drug_release_clustering.py
**Input**
-	weibull_params.csv
**Output**
-	PCA_KMC.csv
-	Fig_S5_screeplot.svg
-	Fig_S6PCA_KMC.svg
-	Fig_S7_WSS.svg
-	Fig_S8_silhoette_coefficient.svg
-	Fig_3b_PCAKMC.svg

8. viz_drug_release_clustering.py
**Input**
-	Weibull_df.csv
-	3_PCA_kmc.CSV
**Output** 
-	Fig_3c_drugrelease.svg
-	Fig_3a_drug_release_preclustering.svg

9. generate_ML_df.py 
**Input**
-	3_PCA_KMC.csv
-   backend_data.csv
**Output**
-	df_7_features_clean.csv
-	df_9_features_clean.csv

10. training_scores.py
**Input**
-ML_9_features_df.csv
**Output**
    -Training_scores.csv
    -Fig4_radarplot.svg

11. SHAP.py 
**Input**
-X, y -> training_scores.py
**Output**
-Fig5_SHAP.svg

12. backward_elimination.py
-X, y from training_scores.py
**Output**
-fig6A_backwardelimination.svg 

13. testing_scores.py
Stratified 5-fold cross validation
**Input**
-ML_7_features_df.csv
**Output**
-train_CV.csv
-test_CV.csv
-Each trained model as a .pkl file

14. perm_test.py
**Input**
    -testing scores -> X, y 
    -cv_test_scores 
**Output**
    -figures6B_permtest.svg

"""