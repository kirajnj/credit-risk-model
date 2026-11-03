# Credit-risk-model
Developing a credit risk management model for a bank. The task is to predict a client's default on a loan.

- **Task type**: binary classification.
- **Data**: the dataset is provided by the Skillbox platform and is not included in this repository. A dataset consisting of 3 GB of Parquet files divided into 12 parts, containing information on borrowers, credit history, and financial metrics. The target variable is flag (where 1 indicates default). Due to the dataset's size, it is processed iteratively, one part at a time.


## Workflow stages
1. **EDA**: Correlation analysis (20 highly correlated enc_paym_* features were removed) and analysis of target class imbalance.
2. **Feature engineering**:
   - credit_count - number of loans per client (aggregated by id)
   - OneHotEncoder for categorical/binarized features
   - Aggregation of records by id (summing OHE features, taking the maximum for credit_count)
   - Removal of constant features and features with zero importance
3. **Modeling** (comparison): RandomForest, LogisticRegression, LightGBM, XGBoost, HistGradientBoosting, and CatBoost. Hyperparameter tuning performed via GridSearchCV.
4. **Final model**: LGBMClassifier (class_weight='balanced', max_depth=3, n_estimators=500).

**ROC-AUC on the test set: ~0.755**
<img width="1522" height="1080" alt="cr_roc" src="https://github.com/user-attachments/assets/bd8348a1-3680-45d6-9acb-c5124fd4e8ca" />


## Artifacts
- credit_risk_assessment.ipynb - a notebook covering all stages (EDA, feature engineering, experiments, and the final model).
- final_pipeline.pkl - a trained scikit-learn Pipeline (preprocessing + LightGBM).
