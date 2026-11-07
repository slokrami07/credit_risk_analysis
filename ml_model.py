import os
import matplotlib
matplotlib.use('Agg')  # Headless backend: prevents Tkinter multi-threading crash with n_jobs=-1
import matplotlib.pyplot as plt
from numpy.random import randint
import pandas as pd 
import numpy as np 
import seaborn as sns 
import joblib 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, roc_auc_score, average_precision_score

# --- Plot saving utility ---
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_and_graphs")
os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(filename):
    """Save the current matplotlib figure in PLOT_DIR and close it."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    filepath = os.path.join(PLOT_DIR, filename)

    plt.savefig(
        filepath,
        bbox_inches="tight",
        dpi=150,
        facecolor="white"
    )

    print(f"[saved] {filepath}")
    plt.close()

def evaluate_model(model_name, model, X_test, y_test, threshold=None):
    
    y_pred_prob = model.predict_proba(X_test)[:,1]
    if threshold is not None:
        y_pred = (y_pred_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    #metrices 
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test,y_pred)
    class_report = classification_report(y_test,y_pred)

    roc_auc = roc_auc_score(y_test, y_pred_prob)

    avg_precision = average_precision_score(
        y_test,
        y_pred_prob
        )

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    print(f"{model_name} Metrics:")
    if threshold is not None: # Only print threshold if provided
        print(f"Used Threshold : {threshold:.2f}")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"F1 : {f1:.2f}")
    print()
    print(f"{model_name} Classification Report:")
    print(class_report)

    #Plotting confusion matrix
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    save_plot(f"{model_name}_confusion_matrix.png")

    #Plotting Precision-Recall curve
    precisions, recalls, threshold = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    save_plot(f"{model_name}_precision_recall_curve.png")

df = pd.read_csv("credit_risk_dataset.csv")
print(df.head())

# EDA
print(df.columns)
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df["loan_status"].unique())
print(df["loan_status"].value_counts())
sns.barplot(x="loan_status",data=df)
save_plot("loan_status_barplot.png")

#outlier check and correaltion check 
numerical_cols = df.select_dtypes(include=np.number).columns

for i in numerical_cols:
    sns.boxplot(x=df[i])
    save_plot(f"boxplot_{i}.png")

print((df['person_age'] > 100).sum())
print((df['person_emp_length'] > 60).sum())

sns.heatmap(df.corr(numeric_only=True),annot=True)
save_plot("correlation_heatmap.png")

# data validation and outlier analysis 
df_copy = df.copy()

#remove duplicate rows 
print(f"original rows : {df_copy.shape[0]}")
df_copy = df_copy.drop_duplicates()
print(f"rows after removing the duplicates :{df_copy.shape[0]}")

#we remove rows with invalid age or work experience (age < 18 and > 100 employment >60)
df_copy = df_copy[df_copy['person_age'] >=18]
df_copy = df_copy[df_copy['person_age'] <=100]

df_copy = df_copy[df_copy['person_emp_length'] <= df_copy['person_age']]
df_copy = df_copy[df_copy['person_emp_length'] <= 60]

#Remove the rows with zero or negative income and loan amount.
df_copy = df_copy[df_copy['loan_amnt'] > 0]

#check if the interest rate looks realistic.
print(f"Loan interest rate range from (min: {df_copy['loan_int_rate'].min()} and max: {df_copy['loan_int_rate'].max()})")

# feature, target and train test split 
X = df_copy.drop('loan_status', axis=1)
y = df_copy['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_model, X_val, y_train_model, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1875,
    random_state=42,
    stratify=y_train
)

numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# handling class imbalance
# smote or class weights 
# class weights 
# logistic regression has inbuild class weights but xgboost does not have in build class weight 
# we will calculate class weights here so that we can use it in xgboost 

neg, pos = np.bincount(y_train_model)
scale_weights = neg/pos

# preprocessing for logistic regression 
# numeric = median imputation + scale numeric features 
# categorical = impute + one-hot encoding 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor_lr = ColumnTransformer(
    transformers=[
        #Pipeline - for numerical cols
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),

        #Pipeline - for Categorical cols.
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

#XGBoost: Impute numeric only (No Scaling), impute + One - Hot Encoding

preprocessor_xg = ColumnTransformer(
    transformers=[
        #Pipeline - for numerical cols
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), numeric_cols),

        #Pipeline - for Categorical cols.
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

preprocessor_rf = ColumnTransformer(
    transformers=[
        #pipeline for numerical columns
        ('num', Pipeline([
            ('imputer',SimpleImputer(strategy="median"))
            ]),numeric_cols),
        #pipeline for categorical cols
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ]),categorical_cols)
    ])
preprocessor_ada_boost = ColumnTransformer(
    transformers=[
        #pipeline for numerical columns
        ('num', Pipeline([
            ('imputer',SimpleImputer(strategy="median"))
            ]),numeric_cols),
        #pipeline for categorical cols
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy="constant", fill_value="Missing")),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ]),categorical_cols)
    ])
# stratified cross validation 
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'roc_auc':'roc_auc','average_precision': 'average_precision', 'accuracy':'accuracy', 'precision':'precision', 'recall':'recall', 'f1':'f1'}

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_cv_pipeline = Pipeline([
    ('preprocessor', preprocessor_lr),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

from xgboost import XGBClassifier
xgb_cv_pipeline = Pipeline([
    ('preprocessor', preprocessor_xg),
    ('classifier', XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_weights,random_state=42
    ))
])

from sklearn.ensemble import RandomForestClassifier
rf_cv_pipeline = Pipeline([
    ('preprocessor',preprocessor_rf),
    ('classifier',RandomForestClassifier(n_estimators=100,
    criterion='gini',
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1))
])
from sklearn.ensemble import AdaBoostClassifier

ada_cv_pipeline = Pipeline([
    ('preprocessor', preprocessor_ada_boost),
    ('classifier', AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])


for cv_name, pipe in [("Logistic Regression", lr_cv_pipeline),
                       ("XGBoost", xgb_cv_pipeline),
                       ("Random Forest Classifier", rf_cv_pipeline),
                       ("AdaBoost Classifier",ada_cv_pipeline)]:
                       cv_results = cross_validate(pipe, X_train_model, y_train_model, cv=cv,scoring=scoring,n_jobs=-1)
                       print(cv_name)
                       print(f"ROC-AUC : {cv_results['test_roc_auc'].mean()}")
                       print(f"Accuracy : {cv_results['test_accuracy'].mean()}")
                       print(f"Precision : {cv_results['test_precision'].mean()}")
                       print(f"Recall : {cv_results['test_recall'].mean()}")
                       print(f"F1 : {cv_results['test_f1'].mean()}")
                       print()

baseline_model = Pipeline([
    ('preprocessor', preprocessor_lr),
    ('classifier', LogisticRegression(class_weight= "balanced"))
])
baseline_model.fit(X_train, y_train)
evaluate_model("Baseline Logistic Model", baseline_model ,X_test, y_test)

xg_model = Pipeline([
    ('preprocessor', preprocessor_xg),
    ('classifier', XGBClassifier(learning_rate=0.1,
        scale_pos_weight=scale_weights,random_state=42))
])
xg_model.fit(X_train, y_train)
rf_model = Pipeline([
    ('preprocessor',preprocessor_rf),
    ('classifier',RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced'))
])
rf_model.fit(X_train,y_train)

ada_model = Pipeline([
    ('preprocessor', preprocessor_ada_boost),
    ('classifier', AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])

ada_model.fit(X_train, y_train)

evaluate_model("baseline_logistic_model",baseline_model,X_test,y_test)


#hyperparameter tuning of xgboost model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform

param_grid_xgb = {
    "classifier__n_estimators": randint(150, 600),
    "classifier__max_depth" : randint(3,9),
    "classifier__learning_rate" : uniform(0.01, 0.5),
    "classifier__subsample": uniform(0.6,0.4), #Rows sampling
    "classifier__colsample_bytree" :uniform(0.6,0.4), #column sampling
    "classifier__min_child_weight": randint(1, 10),
    "classifier__gamma": uniform(0 , 5)
}

random_search_xgb = RandomizedSearchCV(
    xg_model,
    param_grid_xgb,
    n_iter=150,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_xgb.fit(X_train_model,y_train_model)

print(random_search_xgb.best_params_)
print(random_search_xgb.best_score_)

evaluate_model("xgboost_hyper_parameter_tuning",random_search_xgb,X_test,y_test)

param_grid_rf = [
    {
        "classifier__n_estimators": randint(100, 600),
        "classifier__max_depth": [None, 5, 10, 15, 20, 30],
        "classifier__min_samples_split": randint(2, 20),
        "classifier__min_samples_leaf": randint(1, 10),
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [True],
        "classifier__max_samples": [0.6, 0.7, 0.8, 0.9, None],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__class_weight": [
            None, "balanced", "balanced_subsample"
        ]
    },
    {
        "classifier__n_estimators": randint(100, 600),
        "classifier__max_depth": [None, 5, 10, 15, 20, 30],
        "classifier__min_samples_split": randint(2, 20),
        "classifier__min_samples_leaf": randint(1, 10),
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [False],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__class_weight": [
            None, "balanced", "balanced_subsample"
        ]
    }
]

random_search_rf = RandomizedSearchCV(
    rf_model,
    param_grid_rf,
    n_iter=150,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_rf.fit(X_train_model,y_train_model)

print(random_search_rf.best_params_)
print(random_search_rf.best_score_)

evaluate_model("random_forest_hyper_parameter_tuning",random_search_rf,X_test,y_test)

param_grid_ada = {
    "classifier__n_estimators": randint(50, 501),

    "classifier__learning_rate": loguniform(0.01, 1.0)
}

random_search_ada = RandomizedSearchCV(
    estimator=ada_model,
    param_distributions=param_grid_ada,
    n_iter=100,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_ada.fit(X_train_model, y_train_model)

print(random_search_ada.best_params_)
print(random_search_ada.best_score_)

evaluate_model("ada_boost_hyper_parameter_tuning",random_search_ada,X_test,y_test)

#1. get the probabilty of class 1 from xgboost and random forest

# ==========================================
# THRESHOLD OPTIMIZATION: XGBOOST & RANDOM FOREST
# ==========================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

import numpy as np
import matplotlib.pyplot as plt


# 1. Get probability of class 1 from validation data

probab_xgb = random_search_xgb.predict_proba(X_val)[:, 1]
probab_rf = random_search_rf.predict_proba(X_val)[:, 1]


# 2. Define threshold range

thresholds = np.linspace(0.01, 0.99, 99)


# 3. Initialize metric lists

precisions_xgb, recalls_xgb, f1_xgb = [], [], []
precisions_rf, recalls_rf, f1_rf = [], [], []


# 4. Calculate precision, recall and F1 for each threshold

for threshold in thresholds:

    # XGBoost predictions
    pred_xgb = (probab_xgb >= threshold).astype(int)

    # Random Forest predictions
    pred_rf = (probab_rf >= threshold).astype(int)

    # XGBoost metrics
    p_xgb = precision_score(y_val, pred_xgb, zero_division=0)
    r_xgb = recall_score(y_val, pred_xgb, zero_division=0)
    f_xgb = f1_score(y_val, pred_xgb, zero_division=0)

    # Random Forest metrics
    p_rf = precision_score(y_val, pred_rf, zero_division=0)
    r_rf = recall_score(y_val, pred_rf, zero_division=0)
    f_rf = f1_score(y_val, pred_rf, zero_division=0)

    precisions_xgb.append(p_xgb)
    recalls_xgb.append(r_xgb)
    f1_xgb.append(f_xgb)

    precisions_rf.append(p_rf)
    recalls_rf.append(r_rf)
    f1_rf.append(f_rf)


# 5. Find the best threshold based on F1 score

best_idx_xgb = np.argmax(f1_xgb)
best_idx_rf = np.argmax(f1_rf)

best_threshold_xgb = thresholds[best_idx_xgb]
best_threshold_rf = thresholds[best_idx_rf]


# 6. Print best thresholds and validation performance

print("\n========== XGBOOST THRESHOLD ==========")

print("Best Threshold:", best_threshold_xgb)
print("Validation Precision:", precisions_xgb[best_idx_xgb])
print("Validation Recall:", recalls_xgb[best_idx_xgb])
print("Validation F1:", f1_xgb[best_idx_xgb])


print("\n========== RANDOM FOREST THRESHOLD ==========")

print("Best Threshold:", best_threshold_rf)
print("Validation Precision:", precisions_rf[best_idx_rf])
print("Validation Recall:", recalls_rf[best_idx_rf])
print("Validation F1:", f1_rf[best_idx_rf])


# 7. Plot Precision vs Recall

plt.figure(figsize=(10, 6))

plt.plot(recalls_xgb, precisions_xgb, label="XGBoost")
plt.plot(recalls_rf, precisions_rf, label="Random Forest")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Comparison at Different Thresholds")

plt.legend()
plt.grid()
plt.savefig(
    os.path.join(PLOT_DIR, "threshold_precision_recall_comparison.png"),
    bbox_inches="tight",
    dpi=150
)
plt.close()


# 8. Plot F1 score against threshold

plt.figure(figsize=(10, 6))

plt.plot(thresholds, f1_xgb, label="XGBoost")
plt.plot(thresholds, f1_rf, label="Random Forest")

plt.axvline(
    best_threshold_xgb,
    linestyle="--",
    label=f"XGBoost Best: {best_threshold_xgb:.2f}"
)

plt.axvline(
    best_threshold_rf,
    linestyle="--",
    label=f"RF Best: {best_threshold_rf:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Classification Threshold")

plt.legend()
plt.grid()
plt.savefig(
    os.path.join(PLOT_DIR, "threshold_f1_comparison.png"),
    bbox_inches="tight",
    dpi=150
)
plt.close()


# 9. Final evaluation on untouched TEST DATA

print("\n========== FINAL TEST EVALUATION ==========")

evaluate_model(
    "XGBoost Optimized Threshold",
    random_search_xgb,
    X_test,
    y_test,
    threshold=best_threshold_xgb
)

evaluate_model(
    "Random Forest Optimized Threshold",
    random_search_rf,
    X_test,
    y_test,
    threshold=best_threshold_rf
)



# ==========================================
# PROBABILITY CALIBRATION
# XGBOOST AND RANDOM FOREST
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.metrics import (
    brier_score_loss,
    log_loss
)

from sklearn.model_selection import StratifiedKFold


# ==========================================
# 1. GET BEST TUNED MODELS
# ==========================================

best_xgb_model = random_search_xgb.best_estimator_
best_rf_model = random_search_rf.best_estimator_

# Reproducible shuffled calibration folds
calibration_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 2. INITIALIZE CALIBRATION MODELS
# ==========================================

# XGBoost - Sigmoid
calibrated_xgb_sigmoid = CalibratedClassifierCV(
    estimator=best_xgb_model,
    method="sigmoid",
    cv=calibration_cv,
    ensemble=True
)

# XGBoost - Isotonic
calibrated_xgb_isotonic = CalibratedClassifierCV(
    estimator=best_xgb_model,
    method="isotonic",
    cv=calibration_cv,
    ensemble=True
)

# Random Forest - Sigmoid
calibrated_rf_sigmoid = CalibratedClassifierCV(
    estimator=best_rf_model,
    method="sigmoid",
    cv=calibration_cv,
    ensemble=True
)

# Random Forest - Isotonic
calibrated_rf_isotonic = CalibratedClassifierCV(
    estimator=best_rf_model,
    method="isotonic",
    cv=calibration_cv,
    ensemble=True
)


# ==========================================
# 3. FIT CALIBRATION MODELS
# ==========================================

print("\n========== FITTING CALIBRATION MODELS ==========")

calibrated_xgb_sigmoid.fit(X_train_model, y_train_model)
print("XGBoost Sigmoid completed.")

calibrated_xgb_isotonic.fit(X_train_model, y_train_model)
print("XGBoost Isotonic completed.")

calibrated_rf_sigmoid.fit(X_train_model, y_train_model)
print("Random Forest Sigmoid completed.")

calibrated_rf_isotonic.fit(X_train_model, y_train_model)
print("Random Forest Isotonic completed.")


# ==========================================
# 4. GET VALIDATION PROBABILITIES
# ==========================================

# XGBoost
xgb_uncalibrated = best_xgb_model.predict_proba(X_val)[:, 1]

xgb_sigmoid = calibrated_xgb_sigmoid.predict_proba(X_val)[:, 1]

xgb_isotonic = calibrated_xgb_isotonic.predict_proba(X_val)[:, 1]


# Random Forest
rf_uncalibrated = best_rf_model.predict_proba(X_val)[:, 1]

rf_sigmoid = calibrated_rf_sigmoid.predict_proba(X_val)[:, 1]

rf_isotonic = calibrated_rf_isotonic.predict_proba(X_val)[:, 1]


# ==========================================
# 5. CALIBRATION METRICS
# ==========================================

calibration_results = {
    "XGBoost Uncalibrated": xgb_uncalibrated,
    "XGBoost Sigmoid": xgb_sigmoid,
    "XGBoost Isotonic": xgb_isotonic,

    "Random Forest Uncalibrated": rf_uncalibrated,
    "Random Forest Sigmoid": rf_sigmoid,
    "Random Forest Isotonic": rf_isotonic
}

print("\n========== CALIBRATION COMPARISON ==========")

for model_name, probabilities in calibration_results.items():

    brier = brier_score_loss(y_val, probabilities)
    loss = log_loss(y_val, probabilities)

    print(f"\n{model_name}")
    print(f"Brier Score : {brier:.6f}")
    print(f"Log Loss    : {loss:.6f}")


# ==========================================
# 6. CALIBRATION CURVES
# QUANTILE BINNING
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))


# XGBoost
xgb_models = {
    "Uncalibrated": xgb_uncalibrated,
    "Sigmoid": xgb_sigmoid,
    "Isotonic": xgb_isotonic
}

axes[0].plot(
    [0, 1], [0, 1],
    "k--",
    label="Perfect Calibration"
)

for name, probabilities in xgb_models.items():

    actual, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    axes[0].plot(
        predicted,
        actual,
        "o-",
        label=name
    )

axes[0].set_title("XGBoost Calibration")
axes[0].set_xlabel("Mean Predicted Probability")
axes[0].set_ylabel("Fraction of Positives")
axes[0].legend()
axes[0].grid(alpha=0.3)


# Random Forest
rf_models = {
    "Uncalibrated": rf_uncalibrated,
    "Sigmoid": rf_sigmoid,
    "Isotonic": rf_isotonic
}

axes[1].plot(
    [0, 1], [0, 1],
    "k--",
    label="Perfect Calibration"
)

for name, probabilities in rf_models.items():

    actual, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    axes[1].plot(
        predicted,
        actual,
        "o-",
        label=name
    )

axes[1].set_title("Random Forest Calibration")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

save_plot("model_calibration_quantile_comparison.png")


# ==========================================
# 7. BOOTSTRAP STABILITY ANALYSIS
# ==========================================

print("\n========== BOOTSTRAP STABILITY ANALYSIS ==========")

N_BOOTSTRAPS = 1000

rng = np.random.default_rng(42)

y_array = np.asarray(y_val)

# Preserve class proportions through stratified sampling
class_0_indices = np.where(y_array == 0)[0]
class_1_indices = np.where(y_array == 1)[0]


def bootstrap_calibration_comparison(
    probabilities_a,
    probabilities_b,
    name_a,
    name_b
):

    brier_differences = []
    logloss_differences = []

    for _ in range(N_BOOTSTRAPS):

        sampled_0 = rng.choice(
            class_0_indices,
            size=len(class_0_indices),
            replace=True
        )

        sampled_1 = rng.choice(
            class_1_indices,
            size=len(class_1_indices),
            replace=True
        )

        sampled_indices = np.concatenate(
            [sampled_0, sampled_1]
        )

        y_sample = y_array[sampled_indices]

        prob_a = probabilities_a[sampled_indices]
        prob_b = probabilities_b[sampled_indices]

        brier_a = brier_score_loss(y_sample, prob_a)
        brier_b = brier_score_loss(y_sample, prob_b)

        loss_a = log_loss(
            y_sample, prob_a, labels=[0, 1]
        )

        loss_b = log_loss(
            y_sample, prob_b, labels=[0, 1]
        )

        # Negative difference favors model A
        brier_differences.append(brier_a - brier_b)

        logloss_differences.append(loss_a - loss_b)

    brier_differences = np.asarray(brier_differences)
    logloss_differences = np.asarray(logloss_differences)

    brier_mean = np.mean(brier_differences)
    loss_mean = np.mean(logloss_differences)

    brier_ci = np.percentile(
        brier_differences, [2.5, 97.5]
    )

    loss_ci = np.percentile(
        logloss_differences, [2.5, 97.5]
    )

    print("\n------------------------------------------")
    print(f"{name_a} VS {name_b}")
    print("------------------------------------------")

    print("\nBrier Score Difference:")
    print(f"Mean : {brier_mean:.6f}")
    print(f"95% CI : [{brier_ci[0]:.6f}, {brier_ci[1]:.6f}]")

    print("\nLog Loss Difference:")
    print(f"Mean : {loss_mean:.6f}")
    print(f"95% CI : [{loss_ci[0]:.6f}, {loss_ci[1]:.6f}]")

    if brier_ci[1] < 0 and loss_ci[1] < 0:
        print("\nConclusion: Both metrics favor " + name_a)

    elif brier_ci[0] > 0 and loss_ci[0] > 0:
        print("\nConclusion: Both metrics favor " + name_b)

    else:
        print("\nConclusion: Improvement is inconclusive across both metrics.")


# ==========================================
# 8. XGBOOST STABILITY
# ==========================================

bootstrap_calibration_comparison(
    xgb_isotonic,
    xgb_sigmoid,
    "XGBoost Isotonic",
    "XGBoost Sigmoid"
)

bootstrap_calibration_comparison(
    xgb_isotonic,
    xgb_uncalibrated,
    "XGBoost Isotonic",
    "XGBoost Uncalibrated"
)


# ==========================================
# 9. RANDOM FOREST STABILITY
# ==========================================

bootstrap_calibration_comparison(
    rf_isotonic,
    rf_sigmoid,
    "Random Forest Isotonic",
    "Random Forest Sigmoid"
)

bootstrap_calibration_comparison(
    rf_isotonic,
    rf_uncalibrated,
    "Random Forest Isotonic",
    "Random Forest Uncalibrated"
)


# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("CALIBRATION EXPERIMENT COMPLETED")
print("==========================================")

print("Models evaluated:")
print("1. XGBoost Uncalibrated")
print("2. XGBoost Sigmoid")
print("3. XGBoost Isotonic")
print("4. Random Forest Uncalibrated")
print("5. Random Forest Sigmoid")
print("6. Random Forest Isotonic")

print("\nBootstrap iterations:", N_BOOTSTRAPS)

print("\nCalibration analysis completed successfully!")



#SHAP 
import shap 
xgb_classifier = best_xgb_model.named_steps['classifier']
X_test_transformed = best_xgb_model.named_steps['preprocessor'].transform(X_test)
feature_names = best_xgb_model.named_steps['preprocessor'].get_feature_names_out()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test_df)

#global explanation 
# in this type the shap explains which features are important and carry how much wait for the whole dataset rather than one singlle row prediction 
shap.summary_plot(shap_values, X_test_df, show=False)
save_plot("shap_summary_plot.png")

#local expalantion 
# in this the shap explains the features that carry weight for the single output given by the model 
row_index = 5

#2. Draw a waterfall plot showing how each feature pushed the prediction up or down.
shap.plots.waterfall(
    shap.Explanation(
        values = shap_values[row_index],
        base_values = explainer.expected_value,
        data = X_test_df.iloc[row_index],
        feature_names = feature_names
    ),
    show=False
)
save_plot("shap_waterfall_plot.png")

#recalculate the threshold for the calibrated xgboost model 
#Get the prob. of class 1 from calibrated XGBoost
prob = calibrated_xgb_isotonic.predict_proba(X_val)[:, 1]

#For each threshold calculate Precision and Recall
precisions, recalls, thresholds = precision_recall_curve(y_val, prob)


#Calculate F1 score for each threshold
f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

#Get the threshold that produces the highest f1-score
best = np.argmax(f1[:-1])
best_threshold = thresholds[best]

#Evaluate the XGBoost model using that threshold.
evaluate_model("Calibrated XGBoost Isotonic", calibrated_xgb_isotonic, X_test, y_test, best_threshold)

joblib.dump(calibrated_xgb_isotonic, "credit_risk_model.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")
