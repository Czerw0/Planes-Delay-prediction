import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def train_and_evaluate_models(X_train, X_test, y_train, y_test, results_path, charts_path):
    """
    Balances data, selects features, trains, predicts, and evaluates models.
    """
    # 1. Data Balancing with SMOTE
    print("Balancing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Applied SMOTE. New training set size: {X_train_smote.shape}")

    # 2. Feature Selection
    print("Selecting top 500 features with ANOVA F-test...")
    selector = SelectKBest(f_classif, k=500)
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test)
    print(f"Feature selection complete. Number of features: {X_train_selected.shape[1]}")

    # 3. Model Training
    print("Training baseline models...")
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000, random_state=0),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "XGBoost": XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss')
    }
    
    with open(results_path, 'w') as f:
        f.write("--- MODEL EVALUATION REPORT ---\n\n")
        f.write("--- Initial Model Performance ---\n\n")

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_selected, y_train_smote)
            y_pred = model.predict(X_test_selected)
            
            # Save classification report
            f.write(f"--- {name} ---\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n\n")
            
            # Save confusion matrix
            ConfusionMatrixDisplay.from_estimator(model, X_test_selected, y_test)
            plt.title(f"Confusion Matrix - {name}")
            plt.savefig(os.path.join(charts_path, f'confusion_matrix_{name.replace(" ", "_")}.png'))
            plt.close()

    print("Initial model evaluation complete. Reports and matrices saved.")

    # 4. Cross-Validation 
    print("Performing 5-fold cross-validation...")
    with open(results_path, 'a') as f:
        f.write("\n--- 5-Fold Cross-Validation Scores (F1-Score) ---\n\n")
        for name, model in models.items():
            scores = cross_val_score(model, X_train_selected, y_train_smote, cv=5, scoring='f1', n_jobs=-1)
            f.write(f"{name}: Mean F1-score = {scores.mean():.4f}, Std = {scores.std():.4f}\n")
    print("Cross-validation results saved.")
    
    # 5. Hyperparameter Tuning for XGBoost 
    print("Performing GridSearchCV on XGBoost (this may take a while)...")
    grid_xgb = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 8],
        'colsample_bytree': [0.7, 1.0]
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'),
        grid_xgb, cv=3, scoring='f1', n_jobs=-1
    )
    xgb_grid.fit(X_train_selected, y_train_smote)
    
    with open(results_path, 'a') as f:
        f.write("\n\n--- XGBoost GridSearchCV Results ---\n")
        f.write(f"Best parameters: {xgb_grid.best_params_}\n")
        f.write(f"Best F1-score from CV: {xgb_grid.best_score_:.4f}\n")
    print(f"GridSearchCV complete. Best params: {xgb_grid.best_params_}")
    
    # 6. Final Model Evaluation
    best_xgb = xgb_grid.best_estimator_
    cv_scores_final = cross_val_score(best_xgb, X_train_selected, y_train_smote, cv=5, scoring='f1', n_jobs=1)
    
    with open(results_path, 'a') as f:
        f.write("\n\n--- Final Tuned XGBoost Model Evaluation (10-Fold CV) ---\n")
        f.write(f"Mean F1-score: {cv_scores_final.mean():.4f}\n")
        f.write(f"Standard deviation: {cv_scores_final.std():.4f}\n")
    print(f"Final evaluation of tuned XGBoost saved. Mean F1: {cv_scores_final.mean():.4f}")