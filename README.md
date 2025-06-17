# Flight Delay Classification Project

This project implements a complete machine learning pipeline to classify airline flights as either on-time or delayed. It includes data ingestion, preprocessing, feature engineering, exploratory analysis, model training, evaluation, and hyperparameter tuning.

**Data Source**
The dataset used in this project comes from Kaggle:
https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay

## Features

- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering: unit conversion, one-hot encoding, scaling
- Class imbalance handling using SMOTE
- Feature selection using ANOVA F-test
- Model comparison:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - XGBoost
- Hyperparameter tuning using GridSearchCV
- Modular, reproducible pipeline with separate scripts for each task

## Project Structure
├── 00_data_raw/

│ └── Airlines.csv

├── 01_data_processed/

│ └── processed_airlines_data.csv

├── 04_reports_and_results/

│ ├── charts/

│ │ ├── confusion_matrix_.png

│ │ └── distribution_.png

│ └── model_evaluation_report.txt

├── flight_delay_classification.py

├── data_loader.py

├── data_processing.py

├── data_splitter.py

├── eda.py

├── model_trainer.py

└── requirements.txt


## Machine Learning Workflow

The main pipeline is executed through `flight_delay_classification.py`, which orchestrates all components in the following order:

1. **Data Loading**  
   Loads the dataset from `00_data_raw/` using `data_loader.py`.

2. **Data Processing**  
   Uses `data_processing.py` to clean, engineer features, encode, and scale data. Outputs to `01_data_processed/`.

3. **Exploratory Data Analysis**  
   Generates and saves plots using `eda.py` to `04_reports_and_results/charts/`.

4. **Data Splitting**  
   `data_splitter.py` scales and splits the processed data into training and testing sets.

5. **Model Training and Evaluation**  
   Conducted by `model_trainer.py`:
   - Applies SMOTE to balance training data
   - Selects top 500 features using ANOVA F-test
   - Trains five baseline models
   - Performs 10-fold cross-validation
   - Tunes XGBoost using GridSearchCV
  
   
## Model Evaluation Results

### Initial Model Performance (on Test Set)

| Model              | Accuracy | F1-Score |
|-------------------|----------|----------|
| Logistic Regression | 0.65     | 0.65     |
| KNN                | 0.62     | 0.62     |
| Decision Tree      | 0.61     | 0.61     |
| Random Forest      | 0.67     | 0.67     |
| XGBoost            | 0.68     | 0.68     |

### F1-Scores (5-Fold Cross-Validation)

| Model              | Mean F1-Score | Std Dev  |
|-------------------|---------------|----------|
| Logistic Regression | 0.6396        | 0.0045   |
| KNN                | 0.6528        | 0.0362   |
| Decision Tree      | 0.6101        | 0.0281   |
| Random Forest      | 0.6828        | 0.0570   |
| XGBoost            | 0.6439        | 0.0179   |

### XGBoost Hyperparameter Tuning

- Best Parameters:  
  `{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 500}`

- Best Cross-Validation F1-Score: **0.6482**

### Final Tuned XGBoost Evaluation (10-Fold CV)

- Mean F1-Score: **0.6539**  
- Standard Deviation: **0.0259**

## Project Insights
Tree-based models (Random Forest and XGBoost) consistently outperformed simpler models like Logistic Regression and KNN, highlighting the complex, non-linear nature of flight delay prediction. XGBoost was selected for final tuning due to its balance of speed and accuracy.

## Outputs and Reports

Results and visualizations are stored in `04_reports_and_results/`.

- EDA visuals showing distributions of important variables
- Confusion matrices for all models
- Classification reports with performance metrics
- Cross-validation results
- Final report of the optimized XGBoost model

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/flight-delay-classification.git
   cd flight-delay-classification
   ```
2. Create and activate a virtual environment:
   
    **Linux/macOS**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    **Windows**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Add the dataset:
  Place Airlines.csv into the 00_data_raw/ folder.

This will process the data, generate plots, train models, and save evaluation reports.

## Disclaimer
This is an educational and portfolio project based on historical data. It is not intended for real-time or production use. Accurate flight delay prediction in real-world applications requires additional features and integration with real-time data sources.




