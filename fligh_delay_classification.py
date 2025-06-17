import os
import data_loader
import data_processing
import data_splitter
import eda
import model_training 

# Define constants for paths
RAW_DATA_PATH = '00_data_raw/Airlines.csv'
PROCESSED_DATA_PATH = '01_data_processed/processed_airlines_data.csv'
REPORTS_DIR = '04_reports_and_results'
CHARTS_DIR = os.path.join(REPORTS_DIR, 'charts')
RESULTS_PATH = os.path.join(REPORTS_DIR, 'model_evaluation_report.txt')

def main():
    """Main pipeline to run the flight delay classification project."""
    print("Starting Flight Delay Classification Pipeline")

    # 1. Create necessary directories
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"Output directories created/verified.")

    # 2. Load Data
    df_raw = data_loader.load_raw_data(RAW_DATA_PATH)
    if df_raw is None:
        return # Exit if data loading fails
        
    # 3. Process Data and Perform EDA
    df_processed, df_sampled_for_eda = data_processing.process_data(df_raw)
    
    # Save the processed data
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to '{PROCESSED_DATA_PATH}'.")
    
    df_eda_renamed = df_sampled_for_eda.rename(columns={
        'Flight': 'TypesofAirplanes', 'Time': 'Timeofdeparture', 'Length': 'FlightLength'
    })
    eda.perform_eda(df_eda_renamed, CHARTS_DIR)

    # 4. Split and Scale Data for Modeling
    X_train, X_test, y_train, y_test = data_splitter.split_and_scale_data(df_processed)

    # 5. Train and Evaluate Models
    model_training.train_and_evaluate_models(X_train, X_test, y_train, y_test, RESULTS_PATH, CHARTS_DIR)

    print("\n--- Pipeline Execution Complete! ---")
    print(f"All outputs have been saved to the '{REPORTS_DIR}' directory.")

if __name__ == "__main__":
    main()