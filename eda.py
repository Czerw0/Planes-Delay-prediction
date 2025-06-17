import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(df, output_dir):
    """
    Performs EDA and saves plots and summary to the specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        output_dir (str): The directory to save charts and summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Performing Exploratory Data Analysis (EDA)...")

    # Save summary statistics
    summary = df.describe(include='all')
    summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    df.isnull().sum().to_csv(os.path.join(output_dir, 'missing_values.csv'))

    # Distribution of Flight Lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='FlightLength', kde=True)
    plt.title('Distribution of Flight Lengths (in minutes)')
    plt.xlabel('Length (minutes)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'distribution_flight_length.png'))
    plt.close()

    # Boxplot of Flight Lengths by Delay
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Delay', y='FlightLength')
    plt.title('Flight Length by Delay Status')
    plt.xlabel('Delay (0=On Time, 1=Delayed)')
    plt.ylabel('Flight Length (minutes)')
    plt.savefig(os.path.join(output_dir, 'boxplot_flight_length_by_delay.png'))
    plt.close()

    # Distribution of Departure Times
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Timeofdeparture', kde=True)
    plt.title('Distribution of Departure Times (in minutes from midnight)')
    plt.xlabel('Time of Departure (minutes from midnight)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'distribution_departure_time.png'))
    plt.close()

    # Countplot of Airplane Types
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='TypesofAirplanes', order=df['TypesofAirplanes'].value_counts().index)
    plt.title('Distribution of Airplane Types')
    plt.xlabel('Airplane Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_airplane_types.png'))
    plt.close()

    # Distribution of Delays
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Delay')
    plt.title('Distribution of Delays')
    plt.xlabel('Flight Status')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['On Time', 'Delayed'])
    plt.savefig(os.path.join(output_dir, 'distribution_delays.png'))
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Pairplot for selected features
    selected_features = ['FlightLength', 'Timeofdeparture', 'Delay']
    sns.pairplot(df[selected_features], hue='Delay', diag_kind='kde')
    plt.savefig(os.path.join(output_dir, 'pairplot_selected_features.png'))
    plt.close()

    print(f"EDA charts and summary saved to '{output_dir}'.")