import pandas as pd

def load_raw_data(filepath):
    """
    Loads raw data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None