import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def process_data(df_raw, sample_size=100000, random_state=42):
    """
    Cleans, preprocesses, and engineers features for the flight data.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame.
        sample_size (int or None): The number of rows to sample. If None, use all rows.
        random_state (int): The random state for sampling.

    Returns:
        pd.DataFrame: The fully processed and feature-engineered DataFrame.
    """
    print(f"Processing data: renaming, encoding, and scaling...")

    # Sample and Rename
    if sample_size is not None:
        df_sampled = df_raw.sample(n=sample_size, random_state=random_state)
    else:
        df_sampled = df_raw.copy()
    df_sampled = df_sampled.rename(columns={
        'Flight': 'TypesofAirplanes',
        'Time': 'Timeofdeparture',
        'Length': 'FlightLength'
    })

    # Feature Engineering (Unit Conversion)
    df_processed = df_sampled.copy()
    df_processed['Timeofdeparture'] = (df_processed['Timeofdeparture'] / 60).round(2)
    df_processed['FlightLength'] = (df_processed['FlightLength'] / 60).round(2)
    print("Converted time units from minutes to hours.")

    # One-Hot Encoding
    categorical_cols = ['Airline', 'AirportFrom', 'AirportTo']
    encoded_dfs = []

    for col in categorical_cols:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformed_data = enc.fit_transform(df_processed[[col]])
        encoded_df = pd.DataFrame(transformed_data, columns=enc.get_feature_names_out([col]))
        encoded_df.reset_index(drop=True, inplace=True)
        encoded_dfs.append(encoded_df)

    # Combine encoded data and drop original columns
    df_processed.reset_index(drop=True, inplace=True)
    df_processed = df_processed.drop(columns=categorical_cols)
    final_df = pd.concat([df_processed] + encoded_dfs, axis=1)

    # Scaling (MinMaxScaler on specific columns)
    scaler_fl = MinMaxScaler()
    final_df['FlightLength'] = scaler_fl.fit_transform(final_df[['FlightLength']])

    scaler_tod = MinMaxScaler()
    final_df['Timeofdeparture'] = scaler_tod.fit_transform(final_df[['Timeofdeparture']])
    print("Scaled 'FlightLength' and 'Timeofdeparture'.")

    print("Data processing complete.")
    return final_df, df_sampled  