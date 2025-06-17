from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def split_and_scale_data(df, target_column='Delay', test_size=0.3, random_state=42):
    """
    Splits data into features/target, scales features, and creates train/test sets.

    Args:
        df (pd.DataFrame): The processed DataFrame.
        target_column (str): The name of the target variable.
        test_size (float): The proportion of the dataset to allocate to the test set.
        random_state (int): The random state for splitting.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    print("Splitting and scaling data...")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Normalize all features for distance based models and consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle potential NaN values after scaling
    if np.isnan(X_scaled).any():
        print("NaNs found after scaling. Applying SimpleImputer.")
        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)

    # Perform train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")
    return X_train, X_test, y_train, y_test