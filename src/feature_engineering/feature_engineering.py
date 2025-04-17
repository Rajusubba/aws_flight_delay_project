
<<<<<<< HEAD
import pandas as pd
import os
=======
import os
import pandas as pd
>>>>>>> origin/master
from sklearn.model_selection import train_test_split

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
<<<<<<< HEAD
    Perform feature engineering on the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Example feature engineering: create a new feature based on existing ones
    df_processed = df.copy()
    df_processed.drop(columns=['id', 'filght_time'], inplace=True, errors='ignore')
    cat_cols = ['Airline', 'AirportFrom', 'AirportTo']
    for col in cat_cols:
       df_processed[col + '_encoded'], _ = pd.factorize(df_processed[col])
    df_processed.drop(columns=cat_cols, inplace=True, errors='ignore')
    
    return df_processed
   
def split_and_save_dataset(df: pd.DataFrame, label: str,
                           train_filename: str = "train-v-1.csv",
                           test_filename: str = "test-v-1.csv",
                           output_dir: str = "../../data/processed/",
                           test_size: float = 0.2,
                           random_state: int = 42) -> tuple:
    
    os.makedirs(output_dir, exist_ok=True)
    
    if label not in df.columns:
        raise ValueError(f"Label '{label}' not found in DataFrame columns.")
    
    x = df.drop(columns=[label])
    y = df[label]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    train_df = x_train.copy()
    train_df[label] = y_train
    
    test_df = x_test.copy()
    test_df[label] = y_test
    
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)  
    
    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    sample_file = os.path.join("data", "raw", "Airlines_sample.csv")
    df_raw = pd.read_csv(sample_file)
    
    df_features = engineer_features(df_raw)
    

    train_df, test_df = split_and_save_dataset(df_features, label="Delay", output_dir='data/processed')
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    
=======
    Encode categoricals, drop unused cols, etc.
    """
    df = df.copy()
    # drop id, flight_time if you like:
    df.drop(columns=['id','filght_time'], errors='ignore', inplace=True)

    cat_cols = ['Airline','AirportFrom','AirportTo']
    for col in cat_cols:
        df[col + '_enc'] = df[col].astype('category').cat.codes
    df.drop(columns=cat_cols, inplace=True)

    return df

def split_and_save_data(df: pd.DataFrame, label: str, output_dir: str, test_size=0.2, random_state=42):
    """
    Split df into train/test, save to CSV under output_dir/train-*.csv & test-*.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    X = df.drop(columns=[label])
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df  = pd.concat([y_test,  X_test ], axis=1)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path  = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    return train_path, test_path
>>>>>>> origin/master
