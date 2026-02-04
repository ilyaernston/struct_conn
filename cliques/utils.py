import pandas as pd

def load_dataframe(file_path, usecols=None):
    """Load DataFrame from CSV or Parquet file based on extension.
    
    Args:
        file_path: Path to the file (CSV or Parquet)
        usecols: List of columns to load (for memory efficiency), or None to load all
        
    Returns:
        pandas DataFrame
    """
    file_path_str = str(file_path)
    if file_path_str.endswith('.parquet'):
        return pd.read_parquet(file_path, columns=usecols)
    elif file_path_str.endswith('.csv'):
        return pd.read_csv(file_path, usecols=usecols, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Expected .csv or .parquet")