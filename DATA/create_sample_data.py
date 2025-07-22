import pandas as pd
import numpy as np
import json
import os

def create_sample_data():
    """
    Create a sample dataset for regression testing.
    
    This generates a small synthetic dataset with mixed data types:
    - Numerical (age, income)
    - Categorical (education, employed)
    - DateTime (join_date)
    """
    print("Creating sample data for regression testing...")
    
    # Create a sample DataFrame with mixed data types
    np.random.seed(42)  # For reproducibility
    sample_df = pd.DataFrame({
        'age': np.random.randint(18, 90, size=10),
        'income': np.random.randint(20000, 100000, size=10),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=10),
        'employed': np.random.choice([True, False], size=10),
        'join_date': pd.date_range(start='2020-01-01', periods=10)
    })
    
    # Create directories
    os.makedirs('data/test_sample/parquet', exist_ok=True)
    os.makedirs('data/test_sample/config', exist_ok=True)
    os.makedirs('data/test_sample/batched_output', exist_ok=True)
    
    # Save as parquet
    parquet_path = 'data/test_sample/parquet/sample.parquet'
    sample_df.to_parquet(parquet_path)
    print(f"Saved parquet file to {parquet_path}")
    
    # Create a configuration file
    config = {
        "table_name": "sample_data",
        "table_description": "Sample data for testing VAE model",
        "column_types": {
            "age": "numerical",
            "income": "numerical",
            "education": "categorical",
            "employed": "categorical",
            "join_date": "datetime"
        },
        "base_name": "test_sample",
        "table_metadata": "This is a sample dataset for regression testing"
    }
    
    # Save config as JSON
    config_path = 'data/test_sample/config/sample.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config file to {config_path}")
    
    print("Sample data creation complete")
    return sample_df, config

if __name__ == "__main__":
    sample_df, config = create_sample_data()
    
    # Display sample data
    print("\nSample Data:")
    print(sample_df.head())
    
    # Display config
    print("\nConfiguration:")
    print(json.dumps(config, indent=2)) 