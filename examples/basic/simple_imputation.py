"""
Simple imputation example showing basic usage of AutoFillGluon.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autofillgluon import Imputer
from autofillgluon.utils import calculate_missingness_statistics

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(n_rows=None):
    """Load the Titanic dataset and prepare it for the imputation example."""
    # Load the Titanic dataset
    titanic = sns.load_dataset('titanic')
    
    # Select a subset of columns for the example
    cols_to_use = ['age', 'fare', 'sex', 'class', 'embarked', 'survived']
    
    # For evaluation purposes, we create a "complete" dataset by dropping rows with any missing values.
    df_complete = titanic[cols_to_use].dropna().reset_index(drop=True)
    
    if n_rows is not None:
        df_complete = df_complete.head(n_rows)
        
    # Create a copy that will have missing values introduced
    df_missing = df_complete.copy()
    
    # Introduce missingness artificially to demonstrate and evaluate the imputer.
    # In a real-world scenario, you would use your dataset that already has missing values.
    mask = np.random.rand(*df_missing.shape) < 0.15
    df_missing = df_missing.mask(mask)
    
    # The 'survived' column is an integer, but we'll treat it as categorical for this example.
    df_missing['survived'] = df_missing['survived'].astype('category')
    df_complete['survived'] = df_complete['survived'].astype('category')
    
    return df_missing, df_complete

def main():
    # Generate example data
    print("Loading and preparing example data...")
    df_missing, df_complete = load_and_prepare_data(200)
    
    # Show missingness statistics
    missing_stats = calculate_missingness_statistics(df_missing)
    print("\nMissingness statistics:")
    for col, stats in missing_stats.items():
        print(f"{col}: {stats['count_missing']} missing values ({stats['percent_missing']:.1f}%)")
    
    # Initialize imputer with conservative settings for quick example
    print("\nInitializing imputer...")
    imputer = Imputer(
        num_iter=1,
        time_limit=30,
        presets=['medium_quality']
    )
    
    # Fit imputer on data with missing values
    print("\nFitting imputer...")
    df_imputed = imputer.fit(df_missing)
    
    # Evaluate imputation quality
    print("\nEvaluating imputation quality...")
    
    # For numeric columns, we can calculate correlation
    for col in ['age', 'fare']:
        # Find indices with missing values in the original data
        missing_mask = df_missing[col].isnull()
        if missing_mask.sum() > 0:
            # Get true and imputed values
            true_vals = df_complete.loc[missing_mask, col]
            imputed_vals = df_imputed.loc[missing_mask, col]
            
            # Calculate correlation
            corr = np.corrcoef(true_vals, imputed_vals)[0, 1]
            print(f"Correlation for {col}: {corr:.4f}")
            
            # Calculate mean absolute error
            mae = np.abs(true_vals - imputed_vals).mean()
            print(f"Mean absolute error for {col}: {mae:.4f}")
    
    # For categorical columns, we can calculate accuracy
    for col in ['sex', 'class', 'embarked', 'survived']:
        # Find indices with missing values in the original data
        missing_mask = df_missing[col].isnull()
        if missing_mask.sum() > 0:
            # Get true and imputed values
            true_vals = df_complete.loc[missing_mask, col]
            imputed_vals = df_imputed.loc[missing_mask, col]
            
            # Calculate accuracy
            accuracy = (true_vals == imputed_vals).mean()
            print(f"Accuracy for {col}: {accuracy:.4f}")
    
    # Plot a comparison for age
    plt.figure(figsize=(10, 6))
    missing_mask = df_missing['age'].isnull()
    if missing_mask.sum() > 0:
        plt.scatter(df_complete.loc[missing_mask, 'age'], 
                    df_imputed.loc[missing_mask, 'age'], 
                    alpha=0.7)
        plt.plot([df_complete['age'].min(), df_complete['age'].max()], 
                [df_complete['age'].min(), df_complete['age'].max()], 
                'r--')
        plt.xlabel('True Age')
        plt.ylabel('Imputed Age')
        plt.title('True vs Imputed Values for Age')
        plt.tight_layout()
        plt.savefig('age_imputation.png')
        print("\nSaved visualization to 'age_imputation.png'")
    
    # Save imputer models for future use
    print("\nSaving imputer models...")
    imputer.save_models('example_imputer_models')
    print("Saved imputer models to 'example_imputer_models/'")
    
    # Test loading the models
    print("\nTesting load models functionality...")
    new_imputer = Imputer()
    new_imputer.load_models('example_imputer_models')
    print("Successfully loaded imputer models!")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()