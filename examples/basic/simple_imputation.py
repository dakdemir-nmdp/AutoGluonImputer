"""
Simple imputation example showing basic usage of AutoFillGluon.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from autofillgluon import Imputer
from autofillgluon.utils import calculate_missingness_statistics

# Set random seed for reproducibility
np.random.seed(42)

def create_example_data(n_rows=200):
    """Create example data with some missing values."""
    # Create a dataframe with some correlation between columns
    df = pd.DataFrame({
        'age': np.random.normal(40, 10, n_rows),
        'income': np.random.normal(50000, 15000, n_rows),
        'experience': np.random.normal(15, 7, n_rows),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_rows),
        'department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing', 'Support'], n_rows)
    })
    
    # Add some correlations
    df['experience'] = df['age'] * 0.4 + np.random.normal(0, 3, n_rows)
    df['income'] = 20000 + df['experience'] * 2000 + np.random.normal(0, 5000, n_rows)
    
    # Add categorical biases
    df.loc[df['department'] == 'Engineering', 'income'] += 10000
    df.loc[df['department'] == 'Sales', 'income'] += 5000
    
    # Ensure proper data types
    df['satisfaction'] = df['satisfaction'].astype('category')
    df['department'] = df['department'].astype('category')
    
    # Create a complete copy before adding missing values
    df_complete = df.copy()
    
    # Add some missingness
    mask = np.random.random(df.shape) < 0.15
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if mask[i, j]:
                df.iloc[i, j] = np.nan
    
    return df, df_complete

def main():
    # Generate example data
    print("Generating example data...")
    df_missing, df_complete = create_example_data(200)
    
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
    for col in ['age', 'income', 'experience']:
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
    for col in ['satisfaction', 'department']:
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