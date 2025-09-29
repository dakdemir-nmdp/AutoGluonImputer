"""
California housing dataset imputation example.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Import from our package
from autofillgluon import Imputer
from autofillgluon.utils import evaluate_imputation_accuracy

# Fix SSL certificate issues on macOS
def fix_ssl_context():
    """Fix SSL context for macOS certificate issues"""
    try:
        import ssl
        
        # Create unverified HTTPS context as fallback
        ssl._create_default_https_context = ssl._create_unverified_context
    except ImportError:
        # If certifi is not available, create unverified context
        ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("Loading California housing dataset...")
    
    # Fix SSL context for dataset download
    fix_ssl_context()
    
    # Load the California Housing dataset
    housing = fetch_california_housing()
    
    # Create a dataframe with the features and target (use subset for demo)
    df = pd.DataFrame(housing.data[:2000], columns=housing.feature_names)  # Smaller subset
    df['target'] = housing.target[:2000]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {', '.join(housing.feature_names)}")
    print("Target: Housing value (in $100,000s)")
    
    # Split the data into training and testing sets
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    
    # Introduce 20% missingness in the training and test datasets
    print("\nIntroducing artificial missingness (20% of values)...")
    train_missing = train.mask(np.random.random(train.shape) < 0.2)
    test_missing = test.mask(np.random.random(test.shape) < 0.2)
    
    # Display missingness statistics
    train_missing_count = train_missing.isnull().sum()
    print("\nMissing values by column (training data):")
    for col, count in train_missing_count.items():
        print(f"  {col}: {count} ({count/len(train_missing)*100:.1f}%)")
    
    # Initialize and fit the imputer
    print("\nInitializing imputer...")
    imputer = Imputer(
        num_iter=1,       # Number of iterations (reduced for demo)
        time_limit=30,    # Time limit per column (seconds) 
        presets=['good_quality']  # Use good quality presets
    )
    
    print("\nFitting imputer and imputing training data...")
    imputer.fit(train_missing)
    
    print("\nImputing test data...")
    test_imputed = imputer.transform(test_missing)
    
    # Evaluate imputation quality
    print("\nEvaluating imputation quality...")
    
    # Create masks of missing values
    test_mask = pd.DataFrame(np.random.random(test.shape) < 0.2, 
                             index=test.index, 
                             columns=test.columns)
    
    # Evaluate with utility function
    metrics = evaluate_imputation_accuracy(test, test_imputed, test_mask)
    
    # Display metrics for each column
    for col, col_metrics in metrics.items():
        print(f"\nMetrics for {col}:")
        for metric_name, value in col_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Visualize imputation for MedInc (Median Income)
    print("\nVisualizing imputation results for MedInc...")
    
    # Get indices with missing MedInc values
    missing_indices = test_missing['MedInc'].index[test_missing['MedInc'].isna()]
    
    if len(missing_indices) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(test['MedInc'][missing_indices], test_imputed['MedInc'][missing_indices])
        plt.xlabel('Original MedInc Values')
        plt.ylabel('Imputed MedInc Values')
        plt.title('Original vs Imputed Median Income Values')
        
        # Add reference line
        min_val = min(test['MedInc'][missing_indices].min(), test_imputed['MedInc'][missing_indices].min())
        max_val = max(test['MedInc'][missing_indices].max(), test_imputed['MedInc'][missing_indices].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        sns.regplot(x=test['MedInc'][missing_indices], 
                   y=test_imputed['MedInc'][missing_indices], 
                   scatter=False, color='blue')
        
        # Calculate correlation
        corr = np.corrcoef(test['MedInc'][missing_indices], test_imputed['MedInc'][missing_indices])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig('medinc_imputation.png')
        print("Saved visualization to 'medinc_imputation.png'")
    
    # Save the imputer for later use
    imputer.save_models('housing_imputer')
    print("\nSaved imputer to 'housing_imputer/'")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()