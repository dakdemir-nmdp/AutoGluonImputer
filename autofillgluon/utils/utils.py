"""
Utility functions for the AutoFillGluon package.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_missingness_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate missingness statistics for a dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe
        
    Returns
    -------
    dict
        A dictionary with column names as keys and statistics as values.
        Statistics include:
        - percent_missing: percentage of missing values in the column
        - count_missing: count of missing values in the column
        - total_rows: total number of rows in the dataframe
    """
    stats = {}
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = missing_count / total_rows * 100
        
        stats[col] = {
            'percent_missing': missing_percent,
            'count_missing': missing_count,
            'total_rows': total_rows
        }
    
    return stats

def generate_missingness_pattern(df: pd.DataFrame, 
                                percent_missing: float = 0.2, 
                                random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate a missingness pattern by randomly setting values to NaN.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe
    percent_missing : float, optional (default=0.2)
        The percentage of values to set to NaN
    random_state : int, optional (default=None)
        Random state for reproducibility
        
    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with values randomly set to NaN
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    df_missing = df.copy()
    mask = np.random.random(df.shape) < percent_missing
    
    # Convert mask to boolean DataFrame with same index and columns as df
    mask_df = pd.DataFrame(mask, index=df.index, columns=df.columns)
    
    # Apply mask to df (set values to NaN where mask is True)
    for col in df.columns:
        df_missing.loc[mask_df[col], col] = np.nan
    
    return df_missing

def evaluate_imputation_accuracy(original_df: pd.DataFrame, 
                               imputed_df: pd.DataFrame, 
                               missing_mask: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Evaluate imputation accuracy by comparing imputed values with original values.
    
    Parameters
    ----------
    original_df : pandas.DataFrame
        The original dataframe with complete data
    imputed_df : pandas.DataFrame
        The imputed dataframe
    missing_mask : pandas.DataFrame
        Boolean mask indicating which values were missing (True = missing)
        
    Returns
    -------
    dict
        A dictionary with column names as keys and evaluation metrics as values.
        Metrics for numeric columns:
        - mse: mean squared error
        - mae: mean absolute error
        - rmse: root mean squared error
        - r2: R^2 score
        
        Metrics for categorical columns:
        - accuracy: accuracy score
    """
    
    results = {}
    
    for col in original_df.columns:
        # Get indices where values were missing
        missing_indices = missing_mask[col]
        
        if missing_indices.sum() == 0:
            # Skip columns with no missing values
            continue
        
        # Get original and imputed values
        y_true = original_df.loc[missing_indices, col]
        y_pred = imputed_df.loc[missing_indices, col]
        
        # Check if column is numeric or categorical
        if np.issubdtype(original_df[col].dtype, np.number):
            # Calculate metrics for numeric columns
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate R^2 score if possible
            try:
                r2 = r2_score(y_true, y_pred)
            except Exception:
                r2 = np.nan
            
            results[col] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        else:
            # Calculate metrics for categorical columns
            accuracy = accuracy_score(y_true, y_pred)
            results[col] = {
                'accuracy': accuracy
            }
    
    return results

def plot_imputation_evaluation(original_df: pd.DataFrame, 
                            imputed_df: pd.DataFrame, 
                            missing_mask: pd.DataFrame,
                            column: str,
                            figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot imputation evaluation for a specific column.
    
    Parameters
    ----------
    original_df : pandas.DataFrame
        The original dataframe with complete data
    imputed_df : pandas.DataFrame
        The imputed dataframe
    missing_mask : pandas.DataFrame
        Boolean mask indicating which values were missing (True = missing)
    column : str
        The column to evaluate
    figsize : tuple, optional (default=(10, 6))
        Figure size
        
    Returns
    -------
    None
        The function creates a plot
    """
    
    # Get indices where values were missing
    missing_indices = missing_mask[column]
    
    if missing_indices.sum() == 0:
        print(f"No missing values in column '{column}'")
        return
    
    # Get original and imputed values
    y_true = original_df.loc[missing_indices, column]
    y_pred = imputed_df.loc[missing_indices, column]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Check if column is numeric or categorical
    if np.issubdtype(original_df[column].dtype, np.number):
        # Plot for numeric column
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        
        # Add regression line
        sns.regplot(x=y_true, y=y_pred, scatter=False, color='blue')
        
        # Calculate correlation coefficient
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        plt.xlabel('Original Values')
        plt.ylabel('Imputed Values')
        plt.title(f'Imputation Evaluation for {column}')
        
    else:
        # Plot for categorical column
        from sklearn.metrics import confusion_matrix
        import pandas as pd
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert to DataFrame for better visualization
        cm_df = pd.DataFrame(cm, 
                            index=sorted(original_df[column].unique()), 
                            columns=sorted(original_df[column].unique()))
        
        # Plot heatmap
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
        
        # Calculate accuracy
        accuracy = (y_true == y_pred).mean()
        plt.text(0.05, 0.05, f'Accuracy: {accuracy:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {column}')
    
    plt.tight_layout()
    plt.show()