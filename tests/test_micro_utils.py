"""
Micro tests for utility functions.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from autofillgluon.utils import (
    calculate_missingness_statistics,
    generate_missingness_pattern,
    evaluate_imputation_accuracy,
    plot_imputation_evaluation
)


class TestMicroUtils(unittest.TestCase):
    """Test utility functions work correctly."""
    
    def setUp(self):
        """Set up test data."""
        self.df_complete = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        self.df_missing = self.df_complete.copy()
        self.df_missing.loc[0, 'num1'] = np.nan
        self.df_missing.loc[1, 'num2'] = np.nan
        self.df_missing.loc[2, 'cat1'] = np.nan
    
    def test_calculate_missingness_statistics(self):
        """Test calculate_missingness_statistics function."""
        try:
            stats = calculate_missingness_statistics(self.df_missing)
            
            # Check return type
            self.assertIsInstance(stats, dict)
            
            # Check all columns are represented
            expected_columns = ['num1', 'num2', 'cat1', 'cat2']
            self.assertEqual(set(stats.keys()), set(expected_columns))
            
            # Check structure of each column's stats
            for col, col_stats in stats.items():
                self.assertIsInstance(col_stats, dict)
                required_keys = ['percent_missing', 'count_missing', 'total_rows']
                self.assertEqual(set(col_stats.keys()), set(required_keys))
            
            # Check specific values
            self.assertEqual(stats['num1']['count_missing'], 1)
            self.assertEqual(stats['num1']['percent_missing'], 20.0)  # 1/5 * 100
            self.assertEqual(stats['num2']['count_missing'], 1)
            self.assertEqual(stats['cat1']['count_missing'], 1)
            self.assertEqual(stats['cat2']['count_missing'], 0)
            
            # Check total rows
            for col_stats in stats.values():
                self.assertEqual(col_stats['total_rows'], 5)
            
        except Exception as e:
            self.fail(f"calculate_missingness_statistics failed: {e}")
    
    def test_generate_missingness_pattern(self):
        """Test generate_missingness_pattern function."""
        try:
            # Test with default parameters
            df_missing = generate_missingness_pattern(self.df_complete)
            
            # Check return type and shape
            self.assertIsInstance(df_missing, pd.DataFrame)
            self.assertEqual(df_missing.shape, self.df_complete.shape)
            
            # Check that some values are missing (but not all)
            total_missing = df_missing.isnull().sum().sum()
            self.assertGreater(total_missing, 0)
            self.assertLess(total_missing, df_missing.size)
            
            # Test with specific percentage
            df_missing_10 = generate_missingness_pattern(
                self.df_complete, 
                percent_missing=0.1, 
                random_state=42
            )
            
            # Should be repeatable with same random state
            df_missing_10_repeat = generate_missingness_pattern(
                self.df_complete, 
                percent_missing=0.1, 
                random_state=42
            )
            
            pd.testing.assert_frame_equal(
                df_missing_10.isnull(), 
                df_missing_10_repeat.isnull()
            )
            
        except Exception as e:
            self.fail(f"generate_missingness_pattern failed: {e}")
    
    def test_evaluate_imputation_accuracy(self):
        """Test evaluate_imputation_accuracy function."""
        try:
            # Create imputed data (simple mean/mode imputation for testing)
            df_imputed = self.df_missing.copy()
            df_imputed['num1'].fillna(df_imputed['num1'].mean(), inplace=True)
            df_imputed['num2'].fillna(df_imputed['num2'].mean(), inplace=True)
            df_imputed['cat1'].fillna('A', inplace=True)  # Use most frequent
            
            # Create missing mask
            missing_mask = self.df_missing.isnull()
            
            # Evaluate accuracy
            results = evaluate_imputation_accuracy(
                self.df_complete, 
                df_imputed, 
                missing_mask
            )
            
            # Check return type
            self.assertIsInstance(results, dict)
            
            # Check numeric columns have numeric metrics
            if 'num1' in results:
                num_metrics = results['num1']
                expected_metrics = {'mse', 'mae', 'rmse', 'r2'}
                self.assertTrue(expected_metrics.issubset(num_metrics.keys()))
                
                # Check all metrics are numeric
                for metric_val in num_metrics.values():
                    self.assertTrue(isinstance(metric_val, (int, float, np.number)) or np.isnan(metric_val))
            
            # Check categorical columns have accuracy metric
            if 'cat1' in results:
                cat_metrics = results['cat1']
                self.assertIn('accuracy', cat_metrics)
                self.assertIsInstance(cat_metrics['accuracy'], (int, float, np.number))
                self.assertGreaterEqual(cat_metrics['accuracy'], 0.0)
                self.assertLessEqual(cat_metrics['accuracy'], 1.0)
            
        except Exception as e:
            self.fail(f"evaluate_imputation_accuracy failed: {e}")
    
    def test_plot_imputation_evaluation(self):
        """Test plot_imputation_evaluation function."""
        try:
            # Create imputed data
            df_imputed = self.df_missing.copy()
            df_imputed['num1'].fillna(2.5, inplace=True)  # Use a specific value for testing
            
            # Create missing mask
            missing_mask = self.df_missing.isnull()
            
            # Test plotting (should not raise an error)
            with tempfile.TemporaryDirectory() as temp_dir:
                plot_path = os.path.join(temp_dir, 'test_plot.png')
                
                # This should not raise an error
                plot_imputation_evaluation(
                    self.df_complete,
                    df_imputed, 
                    missing_mask,
                    column='num1'
                )
                
                # Note: We don't check if file is created since the function doesn't save by default
                # The test passes if no exception is raised
            
        except Exception as e:
            self.fail(f"plot_imputation_evaluation failed: {e}")
    
    def test_empty_dataframe_handling(self):
        """Test utility functions with edge cases."""
        # Test with empty dataframe
        df_empty = pd.DataFrame()
        
        try:
            stats = calculate_missingness_statistics(df_empty)
            self.assertEqual(stats, {})
        except Exception as e:
            self.fail(f"Empty dataframe handling failed: {e}")
        
        # Test with dataframe with no missing values
        try:
            stats = calculate_missingness_statistics(self.df_complete)
            for col_stats in stats.values():
                self.assertEqual(col_stats['count_missing'], 0)
                self.assertEqual(col_stats['percent_missing'], 0.0)
        except Exception as e:
            self.fail(f"No missing values handling failed: {e}")
    
    def test_all_missing_column(self):
        """Test handling of columns with all missing values."""
        df_all_missing = self.df_complete.copy()
        df_all_missing['all_missing'] = np.nan
        
        try:
            stats = calculate_missingness_statistics(df_all_missing)
            self.assertEqual(stats['all_missing']['count_missing'], 5)
            self.assertEqual(stats['all_missing']['percent_missing'], 100.0)
        except Exception as e:
            self.fail(f"All missing column handling failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)