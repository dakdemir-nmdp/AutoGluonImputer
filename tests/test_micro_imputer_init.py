"""
Micro tests for Imputer class initialization and basic attributes.
"""
import unittest
import pandas as pd
import numpy as np
from autofillgluon import Imputer


class TestMicroImputerInit(unittest.TestCase):
    """Test Imputer initialization and basic methods."""
    
    def test_imputer_default_init(self):
        """Test Imputer can be initialized with default parameters."""
        try:
            imputer = Imputer()
            
            # Check default attributes
            self.assertEqual(imputer.num_iter, 10)
            self.assertEqual(imputer.time_limit, 60)
            self.assertEqual(imputer.presets, ['medium_quality', 'optimize_for_deployment'])
            self.assertFalse(imputer.use_missingness_features)
            self.assertEqual(imputer.simple_impute_columns, [])  # defaults to empty list
            self.assertEqual(imputer.column_settings, {})  # defaults to empty dict
            
        except Exception as e:
            self.fail(f"Default initialization failed: {e}")
    
    def test_imputer_custom_init(self):
        """Test Imputer with custom parameters."""
        try:
            imputer = Imputer(
                num_iter=3,
                time_limit=30,
                presets=['high_quality'],
                use_missingness_features=True,
                simple_impute_columns=['col1'],
                column_settings={'col1': {'time_limit': 10}}
            )
            
            # Check custom attributes
            self.assertEqual(imputer.num_iter, 3)
            self.assertEqual(imputer.time_limit, 30)
            self.assertEqual(imputer.presets, ['high_quality'])
            self.assertTrue(imputer.use_missingness_features)
            self.assertEqual(imputer.simple_impute_columns, ['col1'])
            self.assertEqual(imputer.column_settings, {'col1': {'time_limit': 10}})
            
        except Exception as e:
            self.fail(f"Custom initialization failed: {e}")
    
    def test_imputer_has_required_methods(self):
        """Test that Imputer has all required methods."""
        imputer = Imputer()
        
        required_methods = [
            'fit', 'transform', 'fit_transform',
            'save_models', 'load_models',
            'missingness_matrix', '_simple_impute_column',
            'evaluate_imputation', 'feature_importance'
        ]
        
        for method_name in required_methods:
            with self.subTest(method=method_name):
                self.assertTrue(hasattr(imputer, method_name))
                self.assertTrue(callable(getattr(imputer, method_name)))
    
    def test_missingness_matrix_method(self):
        """Test the missingness_matrix method works correctly."""
        imputer = Imputer()
        
        # Create test data
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [np.nan, 2, 3, 4],
            'col3': [1, 2, 3, 4]  # no missing values
        })
        
        try:
            result = imputer.missingness_matrix(df)
            
            # Check result structure
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result.columns), 3)
            self.assertTrue(all(col.endswith('_missing') for col in result.columns))
            
            # Check values
            expected_col1 = [0, 0, 1, 0]  # missing in row 2 (index 2)
            expected_col2 = [1, 0, 0, 0]  # missing in row 0 (index 0)
            expected_col3 = [0, 0, 0, 0]  # no missing values
            
            self.assertEqual(list(result['col1_missing']), expected_col1)
            self.assertEqual(list(result['col2_missing']), expected_col2)
            self.assertEqual(list(result['col3_missing']), expected_col3)
            
        except Exception as e:
            self.fail(f"missingness_matrix method failed: {e}")
    
    def test_simple_impute_column_method(self):
        """Test the _simple_impute_column method works correctly."""
        imputer = Imputer()
        
        # Test numeric column
        df_num = pd.DataFrame({'num_col': [1.0, 2.0, np.nan, 4.0, np.nan]})
        
        try:
            imputer._simple_impute_column(df_num, 'num_col')
            self.assertFalse(df_num['num_col'].isnull().any())
            # Should fill with mean: (1+2+4)/3 = 2.333...
            self.assertAlmostEqual(df_num.loc[2, 'num_col'], 2.333, places=2)
            
        except Exception as e:
            self.fail(f"Simple imputation for numeric column failed: {e}")
        
        # Test categorical column
        df_cat = pd.DataFrame({'cat_col': ['A', 'B', np.nan, 'A', np.nan]})
        
        try:
            imputer._simple_impute_column(df_cat, 'cat_col')
            self.assertFalse(df_cat['cat_col'].isnull().any())
            # Should fill with mode: 'A' (appears twice)
            self.assertEqual(df_cat.loc[2, 'cat_col'], 'A')
            self.assertEqual(df_cat.loc[4, 'cat_col'], 'A')
            
        except Exception as e:
            self.fail(f"Simple imputation for categorical column failed: {e}")
    
    def test_imputer_str_repr(self):
        """Test string representation of Imputer."""
        imputer = Imputer(num_iter=3, time_limit=30)
        
        # Should not raise an error
        try:
            str(imputer)
            repr(imputer)
        except Exception as e:
            self.fail(f"String representation failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)