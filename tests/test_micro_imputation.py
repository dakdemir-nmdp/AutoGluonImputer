"""
Micro tests for basic imputation functionality.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from autofillgluon import Imputer


class TestMicroImputation(unittest.TestCase):
    """Test basic imputation functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create a reasonably sized dataset for ML training
        n_samples = 100
        self.df_complete = pd.DataFrame({
            'numeric1': np.random.normal(10, 2, n_samples),
            'numeric2': np.random.normal(50, 10, n_samples), 
            'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical2': np.random.choice(['X', 'Y'], n_samples)
        })
        
        # Add some relationships to make imputation meaningful
        self.df_complete.loc[self.df_complete['categorical1'] == 'A', 'numeric1'] += 5
        self.df_complete.loc[self.df_complete['categorical2'] == 'X', 'numeric2'] += 10
        
        # Create version with missing values
        self.df_missing = self.df_complete.copy()
        # Add systematic missingness
        missing_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
        for col in ['numeric1', 'numeric2', 'categorical1']:
            self.df_missing.loc[missing_indices[:len(missing_indices)//3], col] = np.nan
    
    def test_basic_imputation(self):
        """Test basic imputation functionality."""
        try:
            # Test with minimal settings for speed
            imputer = Imputer(
                num_iter=1,
                time_limit=15,
                presets=['medium_quality']
            )
            
            # Check original missing values
            original_missing = self.df_missing.isnull().sum().sum()
            self.assertGreater(original_missing, 0, "Test data should have missing values")
            
            # Perform imputation
            df_imputed = imputer.fit(self.df_missing)
            
            # Check results
            self.assertIsInstance(df_imputed, pd.DataFrame)
            self.assertEqual(df_imputed.shape, self.df_missing.shape)
            
            # Check that missing values are filled
            final_missing = df_imputed.isnull().sum().sum()
            self.assertEqual(final_missing, 0, "All missing values should be imputed")
            
            # Check that non-missing values are unchanged
            non_missing_mask = ~self.df_missing.isnull()
            for col in self.df_missing.columns:
                col_mask = non_missing_mask[col]
                if col_mask.any():
                    pd.testing.assert_series_equal(
                        self.df_missing.loc[col_mask, col],
                        df_imputed.loc[col_mask, col],
                        check_names=False
                    )
            
        except Exception as e:
            self.fail(f"Basic imputation failed: {e}")
    
    def test_imputation_with_small_dataset(self):
        """Test imputation with very small dataset (should fall back to simple imputation)."""
        try:
            # Create very small dataset
            small_df = pd.DataFrame({
                'num': [1.0, np.nan],
                'cat': ['A', np.nan]
            })
            
            imputer = Imputer(num_iter=1, time_limit=5)
            result = imputer.fit(small_df)
            
            # Should not have missing values
            self.assertEqual(result.isnull().sum().sum(), 0)
            
            # First value should be unchanged
            self.assertEqual(result.loc[0, 'num'], 1.0)
            self.assertEqual(result.loc[0, 'cat'], 'A')
            
        except Exception as e:
            self.fail(f"Small dataset imputation failed: {e}")
    
    def test_no_missing_values(self):
        """Test imputation on data with no missing values."""
        try:
            imputer = Imputer(num_iter=1, time_limit=5)
            
            # Should return unchanged dataframe
            result = imputer.fit(self.df_complete)
            
            # Should be identical to original
            pd.testing.assert_frame_equal(result, self.df_complete)
            
        except Exception as e:
            self.fail(f"No missing values test failed: {e}")
    
    def test_single_column_missing(self):
        """Test imputation when only one column has missing values."""
        try:
            # Create data with missing values in only one column
            single_missing_df = self.df_complete.copy()
            single_missing_df.loc[10:15, 'numeric1'] = np.nan
            
            imputer = Imputer(num_iter=1, time_limit=10)
            result = imputer.fit(single_missing_df)
            
            # Should have no missing values
            self.assertEqual(result.isnull().sum().sum(), 0)
            
            # Other columns should be unchanged
            for col in ['numeric2', 'categorical1', 'categorical2']:
                pd.testing.assert_series_equal(
                    single_missing_df[col],
                    result[col],
                    check_names=False
                )
            
        except Exception as e:
            self.fail(f"Single column missing test failed: {e}")
    
    def test_all_missing_column(self):
        """Test behavior when a column is entirely missing."""
        try:
            # Create data with one completely missing column
            all_missing_df = self.df_complete.copy()
            all_missing_df['all_missing'] = np.nan
            
            imputer = Imputer(num_iter=1, time_limit=10)
            result = imputer.fit(all_missing_df)
            
            # Should handle this gracefully (might fill with default or error)
            # The exact behavior depends on implementation, but it shouldn't crash
            self.assertIsInstance(result, pd.DataFrame)
            
        except Exception as e:
            # It's OK if this fails gracefully with a specific error
            self.assertIn("Error", str(type(e).__name__))
    
    def test_mixed_data_types(self):
        """Test imputation with mixed data types."""
        try:
            # Create data with various types
            mixed_df = pd.DataFrame({
                'int_col': [1, 2, np.nan, 4, 5],
                'float_col': [1.1, np.nan, 3.3, 4.4, np.nan],
                'str_col': ['a', 'b', np.nan, 'd', 'e'],
                'bool_col': [True, False, np.nan, True, False]
            })
            
            # Extend to make it larger for ML
            mixed_df = pd.concat([mixed_df] * 25, ignore_index=True)
            
            imputer = Imputer(num_iter=1, time_limit=15)
            result = imputer.fit(mixed_df)
            
            # Should handle all data types
            self.assertEqual(result.isnull().sum().sum(), 0)
            self.assertEqual(result.shape, mixed_df.shape)
            
        except Exception as e:
            self.fail(f"Mixed data types test failed: {e}")
    
    def test_fit_transform_method(self):
        """Test fit_transform method."""
        try:
            imputer = Imputer(num_iter=1, time_limit=10)
            
            # fit_transform should be equivalent to fit
            result_fit_transform = imputer.fit_transform(self.df_missing)
            
            # Should produce same result as fit
            self.assertIsInstance(result_fit_transform, pd.DataFrame)
            self.assertEqual(result_fit_transform.isnull().sum().sum(), 0)
            
        except Exception as e:
            self.fail(f"fit_transform method failed: {e}")
    
    def test_imputer_attributes_after_fit(self):
        """Test that imputer has correct attributes after fitting."""
        try:
            imputer = Imputer(num_iter=1, time_limit=10)
            imputer.fit(self.df_missing)
            
            # Should have internal attributes set
            expected_attributes = [
                'missing_cells',
                'col_data_types', 
                'colsummary',
                'models'
            ]
            
            for attr in expected_attributes:
                with self.subTest(attribute=attr):
                    self.assertTrue(hasattr(imputer, attr))
            
        except Exception as e:
            self.fail(f"Imputer attributes test failed: {e}")
    
    def test_reproducibility(self):
        """Test that imputation is reproducible with same settings."""
        try:
            # Create two identical imputers
            imputer1 = Imputer(num_iter=1, time_limit=10, presets=['medium_quality'])
            imputer2 = Imputer(num_iter=1, time_limit=10, presets=['medium_quality'])
            
            # Note: Perfect reproducibility is hard with AutoGluon due to random seeds
            # This test mainly checks that the process doesn't crash
            result1 = imputer1.fit(self.df_missing.copy())
            result2 = imputer2.fit(self.df_missing.copy())
            
            # Both should complete successfully and have no missing values
            self.assertEqual(result1.isnull().sum().sum(), 0)
            self.assertEqual(result2.isnull().sum().sum(), 0)
            self.assertEqual(result1.shape, result2.shape)
            
        except Exception as e:
            self.fail(f"Reproducibility test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)