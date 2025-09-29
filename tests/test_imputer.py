import unittest
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset
from autofillgluon import Imputer

class TestImputer(unittest.TestCase):
    def setUp(self):
        # Create a simple dataframe for testing
        self.df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'cat1': ['A', 'B', 'C', 'A', 'B'],
        })
        
        # Convert to TabularDataset
        self.df = TabularDataset(self.df)
        
        # Convert categorical column to category type
        self.df['cat1'] = self.df['cat1'].astype('category')
        
        # Create a copy with missing values
        self.df_missing = self.df.copy()
        self.df_missing.loc[0, 'num1'] = np.nan
        self.df_missing.loc[1, 'num2'] = np.nan
        self.df_missing.loc[2, 'cat1'] = np.nan
    
    def test_imputer_initialization(self):
        # Test that the Imputer can be initialized
        imputer = Imputer(num_iter=1, time_limit=30, presets=['medium_quality'])
        self.assertIsInstance(imputer, Imputer)
    
    def test_imputer_fit(self):
        # Test that the Imputer can fit a dataframe with missing values
        # Use basic presets and reasonable time limit for tests
        imputer = Imputer(num_iter=1, time_limit=30, presets=['medium_quality'])
        df_imputed = imputer.fit(self.df_missing)
        
        # Check that the returned dataframe has no missing values
        self.assertFalse(df_imputed.isnull().any().any())
    
    def test_imputer_transform(self):
        # Test that the Imputer can transform a new dataframe
        # Use basic presets and reasonable time limit for tests
        imputer = Imputer(num_iter=1, time_limit=30, presets=['medium_quality'])
        imputer.fit(self.df_missing)
        
        # Create a new dataframe with missing values
        new_df_missing = self.df.copy()
        new_df_missing.loc[3, 'num1'] = np.nan
        new_df_missing.loc[4, 'cat1'] = np.nan
        
        # Transform the new dataframe
        new_df_imputed = imputer.transform(new_df_missing)
        
        # Check that the returned dataframe has no missing values
        self.assertFalse(new_df_imputed.isnull().any().any())
    
    def test_column_types_preserved(self):
        # Test that the Imputer preserves column types
        # Use very simple parameters for this test
        imputer = Imputer(num_iter=1, time_limit=120, presets=['medium_quality'])
        df_imputed = imputer.fit(self.df_missing)
        
        # Check that numerical columns are still numerical
        self.assertTrue(np.issubdtype(df_imputed['num1'].dtype, np.number))
        self.assertTrue(np.issubdtype(df_imputed['num2'].dtype, np.number))
        
        # Check that categorical columns are still categorical
        self.assertEqual(str(df_imputed['cat1'].dtype), 'category')

if __name__ == '__main__':
    unittest.main()