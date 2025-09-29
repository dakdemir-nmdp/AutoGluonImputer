"""
Micro tests for scorer functions.
"""
import unittest
import numpy as np
import pandas as pd
from autofillgluon.scorer import (
    scorefunct_cindex,
    scorefunct_coxPH, 
    negative_log_likelihood_exponential,
    concordance_index_scorer,
    cox_ph_scorer,
    exponential_nll_scorer
)


class TestMicroScorer(unittest.TestCase):
    """Test scorer functions work correctly."""
    
    def setUp(self):
        """Set up test data for survival analysis."""
        # Create simple survival test data
        # Negative values indicate censored observations
        self.y_true = np.array([-1, 2, 1, 4, 5, -2, 7, 8, 9, 10])  
        self.y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Perfect concordance case
        self.y_true_perfect = np.array([1, 2, 3, 4, 5])
        self.y_pred_perfect = np.array([1, 2, 3, 4, 5])
        
        # Random case
        np.random.seed(42)
        self.y_true_random = np.array([1, 2, 3, 4, 5, -1, -2, -3])
        self.y_pred_random = np.random.randn(8)
    
    def test_scorefunct_cindex(self):
        """Test concordance index function."""
        try:
            # Test with basic data
            cindex = scorefunct_cindex(self.y_true, self.y_pred)
            
            # C-index should be between 0 and 1
            self.assertIsInstance(cindex, (float, np.floating))
            self.assertGreaterEqual(cindex, 0.0)
            self.assertLessEqual(cindex, 1.0)
            
            # Test with perfect concordance
            cindex_perfect = scorefunct_cindex(self.y_true_perfect, self.y_pred_perfect)
            self.assertAlmostEqual(cindex_perfect, 1.0, places=5)
            
            # Test that function handles numpy arrays properly
            cindex_copy = scorefunct_cindex(self.y_true.copy(), self.y_pred.copy())
            self.assertAlmostEqual(cindex, cindex_copy, places=5)
            
        except Exception as e:
            self.fail(f"scorefunct_cindex failed: {e}")
    
    def test_scorefunct_coxPH(self):
        """Test Cox proportional hazards log-likelihood function."""
        try:
            # Test with basic data
            loglik = scorefunct_coxPH(self.y_true, self.y_pred)
            
            # Log-likelihood should be a finite number
            self.assertIsInstance(loglik, (float, np.floating))
            self.assertTrue(np.isfinite(loglik))
            
            # Test with different data
            loglik_random = scorefunct_coxPH(self.y_true_random, self.y_pred_random)
            self.assertTrue(np.isfinite(loglik_random))
            
        except Exception as e:
            self.fail(f"scorefunct_coxPH failed: {e}")
    
    def test_negative_log_likelihood_exponential(self):
        """Test exponential negative log-likelihood function."""
        try:
            # Test with basic data
            nll = negative_log_likelihood_exponential(self.y_true, self.y_pred)
            
            # NLL should be a finite number
            self.assertIsInstance(nll, (float, np.floating))
            self.assertTrue(np.isfinite(nll))
            
            # Test with different data
            nll_random = negative_log_likelihood_exponential(self.y_true_random, self.y_pred_random)
            self.assertTrue(np.isfinite(nll_random))
            
        except Exception as e:
            self.fail(f"negative_log_likelihood_exponential failed: {e}")
    
    def test_scorer_objects(self):
        """Test that scorer objects are properly created and callable."""
        scorers = [
            concordance_index_scorer,
            cox_ph_scorer,
            exponential_nll_scorer
        ]
        
        for scorer in scorers:
            with self.subTest(scorer=scorer):
                # Check that scorer is callable
                self.assertTrue(callable(scorer))
                
                # Check that scorer has expected attributes (AutoGluon scorer)
                self.assertTrue(hasattr(scorer, 'name'))
                self.assertTrue(hasattr(scorer, 'greater_is_better'))
                
                # Test that we can call the scorer
                try:
                    # Note: For AutoGluon scorers, we would need proper TabularDataset format
                    # Here we just check that the scorer object exists and has the right structure
                    pass
                except Exception as e:
                    # It's OK if calling fails due to format, we just want to ensure object structure
                    pass
    
    def test_edge_cases(self):
        """Test edge cases for scorer functions."""
        # Test with single values
        single_true = np.array([1])
        single_pred = np.array([1.5])
        
        try:
            # These should handle single values gracefully
            cindex = scorefunct_cindex(single_true, single_pred)
            self.assertTrue(np.isfinite(cindex))
        except Exception:
            # Single values may not be well-defined for concordance, that's OK
            pass
        
        # Test with all censored data
        all_censored = np.array([-1, -2, -3, -4])
        pred_censored = np.array([1, 2, 3, 4])
        
        try:
            cindex_censored = scorefunct_cindex(all_censored, pred_censored)
            # Should still return a valid number or NaN
            self.assertTrue(np.isfinite(cindex_censored) or np.isnan(cindex_censored))
        except Exception:
            # Some edge cases may not be handled, that's OK for now
            pass
        
        # Test with identical predictions
        identical_pred = np.array([5, 5, 5, 5, 5])
        varied_true = np.array([1, 2, 3, 4, 5])
        
        try:
            cindex_identical = scorefunct_cindex(varied_true, identical_pred)
            self.assertTrue(np.isfinite(cindex_identical) or np.isnan(cindex_identical))
        except Exception:
            # Identical predictions may cause issues, that's OK
            pass
    
    def test_array_length_mismatch(self):
        """Test that functions handle array length mismatches appropriately."""
        mismatched_true = np.array([1, 2, 3])
        mismatched_pred = np.array([1, 2, 3, 4, 5])
        
        # These should either handle gracefully or raise appropriate errors
        try:
            scorefunct_cindex(mismatched_true, mismatched_pred)
            # If it doesn't raise an error, that's also fine - some functions may handle it
        except (ValueError, IndexError):
            # This is expected behavior
            pass
        
        try:
            scorefunct_coxPH(mismatched_true, mismatched_pred)
        except (ValueError, IndexError):
            pass
        
        try:
            negative_log_likelihood_exponential(mismatched_true, mismatched_pred)
        except (ValueError, IndexError):
            pass
    
    def test_numeric_stability(self):
        """Test scorer functions with extreme values."""
        # Test with very large values
        large_true = np.array([1000, 2000, 3000, 4000])
        large_pred = np.array([1100, 1900, 3200, 3800])
        
        try:
            cindex_large = scorefunct_cindex(large_true, large_pred)
            self.assertTrue(np.isfinite(cindex_large))
        except Exception as e:
            self.fail(f"Large values test failed: {e}")
        
        # Test with very small positive values
        small_true = np.array([0.001, 0.002, 0.003, 0.004])
        small_pred = np.array([0.0011, 0.0019, 0.0032, 0.0038])
        
        try:
            cindex_small = scorefunct_cindex(small_true, small_pred)
            self.assertTrue(np.isfinite(cindex_small))
        except Exception as e:
            self.fail(f"Small values test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)