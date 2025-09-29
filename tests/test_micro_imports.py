"""
Micro tests for basic imports and package structure.
These tests verify that all components can be imported without errors.
"""
import unittest
import sys
import importlib


class TestMicroImports(unittest.TestCase):
    """Test all basic imports work correctly."""
    
    def test_main_package_import(self):
        """Test importing the main autofillgluon package."""
        try:
            import autofillgluon
            self.assertTrue(hasattr(autofillgluon, '__version__'))
            self.assertEqual(autofillgluon.__version__, '0.1.0')
        except ImportError as e:
            self.fail(f"Failed to import autofillgluon: {e}")
    
    def test_imputer_import(self):
        """Test importing the Imputer class."""
        try:
            from autofillgluon import Imputer
            self.assertTrue(callable(Imputer))
        except ImportError as e:
            self.fail(f"Failed to import Imputer: {e}")
    
    def test_multiple_imputation_import(self):
        """Test importing multiple_imputation function."""
        try:
            from autofillgluon import multiple_imputation
            self.assertTrue(callable(multiple_imputation))
        except ImportError as e:
            self.fail(f"Failed to import multiple_imputation: {e}")
    
    def test_scorer_imports(self):
        """Test importing all scorer functions."""
        scorer_functions = [
            'scorefunct_cindex',
            'scorefunct_coxPH', 
            'negative_log_likelihood_exponential',
            'concordance_index_scorer',
            'cox_ph_scorer',
            'exponential_nll_scorer'
        ]
        
        for func_name in scorer_functions:
            with self.subTest(function=func_name):
                try:
                    func = getattr(
                        importlib.import_module('autofillgluon.scorer'),
                        func_name
                    )
                    self.assertTrue(callable(func))
                except (ImportError, AttributeError) as e:
                    self.fail(f"Failed to import {func_name}: {e}")
    
    def test_utils_imports(self):
        """Test importing utility functions."""
        util_functions = [
            'calculate_missingness_statistics',
            'generate_missingness_pattern',
            'evaluate_imputation_accuracy',
            'plot_imputation_evaluation'
        ]
        
        for func_name in util_functions:
            with self.subTest(function=func_name):
                try:
                    func = getattr(
                        importlib.import_module('autofillgluon.utils'),
                        func_name
                    )
                    self.assertTrue(callable(func))
                except (ImportError, AttributeError) as e:
                    self.fail(f"Failed to import {func_name}: {e}")
    
    def test_direct_imports(self):
        """Test direct imports from main package."""
        try:
            from autofillgluon import (
                Imputer, 
                multiple_imputation,
                scorefunct_cindex, 
                scorefunct_coxPH, 
                negative_log_likelihood_exponential,
                concordance_index_scorer,
                cox_ph_scorer,
                exponential_nll_scorer,
                calculate_missingness_statistics,
                generate_missingness_pattern,
                evaluate_imputation_accuracy,
                plot_imputation_evaluation
            )
            
            # Check all imports are callable
            callables = [
                Imputer, multiple_imputation, scorefunct_cindex, scorefunct_coxPH,
                negative_log_likelihood_exponential, concordance_index_scorer,
                cox_ph_scorer, exponential_nll_scorer, calculate_missingness_statistics,
                generate_missingness_pattern, evaluate_imputation_accuracy,
                plot_imputation_evaluation
            ]
            
            for func in callables:
                self.assertTrue(callable(func))
                
        except ImportError as e:
            self.fail(f"Failed direct import: {e}")
    
    def test_submodule_structure(self):
        """Test that submodules can be imported."""
        submodules = [
            'autofillgluon.imputer',
            'autofillgluon.scorer', 
            'autofillgluon.utils'
        ]
        
        for module_name in submodules:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)