"""
AutoFillGluon: Machine learning-based missing data imputation using AutoGluon.

This package provides tools for:
1. Missing data imputation using AutoGluon predictive models
2. Survival analysis with AutoGluon
3. Evaluation of imputation quality

Main components:
- Imputer: Machine learning-based imputation using AutoGluon
- Scoring functions: Custom metrics for survival analysis with AutoGluon
- Utility functions: Tools for evaluating and visualizing imputation results
"""

__version__ = "0.1.0"

# Import main components
from .imputer.imputer import Imputer, multiple_imputation
from .scorer.scorer import (
    scorefunct_cindex, 
    scorefunct_coxPH, 
    negative_log_likelihood_exponential,
    concordance_index_scorer,
    cox_ph_scorer,
    exponential_nll_scorer
)
from .utils import (
    calculate_missingness_statistics,
    generate_missingness_pattern,
    evaluate_imputation_accuracy,
    plot_imputation_evaluation
)

# Define public API
__all__ = [
    # Imputation
    'Imputer',
    'multiple_imputation',
    
    # Survival analysis scoring
    'scorefunct_cindex',
    'scorefunct_coxPH',
    'negative_log_likelihood_exponential',
    'concordance_index_scorer',
    'cox_ph_scorer',
    'exponential_nll_scorer',
    
    # Utility functions
    'calculate_missingness_statistics',
    'generate_missingness_pattern',
    'evaluate_imputation_accuracy',
    'plot_imputation_evaluation'
]