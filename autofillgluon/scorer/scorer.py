"""
Survival analysis scoring functions for use with AutoGluon.

This module provides custom scoring functions for survival analysis tasks,
designed to work with AutoGluon TabularPredictor.
"""

import numpy as np
from lifelines.utils import concordance_index
from autogluon.core.metrics import make_scorer


def scorefunct_cindex(y_true, y_pred):
    """
    Compute the concordance index between predictions and true survival times.
    
    The concordance index (C-index) is a metric of how well a model predicts
    the ordering of events or deaths. It ranges from 0.5 (random) to 1.0 (perfect).
    
    Parameters
    ----------
    y_true : array-like
        True values with negative values indicating censored observations
        and positive values indicating events.
    y_pred : array-like
        Predicted risk scores or survival times.
        
    Returns
    -------
    float
        Concordance index value (between 0 and 1)
    
    Notes
    -----
    A common convention in survival analysis is to use negative values to indicate
    censored observations. This function follows that convention.
    """
    # Extract event indicators and absolute times
    event = y_true >= 0
    event = event.astype(int)
    time = np.abs(y_true)
    
    # Calculate concordance index
    return concordance_index(time, y_pred, event_observed=event)


def scorefunct_coxPH(y_true, y_pred):
    """
    Calculate the negative log partial likelihood from the Cox Proportional Hazards model.
    
    This function implements the negative log partial likelihood function from 
    Cox Proportional Hazards. It can be used as a loss function for survival analysis.
    
    Parameters
    ----------
    y_true : array-like
        True values with negative values indicating censored observations
        and positive values indicating events.
    y_pred : array-like 
        Predicted log-hazard ratios.
        
    Returns
    -------
    float
        Negative log partial likelihood value.
        
    Notes
    -----
    - Higher values indicate worse model fit
    - Use with greater_is_better=False in AutoGluon metrics
    """
    # Extract event indicators and times
    event = y_true >= 0
    event = event.astype(int)
    time = np.abs(y_true)
    
    # Sort data by time
    sort_idx = np.argsort(time)
    
    # Apply sorting
    y_pred_sorted = y_pred[sort_idx]
    event_sorted = event[sort_idx]
    time_sorted = time[sort_idx]
    
    # Calculate the log partial likelihood
    log_lik = 0
    for i in range(len(time_sorted)):
        if event_sorted[i] == 1:
            # Compute the risk set at the current event time
            risk_set = np.where(time_sorted >= time_sorted[i])[0]
            
            # Skip if risk set is empty (should not happen with proper data)
            if len(risk_set) == 0:
                continue
                
            # Compute log likelihood contribution for this event
            log_lik += y_pred_sorted[i] - np.log(np.sum(np.exp(y_pred_sorted[risk_set])))
    
    # Return negative log-likelihood (for minimization)
    return -log_lik


def negative_log_likelihood_exponential(y_true, y_pred):
    """
    Calculate negative log-likelihood for exponential survival model.
    
    This function computes the negative log-likelihood for the exponential
    survival distribution with right-censored data.
    
    Parameters
    ----------
    y_true : array-like
        True values with negative values indicating censored observations
        and positive values indicating events.
    y_pred : array-like
        Predicted rate parameters (λ) of the exponential distribution.
        
    Returns
    -------
    float
        Negative log-likelihood value.
        
    Notes
    -----
    - For the exponential distribution, the hazard is constant over time
    - y_pred represents the hazard rate parameter λ
    - The survival function is S(t) = exp(-λt)
    """
    # Extract event indicators and times
    event = y_true >= 0
    event = event.astype(int)
    time = np.abs(y_true)
    
    # Ensure predictions are positive (exponential parameter λ must be positive)
    y_pred_pos = np.maximum(1e-10, y_pred)
    
    # Compute negative log-likelihood
    # For events (uncensored): log(λ) - λt
    # For censored: -λt
    uncensored_ll = event * np.log(y_pred_pos)
    all_ll = uncensored_ll - y_pred_pos * time
    
    # Return negative sum of log-likelihoods
    return -np.sum(all_ll)


# Create AutoGluon-compatible scorers
concordance_index_scorer = make_scorer(
    name='concordance_index',
    score_func=scorefunct_cindex,
    optimum=1,
    greater_is_better=True
)

cox_ph_scorer = make_scorer(
    name='cox_ph_likelihood',
    score_func=scorefunct_coxPH,
    optimum=-np.inf,
    greater_is_better=False
)

exponential_nll_scorer = make_scorer(
    name='exponential_nll',
    score_func=negative_log_likelihood_exponential,
    optimum=-np.inf,
    greater_is_better=False
)


# Example usage (only runs when module is executed directly)
if __name__ == '__main__':
    # Simple test data
    print("Testing scoring functions with sample data...")
    
    # Example 1: Perfect prediction
    y_true = np.array([-1, 2, 3, -4, 5])  # Negative values are censored
    y_pred = np.array([5, 4, 3, 2, 1])    # Higher value = higher risk
    
    print(f"Concordance index (perfect): {scorefunct_cindex(y_true, y_pred):.4f}")
    print(f"Cox PH score: {scorefunct_coxPH(y_true, y_pred):.4f}")
    print(f"Exponential NLL: {negative_log_likelihood_exponential(y_true, y_pred):.4f}")
    
    # Example 2: Random prediction
    y_true = np.array([-1, 2, 3, -4, 5])
    y_pred = np.random.random(5)
    
    print(f"\nConcordance index (random): {scorefunct_cindex(y_true, y_pred):.4f}")
    
    # Example 3: Complete example with AutoGluon
    try:
        from autogluon.tabular import TabularPredictor
        from lifelines.datasets import load_rossi
        
        print("\nRunning complete example with Rossi recidivism dataset...")
        
        # Load Rossi dataset
        df = load_rossi()
        df['week'] = df['week'].astype(float)
        
        # Create a copy for modeling
        df_model = df.copy()
        
        # Encode survival time with negative values for censoring
        df_model['time'] = df_model['week']
        df_model.loc[df_model['arrest'] == 0, 'time'] = -df_model['week']
        
        # Remove original columns used to create 'time'
        df_model = df_model.drop(columns=['week', 'arrest'])
        
        # Train model with Cox PH scorer
        print("Training model with Cox PH scorer...")
        predictor = TabularPredictor(label='time', eval_metric=cox_ph_scorer).fit(
            df_model, 
            presets="good_quality",
            time_limit=60,
            verbosity=0
        )
        
        # Make predictions
        predictions = predictor.predict(df_model)
        
        # Evaluate concordance index
        cindex = concordance_index(df['week'], -predictions, event_observed=df['arrest'])
        print(f"Concordance index on training data: {cindex:.4f}")
        
    except ImportError:
        print("\nSkipping AutoGluon example (required packages not installed)")