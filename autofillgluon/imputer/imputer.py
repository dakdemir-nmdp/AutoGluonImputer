# Standard library imports
import os
import shutil
import pickle
import tempfile
from random import shuffle
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# AutoGluon imports
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import contextlib
import sys
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def redirect_stdout_to_file(file_path: str):
    """
    Context manager for redirecting standard output to a file.

    Parameters
    ----------
    file_path : str
        The path to the file where the standard output will be redirected.
        The file is opened in append mode.

    Yields
    ------
    None
    """
    original_stdout = sys.stdout
    file = open(file_path, 'a')
    sys.stdout = file

    try:
        yield
    finally:
        sys.stdout = original_stdout
        file.close()


class Imputer:
    """
    Imputer leveraging AutoGluon for predictive imputation of missing data.

    Parameters
    ----------
    num_iter : int, default=10
        Number of iterations for the imputation process.
    time_limit : int, default=60
        Time in seconds for each individual model training.
    presets : list[str] or str, default=['medium_quality', 'optimize_for_deployment']
        Configuration presets for AutoGluon.
    column_settings : dict, optional
        Fine-tuning for specific columns with keys: 'time_limit', 'presets', 
        'eval_metric', 'label'.
    use_missingness_features : bool, default=False
        If True, add binary indicators for missing values as features.
    simple_impute_columns : list, default=[]
        Columns to use simple imputation (mean/mode) instead of predictive models.
    """

    def __init__(
        self,
        num_iter: int = 10,
        time_limit: int = 60,
        presets: Union[List[str], str] = None,
        column_settings: Optional[Dict] = None,
        use_missingness_features: bool = False,
        simple_impute_columns: Optional[List[str]] = None
    ):
        self.num_iter = num_iter
        self.time_limit = time_limit
        self.presets = presets if presets is not None else ['medium_quality', 'optimize_for_deployment']
        self.col_data_types: Dict[str, str] = {}
        self.initial_imputes: Dict[str, Union[float, str]] = {}
        self.models: Dict[str, TabularPredictor] = {}
        self.colsummary: Dict[str, Dict] = {}
        self.missing_cells: Optional[pd.DataFrame] = None
        self.column_settings = column_settings or {}
        self.use_missingness_features = use_missingness_features
        self.simple_impute_columns = simple_impute_columns or []

    def dataset_overview(
        self,
        train_data: Optional[pd.DataFrame],
        test_data: Optional[pd.DataFrame],
        label: str
    ) -> None:
        """
        Log basic dataset information.

        Parameters
        ----------
        train_data : DataFrame, optional
            Training dataset.
        test_data : DataFrame, optional
            Testing dataset.
        label : str
            Target variable name.
        """
        logger.info(f"Train data shape: {train_data.shape if train_data is not None else 'N/A'}")
        logger.info(f"Test data shape: {test_data.shape if test_data is not None else 'N/A'}")
        logger.info(f"Target label: {label}")

    def missingness_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicators for missing values.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        DataFrame
            Binary indicators for missing values.
        """
        return pd.DataFrame(
            {f"{col}_missing": X[col].isnull().astype(int) for col in X.columns},
            index=X.index
        )

    def _simple_impute_column(self, X_imputed: pd.DataFrame, col: str) -> None:
        """
        Perform simple imputation (mean/mode) for a column.

        Parameters
        ----------
        X_imputed : DataFrame
            DataFrame to impute (modified in-place).
        col : str
            Column name to impute.
        """
        if X_imputed[col].dtype.kind in 'biufc':  # numeric
            fill_value = X_imputed[col].mean()
            X_imputed[col].fillna(fill_value, inplace=True)
        else:  # categorical
            mode_val = X_imputed[col].mode()
            fill_value = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
            X_imputed[col].fillna(fill_value, inplace=True)

    def _cleanup_autogluon_models(self) -> None:
        """Clean up AutoGluon model directories and cache."""
        patterns = ['AutogluonModels', 'ag-', 'tabular_']
        paths = ['.', os.getcwd(), tempfile.gettempdir()]

        for base_path in paths:
            if not os.path.exists(base_path):
                continue

            try:
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    should_remove = (
                        item == 'AutogluonModels' or
                        (item.startswith('ag-') and os.path.isdir(item_path)) or
                        (item.startswith('tabular_') and os.path.isdir(item_path))
                    )

                    if should_remove and os.path.isdir(item_path):
                        try:
                            shutil.rmtree(item_path)
                            logger.debug(f"Cleaned up: {item_path}")
                        except OSError as e:
                            logger.debug(f"Could not remove {item_path}: {e}")
            except OSError as e:
                logger.debug(f"Could not list directory {base_path}: {e}")

        # Force garbage collection
        import gc
        gc.collect()

    def _initialize_imputation(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize imputation with mean/mode values.

        Parameters
        ----------
        X_missing : DataFrame
            Input data with missing values.

        Returns
        -------
        DataFrame
            Data with initial simple imputation applied.
        """
        X_imputed = X_missing.copy()

        if self.use_missingness_features:
            missingness_cols = self.missingness_matrix(X_missing)
            X_imputed = pd.concat([X_imputed, missingness_cols], axis=1)

        for col in X_imputed.columns:
            is_categorical = X_imputed[col].dtype == 'object' or str(X_imputed[col].dtype) == 'category'
            
            if is_categorical:
                self.col_data_types[col] = 'object'
                mode_val = X_imputed[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                X_imputed.loc[:, col] = X_imputed[col].fillna(fill_value)
                self.initial_imputes[col] = fill_value
            else:
                self.col_data_types[col] = 'numeric'
                mean_value = X_imputed[col].mean()
                X_imputed.loc[:, col] = X_imputed[col].fillna(mean_value)
                self.initial_imputes[col] = mean_value

        return X_imputed

    def _create_model_path(self, save_path: Optional[str], iter_num: int, col: str) -> str:
        """
        Create a unique model save path.

        Parameters
        ----------
        save_path : str, optional
            Base path for saving models.
        iter_num : int
            Current iteration number.
        col : str
            Column name.

        Returns
        -------
        str
            Unique model path.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        unique_id = f"{timestamp}_{os.getpid()}_{iter_num}_{col}"

        if save_path is not None:
            model_path = f"{save_path}/{unique_id}/"
        else:
            model_path = f"AutogluonModels/ag-{unique_id}/"

        # Remove existing path to avoid conflicts
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
                logger.debug(f"Removed existing model path: {model_path}")
            except OSError as e:
                logger.warning(f"Could not remove {model_path}: {e}")

        return model_path

    def _train_column_model(
        self,
        X_imputed: pd.DataFrame,
        col: str,
        iter_num: int,
        save_path: Optional[str]
    ) -> Optional[TabularPredictor]:
        """
        Train a predictive model for a single column.

        Parameters
        ----------
        X_imputed : DataFrame
            Data with current imputations.
        col : str
            Column to train model for.
        iter_num : int
            Current iteration number.
        save_path : str, optional
            Base path for saving models.

        Returns
        -------
        TabularPredictor or None
            Trained predictor or None if training failed.
        """
        if (col.endswith('_missing') and self.use_missingness_features) or col in self.simple_impute_columns:
            return None

        logger.debug(f"Processing column {col}")

        mask = ~self.missing_cells[col]
        train_data = X_imputed[mask]

        # Check for sufficient training data
        if len(train_data) < 2:
            logger.warning(f"Insufficient data for {col} ({len(train_data)} rows). Using simple imputation.")
            self._simple_impute_column(X_imputed, col)
            return None

        # Get column-specific settings
        col_config = self.column_settings.get(col, {})
        col_time_limit = col_config.get('time_limit', self.time_limit)
        col_presets = col_config.get('presets', self.presets)
        col_label = col_config.get('label', col)
        col_eval_metric = col_config.get('eval_metric', None)

        model_path = self._create_model_path(save_path, iter_num, col)

        try:
            predictor = TabularPredictor(
                label=col_label,
                eval_metric=col_eval_metric,
                path=model_path
            ).fit(
                train_data,
                time_limit=col_time_limit,
                presets=col_presets,
                verbosity=0
            )

            # Impute missing values
            mask_missing = self.missing_cells[col]
            if mask_missing.any():
                X_imputed.loc[mask_missing, col] = predictor.predict(X_imputed.loc[mask_missing])

            return predictor

        except Exception as e:
            logger.error(f"Error training model for {col}: {e}")
            self._simple_impute_column(X_imputed, col)
            return None

    def fit(self, X_missing: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the imputer to the data with missing values.

        Parameters
        ----------
        X_missing : DataFrame
            Input data with missing values.
        save_path : str, optional
            Path to save models.

        Returns
        -------
        DataFrame
            Imputed data.
        """
        logger.info("Fitting the imputer to the data...")

        # Set environment variables
        os.environ['AUTOGLUON_DISABLE_MODEL_CACHING'] = '1'
        os.environ['AUTOGLUON_DISABLE_OOF_CACHE'] = '1'

        with redirect_stdout_to_file('autogluon_fit.log'):
            self.missing_cells = X_missing.isnull()
            X_imputed = self._initialize_imputation(X_missing)
            self._cleanup_autogluon_models()

            # Iterative imputation
            for iter_num in range(self.num_iter):
                self._cleanup_autogluon_models()
                columns = list(X_imputed.columns)
                shuffle(columns)

                for col in columns:
                    predictor = self._train_column_model(X_imputed, col, iter_num, save_path)
                    if predictor is not None:
                        self.models[col] = predictor

            # Store column summaries
            self.colsummary = {}
            for col in X_imputed.columns:
                if self.col_data_types[col] == 'numeric':
                    self.colsummary[col] = {
                        'min': X_imputed[col].min(),
                        'max': X_imputed[col].max()
                    }
                else:
                    self.colsummary[col] = {'categories': X_imputed[col].unique()}

        logger.info("Fitting complete.")
        return X_imputed

    def transform(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using pre-trained models.

        Parameters
        ----------
        X_missing : DataFrame
            Data with missing values.

        Returns
        -------
        DataFrame
            Imputed data.
        """
        X_imputed = X_missing.copy()

        if self.use_missingness_features:
            missingness_cols = self.missingness_matrix(X_missing)
            X_imputed = pd.concat([X_imputed, missingness_cols], axis=1)

        # Apply initial imputation
        for col in X_imputed.columns:
            if col in self.initial_imputes:
                if X_imputed[col].dtype.kind in 'biufc':
                    X_imputed[col].fillna(float(self.initial_imputes[col]), inplace=True)
                else:
                    X_imputed[col].fillna(str(self.initial_imputes[col]), inplace=True)

        # Iterative imputation
        for _ in range(self.num_iter):
            columns = list(X_imputed.columns)
            shuffle(columns)

            for col in columns:
                if (col.endswith('_missing') and self.use_missingness_features) or col in self.simple_impute_columns:
                    continue

                mask_missing = X_missing[col].isnull()
                if not mask_missing.any():
                    continue

                if col in self.models:
                    try:
                        X_imputed.loc[mask_missing, col] = self.models[col].predict(X_imputed.loc[mask_missing])
                    except Exception as e:
                        logger.error(f"Error predicting column {col}: {e}")
                else:
                    logger.warning(f"No model for column {col}. Skipping.")

            # Apply categorical constraints
            for col in columns:
                if self.col_data_types.get(col) == 'object' and col in self.colsummary:
                    X_imputed[col] = X_imputed[col].astype('category')
                    X_imputed[col] = X_imputed[col].cat.set_categories(
                        self.colsummary[col]['categories']
                    )

        return X_imputed

    def fit_transform(self, X_missing: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the imputer and transform the data in one step.

        Parameters
        ----------
        X_missing : DataFrame
            Input data with missing values.
        save_path : str, optional
            Path to save models.

        Returns
        -------
        DataFrame
            Imputed data.
        """
        return self.fit(X_missing, save_path)

    def feature_importance(self, X_imputed: pd.DataFrame) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Compute feature importance for each column's model.

        Parameters
        ----------
        X_imputed : DataFrame
            Imputed dataset.

        Returns
        -------
        dict
            Feature importances for each column.
        """
        feature_importances = {}
        for col, model in self.models.items():
            try:
                if hasattr(model, 'feature_importance'):
                    feature_importances[col] = model.feature_importance(X_imputed)
                else:
                    feature_importances[col] = None
                    logger.warning(f"Model for '{col}' does not support feature importance.")
            except Exception as e:
                feature_importances[col] = None
                logger.error(f"Error computing feature importance for '{col}': {e}")

        return feature_importances

    def save_models(self, path: str) -> None:
        """
        Save trained models and metadata to disk.

        Parameters
        ----------
        path : str
            Directory path to save models.
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory {path}")

        # Save each model
        for col, model in self.models.items():
            col_path = path_obj / col
            try:
                if hasattr(model, 'path'):
                    if col_path.exists():
                        shutil.rmtree(col_path)
                    shutil.copytree(model.path, col_path)
                    logger.info(f"Saved model for '{col}' to {col_path}")
                else:
                    logger.warning(f"Model for '{col}' has no path attribute")
            except Exception as e:
                logger.error(f"Failed to save model for '{col}': {e}")

        # Save metadata
        try:
            pd.DataFrame.from_dict(self.initial_imputes, orient='index').to_csv(
                path_obj / 'initial_imputes.csv'
            )
            logger.info("Saved initial imputes")
        except Exception as e:
            logger.error(f"Failed to save initial imputes: {e}")

        for name, data in [
            ('colsummary.pkl', self.colsummary),
            ('model_columns.pkl', list(self.models.keys())),
            ('col_data_types.pkl', self.col_data_types)
        ]:
            try:
                with open(path_obj / name, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Saved {name}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

    def load_models(self, path: str) -> None:
        """
        Load trained models and metadata from disk.

        Parameters
        ----------
        path : str
            Directory path containing saved models.
        """
        path_obj = Path(path)

        try:
            # Load model columns
            with open(path_obj / 'model_columns.pkl', 'rb') as f:
                model_columns = pickle.load(f)
            logger.info("Loaded model column names")

            # Load models
            self.models = {
                col: TabularPredictor.load(str(path_obj / col))
                for col in model_columns
            }

            # Load initial imputes
            self.initial_imputes = pd.read_csv(
                path_obj / 'initial_imputes.csv',
                index_col=0,
                header=None
            ).to_dict()[1]

            # Load column summary
            with open(path_obj / 'colsummary.pkl', 'rb') as f:
                self.colsummary = pickle.load(f)

            # Load column data types
            try:
                with open(path_obj / 'col_data_types.pkl', 'rb') as f:
                    self.col_data_types = pickle.load(f)
                logger.info("Loaded column data types")
            except FileNotFoundError:
                logger.warning("Column data types file not found, inferring from data")
                self.col_data_types = {}

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def add_missingness_at_random(
        self,
        data: pd.DataFrame,
        percentage: float
    ) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        """
        Add missingness at random at a specified percentage.

        Parameters
        ----------
        data : DataFrame
            Input data.
        percentage : float
            Percentage of values to set to NaN (0-1).

        Returns
        -------
        DataFrame
            Data with added missingness.
        dict
            Indices of additional missingness for each column.
        """
        modified_data = data.copy(deep=True)
        missingness_indices = {}

        for col in modified_data.columns:
            if col not in self.models:
                logger.warning(f"No model for column {col}. Skipping...")
                continue

            non_missing_mask = modified_data[col].notna()
            non_missing_indices = modified_data[non_missing_mask].index.to_numpy()

            n_missing = int(len(non_missing_indices) * percentage)
            logger.info(f"Adding missingness to {col}: {n_missing} values")

            missing_idx = np.random.choice(non_missing_indices, size=n_missing, replace=False)
            modified_data.loc[missing_idx, col] = np.nan
            missingness_indices[col] = missing_idx.tolist()

        logger.info("Missingness addition complete")
        return modified_data, missingness_indices

    def evaluate_imputation(
        self,
        data: pd.DataFrame,
        percentage: float,
        ntimes: int = 10
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Evaluate imputation performance by introducing and imputing random missingness.

        Parameters
        ----------
        data : DataFrame
            Original complete dataset.
        percentage : float
            Percentage of data to make missing (0-1).
        ntimes : int, default=10
            Number of evaluation iterations.

        Returns
        -------
        dict
            Nested dictionary with evaluation metrics per iteration and column.
            Format: {iteration: {column: {metric: value}}}
        """
        accuracies = {}

        for rep in range(ntimes):
            results = {}
            modified_data, missingness_indices = self.add_missingness_at_random(data, percentage)
            logger.info(f"Introduced missingness in {percentage*100}% of dataset")
            
            imputed_data = self.transform(modified_data)
            logger.info("Data imputation completed")

            for col in data.columns:
                if col not in self.models:
                    logger.warning(f"No model for column {col}. Skipping...")
                    continue

                y_true = data.loc[missingness_indices[col], col]
                y_pred = imputed_data.loc[missingness_indices[col], col]

                # Remove NaN values
                valid_mask = y_true.notna() & y_pred.notna()
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]

                if len(y_true) == 0:
                    logger.warning(f"No valid predictions for {col}")
                    continue

                logger.info(f"Evaluating column {col}")

                if self.col_data_types[col] == 'object':
                    acc = accuracy_score(y_true, y_pred)
                    results[col] = {'accuracy': acc}
                    logger.info(f"Accuracy for {col}: {acc:.4f}")
                else:
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    results[col] = {'mse': mse, 'mae': mae}
                    logger.info(f"MSE for {col}: {mse:.4f}, MAE: {mae:.4f}")

            accuracies[rep] = results

        logger.info("Imputation evaluation completed")
        return accuracies


def multiple_imputation(
    data: pd.DataFrame,
    n_imputations: int = 5,
    fitonce: bool = False,
    **kwargs
) -> List[pd.DataFrame]:
    """
    Perform multiple imputation on a dataset.

    Parameters
    ----------
    data : DataFrame
        Dataset with missing values.
    n_imputations : int, default=5
        Number of imputations to perform.
    fitonce : bool, default=False
        If True, fit model once and reuse for all imputations.
    **kwargs
        Additional arguments passed to Imputer class.

    Returns
    -------
    list of DataFrame
        List of imputed datasets.
    """
    imputed_datasets = []

    if fitonce:
        logger.info("Fitting model once for multiple imputations...")
        imputer = Imputer(**kwargs)
        imputer.fit(data)
        
        for i in range(n_imputations):
            try:
                imputed_data = imputer.transform(data)
                imputed_datasets.append(imputed_data)
            except Exception as e:
                logger.error(f"Error during imputation {i+1}: {e}")
                break
    else:
        for i in range(n_imputations):
            logger.debug(f"Performing imputation {i+1}/{n_imputations}")
            try:
                imputer = Imputer(**kwargs)
                imputed_data = imputer.fit(data)
                imputed_datasets.append(imputed_data)
            except Exception as e:
                logger.error(f"Error during imputation {i+1}: {e}")
                break

    return imputed_datasets