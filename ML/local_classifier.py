"""
Local Classifier Module
Classifies whether a user is local (lives in Russia or Ukraine) based on their profile features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Dict, List, Any, Tuple
import warnings

# Sklearn imports
from sklearn.model_selection import (
    StratifiedKFold, 
    LeaveOneOut, 
    GridSearchCV,
    cross_val_predict
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)
from sklearn.pipeline import Pipeline

# XGBoost
from xgboost import XGBClassifier

# Local imports
from .experiment_result import ExperimentResult

warnings.filterwarnings('ignore')


class LocalClassifier:
    """
    Classifier for determining if a user is local (lives in Russia or Ukraine).
    
    Supports multiple classification algorithms with hyperparameter tuning,
    K-Fold and Leave-One-Out cross validation, and balanced/imbalanced modes.
    """
    
    # Columns to exclude from features (raw text that would cause data leakage)
    EXCLUDE_COLUMNS = ['name', 'bio']
    
    # Columns to use as categorical features
    FEATURE_COLUMNS = ['classified_country', 'name_language', 'bio_language', 'location_language']
    
    ALGORITHMS = {
        'LogReg': LogisticRegression,
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'SVM': SVC,
        'XGBoost': XGBClassifier,
        'AdaBoost': AdaBoostClassifier
    }
    
    HYPERPARAMETERS = {
        'LogReg': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [1000]
        },
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto']
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__subsample': [0.8, 1.0]
        },
        'AdaBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
    }
    
    def __init__(
        self, 
        country: Literal['Russia', 'Ukraine'],
        label_column: str = 'local:judge',
        random_state: int = 42
    ):
        """
        Initialize the LocalClassifier.
        
        Args:
            country: The country to classify for ('Russia' or 'Ukraine')
            label_column: The column name containing the labels
            random_state: Random state for reproducibility
        """
        self.country = country
        self.label_column = label_column
        self.random_state = random_state
        self.results: List[ExperimentResult] = []
        self.iteration_counter = 0
        
        # Load the data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load the data from the Excel file."""
        project_dir = Path(__file__).resolve().parent.parent
        data_path = project_dir / "lib" / "Outputs" / self.country / f"{self.country}_local.xlsx"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.df = pd.read_excel(data_path)
        print(f"Loaded {len(self.df)} samples from {data_path}")
    
    def _normalize_label(self, value) -> str:
        """
        Normalize label values to consistent strings.
        
        Args:
            value: The label value (can be bool, string, int, etc.)
            
        Returns:
            Normalized string: 'true', 'false', or 'unknown'
        """
        if pd.isna(value):
            return 'unknown'
        
        # Handle boolean values
        if isinstance(value, bool):
            return 'true' if value else 'false'
        
        # Handle numpy bool
        if isinstance(value, np.bool_):
            return 'true' if value else 'false'
        
        # Handle numeric values (1/0)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if value == 1:
                return 'true'
            elif value == 0:
                return 'false'
            return 'unknown'
        
        # Handle string values
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ('true', 't', 'yes', 'y', '1'):
                return 'true'
            elif value_lower in ('false', 'f', 'no', 'n', '0'):
                return 'false'
        
        return 'unknown'
        
    def _prepare_features(
        self, 
        df: pd.DataFrame,
        num_classes: int = 2
    ) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, int]:
        """
        Prepare features and labels for classification.
        
        Args:
            df: DataFrame with the data
            num_classes: 2 for binary (true/false), 3 for multi-class (true/false/unknown)
            
        Returns:
            X: Feature matrix
            y: Label vector
            label_encoder: Fitted label encoder
            num_original_features: Number of original feature columns used
        """
        # Work on a copy
        df_work = df.copy()
        
        # Normalize all label values
        df_work[self.label_column] = df_work[self.label_column].apply(self._normalize_label)
        
        # Filter by number of classes
        if num_classes == 2:
            # Keep only true/false labels
            df_work = df_work[df_work[self.label_column].isin(['true', 'false'])]
        else:
            # Keep all including 'unknown'
            pass
        
        # Separate features and labels
        y = df_work[self.label_column]
        
        # Only use specified feature columns (exclude raw text like name, bio)
        available_features = [col for col in self.FEATURE_COLUMNS if col in df_work.columns]
        num_original_features = len(available_features)
        X = df_work[available_features].copy()
        
        # Encode categorical features
        X = self._encode_features(X)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        return X, pd.Series(y_encoded, index=X.index), label_encoder, num_original_features
    
    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Encoded feature DataFrame
        """
        X_encoded = X.copy()
        
        # Handle categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Fill NaN with 'unknown'
            X_encoded[col] = X_encoded[col].fillna('unknown')
        
        # One-hot encode categorical columns
        if len(categorical_cols) > 0:
            X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols, drop_first=False)
        
        # Fill any remaining NaN with 0
        X_encoded = X_encoded.fillna(0)
        
        return X_encoded
    
    def _balance_dataset(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset by undersampling to match the smallest class.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Balanced X and y
        """
        # Find the minimum class size
        class_counts = y.value_counts()
        min_count = class_counts.min()
        
        # Sample equal numbers from each class
        balanced_indices = []
        for class_label in class_counts.index:
            class_indices = y[y == class_label].index.tolist()
            sampled_indices = np.random.choice(
                class_indices, 
                size=min_count, 
                replace=False
            )
            balanced_indices.extend(sampled_indices)
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        return X.loc[balanced_indices], y.loc[balanced_indices]
    
    def _get_classifier_instance(self, algorithm: str) -> Any:
        """
        Get a classifier instance for the given algorithm.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Classifier instance
        """
        clf_class = self.ALGORITHMS[algorithm]
        
        if algorithm == 'LogReg':
            return clf_class(random_state=self.random_state, max_iter=1000)
        elif algorithm == 'SVM':
            return clf_class(random_state=self.random_state, probability=True)
        elif algorithm == 'XGBoost':
            return clf_class(
                random_state=self.random_state, 
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif algorithm in ['Decision Tree', 'Random Forest', 'AdaBoost']:
            return clf_class(random_state=self.random_state)
        else:
            return clf_class()
    
    def _create_pipeline(self, algorithm: str) -> Pipeline:
        """
        Create a sklearn pipeline with scaling and classifier.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Pipeline object
        """
        classifier = self._get_classifier_instance(algorithm)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def _perform_hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Label vector
            algorithm: Name of the algorithm
            cv: Number of cross-validation folds for tuning
            
        Returns:
            Best parameters
        """
        pipeline = self._create_pipeline(algorithm)
        param_grid = self.HYPERPARAMETERS[algorithm]
        
        # Use stratified k-fold for tuning
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            error_score='raise'
        )
        
        try:
            grid_search.fit(X, y)
            return grid_search.best_params_
        except Exception as e:
            print(f"Warning: Hyperparameter tuning failed for {algorithm}: {e}")
            return {}
    
    def _evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        num_classes: int
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)
            num_classes: Number of classes
            
        Returns:
            Dictionary with metrics
        """
        average = 'binary' if num_classes == 2 else 'weighted'
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Calculate AUC
        try:
            if y_proba is not None:
                if num_classes == 2:
                    # Binary classification
                    if y_proba.ndim == 2:
                        metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_proba)
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(
                        y_true, 
                        y_proba, 
                        multi_class='ovr', 
                        average='weighted'
                    )
            else:
                metrics['auc'] = 0.0
        except Exception:
            metrics['auc'] = 0.0
        
        return metrics
    
    def _run_kfold_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        best_params: Dict[str, Any],
        k: int = 5,
        num_classes: int = 2
    ) -> Dict[str, float]:
        """
        Run K-Fold cross validation.
        
        Args:
            X: Feature matrix
            y: Label vector
            algorithm: Name of the algorithm
            best_params: Best hyperparameters from tuning
            k: Number of folds
            num_classes: Number of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        pipeline = self._create_pipeline(algorithm)
        
        # Apply best parameters
        if best_params:
            pipeline.set_params(**best_params)
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        
        # Collect predictions
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, method='predict')
        
        # Try to get probabilities
        try:
            y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')
        except Exception:
            y_proba = None
        
        return self._evaluate_model(y, y_pred, y_proba, num_classes)
    
    def _run_loocv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        best_params: Dict[str, Any],
        num_classes: int = 2
    ) -> Dict[str, float]:
        """
        Run Leave-One-Out cross validation.
        
        Args:
            X: Feature matrix
            y: Label vector
            algorithm: Name of the algorithm
            best_params: Best hyperparameters from tuning
            num_classes: Number of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        pipeline = self._create_pipeline(algorithm)
        
        # Apply best parameters
        if best_params:
            pipeline.set_params(**best_params)
        
        # Initialize LOOCV
        loo = LeaveOneOut()
        
        # Collect predictions
        y_pred = cross_val_predict(pipeline, X, y, cv=loo, method='predict')
        
        # Try to get probabilities
        try:
            y_proba = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba')
        except Exception:
            y_proba = None
        
        return self._evaluate_model(y, y_pred, y_proba, num_classes)
    
    def run_experiment(
        self,
        algorithm: str,
        training_type: Literal['K-Fold', 'LOOCV'],
        balanced: bool = False,
        num_classes: int = 2,
        k: int = 5
    ) -> ExperimentResult:
        """
        Run a single experiment with the specified configuration.
        
        Args:
            algorithm: Name of the classification algorithm
            training_type: Type of cross validation ('K-Fold' or 'LOOCV')
            balanced: Whether to balance the dataset
            num_classes: Number of classes (2 or 3)
            k: Number of folds for K-Fold CV
            
        Returns:
            ExperimentResult object
        """
        print(f"\n{'='*60}")
        print(f"Running: {algorithm} | {training_type} | {'Balanced' if balanced else 'Imbalanced'} | {num_classes} classes")
        print(f"{'='*60}")
        
        # Prepare features
        X, y, label_encoder, num_original_features = self._prepare_features(self.df, num_classes)
        
        # Balance if requested
        if balanced:
            X, y = self._balance_dataset(X, y)
        
        # Get class counts
        class_counts = y.value_counts().sort_index()
        class_0_count = class_counts.get(0, 0)
        class_1_count = class_counts.get(1, 0)
        class_2_count = class_counts.get(2, 0) if num_classes == 3 else 0
        min_class_size = class_counts.min()
        
        print(f"Dataset size: {len(X)}")
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Number of original features: {num_original_features}")
        
        # Hyperparameter tuning
        print("Performing hyperparameter tuning...")
        best_params = self._perform_hyperparameter_tuning(X, y, algorithm)
        print(f"Best parameters: {best_params}")
        
        # Run cross validation
        if training_type == 'K-Fold':
            print(f"Running {k}-Fold cross validation...")
            metrics = self._run_kfold_cv(X, y, algorithm, best_params, k, num_classes)
        else:
            print("Running Leave-One-Out cross validation...")
            metrics = self._run_loocv(X, y, algorithm, best_params, num_classes)
        
        print(f"Results: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        # Create result object
        self.iteration_counter += 1
        result = ExperimentResult(
            iteration=self.iteration_counter,
            target_column=self.label_column,
            num_classes=num_classes,
            class_0_count=class_0_count,
            class_1_count=class_1_count,
            class_2_count=class_2_count,
            min_class_size=min_class_size,
            training_type=training_type,
            k=k if training_type == 'K-Fold' else None,
            algorithm=algorithm,
            features_count=num_original_features,
            balanced=balanced,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            auc=metrics['auc']
        )
        
        self.results.append(result)
        return result
    
    def run_all_experiments(
        self,
        algorithms: Optional[List[str]] = None,
        training_types: Optional[List[str]] = None,
        balanced_modes: Optional[List[bool]] = None,
        num_classes_list: Optional[List[int]] = None,
        k: int = 5
    ) -> List[ExperimentResult]:
        """
        Run all combinations of experiments.
        
        Args:
            algorithms: List of algorithms to test (default: all)
            training_types: List of training types (default: ['K-Fold', 'LOOCV'])
            balanced_modes: List of balanced modes (default: [True, False])
            num_classes_list: List of num_classes values (default: [2, 3])
            k: Number of folds for K-Fold CV
            
        Returns:
            List of ExperimentResult objects
        """
        if algorithms is None:
            algorithms = list(self.ALGORITHMS.keys())
        if training_types is None:
            training_types = ['K-Fold', 'LOOCV']
        if balanced_modes is None:
            balanced_modes = [True, False]
        if num_classes_list is None:
            num_classes_list = [2, 3]
        
        total_experiments = (
            len(algorithms) * 
            len(training_types) * 
            len(balanced_modes) * 
            len(num_classes_list)
        )
        print(f"\n{'#'*60}")
        print(f"Starting {total_experiments} experiments for {self.country}")
        print(f"{'#'*60}")
        
        for num_classes in num_classes_list:
            for balanced in balanced_modes:
                for training_type in training_types:
                    for algorithm in algorithms:
                        try:
                            self.run_experiment(
                                algorithm=algorithm,
                                training_type=training_type,
                                balanced=balanced,
                                num_classes=num_classes,
                                k=k
                            )
                        except Exception as e:
                            print(f"Error running experiment: {e}")
                            continue
        
        return self.results
    
    def save_results(
        self, 
        output_path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save experiment results to a CSV file.
        
        Args:
            output_path: Directory to save the file (default: best_model_results/)
            filename: Name of the output file (default: {country}_experiments_results.csv)
            
        Returns:
            Path to the saved file
        """
        if not self.results:
            raise ValueError("No results to save. Run experiments first.")
        
        if output_path is None:
            project_dir = Path(__file__).resolve().parent.parent
            output_path = project_dir / "best_model_results"
        else:
            output_path = Path(output_path)
        
        if filename is None:
            filename = f"{self.country}_experiments_results.csv"
        
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save to CSV
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")
        
        return str(filepath)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get experiment results as a pandas DataFrame.
        
        Returns:
            DataFrame with experiment results
        """
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def reset_results(self) -> None:
        """Reset the results and iteration counter."""
        self.results = []
        self.iteration_counter = 0
        print("Results reset.")


def main():
    """Main function to run the classifier experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Local Classifier experiments')
    parser.add_argument(
        '--country', 
        type=str, 
        choices=['Russia', 'Ukraine'], 
        required=True,
        help='Country to classify for'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        choices=['LogReg', 'Decision Tree', 'Random Forest', 'SVM', 'XGBoost', 'AdaBoost'],
        default=None,
        help='Algorithms to run (default: all)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of folds for K-Fold CV (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = LocalClassifier(country=args.country)
    
    # Run all experiments
    classifier.run_all_experiments(
        algorithms=args.algorithms,
        k=args.k
    )
    
    # Save results
    classifier.save_results(output_path=args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    results_df = classifier.get_results_dataframe()
    print(results_df.to_string())


if __name__ == "__main__":
    main()
