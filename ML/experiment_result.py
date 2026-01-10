"""
Experiment Result Module
Data class to store and export ML experiment results.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class ExperimentResult:
    """Data class to store experiment results."""
    iteration: int
    target_column: str
    num_classes: int
    class_0_count: int
    class_1_count: int
    class_2_count: int
    min_class_size: int
    training_type: str  # K-Fold or LOOCV
    k: Optional[int]  # K value for K-Fold, empty for LOOCV
    algorithm: str
    features_count: int
    feature_set: str  # Description of features used (e.g., "name_language,bio_language")
    tfidf_features: str  # TF-IDF columns used (e.g., "bio,name" or empty)
    balanced: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment result to a dictionary for CSV export.
        
        Returns:
            Dictionary with experiment result fields (without iteration)
        """
        return {
            'target_column': self.target_column,
            '#classes': self.num_classes,
            '#class_0': self.class_0_count,
            '#class_1': self.class_1_count,
            '#class_2': self.class_2_count,
            'min_class_size': self.min_class_size,
            'training_type': self.training_type,
            'K': self.k if self.k else '',
            'algorithm': self.algorithm,
            'Features_count': self.features_count,
            'feature_set': self.feature_set,
            'TFIDF_features': self.tfidf_features if self.tfidf_features else '',
            'balanced': self.balanced,
            'accuracy': round(self.accuracy, 3),
            'precision': round(self.precision, 3),
            'recall': round(self.recall, 3),
            'F1': round(self.f1, 3),
            'AUC': round(self.auc, 3)
        }
    
    def __str__(self) -> str:
        """String representation of the experiment result."""
        return (
            f"ExperimentResult(iteration={self.iteration}, "
            f"algorithm={self.algorithm}, "
            f"training_type={self.training_type}, "
            f"balanced={self.balanced}, "
            f"accuracy={self.accuracy:.3f}, "
            f"f1={self.f1:.3f}, "
            f"auc={self.auc:.3f})"
        )

