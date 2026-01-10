
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ML.classifiers.local_classifier import LocalClassifier



# Country to classify: 'Russia' or 'Ukraine'
COUNTRY = 'Russia'

# Label column to use for classification
LABEL_COLUMN = 'local:judge'

# Feature columns to use for prediction
FEATURE_COLUMNS = [
    'name_language',
    'bio_language', 
    'location_language'
]


TFIDF_COLUMNS = [
     #'bio'  
]

# Algorithms to run
# Options: 'LogReg', 'Decision Tree', 'Random Forest', 'SVM', 'XGBoost', 'AdaBoost'
ALGORITHMS = [
    'LogReg',
    'Decision Tree',
    'Random Forest',
    'SVM',
    'XGBoost',
    'AdaBoost'
]

# Cross-validation types to use
# Options: 'K-Fold', 'LOOCV'
TRAINING_TYPES = [
    'K-Fold',
    'LOOCV'
]

# Number of folds for K-Fold cross-validation
K_FOLDS = 5

# Whether to run balanced experiments (same number of samples per class)
# Options: [True], [False], or [True, False] for both
BALANCED_MODES = [True, False]

# Number of classes
# Options: [2] for binary (true/false), [3] for multi-class (true/false/unknown), or [2, 3] for both
NUM_CLASSES = [2, 3]

# Output directory (None = default: ML/best_model_results/)
OUTPUT_DIR = None



def main():
    print("=" * 60)
    print("LOCAL CLASSIFIER EXPERIMENTS")
    print("=" * 60)
    print(f"Country: {COUNTRY}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Categorical Features: {FEATURE_COLUMNS}")
    print(f"TF-IDF Features: {TFIDF_COLUMNS if TFIDF_COLUMNS else 'None'}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Training types: {TRAINING_TYPES}")
    print(f"K-Folds: {K_FOLDS}")
    print(f"Balanced modes: {BALANCED_MODES}")
    print(f"Num classes: {NUM_CLASSES}")
    print("=" * 60)
    
    # Initialize classifier
    classifier = LocalClassifier(
        country=COUNTRY,
        feature_columns=FEATURE_COLUMNS,
        tfidf_columns=TFIDF_COLUMNS if TFIDF_COLUMNS else None,
        label_column=LABEL_COLUMN
    )
    
    # Run experiments
    classifier.run_all_experiments(
        algorithms=ALGORITHMS,
        training_types=TRAINING_TYPES,
        balanced_modes=BALANCED_MODES,
        num_classes_list=NUM_CLASSES,
        k=K_FOLDS
    )
    
    # Save results
    classifier.save_results(output_path=OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    results_df = classifier.get_results_dataframe()
    print(results_df.to_string())
    
    # Print best results
    print("\n" + "=" * 60)
    print("BEST RESULTS BY METRIC")
    print("=" * 60)
    
    if not results_df.empty:
        print(f"\nBest Accuracy: {results_df['accuracy'].max():.3f}")
        best_acc_idx = results_df['accuracy'].idxmax()
        print(f"  -> {results_df.loc[best_acc_idx, 'algorithm']} | {results_df.loc[best_acc_idx, 'training_type']} | Balanced={results_df.loc[best_acc_idx, 'balanced']}")
        tfidf_info = results_df.loc[best_acc_idx, 'TFIDF_features']
        if tfidf_info:
            print(f"     TF-IDF: {tfidf_info}")
        
        print(f"\nBest F1: {results_df['F1'].max():.3f}")
        best_f1_idx = results_df['F1'].idxmax()
        print(f"  -> {results_df.loc[best_f1_idx, 'algorithm']} | {results_df.loc[best_f1_idx, 'training_type']} | Balanced={results_df.loc[best_f1_idx, 'balanced']}")
        tfidf_info = results_df.loc[best_f1_idx, 'TFIDF_features']
        if tfidf_info:
            print(f"     TF-IDF: {tfidf_info}")
        
        print(f"\nBest AUC: {results_df['AUC'].max():.3f}")
        best_auc_idx = results_df['AUC'].idxmax()
        print(f"  -> {results_df.loc[best_auc_idx, 'algorithm']} | {results_df.loc[best_auc_idx, 'training_type']} | Balanced={results_df.loc[best_auc_idx, 'balanced']}")
        tfidf_info = results_df.loc[best_auc_idx, 'TFIDF_features']
        if tfidf_info:
            print(f"     TF-IDF: {tfidf_info}")


if __name__ == "__main__":
    main()

