# Russia–Ukraine War — User Classification Pipeline

This repository contains ML pipelines for classifying social media users related to the Russia-Ukraine war.

## ML Classifiers

Three binary classifiers that analyze user profiles:

| Classifier | Purpose | Features | Best Accuracy |
|------------|---------|----------|---------------|
| **LocalClassifier** | Is user living in Russia/Ukraine? | language detection, location | ~90% |
| **NationalityClassifier** | Is user Russian/Ukrainian by nationality? | language detection, location | ~96% |
| **PrivateClassifier** | Is user private individual or organization? | followers, following, TF-IDF (name/bio) | ~87% |

## Repository Structure

```
├── ML/
│   ├── classifiers/
│   │   ├── local_classifier.py
│   │   ├── nationality_classifier.py
│   │   └── private_classifier.py
│   ├── best_model_results/      # Experiment results (CSV)
│   └── presentation/            # ML Pipeline presentation
├── lib/
│   ├── generate_features.py     # Feature extraction pipeline
│   ├── translate.py             # Translation + language detection
│   ├── classify_location.py     # Location classification (LLM)
│   └── Outputs/                 # Processed data files
└── Russia/, Ukraine/            # Raw labeled data
```

## Quick Start

```python
from ML.classifiers import LocalClassifier, NationalityClassifier, PrivateClassifier

# Run experiments
classifier = LocalClassifier(country='Russia')
classifier.run_all_experiments()
classifier.save_results()
```