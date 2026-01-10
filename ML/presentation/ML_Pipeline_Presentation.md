# Local User Classification - ML Pipeline
## Classifying Local vs Non-Local Users in Russia/Ukraine

---

# Slide 1: Project Overview

## Goal
Classify whether a social media user is **local** (lives in Russia/Ukraine) based on their profile features.

## Dataset
- **253 samples** (Russia dataset)
- **Labels**: True (local) / False (not local)
- **Class distribution**: 80 True (31.6%) | 158 False (62.5%) | 15 Unknown (5.9%)

---

# Slide 2: Input Features

## Raw Data Columns
| Column | Description | Example |
|--------|-------------|---------|
| `name` | User's display name | "Oksana Tserkovna" |
| `bio` | User's bio text (translated to English) | "Free Creator, Author of..." |
| `classified_country` | Location classification | "Ukraine", "Russia", "Estonia" |
| `name_language` | Detected language of name | "russian", "english" |
| `bio_language` | Detected language of bio | "russian", "english" |
| `location_language` | Detected language of location | "russian", "german" |

## Features Used for Classification
1. **Categorical Features**: `name_language`, `bio_language`, `location_language`
2. **Text Features (TF-IDF)**: `bio` content

---

# Slide 3: Data Preprocessing Pipeline

```
┌─────────────────┐
│   Raw Excel     │
│   Data (253)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clean Features │  ← Remove emojis, special characters
│  (FeatureCleaner)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Translate     │  ← Translate to English + detect original language
│   (Translator)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classify Location│ ← Use LLM to classify country from location text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Dataset  │
│  (253 samples)  │
└─────────────────┘
```

---

# Slide 4: Feature Engineering

## 1. One-Hot Encoding (Categorical Features)

```
bio_language = "russian"  →  bio_language_russian=1, bio_language_english=0, ...
bio_language = "english"  →  bio_language_russian=0, bio_language_english=1, ...
```

**Result**: 3 columns → 85 binary features

## 2. TF-IDF Vectorization (Text Features)

```
bio = "I live in Moscow, software developer"
      ↓
TF-IDF scores: {moscow: 0.82, developer: 0.45, live: 0.31, ...}
```

**Parameters**:
- `max_features=100` (top 100 words)
- `min_df=2` (word must appear in ≥2 documents)
- `ngram_range=(1,2)` (unigrams + bigrams)

**Result**: 100 additional features from bio text

---

# Slide 5: ML Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   Features  │───▶│   Scaler    │───▶│  Classifier │       │
│  │  (185 dim)  │    │ (StandardScaler)│ │  (RF/SVM/..)│       │
│  └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                               │
│        85 categorical + 100 TF-IDF = 185 total features      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

# Slide 6: Algorithms Tested

| Algorithm | Description |
|-----------|-------------|
| **Logistic Regression** | Linear classifier with regularization |
| **Decision Tree** | Tree-based splits on features |
| **Random Forest** | Ensemble of decision trees |
| **SVM** | Support Vector Machine with kernels |
| **XGBoost** | Gradient boosting trees |
| **AdaBoost** | Adaptive boosting ensemble |

## Hyperparameter Tuning
- **GridSearchCV** with 3-fold cross-validation
- Optimized for **F1-weighted** score

---

# Slide 7: Cross-Validation Methods

## K-Fold Cross-Validation (K=5)

```
253 samples → 5 folds (~50 samples each)

Fold 1: Train on 202, Test on 51 → 51 predictions
Fold 2: Train on 202, Test on 51 → 51 predictions
Fold 3: Train on 203, Test on 50 → 50 predictions
Fold 4: Train on 203, Test on 50 → 50 predictions
Fold 5: Train on 203, Test on 51 → 51 predictions
                                   ─────────────
                         Total:    253 predictions
```

## Leave-One-Out Cross-Validation (LOOCV)

```
253 iterations:
- Each iteration: Train on 252 samples, Test on 1 sample
- Most thorough evaluation (trains 253 models)
```

---

# Slide 8: Experiment Configurations

## Total Experiments: 48

| Parameter | Options |
|-----------|---------|
| Algorithms | 6 (LogReg, Decision Tree, Random Forest, SVM, XGBoost, AdaBoost) |
| CV Methods | 2 (K-Fold, LOOCV) |
| Balanced | 2 (True, False) |
| Num Classes | 2 (binary: True/False, or 3-class: True/False/Unknown) |

**Calculation**: 6 × 2 × 2 × 2 = **48 experiments**

---

# Slide 9: Best Results

## Top Performance Metrics

| Metric | Score | Model | Configuration |
|--------|-------|-------|---------------|
| **Best Accuracy** | **89.5%** | Decision Tree | LOOCV, Imbalanced, TF-IDF |
| **Best F1** | **0.887** | XGBoost | K-Fold, Balanced, TF-IDF |
| **Best AUC** | **0.949** | Random Forest | K-Fold, Balanced, TF-IDF |

## Key Observation
All best results used **TF-IDF on bio text**!

---

# Slide 10: Why Does the Model Perform So Well?

## 1. Strong Signal in Language Features

| Bio Language | % Local (True) | % Not Local (False) |
|--------------|----------------|---------------------|
| **Russian** | **65.4%** | 34.6% |
| **English** | 18.6% | **81.4%** |

→ Just knowing `bio_language = russian` gives 65% chance of correct prediction!

---

# Slide 11: Why Does the Model Perform So Well? (cont.)

## 2. Feature Combination Effect

```
bio_language=russian + name_language=russian + location_language=russian
                            ↓
              Very strong signal for "local = True"
```

## 3. TF-IDF Adds Content Signal

Bio content provides additional clues:
- Location names (Moscow, St. Petersburg)
- Cultural references specific to locals
- Job descriptions common in Russia

---

# Slide 12: Why Does the Model Perform So Well? (cont.)

## 4. The Task is Inherently "Easy"

The classification question is essentially:
> "Does this person write in Russian and have Russian-sounding content?"

**Easier than**:
- Sentiment analysis (subjective)
- Topic classification (requires deep understanding)
- Intent detection (ambiguous)

## 5. Clean Data Helps

- FeatureCleaner removes noise (emojis, special characters)
- Translated text is consistent (all English)
- Language detection provides reliable signal

---

# Slide 13: Feature Importance Analysis

## Language Distribution in Dataset

```
Bio Language Distribution:
├── Russian:    78 samples (30.8%)
├── English:    59 samples (23.3%)
├── Ukrainian:  45 samples (17.8%)
├── Other:      71 samples (28.1%)
```

## Correlation with Label

| Condition | P(Local=True) |
|-----------|---------------|
| bio_language = russian | 65.4% |
| bio_language = ukrainian | ~40% |
| bio_language = english | 18.6% |

---

# Slide 14: Model Comparison

## Accuracy by Algorithm (Balanced, K-Fold, TF-IDF)

| Algorithm | Accuracy | F1 | AUC |
|-----------|----------|-----|-----|
| Logistic Regression | ~82% | ~0.82 | ~0.90 |
| Decision Tree | ~85% | ~0.85 | ~0.85 |
| **Random Forest** | **~84%** | **~0.84** | **~0.95** |
| SVM | ~83% | ~0.83 | ~0.91 |
| **XGBoost** | **~85%** | **~0.89** | ~0.93 |
| AdaBoost | ~82% | ~0.82 | ~0.89 |

---

# Slide 15: TF-IDF Impact

## With vs Without TF-IDF

| Configuration | Accuracy | AUC |
|---------------|----------|-----|
| Without TF-IDF | 83.1% | 0.933 |
| **With TF-IDF** | **83.8%** | 0.918 |

## Conclusion
TF-IDF provides **slight improvement** in accuracy after data cleaning.

Old (uncleaned) data: TF-IDF **hurt** performance
New (cleaned) data: TF-IDF **helps** performance

---

# Slide 16: Limitations & Considerations

## Small Dataset
- 253 samples may not capture all patterns
- Risk of overfitting
- Results might not generalize to new data

## Class Imbalance
- 158 False (62.5%) vs 80 True (31.6%)
- Balanced mode undersamples to 160 samples

## Feature Leakage Risk
- Language features directly correlate with locality
- Model learns "Russian = local" pattern
- May not work for Russian speakers living abroad

---

# Slide 17: Conclusions

## Key Findings

1. **Best Model**: Decision Tree with LOOCV achieves **89.5% accuracy**

2. **Most Important Features**: Language detection features (bio_language, name_language)

3. **TF-IDF Value**: Adds ~1-5% improvement with cleaned data

4. **Why It Works**: Strong correlation between writing language and locality

## Recommendations

- Use **Random Forest** or **XGBoost** for best balance of accuracy and AUC
- Include **TF-IDF on bio** for additional signal
- Consider **balanced mode** for fair evaluation

---

# Slide 18: Future Work

## Potential Improvements

1. **More Data**: Collect larger dataset for better generalization

2. **Additional Features**:
   - Posting time patterns
   - Network connections
   - Content topics

3. **Advanced NLP**:
   - Word embeddings (Word2Vec, BERT)
   - Sentiment analysis of bio

4. **Ensemble Methods**:
   - Combine best models
   - Stacking classifiers

---

# Thank You!

## Questions?

### Code Repository Structure
```
ukraine_russia_war/
├── ML/
│   ├── main.py                 # Configuration & runner
│   ├── experiment_result.py    # Result data class
│   ├── classifiers/
│   │   └── local_classifier.py # Main classifier
│   └── best_model_results/
│       └── Russia_local_experiments_results.csv
└── lib/
    └── Outputs/Russia/Russia_local.xlsx
```

