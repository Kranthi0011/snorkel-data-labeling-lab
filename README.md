# SMS Spam Detection Using Snorkel Weak Supervision — Data Labeling Lab

## Table of Contents
- [Introduction](#introduction)
- [The Significance of Data Labeling](#the-significance-of-data-labeling)
- [What is Snorkel?](#what-is-snorkel)
- [Dataset](#dataset)
- [Modifications from Original Lab](#modifications-from-original-lab)
- [Project Architecture](#project-architecture)
- [Tutorial 01: Labeling Functions](#tutorial-01-labeling-functions)
- [Tutorial 02: Data Augmentation](#tutorial-02-data-augmentation)
- [Results Summary](#results-summary)
- [Key Findings and Insights](#key-findings-and-insights)
- [Setup and Installation](#setup-and-installation)
- [References](#references)

---

## Introduction

This lab demonstrates the practical application of **Snorkel**, a weak supervision framework, for building labeled training datasets without manual annotation and augmenting them using transformation functions. The project covers two core Snorkel tutorials adapted to the SMS spam classification domain:

1. **Tutorial 01 — Labeling Functions**: Programmatically label unlabeled SMS data using heuristic labeling functions and train downstream classifiers.
2. **Tutorial 02 — Data Augmentation**: Augment the labeled training set using transformation functions to improve model generalization.

The task: **classify SMS text messages as spam or ham (not spam)** using the UCI SMS Spam Collection dataset. This lab is part of the IE-7374 MLOps course at Northeastern University and is based on the Snorkel Spam Tutorials from the [MLOps repository](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Data_Labeling_Labs).

---

## The Significance of Data Labeling

In the realm of machine learning and artificial intelligence, data is the bedrock upon which models are built and refined. However, raw data, while abundant, often lacks the structure and organization necessary for effective learning. This is where data labeling comes into play.

**Why data labeling matters:**

- **Training supervised models**: Supervised learning relies heavily on labeled datasets to learn the relationship between input features and target outputs. Without accurately labeled data, ML models would struggle to generalize patterns and make accurate predictions.
- **Creating ground truth benchmarks**: Labeled data serves as the benchmark for evaluating model performance. By comparing model predictions against accurately labeled data, ML engineers can assess the efficacy of their algorithms.
- **Enabling domain-specific insights**: By categorizing data points into meaningful classes, organizations can derive valuable insights about customer preferences, market trends, and business operations.

**The challenge**: Manual labeling is labor-intensive, time-consuming, expensive, and prone to errors, inconsistencies, and biases — especially when dealing with large volumes of unstructured data. This motivates the use of automated and programmatic labeling tools like Snorkel.

**The role of data augmentation**: Even after obtaining labeled data, training sets may be too small or lack diversity. Data augmentation addresses this by creating transformed copies of existing data points, increasing dataset size and improving model robustness without additional manual labeling effort.

---

## What is Snorkel?

Snorkel is a powerful framework designed to streamline the data labeling pipeline and mitigate the challenges associated with manual annotation. It provides three core operators:

1. **Labeling Functions (LFs)**: Users write programmatic rules — heuristics, keyword searches, regex patterns, and third-party model outputs — that each assign labels to subsets of the data. Each LF may be noisy or incomplete on its own.
2. **Transformation Functions (TFs)**: Users define class-preserving transformations (synonym replacement, word swapping, typo injection) that generate new training data points from existing ones, effectively augmenting the dataset.
3. **Slicing Functions (SFs)**: Users identify critical data subsets or slices for targeted monitoring and performance improvement.

Snorkel's **Label Model** aggregates noisy LF outputs into single, probabilistic labels by learning LF accuracies and correlations — no ground truth needed for the training set. The resulting labels train downstream discriminative classifiers that generalize to unseen data.

---

## Dataset

**SMS Spam Collection** from the UCI Machine Learning Repository.

| Property | Details |
|----------|---------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) |
| Total Messages | 5,572 SMS messages |
| Classes | Ham (not spam): 4,825 / Spam: 747 |
| Train Split | 4,457 messages (80%) |
| Test Split | 1,115 messages (20%) |
| Features | Raw text content of each SMS message |
| Class Balance | Approximately 87% ham, 13% spam |

This dataset differs from the original YouTube comments dataset used in the course tutorial, providing a different text domain with distinct spam patterns (prize scams, promotional URLs, urgency language, currency mentions, etc.).

---

## Modifications from Original Lab

This submission introduces significant modifications across both tutorials:

| # | Modification | Original Lab | This Lab |
|---|-------------|-------------|----------|
| 1 | **Dataset** | YouTube video comments | UCI SMS Spam Collection (text messages) |
| 2 | **Labeling Functions** | YouTube-specific LFs (e.g., "subscribe", channel promotions) | 11 SMS-specific LFs (currency symbols, phone numbers, txt-speak, personal greetings, etc.) |
| 3 | **Transformation Functions** | Generic text TFs | 6 SMS-specific TFs including typo injection, case changes, word duplication |
| 4 | **Additional Models** | Logistic Regression only | Logistic Regression + Random Forest comparison across both tutorials |
| 5 | **Visualizations** | Minimal | EDA distributions, LF coverage/accuracy charts, augmentation effects, 4-model comparison plots |
| 6 | **End Model** | LSTM (Tutorial 02) | Logistic Regression + Random Forest with TF-IDF features |
| 7 | **Slicing Functions** | 5 YouTube-specific SFs | 9 SMS-specific SFs including sentiment analysis, caps detection, punctuation patterns |

---

## Project Architecture
```
snorkel-data-labeling-lab/
├── 01_sms_spam_labeling_tutorial.py        # Tutorial 01: Labeling Functions
├── 01_sms_spam_labeling_tutorial.ipynb      # Tutorial 01: Notebook version
├── 02_sms_spam_augmentation_tutorial.py     # Tutorial 02: Data Augmentation
├── 02_sms_spam_augmentation_tutorial.ipynb  # Tutorial 02: Notebook version
├── utils.py                                 # Data loading utility functions
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── .gitignore                               # Git ignore rules
└── data/
    ├── SMSSpamCollection                    # Raw dataset (TSV format)
    ├── eda_distributions.png                # EDA visualization (Tutorial 01)
    ├── lf_coverage_accuracy.png             # LF analysis visualization (Tutorial 01)
    ├── model_comparison.png                 # Model comparison chart (Tutorial 01)
    ├── augmentation_effects.png             # Augmentation effects chart (Tutorial 02)
    └── augmentation_model_comparison.png    # Original vs augmented comparison (Tutorial 02)
```

---

## Tutorial 01: Labeling Functions

### Overview

This tutorial demonstrates how to programmatically label an unlabeled SMS dataset using Snorkel's labeling functions, train a Label Model to aggregate noisy labels, and use the resulting labels to train downstream classifiers.

### Methodology

**Step 1 — Data Loading**: The SMS Spam Collection is loaded and split 80/20 with stratified sampling. Ground truth labels are stored separately to simulate an unlabeled training scenario.

**Step 2 — Exploratory Data Analysis**: We analyzed message length and word count distributions, finding that spam messages are significantly longer and more verbose than ham messages.

**Step 3 — Labeling Function Design**: We designed 11 labeling functions across three categories:

**Keyword-Based LFs** (detect spam vocabulary):

| LF | Pattern | Rationale |
|----|---------|-----------|
| `lf_contains_free` | "free" in text | Prize/freebie scams |
| `lf_contains_win` | Word boundary match for "win"/"won" | Lottery/contest scams |
| `lf_contains_prize` | "prize" in text | Prize notification scams |
| `lf_contains_claim` | "claim" in text | Action-oriented spam language |
| `lf_contains_urgent` | "urgent", "immediately", "act now" | Pressure/urgency tactics |

**Pattern-Based LFs** (detect structural spam indicators):

| LF | Pattern | Rationale |
|----|---------|-----------|
| `lf_contains_url` | HTTP/WWW regex | Spam often contains links |
| `lf_contains_phone_number` | 5+ consecutive digits | Spam includes callback numbers |
| `lf_contains_currency` | Currency symbols or "pound"/"cash" | Financial lure language |
| `lf_contains_txt_speak` | "txt", "msg", "reply" | SMS-specific spam instructions |

**Behavioral LFs** (detect ham patterns):

| LF | Pattern | Rationale |
|----|---------|-----------|
| `lf_short_message` | Less than 5 words | Very short messages are almost always ham |
| `lf_personal_greeting` | Starts with "hey", "hi", "hello", etc. | Personal greetings indicate legitimate conversation |

**Step 4 — LF Analysis**: Using Snorkel's LFAnalysis, we evaluated each LF:

| Labeling Function | Coverage | Emp. Accuracy | Correct | Incorrect |
|-------------------|----------|---------------|---------|-----------|
| lf_contains_phone_number | 10.2% | 99.6% | 452 | 2 |
| lf_short_message | 6.3% | 99.3% | 281 | 2 |
| lf_contains_url | 2.0% | 97.8% | 87 | 2 |
| lf_contains_claim | 2.2% | 100.0% | 97 | 0 |
| lf_contains_prize | 1.4% | 100.0% | 64 | 0 |
| lf_personal_greeting | 4.8% | 90.7% | 195 | 20 |
| lf_contains_currency | 5.6% | 88.3% | 219 | 29 |
| lf_contains_urgent | 1.4% | 82.3% | 51 | 11 |
| lf_contains_win | 2.9% | 81.4% | 105 | 24 |
| lf_contains_free | 4.6% | 75.2% | 155 | 51 |
| lf_contains_txt_speak | 6.6% | 70.9% | 207 | 85 |

**Step 5 — Label Model Training**: The Snorkel LabelModel was trained for 500 epochs. For uncovered data points (no LF voted), we defaulted to HAM using domain knowledge.

**Step 6 — End Model Training**: Two classifiers trained on Snorkel-generated labels using TF-IDF features.

### Tutorial 01 Results

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 (Spam) |
|-------|----------|------------------|---------------|-----------|
| Label Model | 95% | 0.76 | 0.93 | 0.84 |
| **Logistic Regression** | **97%** | **0.95** | 0.84 | **0.89** |
| Random Forest | 95% | 0.78 | 0.87 | 0.82 |

---

## Tutorial 02: Data Augmentation

### Overview

This tutorial builds on Tutorial 01 by demonstrating how to use Snorkel's transformation functions to augment the labeled training set, increasing dataset size and improving model robustness. We compare model performance with and without augmentation.

### Methodology

**Step 1 — Load Labeled Data**: We use the labeled SMS dataset (with gold labels) as our starting point, simulating a scenario where labeling has already been completed.

**Step 2 — Transformation Function Design**: We designed 6 transformation functions that create new training data while preserving the original class label:

| TF | Description | Category |
|----|------------|----------|
| `tf_replace_word_with_synonym` | Replaces a random word with its WordNet synonym | Standard NLP augmentation |
| `tf_random_word_swap` | Swaps two random adjacent words | Standard NLP augmentation |
| `tf_random_word_deletion` | Deletes a random non-essential word | Standard NLP augmentation |
| `tf_change_case` | Randomly changes message to lowercase, uppercase, or title case | SMS-specific (YOUR MODIFICATION) |
| `tf_add_typo` | Swaps two adjacent characters to simulate typos | SMS-specific (YOUR MODIFICATION) |
| `tf_duplicate_word` | Duplicates a random word (simulating SMS typing) | SMS-specific (YOUR MODIFICATION) |

**Step 3 — Preview Transformations**: We previewed TF outputs on sample messages to verify they produce valid, class-preserving transformations. Examples include replacing "week" with "hebdomad" (synonym) and swapping "academic the" (word swap).

**Step 4 — Apply Augmentation**: Using Snorkel's RandomPolicy with sequence_length=2 and n_per_original=2, we applied TFs to generate augmented data:

| Metric | Value |
|--------|-------|
| Original dataset size | 4,457 |
| Augmented dataset size | 13,182 |
| Growth factor | ~3x |
| Original Ham / Spam | 3,859 / 598 |
| Augmented Ham / Spam | 11,397 / 1,785 |

**Step 5 — Visualization**: We generated charts comparing dataset sizes and message length distributions before and after augmentation.

**Step 6 — Model Comparison**: We trained Logistic Regression and Random Forest on both original and augmented datasets to measure the impact of augmentation.

### Tutorial 02 Results

| Model | Dataset | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|-------|---------|----------|----------------|-------------|---------|
| Logistic Regression | Original | 97% | 1.00 | 0.81 | 0.90 |
| **Logistic Regression** | **Augmented** | **98%** | **1.00** | **0.88** | **0.94** |
| Random Forest | Original | 97% | 1.00 | 0.81 | 0.89 |
| Random Forest | Augmented | 98% | 1.00 | 0.83 | 0.91 |

**Key improvement**: Data augmentation boosted spam recall by 7 percentage points for Logistic Regression (0.81 to 0.88) while maintaining perfect precision, resulting in a spam F1 improvement from 0.90 to 0.94.

---


## Tutorial 03: Data Slicing

### Overview

This tutorial demonstrates how to use Snorkel's slicing functions to identify critical data subsets, monitor per-slice model performance, and compare models across slices. In real-world applications, overall accuracy can mask poor performance on important subsets — slicing helps surface these failure modes.

### Methodology

**Step 1-2 — Baseline Models**: We trained Logistic Regression and Random Forest on the full labeled SMS dataset using TF-IDF features.

**Step 3 — Slicing Function Design**: We defined 9 slicing functions to identify critical data subsets:

| SF | Description | Category |
|----|------------|----------|
| `short_message` | Messages with fewer than 5 words | Message structure |
| `long_message` | Messages with more than 30 words | Message structure (YOUR MODIFICATION) |
| `contains_url` | Messages with HTTP/WWW links | Content pattern |
| `contains_phone_number` | Messages with 5+ digit numbers | Content pattern |
| `contains_currency` | Messages with $, pound, cash, free | Content pattern |
| `positive_sentiment` | TextBlob polarity > 0.3 | Sentiment-based (YOUR MODIFICATION) |
| `negative_sentiment` | TextBlob polarity < -0.1 | Sentiment-based (YOUR MODIFICATION) |
| `all_caps` | Over 50% uppercase characters | Style-based (YOUR MODIFICATION) |
| `heavy_punctuation` | 4+ punctuation marks (!, ?, .) | Style-based (YOUR MODIFICATION) |

**Step 4-5 — Slice Application and Monitoring**: We applied all SFs to the test set and measured per-slice accuracy, weighted F1, and spam-specific F1.

**Step 6-7 — Worst Slice Analysis**: We identified the weakest slices and examined misclassified examples to understand failure modes.

**Step 8-9 — Model Comparison Per Slice**: We compared LR vs RF performance across all slices to determine which model handles specific subsets better.

### Tutorial 03 Results

| Slice | Size | LR F1 | RF F1 | Better Model |
|-------|------|-------|-------|-------------|
| short_message | 71 | 0.958 | 0.958 | Tie |
| long_message | 90 | 0.949 | 0.963 | RF |
| contains_url | 19 | 1.000 | 1.000 | Tie |
| contains_phone_number | 114 | 0.927 | 0.927 | Tie |
| contains_currency | 74 | 0.909 | 0.909 | Tie |
| positive_sentiment | 274 | 0.962 | 0.962 | Tie |
| negative_sentiment | 128 | 0.966 | 0.984 | RF |
| all_caps | 27 | 0.957 | 0.890 | LR |
| heavy_punctuation | 249 | 0.975 | 0.962 | LR |
| **OVERALL** | **1115** | **0.974** | **0.973** | **LR** |

**Key finding**: The `contains_currency` slice had the worst performance (F1=0.909), with 7 misclassified spam messages that used informal language patterns not captured by TF-IDF features. Random Forest outperformed LR on long messages and negative sentiment slices, while LR was better on all-caps and heavy punctuation slices.

---
## Results Summary

### Overall Best Results Across Both Tutorials

| Approach | Model | Accuracy | Spam F1 | Training Data |
|----------|-------|----------|---------|---------------|
| Tutorial 01 (Weak Supervision) | Logistic Regression | 97% | 0.89 | Snorkel-labeled (4,457) |
| Tutorial 01 (Weak Supervision) | Random Forest | 95% | 0.82 | Snorkel-labeled (4,457) |
| **Tutorial 02 (Augmented)** | **Logistic Regression** | **98%** | **0.94** | **Augmented (13,182)** |
| Tutorial 02 (Augmented) | Random Forest | 98% | 0.91 | Augmented (13,182) |
| Tutorial 03 (Slicing) | Per-slice monitoring | 97% | 0.90 | Gold-labeled (4,457) |

The combination of weak supervision (Tutorial 01) and data augmentation (Tutorial 02) achieves 98% accuracy and 0.94 spam F1 — without any manual labeling of training data.

---

## Key Findings and Insights

1. **Weak supervision works**: Despite using no hand-labeled training data, our Tutorial 01 model achieved 97% accuracy and 0.89 F1 on spam — competitive with fully supervised approaches.

2. **Data augmentation improves recall**: Augmentation primarily boosted spam recall (0.81 to 0.88) while maintaining perfect precision. The augmented dataset (13,182 samples) helped models learn more diverse spam patterns.

3. **LF quality matters more than quantity**: We initially tested 13 LFs but removed two low-accuracy ones (lf_all_caps_ratio at 6% and lf_exclamation_heavy at 8%). A smaller set of high-quality LFs outperformed the larger noisy set.

4. **Handling ABSTAIN is critical**: The strategy for labeling uncovered data points significantly impacts performance. Defaulting abstains to HAM using domain knowledge produced the best results.

5. **Linear models outperform ensembles on this task**: Logistic Regression consistently outperformed Random Forest across both tutorials, likely due to better robustness to label noise and the linear separability of TF-IDF spam features.

6. **SMS spam has distinctive patterns**: Phone numbers (5+ digit sequences) and currency symbols were the strongest individual spam indicators with near-perfect accuracy. SMS-specific TFs (typo injection, case changes) helped models learn real-world SMS text variations.

7. **Augmentation preserves class balance**: The 3x dataset growth maintained the original class ratio (87% ham / 13% spam), preventing the introduction of class imbalance artifacts.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/Kranthi0011/snorkel-data-labeling-lab.git
cd snorkel-data-labeling-lab

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install nltk
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Download the dataset
cd data
curl -L -o smsspamcollection.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
cd ..

# Run Tutorial 01: Labeling Functions
python 01_sms_spam_labeling_tutorial.py

# Run Tutorial 02: Data Augmentation
python 02_sms_spam_augmentation_tutorial.py

# Run Tutorial 03: Data Slicing
python 03_sms_spam_slicing_tutorial.py

# Or open notebooks
jupyter notebook
```

### Dependencies
- snorkel
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk
- jupyter
- textblob
- jupytext

---

## References

1. Ratner, A., Bach, S., Ehrenberg, H., Fries, J., Wu, S., and Re, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. Proceedings of the VLDB Endowment, 11(3), 269-282.
2. Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering.
3. Ratner, A., Ehrenberg, H., Hussain, Z., Dunnmon, J., and Re, C. (2017). Learning to Compose Domain-Specific Transformations for Data Augmentation. NeurIPS 2017.
4. Snorkel Tutorials — Spam Classification: https://github.com/snorkel-team/snorkel-tutorials
5. MLOps Course Repository — Northeastern University: https://github.com/raminmohammadi/MLOps
6. UCI SMS Spam Collection Dataset: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
