# SMS Spam Detection Using Snorkel Weak Supervision — Data Labeling Lab

## Table of Contents
- [Introduction](#introduction)
- [The Significance of Data Labeling](#the-significance-of-data-labeling)
- [What is Snorkel?](#what-is-snorkel)
- [Dataset](#dataset)
- [Modifications from Original Lab](#modifications-from-original-lab)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings and Insights](#key-findings-and-insights)
- [Setup and Installation](#setup-and-installation)
- [References](#references)

---

## Introduction

This lab demonstrates the practical application of **Snorkel**, a weak supervision framework, for building labeled training datasets without manual annotation. Instead of relying on hand-labeled data, we programmatically generate training labels using heuristic **Labeling Functions (LFs)** — noisy, rule-based functions that encode domain knowledge.

The task: **classify SMS text messages as spam or ham (not spam)** using the UCI SMS Spam Collection dataset. This lab is part of the IE-7374 MLOps course at Northeastern University and is based on the Snorkel Spam Tutorials from the [MLOps repository](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Data_Labeling_Labs).

---

## The Significance of Data Labeling

In the realm of machine learning and artificial intelligence, data is the bedrock upon which models are built and refined. However, raw data, while abundant, often lacks the structure and organization necessary for effective learning. This is where data labeling comes into play.

**Why data labeling matters:**

- **Training supervised models**: Supervised learning relies heavily on labeled datasets to learn the relationship between input features and target outputs. Without accurately labeled data, ML models would struggle to generalize patterns and make accurate predictions.
- **Creating ground truth benchmarks**: Labeled data serves as the benchmark for evaluating model performance. By comparing model predictions against accurately labeled data, ML engineers can assess the efficacy of their algorithms.
- **Enabling domain-specific insights**: By categorizing data points into meaningful classes, organizations can derive valuable insights about customer preferences, market trends, and business operations.

**The challenge**: Manual labeling is labor-intensive, time-consuming, expensive, and prone to errors, inconsistencies, and biases — especially when dealing with large volumes of unstructured data. This motivates the use of automated and programmatic labeling tools like Snorkel.

---

## What is Snorkel?

Snorkel is a powerful framework designed to streamline the data labeling pipeline and mitigate the challenges associated with manual annotation. It works through three core operations:

1. **Labeling Functions (LFs)**: Users write programmatic rules — heuristics, keyword searches, regex patterns, and third-party model outputs — that each assign labels to subsets of the data. Each LF may be noisy or incomplete on its own.
2. **Label Model**: Snorkel models the accuracies and correlations of the LFs using only their agreements and disagreements (no ground truth needed for training data). It produces a single, probabilistic label per data point by intelligently aggregating all LF outputs.
3. **End Model Training**: The probabilistic labels from the Label Model are used to train a downstream discriminative classifier (e.g., Logistic Regression, Random Forest) that can generalize to new, unseen data.

This approach accelerates the labeling process, improves label quality through aggregation, and empowers ML engineers to focus on high-level model design rather than tedious annotation work.

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

This submission introduces four significant modifications from the original Snorkel Spam Tutorial:

| # | Modification | Original Lab | This Lab |
|---|-------------|-------------|----------|
| 1 | **Dataset** | YouTube video comments | UCI SMS Spam Collection (text messages) |
| 2 | **Labeling Functions** | YouTube-specific LFs (e.g., "subscribe", channel promotions) | 11 SMS-specific LFs (currency symbols, phone numbers, txt-speak, personal greetings, etc.) |
| 3 | **Additional Model** | Logistic Regression only | Logistic Regression + Random Forest comparison |
| 4 | **Visualizations** | Minimal | EDA distributions, LF coverage/accuracy bar charts, 3-model comparison plot |

---

## Project Architecture
```
snorkel-data-labeling-lab/
├── 01_sms_spam_labeling_tutorial.py      # Main Python script
├── 01_sms_spam_labeling_tutorial.ipynb   # Jupyter notebook (auto-converted)
├── utils.py                              # Data loading utility functions
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .gitignore                            # Git ignore rules
└── data/
    ├── SMSSpamCollection                 # Raw dataset (TSV format)
    ├── eda_distributions.png             # EDA visualization output
    ├── lf_coverage_accuracy.png          # LF analysis visualization
    └── model_comparison.png              # Final model comparison chart
```

---

## Methodology

### Step 1: Data Loading and Exploration

The SMS Spam Collection is loaded from a TSV file, with text labels ("ham"/"spam") converted to binary integers (0/1). The dataset is split 80/20 into train and test sets using stratified sampling to preserve class distribution. Ground truth labels for the training set are stored separately — simulating the real-world scenario where training data is unlabeled.

### Step 2: Exploratory Data Analysis

Before writing labeling functions, we analyze structural differences between spam and ham messages:

- **Message Length Distribution**: Spam messages tend to be significantly longer than ham messages, often containing detailed promotional content, URLs, and instructions.
- **Word Count Distribution**: Similarly, spam messages have higher word counts, reflecting their verbose and persuasive nature.

These insights directly informed the design of our labeling functions (e.g., `lf_short_message` leverages the observation that very short messages are almost always legitimate).

### Step 3: Labeling Function Design

We designed 11 labeling functions across three categories:

**Keyword-Based LFs** (detect common spam vocabulary):

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

**Behavioral LFs** (detect ham/legitimate patterns):

| LF | Pattern | Rationale |
|----|---------|-----------|
| `lf_short_message` | Less than 5 words | Very short messages are almost always ham |
| `lf_personal_greeting` | Starts with "hey", "hi", "hello", etc. | Personal greetings indicate legitimate conversation |

### Step 4: Labeling Function Analysis

Using Snorkel's `LFAnalysis`, we evaluated each LF against gold labels:

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

Key observations: `lf_contains_phone_number` and `lf_short_message` achieved near-perfect accuracy. `lf_contains_prize` and `lf_contains_claim` achieved 100% accuracy but with lower coverage. The txt-speak LF had the highest false positive rate, as words like "reply" occasionally appear in legitimate messages.

### Step 5: Snorkel Label Model Training

The Snorkel `LabelModel` was trained for 500 epochs to learn the accuracies and correlations of the labeling functions. It models LF outputs probabilistically — reweighting reliable LFs and downweighting noisy ones.

For data points where no LF voted (ABSTAIN from all LFs), we assigned them the majority class label (HAM), reflecting domain knowledge that most SMS messages are legitimate. This is critical: without it, the Label Model assigns near-50/50 probabilities to uncovered points, creating severely imbalanced training data.

### Step 6: End Model Training

Using the Snorkel-generated labels, we trained two classifiers:

1. **Logistic Regression** (from original tutorial): A linear model using TF-IDF features with unigrams and bigrams (max 5,000 features).
2. **Random Forest** (our modification): An ensemble of 100 decision trees using the same TF-IDF features, allowing comparison between a linear model and a non-linear ensemble.

### Step 7: Model Comparison and Evaluation

All three models were evaluated on the held-out test set with ground truth labels, using accuracy, precision, recall, and F1-score.

---

## Results

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 (Spam) | F1 (Weighted) |
|-------|----------|------------------|---------------|-----------|---------------|
| **Label Model** | 95% | 0.76 | 0.93 | 0.84 | 0.95 |
| **Logistic Regression** | **97%** | **0.95** | 0.84 | **0.89** | **0.97** |
| **Random Forest** | 95% | 0.78 | 0.87 | 0.82 | 0.95 |

**Training label distribution** (generated by Snorkel): 3,700 ham / 757 spam — closely approximating the true distribution (3,859 ham / 598 spam), validating the effectiveness of our labeling approach.

---

## Key Findings and Insights

1. **Weak supervision works**: Despite using no hand-labeled training data, our best model (Logistic Regression) achieved 97% accuracy and 0.89 F1 on spam — competitive with fully supervised approaches.

2. **LF quality matters more than quantity**: We initially tested 13 LFs but removed two low-accuracy ones (`lf_all_caps_ratio` at 6% and `lf_exclamation_heavy` at 8%). A smaller set of high-quality LFs outperformed the larger noisy set.

3. **Handling ABSTAIN is critical**: The strategy for labeling uncovered data points significantly impacts performance. Defaulting abstains to HAM using domain knowledge produced the best results.

4. **Linear models can outperform ensembles on noisy labels**: Logistic Regression outperformed Random Forest, likely because linear models are more robust to label noise from imperfect labeling functions.

5. **SMS spam has distinctive patterns**: Phone numbers (5+ digit sequences) and currency symbols were the strongest individual spam indicators with near-perfect accuracy, highly specific to the SMS domain.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/snorkel-data-labeling-lab.git
cd snorkel-data-labeling-lab

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
cd data
curl -L -o smsspamcollection.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
cd ..

# Run the script
python 01_sms_spam_labeling_tutorial.py

# Or open the notebook
jupyter notebook 01_sms_spam_labeling_tutorial.ipynb
```

### Dependencies
- snorkel
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter
- jupytext

---

## References

1. Ratner, A., Bach, S., Ehrenberg, H., Fries, J., Wu, S., and Re, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. Proceedings of the VLDB Endowment, 11(3), 269-282.
2. Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering.
3. Snorkel Tutorials — Spam Classification: https://github.com/snorkel-team/snorkel-tutorials
4. MLOps Course Repository — Northeastern University: https://github.com/raminmohammadi/MLOps
5. UCI SMS Spam Collection Dataset: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
