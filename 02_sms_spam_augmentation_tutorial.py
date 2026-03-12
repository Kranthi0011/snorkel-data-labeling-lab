# ============================================================
# Tutorial 02: SMS Spam Data Augmentation with Transformation Functions
# Modification: SMS dataset, custom TFs, comparison with/without augmentation
# ============================================================

import pandas as pd
import numpy as np
import random
import re
import copy
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from snorkel.augmentation import transformation_function, PandasTFApplier, RandomPolicy
from utils import load_sms_spam_dataset

import nltk
from nltk.corpus import wordnet

ABSTAIN = -1
HAM = 0
SPAM = 1

# ============================================================
# STEP 1: Load Labeled Data (from Tutorial 01 output)
# ============================================================
print("=" * 60)
print("STEP 1: Loading SMS Spam Dataset (with labels)")
print("=" * 60)

df_train, df_test, train_gold_labels = load_sms_spam_dataset()
Y_test = df_test.label.values

# Use gold labels for augmentation tutorial (simulating labeled data)
df_train["label"] = train_gold_labels

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(f"Train class distribution:")
print(df_train.label.value_counts())
print(f"\nSample data:")
print(df_train[["text", "label"]].head(5))

# ============================================================
# STEP 2: Define Transformation Functions
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Defining Transformation Functions")
print("=" * 60)

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word and "_" not in lemma.name():
                synonyms.add(lemma.name())
    return list(synonyms)

@transformation_function()
def tf_replace_word_with_synonym(x):
    """Replace a random word with its synonym."""
    words = x.text.split()
    if len(words) < 3:
        return None
    idx = random.randint(0, len(words) - 1)
    syns = get_synonyms(words[idx].lower())
    if syns:
        x = x.copy()
        words[idx] = random.choice(syns)
        x.text = " ".join(words)
        return x
    return None

@transformation_function()
def tf_random_word_swap(x):
    """Swap two random adjacent words."""
    words = x.text.split()
    if len(words) < 4:
        return None
    idx = random.randint(0, len(words) - 2)
    x = x.copy()
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    x.text = " ".join(words)
    return x

@transformation_function()
def tf_random_word_deletion(x):
    """Randomly delete a non-essential word."""
    words = x.text.split()
    if len(words) < 6:
        return None
    idx = random.randint(1, len(words) - 2)
    x = x.copy()
    del words[idx]
    x.text = " ".join(words)
    return x

# YOUR MODIFICATION: SMS-specific TFs
@transformation_function()
def tf_change_case(x):
    """Randomly change case of the message."""
    x = x.copy()
    choice = random.randint(0, 2)
    if choice == 0:
        x.text = x.text.lower()
    elif choice == 1:
        x.text = x.text.upper()
    else:
        x.text = x.text.title()
    return x

@transformation_function()
def tf_add_typo(x):
    """Introduce a random typo by swapping adjacent characters."""
    text = x.text
    if len(text) < 5:
        return None
    idx = random.randint(1, len(text) - 2)
    text_list = list(text)
    text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
    x = x.copy()
    x.text = "".join(text_list)
    return x

@transformation_function()
def tf_duplicate_word(x):
    """Randomly duplicate a word (simulating SMS typing patterns)."""
    words = x.text.split()
    if len(words) < 3:
        return None
    idx = random.randint(0, len(words) - 1)
    x = x.copy()
    words.insert(idx, words[idx])
    x.text = " ".join(words)
    return x

tfs = [
    tf_replace_word_with_synonym,
    tf_random_word_swap,
    tf_random_word_deletion,
    tf_change_case,
    tf_add_typo,
    tf_duplicate_word,
]

print(f"Defined {len(tfs)} transformation functions")

# ============================================================
# STEP 3: Preview Transformations
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Previewing Transformations")
print("=" * 60)

random.seed(42)
sample_df = df_train.sample(5, random_state=42)
for _, row in sample_df.iterrows():
    print(f"\nOriginal: {row.text[:80]}...")
    for tf in tfs:
        result = tf(row)
        if result is not None:
            print(f"  {tf.name}: {result.text[:80]}...")
            break

# ============================================================
# STEP 4: Apply TFs to Augment Dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Applying Transformation Functions")
print("=" * 60)

random.seed(42)
np.random.seed(42)

policy = RandomPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
)

tf_applier = PandasTFApplier(tfs, policy)
df_train_augmented = tf_applier.apply(df_train)

print(f"Original training set size: {len(df_train)}")
print(f"Augmented training set size: {len(df_train_augmented)}")
print(f"\nAugmented class distribution:")
print(df_train_augmented.label.value_counts())

# ============================================================
# STEP 5: Visualize Augmentation Effects (YOUR MODIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Visualizing Augmentation Effects")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = ["Original", "Augmented"]
ham_counts = [
    len(df_train[df_train.label == HAM]),
    len(df_train_augmented[df_train_augmented.label == HAM]),
]
spam_counts = [
    len(df_train[df_train.label == SPAM]),
    len(df_train_augmented[df_train_augmented.label == SPAM]),
]

x = np.arange(len(labels))
width = 0.3
axes[0].bar(x - width / 2, ham_counts, width, label="Ham", color="steelblue")
axes[0].bar(x + width / 2, spam_counts, width, label="Spam", color="coral")
axes[0].set_ylabel("Count")
axes[0].set_title("Dataset Size: Original vs Augmented")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].legend()
for i, (h, s) in enumerate(zip(ham_counts, spam_counts)):
    axes[0].text(i - width / 2, h + 50, str(h), ha="center", fontsize=9)
    axes[0].text(i + width / 2, s + 50, str(s), ha="center", fontsize=9)

df_train["text_length"] = df_train["text"].str.len()
df_train_augmented["text_length"] = df_train_augmented["text"].str.len()
axes[1].hist(df_train["text_length"], bins=50, alpha=0.6, label="Original", color="steelblue")
axes[1].hist(df_train_augmented["text_length"], bins=50, alpha=0.6, label="Augmented", color="coral")
axes[1].set_title("Message Length Distribution")
axes[1].set_xlabel("Character Length")
axes[1].legend()

plt.tight_layout()
plt.savefig("data/augmentation_effects.png", dpi=150)
plt.close()
print("Saved: data/augmentation_effects.png")

# ============================================================
# STEP 6: Train and Compare Models (YOUR MODIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models - Original vs Augmented")
print("=" * 60)

# Train on ORIGINAL data
vectorizer_orig = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_orig = vectorizer_orig.fit_transform(df_train.text)
X_test_orig = vectorizer_orig.transform(df_test.text)

lr_orig = LogisticRegression(solver="lbfgs", max_iter=1000)
lr_orig.fit(X_train_orig, df_train.label.values)
lr_orig_preds = lr_orig.predict(X_test_orig)

rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
rf_orig.fit(X_train_orig, df_train.label.values)
rf_orig_preds = rf_orig.predict(X_test_orig)

# Train on AUGMENTED data
vectorizer_aug = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_aug = vectorizer_aug.fit_transform(df_train_augmented.text)
X_test_aug = vectorizer_aug.transform(df_test.text)

lr_aug = LogisticRegression(solver="lbfgs", max_iter=1000)
lr_aug.fit(X_train_aug, df_train_augmented.label.values)
lr_aug_preds = lr_aug.predict(X_test_aug)

rf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
rf_aug.fit(X_train_aug, df_train_augmented.label.values)
rf_aug_preds = rf_aug.predict(X_test_aug)

print("ORIGINAL DATA - Logistic Regression:")
print(classification_report(Y_test, lr_orig_preds, target_names=["Ham", "Spam"]))

print("AUGMENTED DATA - Logistic Regression:")
print(classification_report(Y_test, lr_aug_preds, target_names=["Ham", "Spam"]))

print("ORIGINAL DATA - Random Forest:")
print(classification_report(Y_test, rf_orig_preds, target_names=["Ham", "Spam"]))

print("AUGMENTED DATA - Random Forest:")
print(classification_report(Y_test, rf_aug_preds, target_names=["Ham", "Spam"]))

# ============================================================
# STEP 7: Model Comparison Chart (YOUR MODIFICATION)
# ============================================================
models = [
    "LR\nOriginal", "LR\nAugmented",
    "RF\nOriginal", "RF\nAugmented",
]
accuracies = [
    accuracy_score(Y_test, lr_orig_preds),
    accuracy_score(Y_test, lr_aug_preds),
    accuracy_score(Y_test, rf_orig_preds),
    accuracy_score(Y_test, rf_aug_preds),
]
f1_list = [
    f1_score(Y_test, lr_orig_preds, average="weighted"),
    f1_score(Y_test, lr_aug_preds, average="weighted"),
    f1_score(Y_test, rf_orig_preds, average="weighted"),
    f1_score(Y_test, rf_aug_preds, average="weighted"),
]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.3
bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="steelblue")
bars2 = ax.bar(x + width / 2, f1_list, width, label="F1 (weighted)", color="coral")
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Original vs Augmented Training Data")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0.8, 1.05)
for i, (a, f) in enumerate(zip(accuracies, f1_list)):
    ax.text(i - width / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + width / 2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("data/augmentation_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: data/augmentation_model_comparison.png")

print("\n" + "=" * 60)
print("ALL DONE! Check data/ folder for saved plots.")
print("=" * 60)
