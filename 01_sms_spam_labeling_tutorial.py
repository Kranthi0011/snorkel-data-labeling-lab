# ============================================================
# SMS Spam Detection Using Snorkel - Data Labeling Lab
# Modification: SMS dataset, custom LFs, Random Forest comparison
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import re

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from utils import load_sms_spam_dataset

ABSTAIN = -1
HAM = 0
SPAM = 1

print("=" * 60)
print("STEP 1: Loading SMS Spam Dataset")
print("=" * 60)

df_train, df_test, train_gold_labels = load_sms_spam_dataset()
Y_test = df_test.label.values

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(f"Train class distribution:")
print(df_train.label.value_counts())
print(f"Sample data:")
print(df_train[["text", "label"]].sample(10, random_state=42))

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

df_train["text_length"] = df_train["text"].str.len()
df_train["word_count"] = df_train["text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_train[df_train.label == HAM]["text_length"].hist(ax=axes[0], bins=50, alpha=0.7, label="Ham", color="steelblue")
df_train[df_train.label == SPAM]["text_length"].hist(ax=axes[0], bins=50, alpha=0.7, label="Spam", color="coral")
axes[0].set_title("Message Length Distribution")
axes[0].legend()
df_train[df_train.label == HAM]["word_count"].hist(ax=axes[1], bins=30, alpha=0.7, label="Ham", color="steelblue")
df_train[df_train.label == SPAM]["word_count"].hist(ax=axes[1], bins=30, alpha=0.7, label="Spam", color="coral")
axes[1].set_title("Word Count Distribution")
axes[1].legend()
plt.tight_layout()
plt.savefig("data/eda_distributions.png", dpi=150)
plt.close()
print("Saved: data/eda_distributions.png")

print("\n" + "=" * 60)
print("STEP 3: Defining Labeling Functions")
print("=" * 60)

@labeling_function()
def lf_contains_free(x):
    return SPAM if "free" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_win(x):
    return SPAM if re.search(r"\bw[io]n\b", x.text.lower()) else ABSTAIN

@labeling_function()
def lf_contains_prize(x):
    return SPAM if "prize" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_claim(x):
    return SPAM if "claim" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_urgent(x):
    return SPAM if re.search(r"urgent|immediately|act now", x.text.lower()) else ABSTAIN

@labeling_function()
def lf_contains_url(x):
    return SPAM if re.search(r"http[s]?://|www\.", x.text.lower()) else ABSTAIN

@labeling_function()
def lf_contains_phone_number(x):
    return SPAM if re.search(r"\b\d{5,}\b", x.text) else ABSTAIN

@labeling_function()
def lf_all_caps_ratio(x):
    if len(x.text) == 0:
        return ABSTAIN
    caps_ratio = sum(1 for c in x.text if c.isupper()) / len(x.text)
    return SPAM if caps_ratio > 0.3 and len(x.text) > 20 else ABSTAIN

@labeling_function()
def lf_contains_currency(x):
    return SPAM if re.search(r"[$\xa3\u20ac]|\bpound\b|\bcash\b", x.text.lower()) else ABSTAIN

@labeling_function()
def lf_short_message(x):
    return HAM if len(x.text.split()) < 5 else ABSTAIN

@labeling_function()
def lf_contains_txt_speak(x):
    return SPAM if re.search(r"\btxt\b|\bmsg\b|\breply\b", x.text.lower()) else ABSTAIN

@labeling_function()
def lf_exclamation_heavy(x):
    return SPAM if x.text.count("!") >= 5 else ABSTAIN

@labeling_function()
def lf_personal_greeting(x):
    greetings = ["hey", "hi ", "hello", "dear", "good morning", "good night"]
    return HAM if any(x.text.lower().startswith(g) for g in greetings) else ABSTAIN

lfs = [
    lf_contains_free, lf_contains_win, lf_contains_prize,
    lf_contains_claim, lf_contains_urgent, lf_contains_url,
    lf_contains_phone_number, lf_contains_currency,
    lf_short_message, lf_contains_txt_speak,
    lf_personal_greeting
]
print(f"Defined {len(lfs)} labeling functions")

print("\n" + "=" * 60)
print("STEP 4: Applying Labeling Functions")
print("=" * 60)

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

print(f"Label matrix shape: {L_train.shape}")
lf_analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary(Y=train_gold_labels)
print("\nLF Analysis:")
print(lf_analysis)

fig, ax = plt.subplots(figsize=(12, 6))
lf_names = lf_analysis.index.tolist()
coverage = lf_analysis["Coverage"].values
accuracy = lf_analysis["Emp. Acc."].values
x_pos = np.arange(len(lf_names))
width = 0.35
ax.bar(x_pos - width/2, coverage, width, label="Coverage", color="steelblue", alpha=0.8)
ax.bar(x_pos + width/2, accuracy, width, label="Empirical Accuracy", color="coral", alpha=0.8)
ax.set_ylabel("Score")
ax.set_title("Labeling Function Coverage vs Accuracy")
ax.set_xticks(x_pos)
ax.set_xticklabels(lf_names, rotation=45, ha="right", fontsize=8)
ax.legend()
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("data/lf_coverage_accuracy.png", dpi=150)
plt.close()
print("Saved: data/lf_coverage_accuracy.png")

print("\n" + "=" * 60)
print("STEP 5: Training Snorkel Label Model")
print("=" * 60)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr=0.01, log_freq=100, seed=42)

probs_train = label_model.predict_proba(L=L_train)
preds_train = probs_to_preds(probs=probs_train)

preds_test_lm = label_model.predict(L=L_test)
preds_test_lm_clean = np.where(preds_test_lm == ABSTAIN, HAM, preds_test_lm)

print(f"\nLabel Model: {np.sum(preds_test_lm == ABSTAIN)} abstains out of {len(preds_test_lm)} test points")
print("\nLabel Model Test Performance:")
print(classification_report(Y_test, preds_test_lm_clean, target_names=["Ham", "Spam"]))

print("\n" + "=" * 60)
print("STEP 6: Training End Models")
print("=" * 60)

# Use Label Model only where at least one LF voted
lf_coverage_mask = (L_train != ABSTAIN).any(axis=1)
final_labels = np.full(len(df_train), HAM)
final_labels[lf_coverage_mask] = label_model.predict(L=L_train)[lf_coverage_mask]
# Clean up any remaining abstains
final_labels = np.where(final_labels == ABSTAIN, HAM, final_labels)
df_train_labeled = df_train.copy()
df_train_labeled["snorkel_label"] = final_labels

print(f"Labeled {len(df_train_labeled)} out of {len(df_train)} training examples")
print(f"Label distribution:")
print(pd.Series(df_train_labeled["snorkel_label"]).value_counts())

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train = vectorizer.fit_transform(df_train_labeled.text)
X_test = vectorizer.transform(df_test.text)
y_train = df_train_labeled.snorkel_label.values

lr_model = LogisticRegression(solver="lbfgs", max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION Results:")
print("=" * 60)
print(classification_report(Y_test, lr_preds, target_names=["Ham", "Spam"]))

print("=" * 60)
print("RANDOM FOREST Results (YOUR MODIFICATION):")
print("=" * 60)
print(classification_report(Y_test, rf_preds, target_names=["Ham", "Spam"]))

models = ["Label Model", "Logistic Regression", "Random Forest"]
accuracies = [
    accuracy_score(Y_test, preds_test_lm_clean),
    accuracy_score(Y_test, lr_preds),
    accuracy_score(Y_test, rf_preds)
]
f1_scores_list = [
    f1_score(Y_test, preds_test_lm_clean, average="weighted"),
    f1_score(Y_test, lr_preds, average="weighted"),
    f1_score(Y_test, rf_preds, average="weighted")
]

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(models))
width = 0.3
ax.bar(x_pos - width/2, accuracies, width, label="Accuracy", color="steelblue")
ax.bar(x_pos + width/2, f1_scores_list, width, label="F1 Score (weighted)", color="coral")
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Snorkel-Labeled SMS Spam Detection")
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.1)
for i, (acc, f1) in enumerate(zip(accuracies, f1_scores_list)):
    ax.text(i - width/2, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9)
    ax.text(i + width/2, f1 + 0.02, f"{f1:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("data/model_comparison.png", dpi=150)
plt.close()
print("\nSaved: data/model_comparison.png")

print("\n" + "=" * 60)
print("ALL DONE! Check the data/ folder for saved plots.")
print("=" * 60)
