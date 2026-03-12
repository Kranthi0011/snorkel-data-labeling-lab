# ============================================================
# Tutorial 03: SMS Spam Data Slicing with Slicing Functions
# Modification: SMS dataset, custom SFs, per-slice performance monitoring
# ============================================================

import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from snorkel.slicing import slicing_function, PandasSFApplier
from utils import load_sms_spam_dataset

from textblob import TextBlob

HAM = 0
SPAM = 1

# ============================================================
# STEP 1: Load Labeled Data
# ============================================================
print("=" * 60)
print("STEP 1: Loading SMS Spam Dataset (with labels)")
print("=" * 60)

df_train, df_test, train_gold_labels = load_sms_spam_dataset()
Y_test = df_test.label.values
Y_train = train_gold_labels
df_train["label"] = train_gold_labels

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(f"Train class distribution:")
print(df_train.label.value_counts())

# ============================================================
# STEP 2: Train Baseline Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Training Baseline Models")
print("=" * 60)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train = vectorizer.fit_transform(df_train.text)
X_test = vectorizer.transform(df_test.text)

lr_model = LogisticRegression(solver="lbfgs", max_iter=1000)
lr_model.fit(X_train, Y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
rf_preds = rf_model.predict(X_test)

print("Baseline Logistic Regression:")
print(classification_report(Y_test, lr_preds, target_names=["Ham", "Spam"]))
print("Baseline Random Forest:")
print(classification_report(Y_test, rf_preds, target_names=["Ham", "Spam"]))

# ============================================================
# STEP 3: Define Slicing Functions (as plain functions)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Defining Slicing Functions")
print("=" * 60)

def sf_short_message(text):
    return len(text.split()) < 5

def sf_long_message(text):
    return len(text.split()) > 30

def sf_contains_url(text):
    return bool(re.search(r"http[s]?://|www\.", text.lower()))

def sf_contains_phone_number(text):
    return bool(re.search(r"\b\d{5,}\b", text))

def sf_contains_currency(text):
    return bool(re.search(r"[$]|pound|cash|free", text.lower()))

def sf_positive_sentiment(text):
    return TextBlob(text).sentiment.polarity > 0.3

def sf_negative_sentiment(text):
    return TextBlob(text).sentiment.polarity < -0.1

def sf_all_caps(text):
    if len(text) < 5:
        return False
    return sum(1 for c in text if c.isupper()) / len(text) > 0.5

def sf_heavy_punctuation(text):
    return sum(1 for c in text if c in "!?.") >= 4

slice_funcs = {
    "short_message": sf_short_message,
    "long_message": sf_long_message,
    "contains_url": sf_contains_url,
    "contains_phone_number": sf_contains_phone_number,
    "contains_currency": sf_contains_currency,
    "positive_sentiment": sf_positive_sentiment,
    "negative_sentiment": sf_negative_sentiment,
    "all_caps": sf_all_caps,
    "heavy_punctuation": sf_heavy_punctuation,
}

print(f"Defined {len(slice_funcs)} slicing functions")

# ============================================================
# STEP 4: Apply Slicing Functions to Test Set
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Applying Slicing Functions to Test Set")
print("=" * 60)

slice_masks = {}
for name, func in slice_funcs.items():
    mask = df_test["text"].apply(func).values
    slice_masks[name] = mask
    count = mask.sum()
    pct = count / len(df_test) * 100
    print(f"  {name}: {count} test points ({pct:.1f}%)")

# ============================================================
# STEP 5: Monitor Per-Slice Performance
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Per-Slice Performance (Logistic Regression)")
print("=" * 60)

print(f"\n{'Slice':<25} {'Size':>6} {'Acc':>8} {'F1':>8} {'Spam_F1':>8}")
print("-" * 60)

slice_results = []
for name, mask in slice_masks.items():
    if mask.sum() < 2:
        continue
    acc = accuracy_score(Y_test[mask], lr_preds[mask])
    f1 = f1_score(Y_test[mask], lr_preds[mask], average="weighted", zero_division=0)
    spam_count = Y_test[mask].sum()
    if spam_count > 0:
        spam_f1 = f1_score(Y_test[mask], lr_preds[mask], average="binary", zero_division=0)
    else:
        spam_f1 = float("nan")
    slice_results.append({"name": name, "size": mask.sum(), "acc": acc, "f1": f1, "spam_f1": spam_f1})
    print(f"  {name:<23} {mask.sum():>6} {acc:>8.3f} {f1:>8.3f} {spam_f1:>8.3f}")

# Overall
overall_acc = accuracy_score(Y_test, lr_preds)
overall_f1 = f1_score(Y_test, lr_preds, average="weighted")
overall_spam_f1 = f1_score(Y_test, lr_preds, average="binary")
print(f"  {'OVERALL':<23} {len(Y_test):>6} {overall_acc:>8.3f} {overall_f1:>8.3f} {overall_spam_f1:>8.3f}")

# ============================================================
# STEP 6: Visualize Slice Performance (YOUR MODIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Visualizing Slice Performance")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Slice sizes
names = [r["name"] for r in slice_results]
sizes = [r["size"] for r in slice_results]

axes[0].barh(names, sizes, color="steelblue", alpha=0.8)
axes[0].set_xlabel("Number of Test Points")
axes[0].set_title("Slice Sizes in Test Set")
for i, v in enumerate(sizes):
    axes[0].text(v + 2, i, str(v), va="center", fontsize=9)

# Chart 2: F1 scores per slice
f1_vals = [r["f1"] for r in slice_results]
colors = ["coral" if f < 0.9 else "steelblue" for f in f1_vals]

axes[1].barh(names, f1_vals, color=colors, alpha=0.8)
axes[1].set_xlabel("F1 Score (weighted)")
axes[1].set_title("F1 Score by Slice (Red = Below 0.9)")
axes[1].axvline(x=0.9, color="gray", linestyle="--", alpha=0.5)
for i, v in enumerate(f1_vals):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("data/slice_performance.png", dpi=150)
plt.close()
print("Saved: data/slice_performance.png")

# ============================================================
# STEP 7: Analyze Worst-Performing Slices (YOUR MODIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Analyzing Worst-Performing Slices")
print("=" * 60)

sorted_results = sorted(slice_results, key=lambda r: r["f1"])
print("\nTop 3 worst-performing slices:")
for r in sorted_results[:3]:
    print(f"  {r['name']}: F1={r['f1']:.3f}, Acc={r['acc']:.3f}, Size={r['size']}")

# Show misclassified examples from worst slice
worst = sorted_results[0]
mask = slice_masks[worst["name"]]
slice_df = df_test[mask].copy()
slice_df["predicted"] = lr_preds[mask]
misclassified = slice_df[slice_df.label != slice_df.predicted]
print(f"\nMisclassified in '{worst['name']}' slice ({len(misclassified)} errors):")
for _, row in misclassified.head(5).iterrows():
    true_label = "SPAM" if row.label == 1 else "HAM"
    pred_label = "SPAM" if row.predicted == 1 else "HAM"
    print(f"  True={true_label}, Pred={pred_label}: {row.text[:80]}...")

# ============================================================
# STEP 8: Compare LR vs RF Per Slice (YOUR MODIFICATION)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Comparing LR vs RF Per Slice")
print("=" * 60)

print(f"\n{'Slice':<25} {'LR F1':>8} {'RF F1':>8} {'Better':>8}")
print("-" * 54)

lr_f1s = []
rf_f1s = []
plot_names = []

for name, mask in slice_masks.items():
    if mask.sum() < 5:
        continue
    lr_f1 = f1_score(Y_test[mask], lr_preds[mask], average="weighted", zero_division=0)
    rf_f1 = f1_score(Y_test[mask], rf_preds[mask], average="weighted", zero_division=0)
    better = "LR" if lr_f1 >= rf_f1 else "RF"
    print(f"  {name:<23} {lr_f1:>8.3f} {rf_f1:>8.3f} {better:>8}")
    plot_names.append(name)
    lr_f1s.append(lr_f1)
    rf_f1s.append(rf_f1)

lr_overall = f1_score(Y_test, lr_preds, average="weighted")
rf_overall = f1_score(Y_test, rf_preds, average="weighted")
print(f"  {'OVERALL':<23} {lr_overall:>8.3f} {rf_overall:>8.3f} {'LR' if lr_overall >= rf_overall else 'RF':>8}")
plot_names.append("OVERALL")
lr_f1s.append(lr_overall)
rf_f1s.append(rf_overall)

# ============================================================
# STEP 9: Final Visualization (YOUR MODIFICATION)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(plot_names))
width = 0.3

ax.bar(x - width / 2, lr_f1s, width, label="Logistic Regression", color="steelblue", alpha=0.8)
ax.bar(x + width / 2, rf_f1s, width, label="Random Forest", color="coral", alpha=0.8)
ax.set_ylabel("F1 Score (weighted)")
ax.set_title("Per-Slice Model Comparison: LR vs RF")
ax.set_xticks(x)
ax.set_xticklabels(plot_names, rotation=45, ha="right", fontsize=8)
ax.legend()
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig("data/slice_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: data/slice_model_comparison.png")

print("\n" + "=" * 60)
print("ALL DONE! Check data/ folder for saved plots.")
print("=" * 60)
