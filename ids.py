# =============================================================================
# CIC-IDS2017 — Full ML Pipeline
# Preprocessing → Feature Selection → Training → Evaluation
# =============================================================================
# Install once:
#   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost
#
# Run:
#   python ids_full_pipeline.py
# =============================================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Preprocessing & feature selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

# Sampling
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline       import Pipeline as ImbPipeline

# Models
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from xgboost                  import XGBClassifier

# Evaluation
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)

import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — change path to your folder
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH       = r"D:\Downloads\MachineLearningCVE"   # ← your folder
OUTPUT_DIR      = r"D:\Downloads\ids_output"            # ← where results go
N_FEATURES      = 20          # top-K features to keep after selection
TEST_SIZE       = 0.15        # 15 % test
VAL_SIZE        = 0.15        # 15 % validation  (from remaining 85 %)
RANDOM_STATE    = 42
MAX_SMOTE       = 5000        # cap minority classes before SMOTE (speed)
MAX_BENIGN      = 200_000     # undersample BENIGN to this count
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD
# =============================================================================
print("\n" + "═"*60)
print("  STEP 1 — LOAD")
print("═"*60)

csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []
for f in csv_files:
    print(f"  Loading {os.path.basename(f)} …")
    try:
        dfs.append(pd.read_csv(f, low_memory=False))
    except Exception as e:
        print(f"  ✖  Error: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"\nRaw shape : {df.shape}")


# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "═"*60)
print("  STEP 2 — PREPROCESSING")
print("═"*60)

# ── 2a. Strip whitespace from column names ───────────────────────────────────
df.columns = df.columns.str.strip()
print(f"Columns cleaned. Label column: 'Label'")

# ── 2b. Drop duplicates ──────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed : {before - len(df):,}  ({len(df):,} remain)")

# ── 2c. Replace inf / -inf → NaN, then drop NaN rows ────────────────────────
df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"Rows with NaN removed : {before - len(df):,}  ({len(df):,} remain)")

# ── 2d. Drop non-numeric / identifier columns ────────────────────────────────
drop_cols = ["Flow ID", "Source IP", "Source Port",
             "Destination IP", "Destination Port", "Timestamp"]
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped identifier columns: {drop_cols}")

# ── 2e. Label distribution ───────────────────────────────────────────────────
print("\nClass distribution (raw):")
print(df["Label"].value_counts().to_string())


# =============================================================================
# 3. ENCODE LABELS
# =============================================================================
print("\n" + "═"*60)
print("  STEP 3 — LABEL ENCODING")
print("═"*60)

# Map verbose labels → short names (easier to read in plots)
LABEL_MAP = {
    "BENIGN"                       : "BENIGN",
    "DoS Hulk"                     : "DoS-Hulk",
    "PortScan"                      : "PortScan",
    "DDoS"                          : "DDoS",
    "DoS GoldenEye"                 : "DoS-GoldenEye",
    "FTP-Patator"                   : "FTP-Patator",
    "SSH-Patator"                   : "SSH-Patator",
    "DoS slowloris"                 : "DoS-Slowloris",
    "DoS Slowhttptest"              : "DoS-Slowhttptest",
    "Bot"                           : "Bot",
    "Web Attack \x96 Brute Force"   : "Web-BruteForce",
    "Web Attack \x96 XSS"           : "Web-XSS",
    "Web Attack \x96 Sql Injection" : "Web-SQLi",
    "Infiltration"                  : "Infiltration",
    "Heartbleed"                    : "Heartbleed",
    # handle encoding variants
    "Web Attack Brute Force"    : "Web-BruteForce",
    "Web Attack XSS"            : "Web-XSS",
    "Web Attack Sql Injection"  : "Web-SQLi",
}
df["Label"] = df["Label"].map(lambda x: LABEL_MAP.get(x, x))

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["Label"])
classes = list(le.classes_)
print(f"Classes ({len(classes)}): {classes}")


# =============================================================================
# 4. FEATURE SELECTION
# =============================================================================
print("\n" + "═"*60)
print("  STEP 4 — FEATURE SELECTION")
print("═"*60)

feature_cols = [c for c in df.columns if c not in ("Label", "label_enc")]
X_all = df[feature_cols].values.astype(np.float32)
y_all = df["label_enc"].values

# ── 4a. Remove zero-variance columns ─────────────────────────────────────────
var = X_all.var(axis=0)
nonzero_mask = var > 0
removed_zero_var = [feature_cols[i] for i in range(len(feature_cols)) if not nonzero_mask[i]]
X_all = X_all[:, nonzero_mask]
feature_cols = [f for f, keep in zip(feature_cols, nonzero_mask) if keep]
print(f"Zero-variance features removed: {len(removed_zero_var)}  → {removed_zero_var}")

# ── 4b. Correlation filter — drop if |corr| > 0.97 with another feature ─────
print("Running correlation filter …")
corr_df   = pd.DataFrame(X_all, columns=feature_cols).corr().abs()
upper     = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
drop_corr = [col for col in upper.columns if any(upper[col] > 0.97)]
keep_mask = [f not in drop_corr for f in feature_cols]
X_all     = X_all[:, keep_mask]
feature_cols = [f for f, k in zip(feature_cols, keep_mask) if k]
print(f"Highly correlated features removed: {len(drop_corr)}  → {len(feature_cols)} remain")

# ── 4c. Mutual information on a 50k sample (fast) ───────────────────────────
print(f"Computing mutual information (sample=50k) …")
sample_idx = np.random.choice(len(X_all), size=min(50_000, len(X_all)), replace=False)
selector   = SelectKBest(mutual_info_classif, k=min(N_FEATURES, X_all.shape[1]))
selector.fit(X_all[sample_idx], y_all[sample_idx])

top_mask     = selector.get_support()
top_features = [f for f, m in zip(feature_cols, top_mask) if m]
mi_scores    = selector.scores_[top_mask]

print(f"\nTop {N_FEATURES} features by mutual information:")
for feat, score in sorted(zip(top_features, mi_scores), key=lambda x: -x[1]):
    bar = "█" * int(score * 30 / max(mi_scores))
    print(f"  {feat:<40} {score:.4f}  {bar}")

X_selected   = X_all[:, top_mask]

# ── 4d. Save feature importance plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
feat_order = sorted(zip(top_features, mi_scores), key=lambda x: x[1])
ax.barh([f[0] for f in feat_order], [f[1] for f in feat_order], color="#378ADD")
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top Feature Importance (Mutual Information vs Label)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"\nFeature importance plot → {OUTPUT_DIR}/feature_importance.png")


# =============================================================================
# 5. BALANCE THE DATASET
# =============================================================================
print("\n" + "═"*60)
print("  STEP 5 — CLASS BALANCING")
print("═"*60)

print("Class counts before balancing:")
for cls, cnt in sorted(Counter(y_all).items(), key=lambda x: -x[1]):
    print(f"  {le.inverse_transform([cls])[0]:<25} {cnt:>8,}")

# Strategy:
#  • Undersample BENIGN to MAX_BENIGN
#  • Cap tiny classes at MAX_SMOTE then SMOTE up to a moderate count
counts      = Counter(y_all)
benign_cls  = le.transform(["BENIGN"])[0]

under_strategy = {cls: min(cnt, MAX_BENIGN if cls == benign_cls else cnt)
                  for cls, cnt in counts.items()}

# SMOTE target: bring every class to at least the median count (capped)
median_count = int(np.median(list(counts.values())))
smote_target = max(median_count, 2000)
over_strategy = {cls: max(cnt, smote_target)
                 for cls, cnt in under_strategy.items()
                 if cnt >= 6}   # SMOTE needs ≥ 6 samples

print(f"\nUndersampling BENIGN → {MAX_BENIGN:,}")
print(f"SMOTE minority classes → ~{smote_target:,} each")

# Use imbalanced-learn Pipeline
sampler = ImbPipeline([
    ("under", RandomUnderSampler(sampling_strategy={benign_cls: MAX_BENIGN},
                                  random_state=RANDOM_STATE)),
    ("over",  SMOTE(sampling_strategy=over_strategy,
                    k_neighbors=5, random_state=RANDOM_STATE))
])

X_bal, y_bal = sampler.fit_resample(X_selected, y_all)

print("\nClass counts after balancing:")
for cls, cnt in sorted(Counter(y_bal).items(), key=lambda x: -x[1]):
    print(f"  {le.inverse_transform([cls])[0]:<25} {cnt:>8,}")

print(f"\nBalanced dataset shape: {X_bal.shape}")


# =============================================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
print("\n" + "═"*60)
print("  STEP 6 — TRAIN / VAL / TEST SPLIT")
print("═"*60)

# First split off test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X_bal, y_bal,
    test_size=TEST_SIZE,
    stratify=y_bal,
    random_state=RANDOM_STATE
)
# Then split validation from remaining
val_ratio = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Train      : {X_train.shape[0]:>8,} rows  ({X_train.shape[0]/len(X_bal)*100:.1f}%)")
print(f"Validation : {X_val.shape[0]:>8,} rows  ({X_val.shape[0]/len(X_bal)*100:.1f}%)")
print(f"Test       : {X_test.shape[0]:>8,} rows  ({X_test.shape[0]/len(X_bal)*100:.1f}%)")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit ONLY on train
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# =============================================================================
# 7. TRAIN MODELS
# =============================================================================
print("\n" + "═"*60)
print("  STEP 7 — TRAIN MODELS")
print("═"*60)

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
    ),
}

val_results = {}

for name, model in MODELS.items():
    print(f"\n  Training {name} …")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    f1  = f1_score(y_val, y_val_pred, average="weighted")
    print(f"  Val accuracy: {acc:.4f}   Val F1 (weighted): {f1:.4f}")
    val_results[name] = {"model": model, "acc": acc, "f1": f1}

# Pick best model by validation F1
best_name  = max(val_results, key=lambda n: val_results[n]["f1"])
best_model = val_results[best_name]["model"]
print(f"\n  ✔  Best model: {best_name}  (val F1={val_results[best_name]['f1']:.4f})")


# =============================================================================
# 8. EVALUATION ON TEST SET
# =============================================================================
print("\n" + "═"*60)
print("  STEP 8 — FINAL EVALUATION ON TEST SET")
print("═"*60)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
except Exception:
    auc = float("nan")

print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")

print(f"\n  Per-class report:\n")
print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name} (test set)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"  Confusion matrix → {OUTPUT_DIR}/confusion_matrix.png")

# ── Per-class F1 bar chart ────────────────────────────────────────────────────
per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(classes, per_class_f1, color=["#E24B4A" if f < 0.8 else "#378ADD" for f in per_class_f1])
ax.axhline(0.8, color="orange", linestyle="--", linewidth=1, label="0.80 threshold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score on Test Set")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150)
plt.close()
print(f"  Per-class F1 chart → {OUTPUT_DIR}/per_class_f1.png")

# ── Validation comparison table ───────────────────────────────────────────────
print(f"\n  Model comparison (validation set):")
print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (weighted)':>15}")
print(f"  {'-'*50}")
for name, res in val_results.items():
    star = " ← best" if name == best_name else ""
    print(f"  {name:<25} {res['acc']:>10.4f} {res['f1']:>15.4f}{star}")


# =============================================================================
# 9. SAVE ARTEFACTS
# =============================================================================
print("\n" + "═"*60)
print("  STEP 9 — SAVE ARTEFACTS")
print("═"*60)

artefact = {
    "model":         best_model,
    "model_name":    best_name,
    "scaler":        scaler,
    "label_encoder": le,
    "features":      top_features,
    "classes":       classes,
    "metrics": {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
    }
}

model_path = os.path.join(OUTPUT_DIR, "ids_model.pkl")
joblib.dump(artefact, model_path, compress=3)
print(f"  Model saved  → {model_path}")

# Save feature list
feat_path = os.path.join(OUTPUT_DIR, "selected_features.txt")
with open(feat_path, "w") as f:
    for feat in top_features:
        f.write(feat + "\n")
print(f"  Features saved → {feat_path}")

print(f"\n{'═'*60}")
print(f"  Done. All outputs in: {OUTPUT_DIR}")
print(f"  Best model : {best_name}")
print(f"  Test acc   : {acc*100:.2f}%   F1: {f1:.4f}")
print(f"{'═'*60}\n")