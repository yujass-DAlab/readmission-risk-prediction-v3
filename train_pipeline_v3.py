# ============================================================
# TRAINING PIPELINE V3
# Version: v3
# Authors: JasmineYu, Deepseek, ChatGPT
#
# Split: 60% train / 20% validation / 20% test
#   - Train: fit imputer + SMOTE + model
#   - Validation: select threshold (recall target + max precision)
#   - Test: UNTOUCHED evaluation + deployment-readiness analysis
#
# Includes:
#   - Experiment A: threshold sweep (recall / precision / F1 / alert rate)
#   - Experiment B: PR-AUC lift over random baseline
#   - Experiment C: subgroup analysis (age, gender, admission type)
#
# ⚠️ KNOWN LIMITATIONS (documented, not fixed in V3):
#   1. Patient-level leakage: random stratified split may place multiple
#      encounters from the same patient in train AND test.
#   2. SMOTE on categoricals: SMOTE interpolates encoded categorical
#      features, potentially creating fractional category values.
#   3. Prediction time: model predicts AT DISCHARGE.
#   4. Multi-worker threshold: each worker maintains process-local threshold.
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
import os
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from shared_feature_extraction import (
    extract_raw_features_from_text, validate_feature_ranges,
    FEATURE_NAMES, EXPECTED_FEATURE_COUNT, FEATURE_SCHEMA_HASH, EXTRACTOR_VERSION,
)

# --- 1. CONFIGURATION ---
DATA_PATH = "readmission_data.csv"
MODEL_VERSION = "v3"
RANDOM_STATE = 42
TARGET_RECALL = 0.85

# ⚠️ EXPERIMENTAL safety floor — NOT a validated clinical criterion.
MIN_ACCEPTABLE_PRECISION = 0.05

# Maximum tolerance for invalid/missing labels. Exceeding this aborts training.
MAX_ALLOWED_LABEL_DROP_RATE = 0.02

# Threshold grid for Experiment A (kept explicit for reproducibility).
THRESHOLD_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                   0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ============================================================
# ANALYSIS HELPER FUNCTIONS (Experiments A, B, C)
# ============================================================

def _binary_class2_metrics(y_true, y_pred_binary):
    """Compute precision / recall / F1 / alert_rate for class 2 as positive."""
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec  = recall_score(y_true, y_pred_binary, zero_division=0)
    f1   = f1_score(y_true, y_pred_binary, zero_division=0)
    alert_rate = float(y_pred_binary.mean())
    return float(prec), float(rec), float(f1), alert_rate


def run_experiment_a_threshold_sweep(y_test_bin2, test_probs_class2):
    """
    Experiment A — Threshold sweep on UNTOUCHED TEST set.
    Answers: 'Is there a more clinically reasonable operating point
    than the 85%-recall target inherited from V1?'
    """
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT A — THRESHOLD SWEEP (on TEST set)")
    print("=" * 70)
    print(f"{'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'Alert%':>7}")
    print("-" * 55)

    sweep = []
    for t in THRESHOLD_SWEEP:
        pred = (test_probs_class2 >= t).astype(int)
        prec, rec, f1, alert_rate = _binary_class2_metrics(y_test_bin2, pred)
        sweep.append({
            "threshold": float(t),
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "alert_rate": alert_rate,
        })
        print(f"{t:>7.2f} | {rec:>7.4f} | {prec:>7.4f} | {f1:>7.4f} | {alert_rate*100:>6.1f}%")

    return sweep


def run_experiment_b_pr_auc_lift(y_test_bin2, test_probs_class2):
    """
    Experiment B — PR-AUC lift over random baseline.
    Random baseline PR-AUC ≈ class-2 prevalence. Lift = PR-AUC / baseline.
    """
    base_rate = float(y_test_bin2.mean())
    pr_auc = float(average_precision_score(y_test_bin2, test_probs_class2))
    lift = pr_auc / base_rate if base_rate > 0 else 0.0

    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT B — PR-AUC LIFT OVER RANDOM BASELINE")
    print("=" * 70)
    print(f"   Class-2 prevalence (random baseline): {base_rate:.4f}")
    print(f"   Model PR-AUC:                         {pr_auc:.4f}")
    print(f"   Lift over random:                     {lift:.3f}x")

    if lift < 1.1:
        print("   ⚠️ Weak discriminative power — barely above random.")
    elif lift < 1.5:
        print("   🟡 Modest lift — real but limited signal.")
    else:
        print("   🟢 Meaningful lift — the model has genuine discriminative power.")

    return {
        "test_class2_prevalence": base_rate,
        "test_pr_auc": pr_auc,
        "test_pr_auc_lift_over_random": lift,
    }


def run_experiment_c_subgroup_analysis(X_test_imp, y_test, y_test_bin2, test_probs_class2, threshold):
    """
    Experiment C — Subgroup analysis.
    For each subgroup, report: N, class-2 recall, precision, alert rate.
    Detects whether the model behaves uniformly across the population.
    """
    print("\n" + "=" * 70)
    print(f"🧪 EXPERIMENT C — SUBGROUP ANALYSIS (threshold={threshold:.4f})")
    print("=" * 70)

    pred = (test_probs_class2 >= threshold).astype(int)

    # Pick interpretable features to slice on
    slice_specs = [
        ("age_bin", {
            0.0: "age <20", 1.0: "age 20-39", 2.0: "age 40-59",
            3.0: "age 60-79", 4.0: "age 80+",
        }),
        ("gender", {0.0: "Female", 1.0: "Male", 2.0: "Unknown"}),
        ("admission_type", {
            0.0: "Elective", 1.0: "Emergency", 2.0: "Newborn",
            3.0: "Unknown", 4.0: "Urgent",
        }),
    ]

    results = {}
    for feature_name, label_map in slice_specs:
        if feature_name not in FEATURE_NAMES:
            continue
        idx = FEATURE_NAMES.index(feature_name)
        print(f"\n   ── Slicing by '{feature_name}' ──")
        print(f"   {'Group':<12} | {'N':>6} | {'Recall':>7} | {'Prec':>7} | {'Alert%':>7}")
        print("   " + "-" * 58)

        feature_results = []
        for val, label in label_map.items():
            mask = (X_test_imp[:, idx] == val)
            n = int(mask.sum())
            if n == 0:
                continue
            subgroup_recall = recall_score(
                y_test_bin2[mask], pred[mask], zero_division=0
            )
            subgroup_prec = precision_score(
                y_test_bin2[mask], pred[mask], zero_division=0
            )
            subgroup_alert = float(pred[mask].mean())
            feature_results.append({
                "group": label,
                "n": n,
                "recall": float(subgroup_recall),
                "precision": float(subgroup_prec),
                "alert_rate": subgroup_alert,
            })
            print(f"   {label:<12} | {n:>6} | {subgroup_recall:>7.4f} | "
                  f"{subgroup_prec:>7.4f} | {subgroup_alert*100:>6.1f}%")
        results[feature_name] = feature_results

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

# --- 2. DATA LOADING ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
n_total = len(df)
print(f"✅ Loaded {n_total} rows")
print(f"🔐 Extractor version: {EXTRACTOR_VERSION} | Schema hash: {FEATURE_SCHEMA_HASH}")

# --- 2b. DEFENSIVE LABEL FILTER ---
valid_mask = df['readmitted_label'].isin([0, 1, 2])
n_dropped = int((~valid_mask).sum())
drop_rate = n_dropped / n_total if n_total > 0 else 0.0
if n_dropped > 0:
    print(f"⚠️ Dropping {n_dropped} rows with invalid/missing labels ({drop_rate*100:.2f}%).")
if drop_rate > MAX_ALLOWED_LABEL_DROP_RATE:
    raise ValueError(
        f"Label drop rate {drop_rate*100:.2f}% exceeds tolerance "
        f"{MAX_ALLOWED_LABEL_DROP_RATE*100:.2f}%. Aborting — data may be contaminated."
    )
df = df[valid_mask].reset_index(drop=True)

# --- 3. EXTRACT FEATURES ---
print(f"⚙️ Extracting features (expected: {EXPECTED_FEATURE_COUNT})...")
X_raw = np.array([extract_raw_features_from_text(t) for t in df['clinical_text']], dtype=np.float32)
y = df['readmitted_label'].values
assert X_raw.shape[1] == EXPECTED_FEATURE_COUNT, (
    f"Feature count mismatch! Got {X_raw.shape[1]}, expected {EXPECTED_FEATURE_COUNT}."
)

# --- 3b. VALUE-RANGE VALIDATION (report only) ---
print("\n🛡️ Running value-range validation (report only, no mutation)...")
range_issue_count = 0
sample_issues = []
for row in X_raw[:1000]:
    _, issues = validate_feature_ranges(list(row), warn=False)
    range_issue_count += len(issues)
    if issues and len(sample_issues) < 3:
        sample_issues.append(issues[0])
print(f"   → {range_issue_count} out-of-range values detected in first 1000 rows.")
for ex in sample_issues:
    print(f"     e.g., {ex}")

# --- 4. THREE-WAY SPLIT ---
print("\n📊 Splitting 60% train / 20% validation / 20% test...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"   Train:      {len(X_train):>6} rows")
print(f"   Validation: {len(X_val):>6} rows  ← for threshold tuning")
print(f"   Test:       {len(X_test):>6} rows  ← untouched")

# ============================================================
# 🔍 FEATURE ANALYSIS (on TRAIN only)
# ============================================================
print("\n" + "=" * 70)
print("🔍 FEATURE ANALYSIS REPORT")
print("=" * 70)

print("\n[4a] Missingness (% NaN) on training set:")
missing_pct = np.isnan(X_train).mean(axis=0) * 100
for name, pct in zip(FEATURE_NAMES, missing_pct):
    flag = "  ⚠️" if pct > 30 else ""
    print(f"   {name:<22} {pct:6.2f}%{flag}")

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)
medians_list = imputer.statistics_.tolist()

print("\n[4b] Pearson r and p-value with each class (|r| ≥ 0.05 or p < 0.01 shown):")
print("     NOTE: Pearson r measures LINEAR correlation with a single class.")
print("     For CATEGORICAL features the encoding implies artificial order.")
print("     Pearson and RF importance answer DIFFERENT questions:")
print("       - Pearson: monotonic linear trend with one class")
print("       - RF importance: nonlinear usefulness for the full multiclass problem")
print("     They will NOT rank features the same way. That's expected, not a bug.")
corr_report = []
for i, name in enumerate(FEATURE_NAMES):
    row = {"feature": name}
    any_sig = False
    for cls in [0, 1, 2]:
        y_bin = (y_train == cls).astype(int)
        if np.std(X_train_imp[:, i]) == 0:
            r, p_val = 0.0, 1.0
        else:
            r, p_val = pearsonr(X_train_imp[:, i], y_bin)
        row[f"r_class{cls}"] = r
        row[f"p_class{cls}"] = p_val
        if abs(r) >= 0.05 or p_val < 0.01:
            any_sig = True
    row["_any_sig"] = any_sig
    corr_report.append(row)

corr_df = pd.DataFrame(corr_report).sort_values(
    by=["_any_sig", "r_class2"], ascending=[False, False]
)
for _, row in corr_df.iterrows():
    marker = "✅" if row["_any_sig"] else "  "
    print(f"   {marker} {row['feature']:<22} "
          f"r0={row['r_class0']:+.3f}(p={row['p_class0']:.1e})  "
          f"r1={row['r_class1']:+.3f}(p={row['p_class1']:.1e})  "
          f"r2={row['r_class2']:+.3f}(p={row['p_class2']:.1e})")

# ============================================================
# 🎯 TRAINING WITH SMOTE
# ============================================================
print("\n" + "=" * 70)
print("🔄 Applying SMOTE to TRAIN only...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train)
print(f"   Before SMOTE: {X_train_imp.shape[0]} samples")
print(f"   After  SMOTE: {X_train_bal.shape[0]} samples")

print("\n[4c] Feature importance from a quick RandomForest (top 15):")
print("     (nonlinear, multiclass view — see Pearson NOTE above)")
quick_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
quick_rf.fit(X_train_imp, y_train)
imp_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": quick_rf.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)
for _, row in imp_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"   {row['feature']:<22} {row['importance']:.4f}  {bar}")

print("\n" + "=" * 70)
print("🧠 Training Stacking Ensemble on SMOTE-balanced TRAIN...")
model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=RANDOM_STATE)),
        ('lgb', LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(),
    cv=5, n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

# ============================================================
# 🎯 THRESHOLD SELECTION ON VALIDATION SET
# ============================================================
print("\n" + "=" * 70)
print(f"🎯 Selecting threshold on VALIDATION set (recall >= {TARGET_RECALL:.2f}, max precision)...")

val_probs = model.predict_proba(X_val_imp)[:, 2]
precisions, recalls, thresholds = precision_recall_curve(y_val == 2, val_probs)
recalls_aligned = recalls[:-1]
precisions_aligned = precisions[:-1]

valid_idx = np.where(recalls_aligned >= TARGET_RECALL)[0]
if len(valid_idx) == 0:
    raise ValueError(f"Target recall {TARGET_RECALL} not achievable. "
                     f"Best: {recalls_aligned.max():.4f}.")

best_local = int(np.argmax(precisions_aligned[valid_idx]))
chosen_idx = valid_idx[best_local]
best_threshold = float(thresholds[chosen_idx])
val_recall = float(recalls_aligned[chosen_idx])
val_precision = float(precisions_aligned[chosen_idx])

print(f"   ✅ Threshold chosen: {best_threshold:.4f}")
print(f"      Validation recall:    {val_recall:.4f}")
print(f"      Validation precision: {val_precision:.4f}")

if val_precision < MIN_ACCEPTABLE_PRECISION:
    raise ValueError(
        f"Validation precision {val_precision:.4f} < floor {MIN_ACCEPTABLE_PRECISION}. "
        f"NOTE: floor is experimental, not clinically validated."
    )

# ============================================================
# 📊 FINAL EVALUATION ON UNTOUCHED TEST SET
# ============================================================
print("\n" + "=" * 70)
print("📊 FINAL EVALUATION ON UNTOUCHED TEST SET")
print("=" * 70)

test_probs = model.predict_proba(X_test_imp)
test_probs_class2 = test_probs[:, 2]
test_pred_class2 = (test_probs_class2 >= best_threshold).astype(int)
y_test_bin2 = (y_test == 2).astype(int)

test_precision = precision_score(y_test_bin2, test_pred_class2, zero_division=0)
test_recall    = recall_score(y_test_bin2, test_pred_class2, zero_division=0)
test_f1        = f1_score(y_test_bin2, test_pred_class2, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test_bin2, test_pred_class2, labels=[0, 1]).ravel()
test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

alert_rate = float(test_pred_class2.mean())
pr_auc = float(average_precision_score(y_test_bin2, test_probs_class2))

final_preds = np.where(test_probs_class2 > best_threshold, 2, np.argmax(test_probs, axis=1))
cm = confusion_matrix(y_test, final_preds, labels=[0, 1, 2])
macro_f1 = float(f1_score(y_test, final_preds, average='macro', zero_division=0))

print(f"\n[Class 2 — '<30 days']")
print(f"   Precision:   {test_precision:.4f}")
print(f"   Recall:      {test_recall:.4f}")
print(f"   F1:          {test_f1:.4f}")
print(f"   Specificity: {test_specificity:.4f}")
print(f"   PR-AUC:      {pr_auc:.4f}")
print(f"   Alert rate:  {alert_rate:.4f}  ({alert_rate*100:.1f} alerts per 100 encounters)")
print(f"\n[Multiclass]")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"\n[Confusion Matrix]")
print(f"              Pred NO  Pred >30  Pred <30")
print(f"   True NO    {cm[0,0]:8d} {cm[0,1]:8d} {cm[0,2]:8d}")
print(f"   True >30   {cm[1,0]:8d} {cm[1,1]:8d} {cm[1,2]:8d}")
print(f"   True <30   {cm[2,0]:8d} {cm[2,1]:8d} {cm[2,2]:8d}")

# ============================================================
# 🧪 DEPLOYMENT-READINESS ANALYSIS (Experiments A, B, C)
# ============================================================
print("\n" + "=" * 70)
print("🧪 DEPLOYMENT-READINESS ANALYSIS")
print("=" * 70)
print("These experiments answer: 'Is there a more clinically reasonable")
print("operating point than the 85%-recall target inherited from V1?'")

experiment_a_sweep = run_experiment_a_threshold_sweep(y_test_bin2, test_probs_class2)
experiment_b_lift  = run_experiment_b_pr_auc_lift(y_test_bin2, test_probs_class2)
experiment_c_subgroups = run_experiment_c_subgroup_analysis(
    X_test_imp, y_test, y_test_bin2, test_probs_class2, best_threshold
)

# ============================================================
# 💾 SAVE ARTIFACTS
# ============================================================
print("\n" + "=" * 70)
print("💾 Saving local artifacts...")

joblib.dump(model, "model.pkl")
print("   ✅ model.pkl saved")

with open("imputer.json", "w") as f:
    json.dump({"medians": medians_list, "feature_count": len(medians_list)}, f)
print("   ✅ imputer.json saved")

metadata = {
    # --- Core (read by app_v3.py) ---
    "version": MODEL_VERSION,
    "extractor_version": EXTRACTOR_VERSION,
    "feature_schema_hash": FEATURE_SCHEMA_HASH,
    "features": len(medians_list),
    "feature_names": FEATURE_NAMES,
    "alert_threshold": best_threshold,
    "prediction_time_point": "discharge",
    "prediction_time_note": (
        "V3 predicts at discharge. Features like LOS, num_meds, num_labs, "
        "num_proc assume post-admission information. Admission-time prediction "
        "would require removing these features (V4)."
    ),

    # --- Validation & Test metrics ---
    "validation_recall": val_recall,
    "validation_precision": val_precision,
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_specificity": float(test_specificity),
    "test_pr_auc": float(pr_auc),
    "test_alert_rate": float(alert_rate),
    "test_macro_f1": float(macro_f1),

    # --- Experiment B — PR-AUC lift ---
    "experiment_b_pr_auc_lift": experiment_b_lift,

    # --- Experiment A — Threshold sweep ---
    "experiment_a_threshold_sweep": experiment_a_sweep,

    # --- Experiment C — Subgroup analysis ---
    "experiment_c_subgroup_analysis": experiment_c_subgroups,

    # --- Metadata / governance ---
    "target_recall": float(TARGET_RECALL),
    "min_acceptable_precision": float(MIN_ACCEPTABLE_PRECISION),
    "min_acceptable_precision_note": "experimental safety floor, NOT clinically validated",
    "random_state": RANDOM_STATE,
    "known_limitations": [
        "Random stratified split — NOT patient-level or temporal. Multiple encounters from same patient may cross splits.",
        "SMOTE applied to encoded categoricals — may create fractional category values.",
        "Prediction time is discharge; admission-time prediction is V4.",
        "Process-local threshold in multi-worker deployments.",
    ],
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("   ✅ metadata.json saved")

print("\n🎉 Training pipeline v3 complete!")
print(f"   Deploy-readiness analysis saved into metadata.json")