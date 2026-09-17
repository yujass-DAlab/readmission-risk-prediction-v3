# ============================================================
# SHARED FEATURE EXTRACTION - V3
# Version: v3
# Authors: JasmineYu, Deepseek, ChatGPT
#
# ⚠️ DO NOT DUPLICATE THIS LOGIC.
# Training and API must import from this single source of truth.
#
# 📌 FEATURE COUNT: 25
# 📌 ICD-9 GROUPS: Project-defined categories (NOT official ICD-9 chapters)
# 📌 SCHEMA HASH: Detects feature-NAME/ORDER drift (NOT formula changes)
#
# 🔴 PREDICTION TIME POINT: AT DISCHARGE (see note below)
# ------------------------------------------------------------
# This extractor assumes all inputs are available AT THE TIME OF
# DISCHARGE. Specifically:
#   - LOS (length of stay): known at discharge
#   - medications administered: known at discharge
#   - labs/procedures: known at discharge
#   - prior-year utilization: known at discharge
# If the intended prediction time is EARLIER (admission, 24h, etc.),
# several features (LOS, meds, labs, procedures) become TARGET LEAKAGE
# and MUST be removed or replaced with point-in-time-safe variants.
# This is a project-definition decision, not a code issue.
# ============================================================

import re
import hashlib
import numpy as np
from typing import List, Optional, Tuple

# --- MODULE-LEVEL CONSTANTS ---
EXPECTED_FEATURE_COUNT = 25
EXTRACTOR_VERSION = "v3"
PREDICTION_TIME_POINT = "discharge"  # or "admission", "24h", etc.

FEATURE_NAMES = [
    "age", "num_meds", "num_labs", "num_diag", "out_vis", "em_vis", "in_vis", "num_proc", "LOS",
    "insulin_change", "diabetic_med_change", "any_diabetic_med",
    "admission_type", "gender", "race",
    "total_visits", "med_per_day", "high_utilization", "age_med_interaction", "acute_ratio", "icd9_group",
    "age_bin", "med_per_diag", "lab_med_ratio", "visit_intensity"
]
assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT, "FEATURE_NAMES length mismatch!"

FEATURE_SCHEMA_HASH = hashlib.sha256(",".join(FEATURE_NAMES).encode()).hexdigest()[:16]

DISCRETE_FEATURES = {
    "insulin_change", "diabetic_med_change", "any_diabetic_med",
    "admission_type", "gender", "race",
    "high_utilization", "icd9_group", "age_bin",
}

FEATURE_BOUNDS = {
    "age": (0.0, 150.0), "num_meds": (0.0, 100.0), "num_labs": (0.0, 500.0),
    "num_diag": (0.0, 50.0), "out_vis": (0.0, 100.0), "em_vis": (0.0, 100.0),
    "in_vis": (0.0, 100.0), "num_proc": (0.0, 50.0), "LOS": (0.0, 365.0),
    "insulin_change": (0.0, 1.0), "diabetic_med_change": (0.0, 1.0), "any_diabetic_med": (0.0, 1.0),
    "admission_type": (0.0, 4.0), "gender": (0.0, 2.0), "race": (0.0, 4.0),
    "total_visits": (0.0, 300.0), "med_per_day": (0.0, 100.0), "high_utilization": (0.0, 1.0),
    "age_med_interaction": (0.0, 15000.0), "acute_ratio": (0.0, 1.0), "icd9_group": (0.0, 17.0),
    "age_bin": (0.0, 4.0), "med_per_diag": (0.0, 100.0), "lab_med_ratio": (0.0, 500.0),
    "visit_intensity": (0.0, 100.0),
}


def validate_feature_ranges(features, warn: bool = True):
    """Clamp numeric features to FEATURE_BOUNDS; round DISCRETE_FEATURES to ints."""
    cleaned, issues = [], []
    for name, val in zip(FEATURE_NAMES, features):
        if val is None:
            cleaned.append(None); continue
        if name not in FEATURE_BOUNDS:
            cleaned.append(val); continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            issues.append(f"{name}={val!r} not numeric → None")
            cleaned.append(None); continue
        if name in DISCRETE_FEATURES:
            rounded = round(fval)
            if rounded != fval:
                issues.append(f"{name}={fval} is discrete → rounded to {rounded}")
            fval = float(rounded)
        lo, hi = FEATURE_BOUNDS[name]
        if fval < lo or fval > hi:
            issues.append(f"{name}={fval} → clamped to [{lo}, {hi}]")
            fval = max(lo, min(hi, fval))
        cleaned.append(fval)
    if warn and issues:
        print(f"⚠️ Feature range issues ({len(issues)}):")
        for issue in issues[:5]: print(f"   - {issue}")
        if len(issues) > 5: print(f"   ... and {len(issues) - 5} more")
    return cleaned, issues


def extract_number(pattern, text, default=None):
    match = re.search(pattern, text)
    if match:
        try: return float(match.group(1))
        except (ValueError, IndexError): return default
    return default


def extract_category(pattern, text, default="Unknown"):
    match = re.search(pattern, text)
    if match: return match.group(1).strip()
    return default


def map_icd9_to_group(icd: str) -> int:
    """
    Maps an ICD-9 code to a PROJECT-DEFINED clinical group (0–17).
    ⚠️ NOT the official ICD-9 chapter numbering. Group 17 = Other/Unknown/Missing.
    """
    try:
        code = int(str(icd).split('.')[0])
    except (ValueError, TypeError): return 17
    if code == 0: return 17
    if 1 <= code <= 139: return 0
    elif 140 <= code <= 239: return 1
    elif 240 <= code <= 279: return 2
    elif 280 <= code <= 289: return 3
    elif 290 <= code <= 319: return 4
    elif 320 <= code <= 389: return 5
    elif 390 <= code <= 459: return 6
    elif 460 <= code <= 519: return 7
    elif 520 <= code <= 579: return 8
    elif 580 <= code <= 629: return 9
    elif 630 <= code <= 679: return 10
    elif 680 <= code <= 709: return 11
    elif 710 <= code <= 739: return 12
    elif 740 <= code <= 759: return 13
    elif 760 <= code <= 779: return 14
    elif 780 <= code <= 799: return 15
    elif 800 <= code <= 999: return 16
    else: return 17


map_icd9_correct = map_icd9_to_group  # backward-compat alias


def extract_raw_features_from_text(text: str) -> List[Optional[float]]:
    """
    Returns exactly EXPECTED_FEATURE_COUNT (25) raw features from clinical text.
    Missing numeric fields → None. Engineered features PROPAGATE missingness:
    if a required input is None, the engineered output is also None
    (so the imputer fills it with the median, not a misleading 0).
    🔒 SINGLE SOURCE OF TRUTH — used by both training and inference.
    """
    # --- Numeric ---
    age = extract_number(r'Age:\s*\[(\d+)', text)
    num_meds = extract_number(r'Number of medications administered during the encounter:\s*(\d+)', text)
    num_labs = extract_number(r'Number of lab tests performed during the encounter:\s*(\d+)', text)
    num_diag = extract_number(r'Number of diagnosis:\s*(\d+)', text)
    out_vis = extract_number(r'Number of outpatient visits of the patient in the year preceding the encounter:\s*(\d+)', text)
    em_vis = extract_number(r'Number of emergency visits of the patient in the year preceding the encounter:\s*(\d+)', text)
    in_vis = extract_number(r'Number of inpatient visits of the patient in the year preceding the encounter:\s*(\d+)', text)
    num_proc = extract_number(r'Number of procedures \(other than lab tests\) performed during the encounter:\s*(\d+)', text)
    LOS = extract_number(r'Number of days between admission and discharge:\s*(\d+)', text)

    # --- Flags ---
    insulin = extract_category(r'insulin dosage change:\s*(\w+)', text, default="No")
    insulin_change = 1 if insulin in ["Up", "Down"] else 0
    diabetic_med = extract_category(r'Change in diabetic medication dosage:\s*(\w+)', text, default="No")
    diabetic_med_change = 1 if diabetic_med == "Ch" else 0
    any_med = extract_category(r'Any diabetic medicine prescribed:\s*(\w+)', text, default="No")
    any_diabetic_med = 1 if any_med == "Yes" else 0

    # --- Categoricals ---
    admission_type_raw = extract_category(r'admission type:\s*(\w+)', text, default="Unknown").title()
    gender_raw = extract_category(r'Gender:\s*(\w+)', text, default="Unknown")
    race_raw = extract_category(r'Race:\s*([^.]+)', text, default="Unknown").strip()
    race_raw = ''.join(race_raw.split())
    icd9_raw = extract_category(r'primary diagnosis \(coded as first three digits of ICD9\):\s*(\d+)', text, default="0")

    # --- Engineered (missingness-aware) ---
    # total_visits requires ALL THREE visit counts (else None — imputer handles)
    if out_vis is None or em_vis is None or in_vis is None:
        total_visits = None
    else:
        total_visits = out_vis + em_vis + in_vis

    # LOS_safe: fallback to 1 if missing or non-positive (Laplace smoothing)
    LOS_safe = LOS if (LOS is not None and LOS > 0) else 1

    # med_per_day: needs num_meds (LOS is smoothed via LOS_safe)
    med_per_day = None if num_meds is None else num_meds / (LOS_safe + 1)

    # high_utilization: needs total_visits
    high_utilization = None if total_visits is None else (1 if total_visits > 5 else 0)

    # age_med_interaction: needs age AND num_meds
    age_med_interaction = None if (age is None or num_meds is None) else age * num_meds

    # acute_ratio: needs total_visits (which already requires all three)
    acute_ratio = None if total_visits is None else (em_vis + in_vis) / (total_visits + 1)

    icd9_group = map_icd9_to_group(icd9_raw)

    # Age bins (uniform 20-year width): 0:<20, 1:20–39, 2:40–59, 3:60–79, 4:≥80
    age_bin = None if age is None else (
        0 if age < 20 else 1 if age < 40 else 2 if age < 60 else 3 if age < 80 else 4
    )

    # med_per_diag: needs num_meds AND num_diag
    med_per_diag = None if (num_meds is None or num_diag is None) else num_meds / (num_diag + 1)

    # lab_med_ratio: needs num_labs AND num_meds
    lab_med_ratio = None if (num_labs is None or num_meds is None) else num_labs / (num_meds + 1)

    # visit_intensity: needs total_visits (LOS is smoothed via LOS_safe)
    visit_intensity = None if total_visits is None else total_visits / (LOS_safe + 1)

    # --- Encodings ---
    admission_map = {'Elective': 0, 'Emergency': 1, 'Newborn': 2, 'Unknown': 3, 'Urgent': 4}
    gender_map = {'Female': 0, 'Male': 1, 'Unknown': 2}
    race_map = {'AfricanAmerican': 0, 'Asian': 1, 'Caucasian': 2, 'Hispanic': 3, 'Unknown': 4}
    admission_type = admission_map.get(admission_type_raw, 3)
    gender = gender_map.get(gender_raw, 2)
    race = race_map.get(race_raw, 4)

    return [
        age, num_meds, num_labs, num_diag, out_vis, em_vis, in_vis, num_proc, LOS,
        insulin_change, diabetic_med_change, any_diabetic_med,
        admission_type, gender, race,
        total_visits, med_per_day, high_utilization, age_med_interaction, acute_ratio, icd9_group,
        age_bin, med_per_diag, lab_med_ratio, visit_intensity
    ]