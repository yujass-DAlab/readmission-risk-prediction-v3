# ============================================================
# READMISSION RISK API V3
# Version: v3
# Authors: JasmineYu, Deepseek, ChatGPT
#
# Loads model, imputer, metadata from S3 via SSM pointer.
# Validates feature schema hash on startup.
# Clamps inputs before inference. Refreshes threshold every 60s.
#
# NOTE: Interface explainability lives on the /predict endpoint.
# ============================================================

import json
import threading
from io import BytesIO
import time
import numpy as np
import joblib
import boto3
from botocore.exceptions import ClientError, ConnectTimeoutError
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict

from shared_feature_extraction import (
    extract_raw_features_from_text, validate_feature_ranges,
    EXPECTED_FEATURE_COUNT, FEATURE_SCHEMA_HASH, EXTRACTOR_VERSION,
)

# --- GLOBAL STATE ---
MODEL = None
IMPUTER_MEDIANS = None
METADATA = None
THRESHOLD = 0.25
FEATURE_COUNT = None
SSM_REGION = "us-east-2"

SSM_THRESHOLD_PATH = "/readmission/v3/active_threshold"
SSM_PREFIX_PATH = "/readmission/v3/active_model_prefix"

STALENESS_THRESHOLD_SECONDS = 300

LAST_REFRESH_TIME = None
LAST_REFRESH_ERROR = None
CLAMP_WARNINGS_TOTAL = 0


def _is_threshold_stale():
    if LAST_REFRESH_TIME is None:
        return True
    try:
        last = datetime.fromisoformat(LAST_REFRESH_TIME.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age > STALENESS_THRESHOLD_SECONDS
    except Exception:
        return True


def extract_features(text: str, medians: list) -> np.ndarray:
    """Extract -> clamp -> validate length -> impute."""
    global CLAMP_WARNINGS_TOTAL
    raw = extract_raw_features_from_text(text)
    clamped, issues = validate_feature_ranges(raw, warn=False)
    if issues:
        CLAMP_WARNINGS_TOTAL += len(issues)
        print(f"WARNING inference clamping: {issues}")

    if len(clamped) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Request-time feature count mismatch! "
            f"Extractor produced {len(clamped)}, expected {EXPECTED_FEATURE_COUNT}."
        )

    imputed = [float(medians[i]) if val is None else float(val)
               for i, val in enumerate(clamped)]
    return np.array([imputed], dtype=np.float32)


def refresh_threshold_periodically():
    global THRESHOLD, LAST_REFRESH_TIME, LAST_REFRESH_ERROR
    ssm = boto3.client('ssm', region_name=SSM_REGION)

    while True:
        try:
            response = ssm.get_parameter(Name=SSM_THRESHOLD_PATH)
            new_threshold = float(response['Parameter']['Value'])
            if not (0.0 <= new_threshold <= 1.0):
                raise ValueError(f"Threshold must be between 0 and 1. Received: {new_threshold}")
            if new_threshold != THRESHOLD:
                print(f"Threshold updated: {THRESHOLD:.4f} -> {new_threshold:.4f}")
                THRESHOLD = new_threshold
            LAST_REFRESH_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")            
            LAST_REFRESH_ERROR = None
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == 'AccessDeniedException':
                msg = "Permission denied for SSM GetParameter"
            elif code == 'ParameterNotFound':
                msg = f"Parameter '{SSM_THRESHOLD_PATH}' not found"
            elif code == 'ThrottlingException':
                msg = "Throttled by SSM"
            else:
                msg = f"AWS ClientError: {e}"
            LAST_REFRESH_ERROR = msg
            print(f"WARNING {msg}")
        except ValueError as e:
            LAST_REFRESH_ERROR = f"Invalid float in SSM threshold: {e}"
            print(f"WARNING {LAST_REFRESH_ERROR}")
        except ConnectTimeoutError:
            LAST_REFRESH_ERROR = "SSM connection timeout"
            print(f"WARNING {LAST_REFRESH_ERROR}")
        except Exception as e:
            LAST_REFRESH_ERROR = f"Unexpected refresh error: {e}"
            print(f"WARNING {LAST_REFRESH_ERROR}")
        time.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, IMPUTER_MEDIANS, THRESHOLD, METADATA, FEATURE_COUNT, LAST_REFRESH_TIME, LAST_REFRESH_ERROR
    ssm = boto3.client('ssm', region_name=SSM_REGION)
    s3 = boto3.client('s3', region_name=SSM_REGION)

    try:
        resp = ssm.get_parameter(Name=SSM_PREFIX_PATH)
        s3_prefix = resp['Parameter']['Value']
        bucket = s3_prefix.split('/')[2]
        prefix = '/'.join(s3_prefix.split('/')[3:])

        model_buffer = BytesIO()
        s3.download_fileobj(bucket, f"{prefix}model.pkl", model_buffer)
        model_buffer.seek(0)
        MODEL = joblib.load(model_buffer)
        print(f"Model loaded from {s3_prefix}")

        imputer_buffer = BytesIO()
        s3.download_fileobj(bucket, f"{prefix}imputer.json", imputer_buffer)
        imputer_buffer.seek(0)
        IMPUTER_MEDIANS = json.load(imputer_buffer)['medians']
        print(f"Imputer loaded ({len(IMPUTER_MEDIANS)} medians).")

        metadata_buffer = BytesIO()
        try:
            s3.download_fileobj(bucket, f"{prefix}metadata.json", metadata_buffer)
            metadata_buffer.seek(0)
            METADATA = json.load(metadata_buffer)
            print(f"Metadata loaded. Version: {METADATA.get('version', '?')}, "
                  f"Extractor: {METADATA.get('extractor_version', '?')}")
        except s3.exceptions.NoSuchKey:
            print("WARNING metadata.json not found in S3.")
            METADATA = None

        FEATURE_COUNT = len(IMPUTER_MEDIANS)
        print(f"Feature count derived: {FEATURE_COUNT}")
        if FEATURE_COUNT != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch! Imputer has {FEATURE_COUNT}, "
                f"shared extractor produces {EXPECTED_FEATURE_COUNT}."
            )

        if METADATA is not None:
            stored_hash = METADATA.get("feature_schema_hash")
            if stored_hash and stored_hash != FEATURE_SCHEMA_HASH:
                raise ValueError(
                    f"Feature schema mismatch! Model hash: {stored_hash}, "
                    f"current code hash: {FEATURE_SCHEMA_HASH}. Retrain or restore extractor."
                )
            print(f"Schema hash validated: {FEATURE_SCHEMA_HASH}")

        resp = ssm.get_parameter(Name=SSM_THRESHOLD_PATH)
        THRESHOLD = float(resp['Parameter']['Value'])
        
        LAST_REFRESH_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        LAST_REFRESH_ERROR = None
        print(f"Initial Threshold: {THRESHOLD:.4f}")

        thread = threading.Thread(target=refresh_threshold_periodically, daemon=True)
        thread.start()
        print("Background refresher started.")

    except Exception as e:
        print(f"Lifespan load failed: {e}")
        raise e

    yield
    print("Shutting down.")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Readmission V3 - Dynamic Threshold",
    version="v3",
    lifespan=lifespan,
    description=(
        "**WARNING:** This is a **SCREENING tool**, not a diagnostic classifier. "
        "`risk_class` reflects the category AFTER threshold override and may "
        "differ from the highest probability. See the POST /predict section "
        "below for the full interpretation guide and one-click test cases.\n\n"
        "**NOTE ABOUT THE '422 VALIDATION ERROR' IN THE RESPONSES PANEL:** "
        "Swagger UI automatically lists 422 as a **possible** response for every "
        "endpoint that accepts a request body. It does **not** mean your request "
        "failed. Your actual result is shown in the **'Server response'** area "
        "right below the Execute button. A green 200 means success; a red 422 "
        "there means the request body was malformed."
    ),
)


# --- SCHEMAS ---
class PatientContext(BaseModel):
    patient_id: str = Field(
        description="Unique identifier for the patient encounter (echoed back)."
    )
    clinical_text: str = Field(
        description=(
            "Free-text clinical context. Should include: Age, Gender, admission "
            "type, LOS, medications, labs, procedures, prior utilization, and "
            "primary diagnosis (ICD-9). Use one of the preloaded examples above, "
            "then REPLACE its values with your actual patient data."
        )
    )


class PredictionResponse(BaseModel):
    patient_id: str = Field(description="Echoed from the request.")
    risk_class: str = Field(
        description=(
            "Operational screening category AFTER threshold override. "
            "May NOT match the highest-probability class. "
            "One of: 'NO', '>30 days', '<30 days'."
        )
    )
    probabilities: Dict[str, float] = Field(
        description="Raw model probability for each class. Sum = 1.0."
    )
    risk_alert: bool = Field(
        description=(
            "The actionable field. true means <30 days probability exceeded "
            "the active threshold, so clinical review is recommended."
        )
    )


# --- PREDICTION ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(
    patient: PatientContext = Body(
        ...,
        openapi_examples={
            "low_risk": {
                "summary": "LOW RISK - young, elective, no prior acute care",
                "description": "20-30yo, elective, 1-day LOS, zero prior ER/inpatient visits.",
                "value": {
                    "patient_id": "TEST-LOW-001",
                    "clinical_text": (
                        "Patient's context Race: Caucasian. Gender: Male. Age: [20-30). "
                        "admission type: Elective. Number of days between admission and discharge: 1. "
                        "Number of lab tests performed during the encounter: 5. "
                        "Number of procedures (other than lab tests) performed during the encounter: 0. "
                        "Number of medications administered during the encounter: 2. "
                        "Number of outpatient visits of the patient in the year preceding the encounter: 0. "
                        "Number of emergency visits of the patient in the year preceding the encounter: 0. "
                        "Number of inpatient visits of the patient in the year preceding the encounter: 0. "
                        "primary diagnosis (coded as first three digits of ICD9): 250. "
                        "Number of diagnosis: 2. insulin dosage change: No. "
                        "Change in diabetic medication dosage: No. Any diabetic medicine prescribed: No"
                    ),
                },
            },
            "moderate_risk": {
                "summary": "MODERATE RISK - older, emergency, chronic condition",
                "description": "60-70yo, emergency, 4-day LOS, 2 prior acute visits, heart failure ICD-9.",
                "value": {
                    "patient_id": "TEST-MOD-003",
                    "clinical_text": (
                        "Patient's context Race: Caucasian. Gender: Female. Age: [60-70). "
                        "admission type: Emergency. Number of days between admission and discharge: 4. "
                        "Number of lab tests performed during the encounter: 30. "
                        "Number of procedures (other than lab tests) performed during the encounter: 1. "
                        "Number of medications administered during the encounter: 12. "
                        "Number of outpatient visits of the patient in the year preceding the encounter: 2. "
                        "Number of emergency visits of the patient in the year preceding the encounter: 1. "
                        "Number of inpatient visits of the patient in the year preceding the encounter: 1. "
                        "primary diagnosis (coded as first three digits of ICD9): 428. "
                        "Number of diagnosis: 5. insulin dosage change: No. "
                        "Change in diabetic medication dosage: No. Any diabetic medicine prescribed: Yes"
                    ),
                },
            },
            "high_risk": {
                "summary": "HIGH RISK - elderly, emergency, heavy prior utilization",
                "description": "80-90yo, emergency, 10-day LOS, 10 prior acute visits, unstable meds.",
                "value": {
                    "patient_id": "TEST-VERYHIGH-005",
                    "clinical_text": (
                        "Patient's context Race: African American. Gender: Female. Age: [80-90). "
                        "admission type: Emergency. Number of days between admission and discharge: 10. "
                        "Number of lab tests performed during the encounter: 80. "
                        "Number of procedures (other than lab tests) performed during the encounter: 5. "
                        "Number of medications administered during the encounter: 30. "
                        "Number of outpatient visits of the patient in the year preceding the encounter: 5. "
                        "Number of emergency visits of the patient in the year preceding the encounter: 6. "
                        "Number of inpatient visits of the patient in the year preceding the encounter: 4. "
                        "primary diagnosis (coded as first three digits of ICD9): 428. "
                        "Number of diagnosis: 12. insulin dosage change: Up. "
                        "Change in diabetic medication dosage: Ch. Any diabetic medicine prescribed: Yes"
                    ),
                },
            },
        },
    )
):
    """
    **Score a patient's 30-day readmission risk:**

    ### HOW TO USE FOR REAL PREDICTION

    1. Click **"Try it out"** (top-right of this section).
    2. In the **Request body**, **REPLACE** the example values with your
       actual patient's data:
       - `patient_id`: your patient / encounter ID
       - `clinical_text`: the patient's clinical context — including Age,
         Gender, admission type, LOS, medications, labs, procedures, prior
         utilization, and primary diagnosis (ICD-9)
    3. Click **Execute**.
    4. Read the **"Server response"** area below the Execute button
       (**NOT** the "Responses" list at the very bottom).
    5. Interpret the result:
       - `risk_alert: true` → **recommend clinical review**
       - `risk_alert: false` → no screening action needed

    The three preloaded examples (LOW / MODERATE / HIGH risk) are for
    **demonstration only**. Replace their values with real patient data
    to make an actual prediction.

    ---

    ### WHY IS THERE A "422 VALIDATION ERROR" AT THE BOTTOM?

    That section at the bottom of this page lists **ALL POSSIBLE** response
    formats this endpoint knows how to produce — it is **documentation**,
    not your result. FastAPI automatically includes 422 (validation error)
    for any endpoint that accepts a request body. It does **not** mean your
    request failed.

    Your **actual** result appears in the **"Server response"** area right
    below the Execute button. Look for a green **200** (success) or a red
    **422** (validation failed) **there**.

    ---

    ### HOW TO READ THE RESPONSE

    - `risk_class`: Operational category **after** threshold. May differ
      from the highest-probability class.
    - `probabilities`: Raw model output; sum = 1.0.
    - `risk_alert`: The **actionable** flag. `true` → recommend clinical review.

    ---

    ### WHY `risk_class` MAY NOT MATCH THE TOP PROBABILITY

    The screening threshold is set **low (~0.07)** to catch ~85% of true
    `<30 days` cases. This means many patients are flagged even when
    `>30 days` has a higher probability.

    Example:
    - probabilities: `{NO: 0.16, >30 days: 0.70, <30 days: 0.14}`
    - threshold: `0.07`
    - result: `risk_class = '<30 days'` (because 0.14 > 0.07), `risk_alert = true`

    This is **expected behavior** — the accepted cost of high recall.

    ---

    ### EXPECTED PERFORMANCE (held-out test data)

    | Metric | Value |
    | :--- | :--- |
    | Recall (`<30 days`) | ~85% |
    | Precision (`<30 days`) | ~12% |
    | Alert rate | ~77% of encounters |

    Out of 100 flagged patients: ~12 are true positives, ~88 are false alarms.
    This is **intentional** — the design favors *not missing* high-risk
    patients over minimizing false alerts.

    ---

    ### CLINICAL INTERPRETATION

    - `risk_alert: true` → Review patient for early intervention.
    - `risk_alert: false` → No screening action needed.
    - `risk_class` differs from top probability → Expected (threshold promotion).

    ---

    ### PREDICTION TIME NOTE

    This model predicts at **DISCHARGE**. It uses information available by
    the end of the encounter (LOS, meds, labs, procedures). It is **NOT**
    designed for admission-time prediction.
    """
    try:
        current_threshold = THRESHOLD
        features = extract_features(patient.clinical_text, IMPUTER_MEDIANS)
        probs = MODEL.predict_proba(features)[0]

        # Operational category (not raw argmax): threshold deliberately
        # promotes patients to '<30 days' to achieve high recall.
        pred_idx = 2 if probs[2] > current_threshold else int(np.argmax(probs))

        class_map = {0: "NO", 1: ">30 days", 2: "<30 days"}
        return PredictionResponse(
            patient_id=patient.patient_id,
            risk_class=class_map[pred_idx],
            probabilities={
                "NO": round(float(probs[0]), 4),
                ">30 days": round(float(probs[1]), 4),
                "<30 days": round(float(probs[2]), 4)
            },
            risk_alert=bool(probs[2] > current_threshold)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Input parsing error: {str(e)}")
    except AttributeError as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    """Operational health. 'degraded' = threshold refresher stalled > 5 min."""
    if MODEL is None:
        return {"status": "loading"}

    stale = _is_threshold_stale()
    status = "degraded" if stale else "healthy"

    return {
        "status": status,
        "version": METADATA.get('version', 'unknown') if METADATA else "unknown",
        "extractor_version": METADATA.get('extractor_version', 'unknown') if METADATA else "unknown",
        "prediction_time_point": METADATA.get('prediction_time_point', 'unknown') if METADATA else "unknown",
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "features": FEATURE_COUNT,
        "current_threshold": THRESHOLD,
        "threshold_is_stale": stale,
        "last_threshold_refresh": LAST_REFRESH_TIME,
        "last_refresh_error": LAST_REFRESH_ERROR,
        "clamp_warnings_total": CLAMP_WARNINGS_TOTAL
    }


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    """Brief overview. Full guidance is on /docs under POST /predict."""
    return {
        "message": "Readmission Risk V3 API is running.",
        "docs": "/docs",
        "health": "/health",
        "note": (
            "SCREENING tool, not a diagnostic classifier. risk_class reflects "
            "category after threshold override and may not match the top "
            "probability. See /docs -> POST /predict for the full interpretation "
            "guide and one-click test cases."
        )
    }