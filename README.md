<img width="799" height="908" alt="readmission_v3UI3" src="https://github.com/user-attachments/assets/d8ca26f2-7bd6-4137-9a05-7d8d5ec0ec9c" />
<img width="540" height="702" alt="readmission_v3UI4" src="https://github.com/user-attachments/assets/ca2d76c0-49e3-4c77-ad61-601be8fee6cb" />
<img width="1399" height="717" alt="V3-AWS-Deployment" src="https://github.com/user-attachments/assets/0ac2718b-5480-456e-9301-48cb47c1221a" />
# Readmission Risk Screening API — V3

URL: http://18.222.170.48:8000/docs

An end-to-end **healthcare ML engineering project** that predicts hospital readmission risk and exposes the model through a cloud-deployed FastAPI service.

> ⚠️ **Screening/portfolio project — not a diagnostic or clinical decision-support tool.**
> Do not use for actual medical decisions.

## 🎯 What It Does

The API accepts patient encounter information—including age, admission type, length of stay, medications, labs, procedures, prior utilization, and ICD-9 information—and returns:

* **`risk_class`** — operational category: `NO`, `>30 days`, or `<30 days`
* **`probabilities`** — model probabilities for all three classes
* **`risk_alert`** — screening flag recommending clinical review

The production threshold is externally configurable through **AWS Systems Manager Parameter Store**, allowing the operating point to be changed without redeploying the application.

---

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │      AWS S3           │
                 │                      │
                 │  model.pkl           │
                 │  imputer.json        │
                 │  metadata.json       │
                 └──────────▲───────────┘
                            │
                            │ artifacts
                            │
┌────────────────┐   ┌──────┴───────────┐
│ AWS SSM        │──▶│   AWS EC2        │
│ Parameter      │   │   FastAPI        │
│ Store          │   │   Uvicorn        │
│                │   │                  │
│ Dynamic        │   │   /predict       │
│ threshold      │   │   /health        │
└────────────────┘   │   /docs          │
                     └──────────────────┘
```

### Key Engineering Decisions

| Design                         | Purpose                                                                  |
| ------------------------------ | ------------------------------------------------------------------------ |
| **Stacking ensemble**          | XGBoost + LightGBM + Random Forest with Logistic Regression meta-learner |
| **SMOTE on training data**     | Addresses class imbalance while keeping validation/test data untouched   |
| **Validation-based threshold** | Separates operating-point selection from final test evaluation           |
| **Dynamic SSM threshold**      | Allows runtime threshold updates without model redeployment              |
| **Schema hash validation**     | Detects feature-order/schema drift between training and inference        |
| **IAM role**                   | Avoids storing AWS credentials on the EC2 instance                       |
| **Systemd service**            | Keeps the API running as a managed Linux service                         |
| **Health endpoint**            | Reports model/schema/threshold operational status                        |

---

## 📊 Model Performance

The model was trained on **91,589 encounters** from the UCI Diabetes 130-US Hospitals dataset.

**Split:** 60% train / 20% validation / 20% test

| Metric                 | V3 Result |
| ---------------------- | --------: |
| `<30 days` recall      |      ~85% |
| `<30 days` precision   |      ~12% |
| Alert rate             |      ~77% |
| `<30 days` PR-AUC      |     ~0.16 |
| Binary variant ROC-AUC |     ~0.67 |

The operating point intentionally emphasizes recall. At approximately 85% recall, the model produces a substantial number of screening alerts.

**Important:** These metrics are from a historical, diabetes-specific dataset and have not been externally or clinically validated.

---

## ⚠️ Interpretation

`risk_class` represents the **operational category after threshold processing** and may differ from the class with the highest raw probability.

For example:

```text
Probabilities:
NO         = 0.16
>30 days   = 0.70
<30 days   = 0.14

Threshold = 0.07

Result:
risk_class = "<30 days"
risk_alert = true
```

This occurs because the `<30 days` probability exceeds the configured screening threshold.

The Swagger `/docs` interface provides additional examples and interpretation guidance.

---

## 🧪 API Endpoints

| Method | Endpoint   | Purpose                                 |
| ------ | ---------- | --------------------------------------- |
| `POST` | `/predict` | Score a patient encounter               |
| `GET`  | `/health`  | Operational health and threshold status |
| `GET`  | `/`        | API root                                |
| `GET`  | `/docs`    | Interactive Swagger documentation       |

### Example

```bash
uvicorn app_v3:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface includes preloaded LOW / MODERATE / HIGH demonstration cases.

---

## 🖥️ Screenshots

![Readmission V3 Swagger overview](screenshots/readmission-v3-overview.png)


![Readmission V3 prediction endpoint](screenshots/readmission-v3-predict.png)


![Readmission V3 health endpoint](screenshots/readmission-v3-health.png)


---

## 🔬 Known Limitations

These limitations are intentionally documented because they affect how the results should be interpreted.

1. **Historical dataset**
   UCI Diabetes 130-US Hospitals represents encounters from 1999–2008 and is diabetes-specific. Generalizability to other populations has not been established.

2. **Random encounter-level split**
   The current V3 split is not patient-level or temporal. Multiple encounters from the same patient may therefore appear across splits. V4 would use patient-level grouping and/or temporal validation.

3. **SMOTE with encoded categoricals**
   SMOTE can interpolate integer-encoded categorical variables, potentially producing semantically invalid intermediate values.

4. **Discharge-time prediction**
   V3 uses information available at discharge, including LOS, medications, labs, and procedures. It is **not an admission-time prediction model**.

5. **Process-local dynamic threshold**
   Each Uvicorn worker maintains its own threshold. A shared configuration/cache would be preferable for strict multi-worker consistency.

6. **No external clinical validation**
   This is an ML engineering and portfolio prototype. Performance should not be interpreted as evidence of clinical effectiveness.

---

## 📁 Project Structure

```text
readmission_api/
│
├── app_v3.py
├── train_pipeline_v3.py
├── shared_feature_extraction.py
│
├── model.pkl
├── imputer.json
├── metadata.json
│
├── requirements.txt
├── README.md
│
├── sample_requests/
│   ├── low_risk.json
│   ├── moderate_risk.json
│   └── high_risk.json
│
└── screenshots/
    ├── readmission-v3-overview.png
    ├── readmission-v3-predict.png
    └── readmission-v3-health.png
```

---

## 🛠️ Tech Stack

**Modeling**

* Python
* scikit-learn
* XGBoost
* LightGBM
* imbalanced-learn / SMOTE

**API**

* FastAPI
* Uvicorn
* Pydantic

**AWS**

* EC2
* S3
* Systems Manager Parameter Store
* IAM
* AWS Budgets

**Operations**

* systemd
* schema hashing
* dynamic configuration
* health monitoring

---

## 📖 Dataset

**Source:** UCI Diabetes 130-US Hospitals for Years 1999–2008

* 91,589 encounters
* 54% — no readmission
* 35% — readmission after 30 days
* 11% — readmission within 30 days

The dataset is historical and diabetes-specific, so the model's transportability to a broader hospital population remains an open question.

---

## 🚀 Deployment

V3 is designed for AWS deployment using:

```text
EC2
 ├── FastAPI / Uvicorn
 ├── IAM role
 └── systemd

S3
 ├── model.pkl
 ├── imputer.json
 └── metadata.json

SSM Parameter Store
 ├── active_model_prefix
 └── active_threshold
```

The deployment workflow is:

```text
Train
  ↓
Validate
  ↓
Select threshold
  ↓
Save artifacts
  ↓
Upload to S3
  ↓
Configure SSM
  ↓
Deploy FastAPI to EC2
  ↓
Health check
  ↓
Prediction tests
```

---

## 📄 License

MIT

This project is provided as a **demo/learning project** and is not intended for clinical use.

---

## 🙏 Acknowledgments

Built as a solo learning project to explore **healthcare ML engineering from training through cloud deployment**, including model evaluation, feature consistency, artifact management, dynamic configuration, API serving, and honest documentation of clinical trade-offs.
