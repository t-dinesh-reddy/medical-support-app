```markdown
# AI Medical Support System

This project is an **AI-powered medical support system** that uses machine learning models (XGBoost/Logistic Regression) to provide **heart disease risk predictions** and explainability using **SHAP**. It includes:

- **Model training** (`train.py`)
- **FastAPI backend** (`api.py`, served with Uvicorn)
- **Streamlit frontend** (`app.py`)
- **Model persistence** using Joblib
- **SQLite logging** of predictions

---

## 📂 Project Structure

```

AI-Medical-Support/
│
├── train.py          # Train the ML model and save joblib artifacts
├── api.py            # FastAPI app exposing prediction API
├── app.py            # Streamlit frontend UI
├── run_api.py        # Helper script to run Uvicorn
├── db.py             # SQLite logging of predictions
├── models/           # Contains model.joblib and explainer.joblib
├── data/             # Dataset folder (heart.csv goes here)
└── requirements.txt  # Python dependencies

````

---

## ⚙️ Setup Instructions

### 1. Clone and Create Virtual Environment

```bash
git clone <repo-url>
cd AI-Medical-Support

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # (Windows PowerShell)
````

### 2. Install Dependencies

```bash
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If SHAP fails to install on Windows (compiler issues), use a prebuilt version:

```bash
.\.venv\Scripts\python.exe -m pip install shap>=0.45.1
```

---

## 📊 Dataset

This project uses the **UCI Heart Disease Dataset (Cleveland subset)**, a classic benchmark dataset for predicting the presence of heart disease.

* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
* **Records**: ~303 patients
* **Features**:

| Feature    | Description                                                           |
| ---------- | --------------------------------------------------------------------- |
| `age`      | Age of patient (years)                                                |
| `sex`      | Sex (1 = male, 0 = female)                                            |
| `cp`       | Chest pain type (0–3: typical, atypical, non-anginal, asymptomatic)   |
| `trestbps` | Resting blood pressure (mm Hg)                                        |
| `chol`     | Serum cholesterol (mg/dl)                                             |
| `fbs`      | Fasting blood sugar >120 mg/dl (1 = true, 0 = false)                  |
| `restecg`  | Resting ECG results (0–2: normal/abnormal patterns)                   |
| `thalach`  | Maximum heart rate achieved                                           |
| `exang`    | Exercise-induced angina (1 = yes, 0 = no)                             |
| `oldpeak`  | ST depression induced by exercise                                     |
| `slope`    | Slope of peak exercise ST segment (0 = up, 1 = flat, 2 = downsloping) |
| `ca`       | Number of major vessels (0–3) colored by fluoroscopy                  |
| `thal`     | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)     |
| **target** | 0 = No heart disease, 1 = Heart disease                               |

👉 Place the dataset CSV at:

```
data/heart.csv
```

---

## 🤖 Model

The training pipeline (`train.py`) builds and evaluates multiple models:

* **Logistic Regression** (baseline, interpretable)
* **XGBoost Classifier** (main model, tree-based boosting, best AUROC ≈ 0.89)

Processing includes:

* Median imputation for missing values
* Calibration (`CalibratedClassifierCV`) for well-calibrated probabilities
* Evaluation metrics: **AUROC** and **Brier Score**

Artifacts saved:

* `models/model.joblib` → best model + preprocessor + calibration
* `models/explainer.joblib` → optional SHAP explainer for interpretability

---

## 🏋️ Training the Model

Run training script (saves `models/model.joblib` and `models/explainer.joblib`):

```bash
.\.venv\Scripts\python.exe train.py
```

You should see output like:

```
AUROC: 0.87
Brier: 0.19
Saved: models/model.joblib
Saved: models/explainer.joblib
```

---

## 🚀 Running the API

Use helper script:

```bash
.\.venv\Scripts\python.exe run_api.py
```

Or run manually:

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Check in browser:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 💻 Running the Streamlit Frontend

In a new terminal (keep API running):

```bash
streamlit run app.py
```

Access UI at:
👉 [http://localhost:8501](http://localhost:8501)

---

## ✅ API Example (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{
    \"features\": {
      \"age\":54,\"sex\":1,\"cp\":0,\"trestbps\":130,
      \"chol\":246,\"fbs\":0,\"restecg\":1,
      \"thalach\":150,\"exang\":0,
      \"oldpeak\":1.0,\"slope\":1,\"ca\":0,\"thal\":2
    }
  }"
```

Response:

```json
{
  "prob": 0.82,
  "label": 1,
  "model_version": "v1.0",
  "top_factors": [
    {"feature":"thalach","impact":0.23},
    {"feature":"chol","impact":0.18}
  ]
}
```

---

## ⚠️ Notes

* Ensure `models/model.joblib` exists before starting API.
* If `EOFError` occurs when loading model → retrain (`train.py`).
* Avoid `--reload` flag on Windows (`uvicorn` sometimes aborts).
* If SHAP fails, predictions still work but explanations may be unavailable.

---

