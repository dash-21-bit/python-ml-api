#  Python ML API — Single-File End-to-End Project

**Author:** Adarsh Ravi  
**Tech:** Python · FastAPI · Scikit-Learn · Docker  
**Project Type:** ML Model Deployment (Production-Style)

---

##  Overview

This project demonstrates a **complete machine-learning deployment workflow**:

- Dataset creation
- Model training & evaluation
- REST API using FastAPI
- Dockerised deployment
- Live predictions via HTTP

Everything required to understand and run this project is documented **in this single README file**.

---

##  What This Project Proves

✔ End-to-end ML workflow  
✔ API engineering with FastAPI  
✔ Model persistence (Joblib)  
✔ Evaluation with metrics & plots  
✔ Docker containerisation  
✔ Tested via curl & Swagger UI  

This is **deployment-focused ML**, not just experimentation.

---

##  Problem Statement

Many ML projects stop at training a model.  
Real-world systems require:

- reproducibility  
- deployable APIs  
- consistent environments  
- reliable inference  

This project solves that by packaging **data, model, evaluation, and API** into a **single Docker-ready service**.

---

##  Dataset

- **Source:** Scikit-learn Breast Cancer dataset
- **Type:** Binary classification
- **Samples:** 569
- **Features:** 30 numerical features
- **Target:** `0 = malignant`, `1 = benign`

The dataset is exported to CSV to ensure reproducibility.

---

##  Model

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler
- **Implementation:** Scikit-learn Pipeline
- **Saved As:** `.joblib`

Using a pipeline ensures **identical preprocessing during training and inference**.

---

##  Evaluation

Generated artifacts:

- Accuracy
- ROC-AUC
- Confusion Matrix (PNG)
- ROC Curve (PNG)
- Metrics CSV

These prove the model was evaluated properly before deployment.

---

##  API Endpoints

### Health Check

GET/ Health

Response:
```json
##{"status":"ok"}

Model Info

{"model_type":"Pipeline"}
Prediction
POST /predict

Request 
{
  "features": {
    "mean radius": 17.99,
    "mean texture": 10.38
  }
}
response:
{
  "prediction": 0,
  "probability_class_1": 0.0123
}

Dockerfile

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install -e .

RUN python scripts/make_dataset.py \
 && python scripts/train_model.py \
 && python scripts/evaluate_model.py

EXPOSE 8000
CMD ["uvicorn", "ml_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build Image 
docker build -t python-ml-api .
Run container
docker run --rm -p 8000:8000 python-ml-api
Test 
curl http://127.0.0.1:8000/health
Swagger UI

Example prediction 
python - <<'EOF' > payload.json
import pandas as pd, json
df = pd.read_csv("data/raw/breast_cancer.csv")
row = df.drop(columns=["target"]).iloc[0].to_dict()
print(json.dumps({"features": row}))
EOF

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @payload.json

O/p
{"prediction":0,"probability_class_1":0.0123}


 Design Decisions
	•	FastAPI: automatic validation & docs
	•	Docker: environment consistency
	•	Pipeline: avoids training/inference mismatch
	•	Joblib: fast model loading
	•	CSV & PNG outputs: portable evaluation

⸻

Limitations
	•	Single baseline model
	•	No authentication
	•	No batch inference
	•	No CI/CD pipeline

⸻

 Future Improvements
	•	Add Random Forest / XGBoost
	•	Batch prediction endpoint
	•	Model versioning
	•	Docker Compose
	•	Cloud deployment
	•	Automated tests##

