# AI-Based IP Evidence Classification Model

## 1. Project Overview

This folder contains the trained machine learning model developed for the CIP-B107 AI Forensics Capstone and Intelligence Analysis project.

The model uses character-level TF-IDF feature extraction and Logistic Regression to classify text evidence as either positive or negative for IP address identification.

The objective is to support the identification and classification of potential IP address evidence in digital forensic investigations.

## 2. Model Architecture

The model uses a Scikit-learn pipeline consisting of:

- **Feature extraction:** TF-IDF vectorization using character-level n-grams (1-4 characters).
- **Classification:** Logistic Regression.
- **Class balancing:** Balanced class weights.
- **Maximum iterations:** 2,000.
- **Random state:** 42 (as configured in the training pipeline).

The trained pipeline includes both the vectorizer and classifier.

## 3. Repository Structure

```text
AI_Model/
├── model/
│   └── ip_evidence_model.joblib
├── source/
│   ├── lab01_pipeline.py
│   └── requirements_frozen.txt
├── results/
│   ├── metrics_summary.csv
│   ├── confusion_matrices.png
│   └── environment.txt
└── README.md
```

## 4. Model Evaluation

The model was evaluated using a test set of 46 records.

| Metric | Result |
|---|---:|
| Accuracy | 97.83% |
| Precision | 97.14% |
| Recall | 100% |
| F1-score | 98.55% |
| AUROC | 0.9902 |
| AUPRC | 0.9965 |

The confusion matrix recorded 34 true positives, 1 false positive, 11 true negatives and 0 false negatives.

## 5. Important Evaluation Limitation

The training pipeline identified that 30 out of the 46 test prompts were exact duplicates of prompts in the training set.

This overlap creates a potential data leakage problem and may inflate the reported evaluation results.

Consequently, these results should be interpreted as preliminary test performance rather than independent evidence of generalisation.

Further evaluation using deduplicated, independent training and test data is necessary.

## 6. Installation and Model Loading

Install the required dependencies:

```bash
pip install -r AI_Model/source/requirements_frozen.txt
```

Load the trained model using Python:

```python
import joblib

model = joblib.load("AI_Model/model/ip_evidence_model.joblib")

sample = ["Enter text evidence to classify here"]

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", prediction[0])
print("Class probabilities:", probability[0])
```

The model returns the predicted class and the associated class probabilities. The class mapping follows the labels used during training.

## 7. Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- TF-IDF Vectorization
- Logistic Regression

## 8. Reproducibility and Evidence

The following supporting files are included:

- `lab01_pipeline.py`: Model training and evaluation pipeline.
- `requirements_frozen.txt`: Recorded Python dependencies.
- `metrics_summary.csv`: Evaluation metrics.
- `confusion_matrices.png`: Confusion matrix visualisation.
- `environment.txt`: Recorded execution environment.

## 9. Limitations

- Performance depends on the quality and representativeness of the training data.
- Duplicate prompts between training and testing may affect evaluation reliability.
- Predictions should be reviewed by a human forensic analyst.
- A positive classification does not independently establish that an IP address is genuine, malicious, or attributable to a particular person or device.

## 10. Project Context

Developed as part of the CIP-B107 AI Forensics Capstone and Intelligence Analysis laboratory project.
