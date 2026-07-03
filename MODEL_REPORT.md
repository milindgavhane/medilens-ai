# MediLens AI — Model Report
**Author:** Milind Gavhane

## 1. Models & Accuracy

| Step | Model | Accuracy | F1-Score |
|---|---|---|---|
| Step 2 | Logistic Regression | 71% | 0.57 |
| Step 2 | Random Forest | 86% | 0.80 |
| Step 2 | XGBoost (Best) | 89% | 0.84 |
| Step 3 | Scratch CNN | 76% | - |
| Step 3 | MobileNet Transfer Learning | 81.57% | - |

## 2. Why XGBoost?
XGBoost performed best with 89% accuracy and F1-Score of 0.844.
PCA was tested but reduced F1 from 0.844 to 0.752 with no speed benefit.

## 3. Ethics & Bias Considerations
1. **Population Bias:** Dataset contains only female Pima Indian patients age 21+. Model should not be used for other populations.
2. **Missing Data:** Zeros were used as missing values — fixed using median imputation.
3. **Black Box Problem:** Solved using SHAP explanations for every prediction.
4. **Decision Threshold:** Default 0.5 used — real deployment needs clinical validation.
5. **Not a Diagnosis:** Model estimates risk only — not a substitute for doctor's advice.

## 4. LSTM Note
Dataset used (Pima Indians Diabetes) does not contain time-series data.
Each patient has a single recorded observation, so LSTM was not applicable.

## 5. Libraries Used
- NumPy, Pandas, Scikit-Learn, XGBoost
- TensorFlow (CNN + MobileNet)
- HuggingFace Transformers (GPT2)
- SHAP (Explainability)
- Streamlit (Deployment)