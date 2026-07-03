import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="MediLens AI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_diabetes_models():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

@st.cache_resource
def load_nlp_models():
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    clf = joblib.load("models/note_classifier.pkl")
    return vec, clf

@st.cache_resource
def load_xray_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model("models/xray_model.keras")
    except Exception as e:
        return None

st.title("🩺 MediLens AI")
st.caption("AI Diagnostic & Insight System")
st.warning("⚠️ This is a prototype only. Not a substitute for medical advice.")

tab1, tab2, tab3 = st.tabs([
    "🩸 Diabetes Risk", "🫁 X-Ray Screening", "📝 Patient Notes"
])

# TAB 1
with tab1:
    st.subheader("Diabetes Risk Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        glucose = st.number_input("Glucose", 0, 300, 120)
    with col2:
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    with col3:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
    with col4:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.4)
        age = st.number_input("Age", 1, 120, 35)

    if st.button("Predict", type="primary"):
        try:
            model, scaler, feature_names = load_diabetes_models()
            input_df = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }])[feature_names]

            proba = model.predict_proba(input_df)[0, 1]

            if proba >= 0.5:
                st.error(f"High Risk — {proba:.1%} probability of Diabetes")
            else:
                st.success(f"Low Risk — {proba:.1%} probability of Diabetes")

            st.progress(float(proba))

            st.markdown("**Why this prediction? (SHAP)**")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            sv = shap_vals[0]
            contrib = pd.Series(np.array(sv).flatten(), index=feature_names)
            top3 = contrib.abs().sort_values(ascending=False).head(3)
            colors = ["red" if contrib[f] > 0 else "blue" for f in top3.index]
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(top3.index[::-1], contrib[top3.index][::-1], color=colors[::-1])
            ax.set_xlabel("SHAP value")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# TAB 2
with tab2:
    st.subheader("Chest X-Ray Pneumonia Screening")
    uploaded_img = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        from PIL import Image
        xray_model = load_xray_model()

        if xray_model is None:
            st.error("TensorFlow not available in this environment.")
        else:
            img = Image.open(uploaded_img).convert("RGB").resize((160, 160))
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Uploaded X-ray")
            arr = np.expand_dims(np.array(img), axis=0).astype("float32")
            proba = float(xray_model.predict(arr, verbose=0).flatten()[0])
            pred_label = "PNEUMONIA" if proba > 0.5 else "NORMAL"
            confidence = proba if proba > 0.5 else 1 - proba
            with col2:
                if pred_label == "PNEUMONIA":
                    st.error(f"Prediction: Pneumonia ({confidence:.1%} confidence)")
                else:
                    st.success(f"Prediction: Normal ({confidence:.1%} confidence)")

# TAB 3
with tab3:
    st.subheader("Patient Notes Categorizer")
    try:
        vec, note_clf = load_nlp_models()
        note_text = st.text_area(
            "Enter patient note:",
            "Patient reports chest tightness and shortness of breath."
        )
        if st.button("Categorize Note"):
            X = vec.transform([note_text])
            pred_category = note_clf.predict(X)[0]
            proba_arr = note_clf.predict_proba(X)[0]
            proba_df = pd.DataFrame({
                "Category": note_clf.classes_,
                "Confidence": proba_arr
            }).sort_values("Confidence", ascending=False)
            st.success(f"Predicted Category: {pred_category}")
            st.bar_chart(proba_df.set_index("Category"))
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("MediLens AI — Built for AI Internship Major Project. Not for clinical use.")
