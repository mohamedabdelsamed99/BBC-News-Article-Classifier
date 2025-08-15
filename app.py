import streamlit as st
import joblib
import numpy as np

# Load saved model and vectorizer
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Category mapping (if your model outputs numbers)
categories = ["business", "entertainment", "politics", "sport", "tech"]

# Bigger title
st.markdown("<h1 style='font-size:40px;'>📑 BBC News Category Classifier</h1>", unsafe_allow_html=True)
st.write("Enter a news headline or article snippet, and I'll predict its category.")

# User input
user_input = st.text_area("✏️ Enter text here:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip():
        # Transform input text
        X_input = vectorizer.transform([user_input])

        # Predict category
        pred_class_raw = model.predict(X_input)[0]  # could be str or int

        # If prediction is an int, map to category name
        if isinstance(pred_class_raw, (np.integer, int)):
            pred_class = categories[pred_class_raw]
        else:
            pred_class = pred_class_raw  # already string

        pred_proba = model.predict_proba(X_input)[0]

        # Display main prediction in large text
        st.markdown(
            f"<h2 style='color:blue;'>Prediction: {pred_class}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3>Confidence: {np.max(pred_proba) * 100:.2f}%</h3>",
            unsafe_allow_html=True
        )

        # Show all class probabilities in a table
        st.subheader("📊 Class Probabilities")
        prob_dict = {cat: f"{prob * 100:.2f}%" for cat, prob in zip(categories, pred_proba)}
        st.table(prob_dict.items())

    else:
        st.warning("⚠️ Please enter some text before predicting.")
