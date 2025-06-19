import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    h1 {
        color: black !important;
    }

    /* General text overrides */
    .css-1d391kg, .css-1cpxqw2 {
        color: white !important;
    }

    .stSelectbox label, .stNumberInput label {
        color: white !important;
    }

    /* Make Predict button text black */
    .stButton > button {
        color: black !important;
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.9); /* Optional: light background */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Try importing XGBoost safely
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Load encoders and models
log_model = joblib.load("logistic_model.pkl")
le_region = joblib.load("le_region.pkl")
le_dam = joblib.load("le_dam.pkl")

xgb_model = None
if xgb_available:
    xgb_model = joblib.load("xgboost_flood_model.pkl")

# Title
st.title("🌊 Flood Forecasting App")
st.write("Select a model and input conditions to predict flood probability.")

# Model selection
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "XGBoost" if xgb_available else "XGBoost (Not Available)"])

# User Inputs
a = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
b = st.number_input("River Level (meters)", min_value=0.0, step=0.1)
c = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=1.0)
d = st.number_input("Temperature (°C)", min_value=-10.0, step=0.5)
region = st.selectbox("Region", le_region.classes_)
dam = st.selectbox("Dam Type", le_dam.classes_)

# Encode categorical features
e = le_region.transform([region])[0]
f = le_dam.transform([dam])[0]

if st.button("🔍 Predict"):
        model = log_model if model_choice == "Logistic Regression" else xgb_model
        if model is None:
            st.warning("❌ XGBoost is not available. Install it with: `pip install xgboost`.")
        else:
            input_data = np.array([[a, b, c, d, e, f]])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            if prediction == 1:
                st.error(f"⚠️ High risk of flood: {proba[1]*100:.2f}%")
            else:
                st.success(f"✅ Low risk of flood: {proba[0]*100:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(["No Flood", "Flood"], proba, color=["green", "red"])
            ax.set_ylabel("Probability")
            ax.set_title("Flood Prediction Probability")
            st.pyplot(fig)