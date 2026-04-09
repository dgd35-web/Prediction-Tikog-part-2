import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Tikog Requirement Prediction", page_icon="🧺")

# ======================================================
# LOAD TRAINED MODELS
# ======================================================
@st.cache_resource
def load_models():
    lstm_path = "lstm_model.keras"
    xgb_path = "xgb_model.json"

    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"Missing file: {lstm_path}")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Missing file: {xgb_path}")

    lstm_model = load_model(lstm_path)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)

    return lstm_model, xgb_model

try:
    lstm_model, xgb_model = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ======================================================
# APP TITLE & DESCRIPTION
# ======================================================
st.title("Tikog Requirement Prediction Application")
st.write("Enter the following details to predict the required Tikog for your product:")

# ======================================================
# NUMBER OF SIDES PER PRODUCT
# ======================================================
product_sides = {
    "Basket": 1,
    "Mat": 1,
    "Bag": 2,
    "Slippers": 2,
    "Wallet": 2,
    "Others": 1
}

# ======================================================
# DIMENSION OPTIONS
# ======================================================
dimension_options = {
    "27 inches x 16 inches": (27.0, 16.0),
    "11 inches x 14 ½ inches": (11.0, 14.5),
    "12 inches x 7 ½ inches x 3 ½ inches": (12.0, 7.5),
    "Body = 17 ½ x 2, packet (11 ½ x 11 ½), side (5 x 6)": (17.5, 2.0),
    "29 inches x 22 inches": (29.0, 22.0)
}

# ======================================================
# DIMENSION INPUT
# ======================================================
dimension = st.selectbox(
    "Dimension",
    options=list(dimension_options.keys()) + ["Custom"]
)

if dimension != "Custom":
    length, width = dimension_options[dimension]
    st.write(f"Length: {length} inches")
    st.write(f"Width: {width} inches")
else:
    length = st.number_input("Length (in inches)", min_value=0.0, step=0.1)
    width = st.number_input("Width (in inches)", min_value=0.0, step=0.1)

# ======================================================
# OTHER INPUTS
# ======================================================
quantity = st.text_input("Quantity", "10")

product_type = st.selectbox(
    "Product Type",
    ["Basket", "Mat", "Bag", "Slippers", "Wallet", "Others"]
)

sales_trend = st.selectbox(
    "Sales Trend",
    ["Increasing", "Stable", "Decreasing"]
)

# ======================================================
# OPTIONAL RULE-BASED FIX FOR PROBLEM PRODUCTS
# ======================================================
rule_based_tikog = {
    "Slippers": 40,   # change only if your validated value is different
    # "Wallet": 60,   # uncomment if validated by weavers
}

# ======================================================
# PREDICTION LOGIC
# ======================================================
if st.button("Predict"):
    try:
        total_quantity = int(quantity)

        if total_quantity < 1:
            st.error("Quantity must be a positive whole number.")
            st.stop()

        sides = product_sides.get(product_type, 1)
        area = float(length) * float(width)
        trend_map = {"Increasing": 1, "Stable": 0, "Decreasing": -1}

        features = pd.DataFrame([{
            "length": float(length),
            "width": float(width),
            "area": float(area),
            "sides": int(sides),
            "quantity": int(total_quantity),
            "sales_trend": int(trend_map[sales_trend])
        }])

        # LSTM prediction
        lstm_input = np.expand_dims(features.values.astype(np.float32), axis=1)
        lstm_pred = lstm_model.predict(lstm_input, verbose=0)
        lstm_pred_value = float(lstm_pred[0][0])

        # XGBoost prediction
        xgb_pred = xgb_model.predict(features)
        xgb_pred_value = float(xgb_pred[0])

        # Combine predictions
        model_prediction = (lstm_pred_value + xgb_pred_value) / 2

        # Rule-based override for validated problematic products
        if product_type in rule_based_tikog:
            final_tikog_needed = rule_based_tikog[product_type] * total_quantity
            method_used = "Rule-based override"
        else:
            final_tikog_needed = model_prediction
            method_used = "Model ensemble (LSTM + XGBoost)"

        # OUTPUT
        st.success(f"Prediction: {final_tikog_needed:.0f} units of Tikog required")

        st.write("### Breakdown")
        st.write(f"Product: {product_type}")
        st.write(f"Quantity: {total_quantity}")
        st.write(f"LSTM prediction: {lstm_pred_value:.2f}")
        st.write(f"XGBoost prediction: {xgb_pred_value:.2f}")
        st.write(f"Combined model prediction: {model_prediction:.2f}")
        st.write(f"Method used: {method_used}")

    except ValueError:
        st.error("Please enter a valid whole number for Quantity.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
