import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb

# ======================================================
# LOAD TRAINED MODELS
# ======================================================
lstm_model = load_model("lstm_model.keras")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")

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
    "27 inches x 16 inches": (27, 16),
    "11 inches x 14 ½ inches": (11, 14.5),
    "12 inches x 7 ½ inches x 3 ½ inches": (12, 7.5),
    "Body = 17 ½ x 2, packet (11 ½ x 11 ½), side (5 x 6)": (17.5, 2),
    "29 inches x 22 inches": (29, 22)
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
    length = st.number_input("Length (in inches)", min_value=0, step=1)
    width = st.number_input("Width (in inches)", min_value=0, step=1)

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
# PREDICTION LOGIC
# ======================================================
if st.button("Predict"):
    try:
        total_quantity = int(quantity)
        if total_quantity < 1:
            st.error("Quantity must be a positive whole number.")
        else:
            sides = product_sides.get(product_type, 1)
            area = length * width
            trend_map = {"Increasing": 1, "Stable": 0, "Decreasing": -1}

            features = pd.DataFrame([{
                "length": length,
                "width": width,
                "area": area,
                "sides": sides,
                "quantity": total_quantity,
                "sales_trend": trend_map[sales_trend]
            }])

            # LSTM prediction
            lstm_input = np.expand_dims(features.values, axis=1)
            lstm_pred = lstm_model.predict(lstm_input)
            lstm_pred_value = float(lstm_pred[0][0])

            # XGBoost prediction
            xgb_pred = xgb_model.predict(features)
            xgb_pred_value = float(xgb_pred[0])

            # Combine predictions
            final_tikog_needed = (lstm_pred_value + xgb_pred_value) / 2

            # ==================================================
            # OUTPUT
            # ==================================================
            st.success(f"Prediction: {final_tikog_needed:.0f} units of Tikog required")

            st.write("### Breakdown")
            st.write(f"Product: {product_type}")
            st.write(f"Quantity: {total_quantity}")
            st.write(f"LSTM prediction: {lstm_pred_value:.2f}")
            st.write(f"XGBoost prediction: {xgb_pred_value:.2f}")
            st.write(f"Combined (average): {final_tikog_needed:.2f}")

    except ValueError:
        st.error("Please enter a valid whole number for Quantity.")
