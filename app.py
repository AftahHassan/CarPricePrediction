import streamlit as st
import joblib

from predict import predict_price

# ================================
# LOAD ENCODERS
# ================================

encoders = joblib.load("encoders/encoders.pkl")

city_encoder = encoders["city"]
brand_encoder = encoders["brand"]
model_encoder = encoders["model"]
fuel_encoder = encoders["fuel"]

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ================================
# TITLE
# ================================

st.title("🚗 Moroccan Car Price Prediction")

st.write(
    "Predict used car prices in Morocco using Machine Learning"
)

# ================================
# INPUTS
# ================================

city = st.selectbox(
    "Ville",
    city_encoder.classes_
)

brand = st.selectbox(
    "Marque",
    brand_encoder.classes_
)

car_model = st.selectbox(
    "Modèle",
    model_encoder.classes_
)

year = st.number_input(
    "Année-Modèle",
    min_value=1990,
    max_value=2026,
    value=2020
)

mileage = st.number_input(
    "Kilométrage",
    min_value=0,
    value=50000
)

fuel = st.selectbox(
    "Type de carburant",
    fuel_encoder.classes_
)

# ================================
# BUTTON
# ================================

if st.button("Predict Price"):

    price = predict_price(
        city,
        brand,
        car_model,
        year,
        mileage,
        fuel
    )

    st.success(f"💰 Estimated Price : {price:,} MAD")