import pandas as pd
import joblib

# ================================
# LOAD MODEL
# ================================

model = joblib.load("model/model.pkl")
encoders = joblib.load("encoders/encoders.pkl")

# ================================
# LOAD ENCODERS
# ================================

city_encoder = encoders["city"]
brand_encoder = encoders["brand"]
model_encoder = encoders["model"]
fuel_encoder = encoders["fuel"]

# ================================
# PREDICT FUNCTION
# ================================

def predict_price(city, brand, car_model, year, mileage, fuel):

    city_encoded = city_encoder.transform([city])[0]
    brand_encoded = brand_encoder.transform([brand])[0]
    model_encoded = model_encoder.transform([car_model])[0]
    fuel_encoded = fuel_encoder.transform([fuel])[0]

    input_data = pd.DataFrame([{
        "city": city_encoded,
        "brand": brand_encoded,
        "model": model_encoded,
        "year": year,
        "mileage": mileage,
        "fuel": fuel_encoded
    }])

    prediction = model.predict(input_data)[0]

    return int(prediction)