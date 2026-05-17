import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ================================
# CREATE FOLDERS
# ================================

os.makedirs("model", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

# ================================
# LOAD DATASET
# ================================

df = pd.read_csv("data/avito_car_dataset_ALL.csv")

# ================================
# SELECT COLUMNS
# ================================

df = df[[
    "Ville",
    "Marque",
    "Modèle",
    "Année-Modèle",
    "Kilométrage",
    "Type de carburant",
    "Prix"
]]

# ================================
# DROP NULL VALUES
# ================================

df = df.dropna()

# ================================
# RENAME COLUMNS
# ================================

df.columns = [
    "city",
    "brand",
    "model",
    "year",
    "mileage",
    "fuel",
    "price"
]

# ================================
# ENCODERS
# ================================

city_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

df["city"] = city_encoder.fit_transform(df["city"])
df["brand"] = brand_encoder.fit_transform(df["brand"])
df["model"] = model_encoder.fit_transform(df["model"])
df["fuel"] = fuel_encoder.fit_transform(df["fuel"])

# ================================
# FEATURES & TARGET
# ================================

X = df[[
    "city",
    "brand",
    "model",
    "year",
    "mileage",
    "fuel"
]]

y = df["price"]

# ================================
# SPLIT DATA
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================================
# MODEL
# ================================

model = LinearRegression()

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICTIONS
predictions = model.predict(X_test)

# SCORE
mae = mean_absolute_error(y_test, predictions)

print(f"MAE : {mae}")

# ================================
# SAVE MODEL
# ================================

joblib.dump(model, "model/model.pkl")

encoders = {
    "city": city_encoder,
    "brand": brand_encoder,
    "model": model_encoder,
    "fuel": fuel_encoder
}

joblib.dump(encoders, "encoders/encoders.pkl")

print("✅ Model Saved Successfully")