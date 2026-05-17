import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ================================
# CREATE FOLDERS
# ================================
os.makedirs("model", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ================================
# LOAD DATA (FIX ENCODING)
# ================================
df = pd.read_csv(
    "data/avito_car_dataset_ALL.csv",
    encoding="latin1"
)

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
# CLEAN DATA
# ================================

# Clean mileage (IMPORTANT FIX)
df["mileage"] = df["mileage"].astype(str)
df["mileage"] = df["mileage"].str.replace(" ", "")
df["mileage"] = df["mileage"].str.split("-").str[0]
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

# Clean year
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Clean price (remove spaces if string)
df["price"] = df["price"].astype(str).str.replace(" ", "")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Drop missing values
df = df.dropna()

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
# FEATURES / TARGET
# ================================
X = df[["city", "brand", "model", "year", "mileage", "fuel"]]
y = df["price"]

# ================================
# SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================================
# MODEL
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# EVALUATION
# ================================
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("✅ Model trained successfully")
print("📊 MAE:", mae)

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

print("✅ Saved successfully")



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("📊 MODEL PERFORMANCE")
print("MAE :", mae)
print("RMSE :", rmse)
print("R2 Score :", r2)



plt.figure()
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Réel vs Prédit")

plt.savefig("results/real_vs_pred.png")
plt.close()


errors = y_test - pred

plt.figure()
plt.hist(errors, bins=50)
plt.title("Distribution des erreurs")
plt.xlabel("Erreur")
plt.ylabel("Fréquence")

plt.savefig("results/error_distribution.png")
plt.close()


sample = 30

plt.figure()
plt.plot(y_test.values[:sample], label="Réel")
plt.plot(pred[:sample], label="Prédit")
plt.legend()
plt.title("Comparaison sur échantillon")

plt.savefig("results/sample_comparison.png")
plt.close()
