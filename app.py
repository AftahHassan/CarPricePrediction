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
    page_title="Prédiction Prix Auto - Maroc",
    page_icon="🚗",
    layout="wide"
)

# ================================
# CSS
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #4f46e5;
    --primary-dark: #3730a3;
    --secondary: #10b981;
    --bg: #eef0f4;
    --surface: #ffffff;
    --header: #0b1120;
    --text: #1e293b;
    --text-muted: #64748b;
    --text-light: #94a3b8;
    --border: #e2e8f0;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    --shadow-lg: 0 8px 32px rgba(0,0,0,.14);
}

html, body, [class*="css"] {
    font-family: 'Outfit', 'Segoe UI', Tahoma, sans-serif !important;
}

/* Fond + no scroll */
.stApp {
    background: var(--bg) !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* Centrage vertical de tout le contenu */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

/* ---- CARD ---- */
.main-card {
    background: var(--surface);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    width: 100%;
    overflow: hidden;
}

/* ---- HEADER ---- */
.card-header {
    background: var(--header);
    padding: 1.1rem 1.6rem .9rem;
    position: relative;
    overflow: hidden;
}
.card-header::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(79,70,229,.2);
}
.card-header::after {
    content: '';
    position: absolute;
    bottom: -20px; left: 40px;
    width: 70px; height: 70px;
    border-radius: 50%;
    background: rgba(16,185,129,.12);
}
.card-header-tag {
    display: inline-flex;
    gap: 5px;
    background: rgba(79,70,229,.3);
    color: #a5b4fc;
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.card-header h1 {
    font-size: 19px !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 0 2px !important;
    position: relative; z-index: 1;
}
.card-header p {
    font-size: 12px;
    color: #94a3b8;
    margin: 0;
    position: relative; z-index: 1;
}

/* ---- STATS ---- */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-bottom: 1px solid var(--border);
}
.stat-item {
    padding: .5rem 1rem;
    text-align: center;
    border-right: 1px solid var(--border);
}
.stat-item:last-child { border-right: none; }
.stat-item .s-label {
    font-size: 9px;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
}
.stat-item .s-val {
    font-size: 17px;
    font-weight: 700;
}
.stat-item:nth-child(1) .s-val { color: var(--primary); }
.stat-item:nth-child(2) .s-val { color: var(--secondary); }
.stat-item:nth-child(3) .s-val { color: #f59e0b; }

/* ---- BODY ---- */
.card-body {
    padding: 1rem 1.6rem .8rem;
}

.section-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .4rem;
    display: flex;
    align-items: center;
    gap: 5px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 6px;
}

/* ---- INPUTS ---- */
.stSelectbox label,
.stNumberInput label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    margin-bottom: 1px !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    color: var(--text) !important;
    background: var(--surface) !important;
    box-shadow: none !important;
    height: 38px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--primary) !important;
}
.stSelectbox, .stNumberInput { margin-bottom: 6px !important; }

/* ---- BOUTON ---- */
.stButton > button {
    width: 100% !important;
    height: 44px !important;
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.35) !important;
    margin-top: .4rem !important;
}
.stButton > button:hover { background: var(--primary-dark) !important; }
.stButton > button:active { transform: scale(.98) !important; }

/* ---- RÉSULTAT ---- */
.result-block {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: var(--radius-lg);
    padding: .9rem 1.2rem;
    margin-top: .6rem;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
}
.result-block::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 110px; height: 110px;
    border-radius: 50%;
    background: rgba(79,70,229,.15);
}
.r-label {
    font-size: 10px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 2px;
}
.r-price {
    font-size: 30px;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.r-currency {
    font-size: 15px;
    font-weight: 400;
    color: #94a3b8;
    margin-left: 4px;
}
.result-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}
.badge-info    { background: rgba(79,70,229,.3); color: #a5b4fc; }
.badge-success { background: rgba(16,185,129,.2); color: #6ee7b7; }

/* ---- FOOTER ---- */
.card-footer {
    text-align: center;
    font-size: 11px;
    color: var(--text-light);
    padding: .5rem 1.6rem .7rem;
    border-top: 1px solid var(--border);
}

div[data-testid="stAlert"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ================================
# LAYOUT — 3 colonnes pour centrer
# ================================
_, center, _ = st.columns([1, 2, 1])

with center:
    # HEADER + STATS (HTML pur)
    st.markdown("""
    <div class="main-card">
        <div class="card-header">
            <div class="card-header-tag">✨ Machine Learning</div>
            <h1>🚗 Prédiction du Prix Auto — Maroc</h1>
            <p>Estimez le prix d'un véhicule d'occasion avec notre modèle ML</p>
        </div>
        <div class="stats-row">
            <div class="stat-item"><div class="s-label">Villes</div><div class="s-val">18+</div></div>
            <div class="stat-item"><div class="s-label">Marques</div><div class="s-val">40+</div></div>
            <div class="stat-item"><div class="s-label">Précision</div><div class="s-val">~87%</div></div>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)

    # --- Localisation & Véhicule ---
    st.markdown('<div class="section-label">📍 Localisation &amp; Véhicule</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Ville", city_encoder.classes_)
        car_model = st.selectbox("Modèle", model_encoder.classes_)
    with col2:
        brand = st.selectbox("Marque", brand_encoder.classes_)
        year = st.number_input("Année", min_value=1990, max_value=2026, value=2020, step=1)

    # --- Caractéristiques ---
    st.markdown('<div class="section-label" style="margin-top:.4rem">⚙️ Caractéristiques</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        mileage = st.number_input("Kilométrage (km)", min_value=0, value=50000, step=1000)
    with col4:
        fuel = st.selectbox("Carburant", fuel_encoder.classes_)

    # --- Bouton ---
    predict_clicked = st.button("✨ Prédire le prix")

    # --- Résultat inline ---
    if predict_clicked:
        price = predict_price(city, brand, car_model, year, mileage, fuel)
        fi = {"Diesel": "💧", "Essence": "🔥", "Hybride": "🍃",
              "Électrique": "⚡", "GPL": "🛢️"}.get(fuel, "⛽")
        st.markdown(f"""
        <div class="result-block">
            <div>
                <div class="r-label">💰 Prix estimé</div>
                <div class="r-price">{price:,.0f}<span class="r-currency">MAD</span></div>
            </div>
            <div class="result-badges">
                <span class="badge badge-info">📍 {city}</span>
                <span class="badge badge-info">🚗 {brand} {year}</span>
                <span class="badge badge-success">🛣️ {mileage:,} km</span>
                <span class="badge badge-success">{fi} {fuel}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fermer card
    st.markdown("""
        </div>
        <div class="card-footer">
            Modèle entraîné sur des données marocaines — résultat indicatif
        </div>
    </div>
    """, unsafe_allow_html=True)