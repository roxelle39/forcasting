import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import datetime
import os

# ============================
# Chargement des données 
# ============================
@st.cache_data(ttl=600)
def load_data(): 
    base_path = os.path.join(os.path.dirname(__file__), "data")
    dakar = pd.read_csv(os.path.join(base_path, "dakar_new.csv"), parse_dates=["Datetime", "Date"])
    region = pd.read_csv(os.path.join(base_path, "region_new.csv"), parse_dates=["Datetime", "Date"])
    return dakar, region

dakar, region = load_data() 

# ============================
# Fonctions utilitaires
# ============================
def extract_hour(col):
    """Extrait l'heure d'une colonne."""
    return pd.to_datetime(col, errors='coerce').dt.hour.fillna(0).astype(int)

def assigner_saison(mois):
    if mois in [1, 2, 3, 12]:
        return "Bas"
    elif mois in [4, 5, 11]:
        return "Transition"
    elif mois in [6, 7, 8, 9, 10]:
        return "Haute"
    else:
        return "Inconnu"

# Jours fériés
jours_feries = pd.to_datetime([
    "2024-01-01", "2024-04-04", "2024-04-10", "2024-05-01", "2024-06-16", "2024-07-16", "2024-08-15", "2024-12-25",
    "2025-01-01", "2025-04-04", "2025-05-01", "2025-06-09", "2025-07-05", "2025-08-13", "2025-12-25"
])
jours_feries_set = set(jours_feries)

# Ramadan
ramadan_dates = {
    2024: pd.date_range("2024-03-11", "2024-04-09"),
    2025: pd.date_range("2025-03-01", "2025-03-30"),
}

def facteur_ramadan(date):
    date = pd.to_datetime(date)
    year = date.year
    return 1.05 if year in ramadan_dates and date in set(ramadan_dates[year]) else 1.0

def facteur_ferie(date): 
    return 1.05 if pd.to_datetime(date) in jours_feries_set else 1.0

# ============================
# Ajustement consommation
# ============================
def ajuster_consommation(y_pred, df_test):
    """Ajuste la consommation selon température, humidité, saison et heures de pointe."""
    mois = df_test["month"].iloc[0]
    heures = df_test["hour"]
    y_adj = y_pred.copy().astype(float)

    # Facteurs température et humidité
    temp_factor = 1.0 + 0.085 * (df_test["temperature"] - df_test["temperature"].mean())
    humid_factor = 1.0 - 0.01 * (df_test["humidity"] - df_test["humidity"].mean())

    # Facteurs événements
    ferie_factor = df_test["Date_only"].apply(facteur_ferie)
    ramadan_factor = df_test["Date_only"].apply(facteur_ramadan)

    y_adj *= temp_factor * humid_factor * ferie_factor * ramadan_factor

          # Facteur saison + heures combinés
    saison = assigner_saison(mois)
    horaires_coeff = {
            "Bas": [0.7,0.7,0.7,0.7,0.70,0.7,0.7,0.7,0.70,0.75,0.75,0.75,0.75,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],
            "Transition": [1.2]*24,
            "Haute": [0.95,0.95,0.95,0.95,0.95,0.95,0.9,0.85,0.85,0.8,0.8,0.8,0.7,0.7,0.7,0.7,0.75,0.8,0.85,0.9,0.9,0.9,0.95,0.95]
        }
    coeff = horaires_coeff.get(saison, [1.0]*24)
    for i, h in enumerate(heures):
        y_adj[i] *= coeff[h]

    return y_adj
# ============================
# Prédiction de consommation
# ============================
def pred_base(df_base, colonne_cons, date_choisie, temp_hour, humid_hour, ref_year=2024, taux_croissance_annuelle=0.20):
    date_dt = pd.to_datetime(date_choisie)

    df_jour = df_base.copy()
    df_jour["year"] = df_jour["Date"].dt.year
    df_jour["month"] = df_jour["Date"].dt.month
    df_jour["hour"] = extract_hour(df_jour["Heure"])
    df_jour["Date_only"] = df_jour["Date"].dt.date
    df_jour["dayofyear"] = df_jour["Date"].dt.dayofyear
    df_jour["weekofyear"] = df_jour["Date"].dt.isocalendar().week.astype(int)
    df_jour["trend"] = (df_jour["Date"] - df_jour["Date"].min()).dt.days
    df_jour["day_of_week"] = df_jour["Date"].dt.weekday
    df_jour["is_weekend"] = (df_jour["day_of_week"] >= 5).astype(int)

    # Colonnes événementielles pour df_jour
    for col in ["humidity","is_ramadan","is_tabaski","is_korite","is_gamou","is_magal"]:
        if col not in df_jour.columns:
            df_jour[col] = 0.0 if col=="humidity" else 0

    features = ["year","month","hour","temperature","precipitation","humidity",
                "day_of_week","is_weekend","is_ramadan","is_tabaski","is_korite","is_gamou","is_magal",
                "dayofyear","weekofyear","trend"]

    y = df_jour[colonne_cons]
    X = df_jour[features]

    # Entraînement du modèle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Construction de df_test pour 24 heures
    df_test = pd.DataFrame({
        "hour": np.arange(24),
        "temperature": temp_hour,
        "humidity": humid_hour
    })
    df_test["year"] = date_dt.year
    df_test["month"] = date_dt.month
    df_test["day_of_week"] = date_dt.weekday()
    df_test["is_weekend"] = int(date_dt.weekday() >= 5)
    df_test["Date_only"] = date_dt
    df_test["precipitation"] = df_jour.groupby("hour")["precipitation"].mean().reindex(range(24), fill_value=0).values
    df_test["dayofyear"] = date_dt.timetuple().tm_yday
    df_test["weekofyear"] = date_dt.isocalendar().week
    df_test["trend"] = (date_dt - df_jour["Date"].min()).days

    # Colonnes événementielles pour df_test (important pour éviter KeyError)
    for col in ["is_ramadan","is_tabaski","is_korite","is_gamou","is_magal"]:
        df_test[col] = 0

    # Prédiction avec correction annuelle
    y_pred = model.predict(df_test[features])
    df_test["annee_factor"] = 1 + taux_croissance_annuelle * ((df_test["year"] - ref_year) + df_test["month"]/12)
    y_pred = y_pred * df_test["annee_factor"]

    # Ajustement consommation (température, humidité, saison, heures, événements)
    y_pred = ajuster_consommation(y_pred, df_test)

    # Graphique Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test["hour"], y=y_pred, mode="lines+markers", name="Consommation prédite"))
    fig.update_layout(
        title=f"Prévision {colonne_cons} - {date_choisie}",
        xaxis=dict(title="Heure"),
        yaxis=dict(title="Consommation (MW)"),
        template="plotly_white"
    )

    return y_pred, fig



# ============================
# Interface Streamlit
# ============================
st.set_page_config(page_title="Prévision Consommation Sénégal", layout="wide")
st.title("⚡ Prévision de la Consommation Électrique au Sénégal")
st.write("Saisissez manuellement la température et l'humidité pour chaque heure.")

date_choisie = st.date_input("Choisissez une date :", datetime.date(2025, 10, 13))

st.subheader("📍 Saisie horaire de la Température (°C) et Humidité (%)")
temp_hour = []
humid_hour = []

cols = st.columns([1,1])
for h in range(24):
    with cols[0]:
        t = st.number_input(f"T° heure {h}", min_value=-10.0, max_value=50.0, value=25.0, key=f"temp{h}")
        temp_hour.append(t)
    with cols[1]:
        hmd = st.number_input(f"Humidité heure {h} (%)", min_value=0.0, max_value=100.0, value=60.0, key=f"hum{h}")
        humid_hour.append(hmd)

if st.button("Lancer la prédiction"):
    with st.spinner("Calcul des prédictions..."):
        y_dakar, fig_dakar = pred_base(dakar, "cons_dakar", date_choisie, temp_hour, humid_hour)
        y_region, fig_region = pred_base(region, "cons_regions", date_choisie, temp_hour, humid_hour)

        heures = np.arange(24)
        df_table = pd.DataFrame({
            "Heure": heures,
            "Consommation_Dakar": y_dakar,
            "Consommation_Régions": y_region
        })
        df_table["Total_Sénégal"] = df_table["Consommation_Dakar"] + df_table["Consommation_Régions"]

        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(x=heures, y=df_table["Consommation_Dakar"], mode="lines+markers", name="Dakar"))
        fig_total.add_trace(go.Scatter(x=heures, y=df_table["Consommation_Régions"], mode="lines+markers", name="Régions"))
        fig_total.add_trace(go.Scatter(x=heures, y=df_table["Total_Sénégal"], mode="lines+markers", name="Total Sénégal", line=dict(color='red', width=3)))
        fig_total.update_layout(title=f"Prévision Totale - {date_choisie}", xaxis_title="Heure", yaxis_title="Consommation (MW)", template="plotly_white")

        st.success("✅ Prédictions terminées !")
        st.subheader("📊 Graphiques de Consommation")
        st.plotly_chart(fig_dakar, use_container_width=True)
        st.plotly_chart(fig_region, use_container_width=True)
        st.plotly_chart(fig_total, use_container_width=True)
        st.subheader("📋 Tableau des Résultats")
        st.dataframe(df_table.style.format(precision=2))
