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
@st.cache_data
def load_data(): 
    # chemin relatif
    base_path = os.path.join(os.path.dirname(__file__), "data")
    dakar = pd.read_csv(os.path.join(base_path, "dakar_new.csv"), parse_dates=["Datetime", "Date"])
    region = pd.read_csv(os.path.join(base_path, "region_new.csv"), parse_dates=["Datetime", "Date"])
    return dakar, region
dakar, region = load_data()
def assigner_saison(mois):
    if mois in [1, 2, 3, 12]:
        return "Bas"
    elif mois in [4, 5, 11]:
        return "Transition"
    elif mois in [6, 7, 8, 9, 10]:
        return "Haute"
    else:
        return "Inconnu"


def extract_hour(col):
    if col.dtype == "O":
        return pd.to_datetime(col, format="%H:%M:%S").dt.hour
    elif np.issubdtype(col.dtype, np.integer):
        return col
    else: 
        return col.apply(lambda x: x.hour)

# ============================
# Jours fériés et Ramadan
# ============================
jours_feries = pd.to_datetime([
    "2024-01-01", "2024-04-04", "2024-04-10", "2024-05-01", "2024-06-16", "2024-07-16", "2024-08-15", "2024-12-25",
    "2025-01-01", "2025-04-04", "2025-05-01", "2025-06-09", "2025-07-05", "2025-08-13", "2025-12-25"
])

ramadan_dates = {
    2024: pd.date_range("2024-03-11", "2024-04-09"),
    2025: pd.date_range("2025-03-01", "2025-03-30"),
    2026: pd.date_range("2026-02-18", "2026-03-19"),
    2027: pd.date_range("2027-02-07", "2027-03-08")
}

def facteur_ramadan(date):
    date = pd.to_datetime(date)
    year = date.year
    if year in ramadan_dates and date in ramadan_dates[year]:
        return 1.05 
    return 1.0 

def facteur_ferie(date): 
    if pd.to_datetime(date) in jours_feries:
        return 1.05
    return 1.0
# ============================
# Ajustement de consommation
# ============================
def ajuster_consommation(y_pred, df_test):
    mois = df_test["month"].iloc[0]
    heures = df_test["hour"]
    y_adj = y_pred.copy()

    # Facteur température
    temp_factor = 1.0 + 0.085 * (df_test["temperature"] - df_test["temperature"].mean())
    # Facteur humidité
    humid_factor = 1.0 - 0.01 * (df_test["humidity"] - df_test["humidity"].mean())
    # Facteur jours fériés
    ferie_factor = df_test["Date_only"].apply(facteur_ferie)

    y_adj = y_adj * temp_factor * humid_factor * ferie_factor

    # Ajustements saisonniers
    saison = assigner_saison(mois)
    if saison == "Bas":
        y_adj *= 0.62
    elif saison == "Transition":
        y_adj *= 0.85
    elif saison == "Haute":
        y_adj *= 0.95

       # ============================
    # Ajustement heures de pointe par saison
    # ============================
    for i, h in enumerate(heures):

        # -------- Saison BASSE --------
        if saison == "Bas":
            if 0 <= h <= 4:          # nuit
                y_adj[i] *= 1.3
            elif 5 <= h <= 9:        # réveil
                y_adj[i] *= 1
            elif 10 <= h <= 13:      # journée calme
                y_adj[i] *= 0.69
           # elif h == 13:      # journée calme
             #   y_adj[i] *= 0.8  
            elif 14 <= h <= 16:      # pointe soir modérée
                y_adj[i] *= 0.7  
            elif 17 <= h <= 19:      # journée calme
                y_adj[i] *= 0.85  
            elif 20 <= h <= 22:      # pointe soir modérée
                y_adj[i] *= 0.95
            elif h ==23 :
                y_adj[i] *= 0.93   

        # -------- Saison TRANSITION --------
        elif saison == "Transition":
            if  h == 0:
                y_adj[i] *= 1.43 
            elif 1<= h <= 4:
                y_adj[i] *= 1.2
            elif 5 <= h <= 8:
                y_adj[i] *= 1.2
            elif h== 9:        # pointe matin
                y_adj[i] *= 0.98
            elif 10 <= h <= 13:
                y_adj[i] *= 0.87
            elif 14 <= h <= 17:
                y_adj[i] *= 0.86
            elif 18 <= h <= 21:      # pointe soir
                y_adj[i] *= 0.99
            elif h >= 22:
                y_adj[i] *= 1.03
        # -------- Saison HAUTE --------
        
        elif saison == "Haute":
            if h == 0:
                y_adj[i] *= 1.05
            elif 1 <= h <= 5:
                y_adj[i] *= 0.97
            elif 6 <= h <= 8:
                y_adj[i] *= 1
            elif 9 <= h <= 11: 
                y_adj[i] *= 0.8
            elif 12 <= h <= 16:
                y_adj[i] *= 0.62
            elif 17 <= h <= 19:
                y_adj[i] *= 0.75       
            elif 20<= h <= 22:      # TRÈS forte pointe soir
                y_adj[i] *= 0.85
            elif h == 23:
                y_adj[i] *= 0.8


    return y_adj




# ============================
# Prédiction de consommation
# ============================
def pred_base(df_base, colonne_cons, date_choisie, temp_hour, humid_hour, ref_year=2024, taux_croissance_annuelle=0.20):
    date_dt = pd.to_datetime(date_choisie)

    df_jour = df_base.copy()
    df_jour["year"] = df_jour["Date"].dt.year
    df_jour["month"] = df_jour["Date"].dt.month
    if "hour" not in df_jour.columns:
        df_jour["hour"] = extract_hour(df_jour["Heure"])
    df_jour["Date_only"] = df_jour["Date"].dt.date
    df_jour["dayofyear"] = df_jour["Date"].dt.dayofyear
    df_jour["weekofyear"] = df_jour["Date"].dt.isocalendar().week.astype(int)
    df_jour["trend"] = (df_jour["Date"] - df_jour["Date"].min()).dt.days
    df_jour["day_of_week"] = df_jour["Date"].dt.weekday
    df_jour["is_weekend"] = (df_jour["day_of_week"] >= 5).astype(int)

    # Ajouter les colonnes manquantes
    for col in ["humidity", "is_ramadan", "is_tabaski", "is_korite", "is_gamou", "is_magal"]:
        if col not in df_jour.columns:
            df_jour[col] = 0.0 if col == "humidity" else 0

    features = ["year", "month", "hour", "temperature", "precipitation", "humidity",
                "day_of_week", "is_weekend", "is_ramadan",
                "is_tabaski", "is_korite", "is_gamou", "is_magal",
                "dayofyear", "weekofyear", "trend"]

    y = df_jour[colonne_cons]

    X = df_jour[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Construction du df_test avec températures et humidités saisies manuellement
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
    df_test["precipitation"] = df_jour.groupby("hour")["precipitation"].mean().values
    df_test["dayofyear"] = date_dt.timetuple().tm_yday
    df_test["weekofyear"] = date_dt.isocalendar().week
    df_test["trend"] = (date_dt - df_jour["Date"].min()).days

    # Colonnes événementielles
    for col in ["is_ramadan", "is_tabaski", "is_korite", "is_gamou", "is_magal"]:
        df_test[col] = 0

    df_test["annee_factor"] = 1 + taux_croissance_annuelle * ((df_test["year"] - ref_year) + df_test["month"] / 12)
    df_test["ferie_factor"] = df_test["Date_only"].apply(facteur_ferie)
    df_test["ramadan_factor"] = df_test["Date_only"].apply(facteur_ramadan)

    y_pred = model.predict(df_test[features])
    y_pred = y_pred * df_test["annee_factor"] * df_test["ferie_factor"] * df_test["ramadan_factor"]
    y_pred = ajuster_consommation(y_pred, df_test)

    # Graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test["hour"], y=y_pred, mode="lines+markers", name="Consommation prédite"))
    #fig.add_trace(go.Scatter(x=df_test["hour"], y=df_test["temperature"], mode="lines+markers", name="Température (°C)", yaxis="y2"))
    #fig.add_trace(go.Scatter(x=df_test["hour"], y=df_test["humidity"], mode="lines+markers", name="Humidité (%)", yaxis="y3"))

    fig.update_layout(
        title=f"Prévision {colonne_cons} - {date_choisie}",
        xaxis=dict(title="Heure"),
        yaxis=dict(title="Consommation (MW)"),
       # yaxis2=dict(title="Température (°C)", overlaying="y", side="right"),
       # yaxis3=dict(title="Humidité (%)", overlaying="y", side="right", position=0.95),
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
