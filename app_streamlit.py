# ==========================================
# app.py — Prévision de la consommation électrique au Sénégal
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import datetime
import random

# ============================
# Chargement des données
# ============================
@st.cache_data
def load_data():
    dakar = pd.read_csv(
        "/home/rokhayadiop/Téléchargements/stage/final_app/data/dakar_new.csv",
        parse_dates=["Datetime", "Date"]
    )
    region = pd.read_csv(
        "/home/rokhayadiop/Téléchargements/stage/final_app/data/region_new.csv",
        parse_dates=["Datetime", "Date"]
    )
    return dakar, region


dakar, region = load_data()


# ============================
# Fonctions utilitaires
# ============================
def assigner_saison(mois):
    if mois in [1, 2, 3, 12]:
        return "Bas"
    elif mois in [4, 5, 6, 11]:
        return "Transition"
    elif mois in [7, 8, 9, 10]:
        return "Hautes"
    else:
        return "Inconnu"


def corriger_precipitation(df):
    df = df.copy()
    df["month"] = df["Datetime"].dt.month
    df.loc[df["month"].isin([11, 12, 1, 2, 3, 4, 5]), "precipitation"] = 0.0
    return df


dakar = corriger_precipitation(dakar)
region = corriger_precipitation(region)


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
    "2024-01-01", "2024-04-04", "2024-05-01", "2024-12-25",
    "2025-01-01", "2025-04-04", "2025-05-01", "2025-12-25",
    "2026-01-01", "2026-04-04", "2026-05-01", "2026-12-25",
    "2027-01-01", "2027-04-04", "2027-05-01", "2027-12-25"
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
# Fonction de correction horaire pour novembre/décembre
# ============================
def facteur_horaire(date, heure):
    mois = date.month
    if mois not in [2,11, 12]:
        return 1.0
    # Réduire la consommation selon l'heure
    if 0 <= heure <= 6:
        return 0.90  # -5%
    elif 7 <= heure <= 17:
        return 0.92  # -8%
    elif 18 <= heure <= 22:
        return 0.85  # -12%
    else:
        return 0.88
    # ============================
# Fonction de prédiction principale
# ============================
def pred_base(df_base, colonne_cons, date_choisie, ref_year=2024, taux_croissance_annuelle=0.20):
    date_dt = pd.to_datetime(date_choisie)
    jours_fr = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    jours_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    jour_nom = jours_fr[jours_en.index(date_dt.day_name())]

    df_jour = df_base[df_base['day_of_week_name'].str.lower() == jour_nom].copy()
    df_jour = df_jour[~df_jour["Date"].dt.date.isin(jours_feries.date)]
    df_jour["year"] = df_jour["Date"].dt.year
    df_jour["month"] = df_jour["Date"].dt.month
    if "hour" not in df_jour.columns:
        df_jour["hour"] = extract_hour(df_jour["Heure"])
    df_jour["profil_type"] = df_jour["Saison de demande"] + ' ' + jour_nom

    mean_temp = df_jour.groupby("hour")["temperature"].mean().reset_index()
    mean_precip = df_jour.groupby("hour")["precipitation"].mean().reset_index()

    # Simulation d’une température variable chaque jour
    random.seed(date_dt.toordinal())
    daily_temp_shift = random.uniform(-1.5, 1.5)
    hourly_fluctuation = np.sin(np.linspace(0, 2*np.pi, 24)) * random.uniform(0.5, 1.2)
    temperature_day = mean_temp["temperature"] + daily_temp_shift + hourly_fluctuation

    df_test = pd.DataFrame({"hour": np.arange(24)})
    df_test["year"] = date_dt.year
    df_test["month"] = date_dt.month
    df_test["day_of_week"] = date_dt.weekday()
    df_test["is_weekend"] = (df_test["day_of_week"] >= 5).astype(int)
    df_test["day"] = date_dt.day
    df_test["temperature"] = temperature_day
    df_test["precipitation"] = mean_precip["precipitation"]
    df_test["Saison de demande"] = df_test["month"].apply(assigner_saison)
    df_test["profil_type"] = df_test["Saison de demande"] + " " + jour_nom
    for col in ["is_ramadan", "is_tabaski", "is_korite", "is_gamou", "is_magal"]:
        df_test[col] = 0
    df_test["Date_only"] = date_dt

    features = [
        "year", "month", "hour", "temperature", "precipitation",
        "day_of_week", "is_weekend", "is_ramadan", "is_tabaski",
        "is_korite", "is_gamou", "is_magal", "Saison de demande",
        "profil_type", "day"
    ]

    X = pd.get_dummies(df_jour[features], columns=["Saison de demande", "profil_type"], drop_first=True)
    y = df_jour[colonne_cons]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    X_future = pd.get_dummies(df_test[features], columns=["Saison de demande", "profil_type"], drop_first=True)
    X_future = X_future.reindex(columns=X_train.columns, fill_value=0)

    # Facteurs multiplicatifs
    df_test["annee_factor"] = 1 + taux_croissance_annuelle * (df_test["year"] - ref_year)
    df_test["ferie_factor"] = df_test["Date_only"].apply(facteur_ferie)
    df_test["ramadan_factor"] = df_test["Date_only"].apply(facteur_ramadan)
    df_test["horaire_factor"] = df_test.apply(lambda row: facteur_horaire(date_dt, row["hour"]), axis=1)

    # Prédiction ajustée
    y_pred = model.predict(X_future)
    y_pred *= df_test["annee_factor"] * df_test["ferie_factor"] * df_test["ramadan_factor"] * df_test["horaire_factor"]

    # ============================
    # Graphique
    # ============================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test["hour"], y=y_pred, mode="lines+markers", name="Consommation prédite"))
    fig.add_trace(go.Scatter(x=df_test["hour"], y=df_test["temperature"], mode="lines+markers", name="Température", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_test["hour"], y=df_test["precipitation"], mode="lines+markers", name="Précipitation", yaxis="y3"))
    fig.update_layout(
        title=f"Prévision {colonne_cons} - {jour_nom.capitalize()} - {date_choisie}",
        xaxis=dict(title="Heure"),
        yaxis=dict(title="Consommation (MW)"),
        yaxis2=dict(title="Température (°C)", overlaying="y", side="right"),
        yaxis3=dict(title="Précipitation (mm)", overlaying="y", side="right", position=0.95),
        template="plotly_white"
    )

    return y_pred, fig, df_test


# ============================
# Interface Streamlit
# ============================
st.set_page_config(page_title="Prévision Consommation Sénégal", layout="wide")
st.title("⚡ Prévision de la Consommation Électrique au Sénégal")
st.write("Prédiction horaire pour **Dakar**, **les Régions**, et visualisation des températures quotidiennes.")

date_choisie = st.date_input("Choisissez une date :", datetime.date(2025, 11, 11))

if st.button("Lancer la prédiction"):
    with st.spinner("Calcul des prédictions..."):
        y_dakar, fig_dakar, df_dakar_test = pred_base(dakar, "cons_dakar", date_choisie)
        y_region, fig_region, df_region_test = pred_base(region, "cons_regions", date_choisie)

        heures = np.arange(24)
        df_table = pd.DataFrame({
            "Heure": heures,
            "Consommation_Dakar": y_dakar,
            "Consommation_Régions": y_region
        })
        df_table["Total_Sénégal"] = df_table["Consommation_Dakar"] + df_table["Consommation_Régions"]

        # Graphique total
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

        st.subheader("🌡️ Température quotidienne")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df_dakar_test["hour"], y=df_dakar_test["temperature"], mode="lines+markers", name="Température simulée"))
        fig_temp.update_layout(title=f"Température simulée pour {date_choisie}", xaxis_title="Heure", yaxis_title="Température (°C)", template="plotly_white")
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("📋 Tableau des Résultats")
        st.dataframe(df_table.style.format(precision=2))
