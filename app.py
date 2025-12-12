# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import plotly.io as pio

# ============================
# Initialisation Flask
# ============================
app = Flask(__name__)

# ============================
# Chargement des bases
# ============================
dakar = pd.read_csv("/home/rokhayadiop/Téléchargements/stage/final_app/data/dakar_new.csv", parse_dates=["Datetime","Date"])
region = pd.read_csv("/home/rokhayadiop/Téléchargements/stage/final_app/data/region_new.csv", parse_dates=["Datetime","Date"])

# ============================
# Fonctions communes
# ============================
def assigner_saison(mois):
    if mois in [1,2,3,12]:
        return "Bas"
    elif mois in [4,5,6,11]:
        return "Transition"
    elif mois in [7,8,9,10]:
        return "Hautes"
    else:
        return "Inconnu"

def corriger_precipitation(df):
    df = df.copy()
    df["month"] = df["Datetime"].dt.month
    df.loc[df["month"].isin([11,12,1,2,3,4,5]), "precipitation"] = 0.0
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
# Jours fériés Sénégal
# ============================
jours_feries = pd.to_datetime([
    "2024-01-01","2024-04-01","2024-04-04","2024-04-10","2024-05-01","2024-05-09",
    "2024-05-20","2024-06-16","2024-07-16","2024-08-15","2024-08-22","2024-09-15",
    "2024-11-01","2024-12-25",
    "2025-01-01","2025-03-31","2025-04-04","2025-04-21","2025-05-01","2025-05-29",
    "2025-06-06","2025-06-09","2025-07-05","2025-08-13","2025-08-15","2025-09-05",
    "2025-11-01","2025-12-25",
    "2026-01-01","2026-03-20","2026-04-04","2026-04-06","2026-05-01","2026-05-14",
    "2026-05-25","2026-05-27","2026-06-25","2026-08-01","2026-08-15","2026-08-25",
    "2026-11-01","2026-12-25",
    "2027-01-01","2027-03-09","2027-03-29","2027-04-04","2027-05-01","2027-05-06",
    "2027-05-16","2027-05-17","2027-06-15","2027-07-23","2027-08-14","2027-08-15",
    "2027-11-01"
])

# ============================
# Ramadan
# ============================
ramadan_dates = {
    2024: pd.date_range("2024-03-11","2024-04-09"),
    2025: pd.date_range("2025-03-01","2025-03-30"),
    2026: pd.date_range("2026-02-18","2026-03-19"),
    2027: pd.date_range("2027-02-07","2027-03-08")
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
# Fonction de prédiction pour une base
# ============================
def pred_base(df_base, colonne_cons, date_choisie, ref_year=2024, taux_croissance_annuelle=0.20):
    date_dt = pd.to_datetime(date_choisie)
    jours_fr = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
    jours_en = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    jour_nom = jours_fr[jours_en.index(date_dt.day_name())]

    df_jour = df_base[df_base['day_of_week_name'].str.lower()==jour_nom].copy()
    df_jour = df_jour[~df_jour["Date"].dt.date.isin(jours_feries.date)]
    df_jour["year"] = df_jour["Date"].dt.year
    df_jour["month"] = df_jour["Date"].dt.month
    if "hour" not in df_jour.columns:
        df_jour["hour"] = extract_hour(df_jour["Heure"])
    df_jour["profil_type"] = df_jour["Saison de demande"] + ' ' + jour_nom
    df_jour["Date_only"] = df_jour["Date"].dt.date

    mean_temp = df_jour.groupby("hour")["temperature"].mean().reset_index()
    mean_precip = df_jour.groupby("hour")["precipitation"].mean().reset_index()

    features = [
        "year","month","hour","temperature","precipitation",
        "day_of_week","is_weekend","is_ramadan",
        "is_tabaski","is_korite","is_gamou","is_magal",
        "Saison de demande","profil_type","day"
    ]

    X = pd.get_dummies(df_jour[features], columns=["Saison de demande","profil_type"], drop_first=True)
    y = df_jour[colonne_cons]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    df_test = df_jour[df_jour["Date_only"]==date_dt.date()].copy()
    if df_test.empty:
        df_test = pd.DataFrame({"hour":np.arange(24)})
        df_test["year"]=date_dt.year
        df_test["month"]=date_dt.month
        df_test["day_of_week"]=date_dt.weekday()
        df_test["is_weekend"]=int(df_test["day_of_week"].iloc[0]>=5)
        df_test["day"]=date_dt.day
        df_test["temperature"]=df_test["hour"].map(mean_temp.set_index("hour")["temperature"])
        df_test["precipitation"]=df_test["hour"].map(mean_precip.set_index("hour")["precipitation"])
        df_test["Saison de demande"]=df_test["month"].apply(assigner_saison)
        df_test["profil_type"]=df_test["Saison de demande"]+' '+jour_nom
        for col in ["is_ramadan","is_tabaski","is_korite","is_gamou","is_magal"]:
            df_test[col]=0

    df_test["Date_only"]=pd.to_datetime(date_dt.date())
    df_test["temp_factor"]=1.0

    heures_pic = [4, 16, 23]
    date_precedente = date_dt - pd.Timedelta(days=7)
    df_ref = dakar[dakar['Date'].dt.date == date_precedente.date()].copy()
    if df_ref.empty:
        df_ref = df_jour[df_jour["Date_only"] == date_precedente.date()].copy()
    if "hour" not in df_ref.columns and "Heure" in df_ref.columns:
        df_ref["hour"] = extract_hour(df_ref["Heure"])

    df_test["temp_factor"] = 1.0

    for h in heures_pic:
        if h in df_test["hour"].values and h in df_ref["hour"].values:
            temp_diff = df_test.loc[df_test["hour"]==h, "temperature"].values[0] - \
                        df_ref.loc[df_ref["hour"]==h, "temperature"].values[0]
            pct = np.clip(0.03 * abs(temp_diff), 0, 0.05)
            factor = 1 + np.sign(temp_diff) * pct
            df_test.loc[df_test["hour"] == h, "temp_factor"] = factor

    # Lisser et limiter les facteurs
    df_test["temp_factor"] = df_test["temp_factor"].interpolate().fillna(1.0)
    df_test["temp_factor"] = df_test["temp_factor"].clip(0.9, 1.1)

    df_test["annee_factor"]=1+taux_croissance_annuelle*(df_test["year"]-ref_year)
    df_test["ferie_factor"]=df_test["Date_only"].apply(facteur_ferie)
    df_test["ramadan_factor"]=df_test["Date_only"].apply(facteur_ramadan)

    X_future=pd.get_dummies(df_test[features], columns=["Saison de demande","profil_type"], drop_first=True)
    X_future=X_future.reindex(columns=X_train.columns, fill_value=0)
    y_pred=model.predict(X_future)
    y_pred=y_pred*df_test["temp_factor"]*df_test["annee_factor"]*df_test["ferie_factor"]*df_test["ramadan_factor"]

    # Graphique Plotly
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df_test["hour"],y=y_pred,mode="lines+markers",name="Consommation prédite"))
    fig.add_trace(go.Scatter(x=df_test["hour"],y=df_test["temperature"],mode="lines+markers",name="Température",yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_test["hour"],y=df_test["precipitation"],mode="lines+markers",name="Précipitation",yaxis="y3"))
    fig.update_layout(
        title=f"Prévision {colonne_cons} - {jour_nom.capitalize()} - {date_choisie}",
        xaxis=dict(title="Heure"),
        yaxis=dict(title="Consommation (MW)"),
        yaxis2=dict(title="Température (°C)", overlaying="y", side="right"),
        yaxis3=dict(title="Précipitation (mm)", overlaying="y", side="right", position=0.95),
        template="plotly_white"
    )

    return y_pred, fig

# ============================
# Route Flask
# ============================
@app.route("/", methods=["GET","POST"])
def index():
    resultats = {}
    graphs = {}
    df_table = pd.DataFrame()  # Pour le tableau HTML

    if request.method == "POST":
        date_choisie = request.form.get("date")
        y_dakar, fig_dakar = pred_base(dakar, "cons_dakar", date_choisie)
        y_region, fig_region = pred_base(region, "cons_regions", date_choisie)

        # Création DataFrame fusionné pour le total
        heures = np.arange(24)
        df_table = pd.DataFrame({
            "Heure": heures,
            "Consommation_Dakar": y_dakar,
            "Consommation_Régions": y_region
        })

        # Calcul total et diminution de 3% entre 8h et 19h
        df_table["Total_Sénégal"] = df_table["Consommation_Dakar"] + df_table["Consommation_Régions"]
        #df_table.loc[(df_table["Heure"] >= 1) & (df_table["Heure"] <= 20), "Total_Sénégal"] *= 0.95

        # Récupération des températures moyennes pour affichage
        df_table["Température_Dakar"] = dakar.groupby(dakar["Datetime"].dt.hour)["temperature"].mean().reindex(heures).values
        df_table["Température_Régions"] = region.groupby(region["Datetime"].dt.hour)["temperature"].mean().reindex(heures).values

        # Graphique du total
        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(x=df_table["Heure"], y=df_table["Consommation_Dakar"], mode="lines+markers", name="Dakar"))
        fig_total.add_trace(go.Scatter(x=df_table["Heure"], y=df_table["Consommation_Régions"], mode="lines+markers", name="Régions"))
        fig_total.add_trace(go.Scatter(x=df_table["Heure"], y=df_table["Total_Sénégal"], mode="lines+markers", name="Total Sénégal", line=dict(color='red', width=3)))

        fig_total.add_trace(go.Scatter(x=df_table["Heure"], y=df_table["Température_Dakar"], mode="lines", name="Temp Dakar", yaxis="y2", line=dict(dash="dot")))
        fig_total.add_trace(go.Scatter(x=df_table["Heure"], y=df_table["Température_Régions"], mode="lines", name="Temp Régions", yaxis="y2", line=dict(dash="dot")))

        fig_total.update_layout(
            title=f"Prévision Consommation Totale - {date_choisie}",
            xaxis=dict(title="Heure"),
            yaxis=dict(title="Consommation (MW)"),
            yaxis2=dict(title="Température (°C)", overlaying="y", side="right"),
            template="plotly_white"
        )

        resultats = df_table.to_dict(orient="records")
        graphs = {
            "Dakar": pio.to_html(fig_dakar, full_html=False),
            "Régions": pio.to_html(fig_region, full_html=False),
            "Total": pio.to_html(fig_total, full_html=False)
        }

    return render_template("index.html", resultats=resultats, graphs=graphs)

# ============================
# Lancer l'application
# ============================
if __name__=="__main__":
    app.run(debug=True)
