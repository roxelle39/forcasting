# utils.py
import pandas as pd
import numpy as np
from datetime import datetime


# Jours fériés (déjà fournis par vous) — gardés ici
JOURS_FERIES = pd.to_datetime([
# 2024
"2024-01-01","2024-04-01","2024-04-04","2024-04-10","2024-05-01","2024-05-09",
"2024-05-20","2024-06-16","2024-07-16","2024-08-15","2024-08-22","2024-09-15",
"2024-11-01","2024-12-25",
# 2025
"2025-01-01","2025-03-31","2025-04-04","2025-04-21","2025-05-01","2025-05-29",
"2025-06-06","2025-06-09","2025-07-05","2025-08-13","2025-08-15","2025-09-05",
"2025-11-01","2025-12-25",
# 2026
"2026-01-01","2026-03-20","2026-04-04","2026-04-06","2026-05-01","2026-05-14",
"2026-05-25","2026-05-27","2026-06-25","2026-08-01","2026-08-15","2026-08-25",
"2026-11-01","2026-12-25",
# 2027
"2027-01-01","2027-03-09","2027-03-29","2027-04-04","2027-05-01","2027-05-06",
"2027-05-16","2027-05-17","2027-06-15","2027-07-23","2027-08-14","2027-08-15",
"2027-11-01"
])


# Ramadan approximatif (gardé)
RAMADAN_DATES = {
2024: pd.date_range("2024-03-11", "2024-04-09"),
2025: pd.date_range("2025-03-01", "2025-03-30"),
2026: pd.date_range("2026-02-18", "2026-03-19"),
2027: pd.date_range("2027-02-07", "2027-03-08")
}




def assigner_saison(mois):
if mois in [1, 2, 3, 12]:
return "Bas"
elif mois in [4,5,6,11]:
return "Transition"
elif mois in [7,8,9,10]:
return "Hautes"
else:
return "Inconnu"




def corriger_precipitation(df, datetime_col='Datetime'):
df = df.copy()
df[datetime_col] = pd.to_datetime(df[datetime_col])
df["month"] = df[datetime_col].dt.month
df.loc[df["month"].isin([11,12,1,2,3,4,5]), "precipitation"] = 0.0
return df




def extract_hour(col):
# gère object, int, datetime
if pd.api.types.is_object_dtype(col.dtype):
return pd.to_datetime(col, errors='coerce').dt.hour.fillna(0).astype(int)
elif np.issubdtype(col.dtype, np.integer):
return col
else:
return pd.to_datetime(col, errors='coerce').apply(lambda x: x.hour if not pd.isna(x) else 0)




def facteur_saison(temp_diff, saison):
if saison == "Bas":
coef = 0.03
elif saison == "Transition":
coef = 0.035
else:
coef = 0.05
return 1 + coef * temp_diff




def facteur_ramadan(date):
date = pd.to_datetime(date)
return df