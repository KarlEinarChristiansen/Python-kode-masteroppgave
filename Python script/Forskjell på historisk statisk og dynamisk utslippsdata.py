# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#Inndata
utslipps_csv = r"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/Raadata/Alle_timer_2015-2024_utslippsintensitet.csv"
start_year, end_year = 2015, 2024

#Last inn
df = pd.read_csv(utslipps_csv)
df["Dato"] = pd.to_datetime({
    "year": df["Aar"], "month": df["Maaned"],
    "day":  df["Dag"], "hour":  df["time"]})
df = df[(df["Aar"] >= start_year) & (df["Aar"] <= end_year)].copy()
df = df.set_index("Dato").sort_index()

# Typisk årsprofil: snitt per måned, dag og time over alle år
df["month"] = df.index.month
df["day"]   = df.index.day
df["hour"]  = df.index.hour

profil = (df.groupby(["month", "day", "hour"], as_index=False)
            ["Utslippsintensitet"].mean())

basis = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
basis_df = pd.DataFrame({
    "month": basis.month, "day": basis.day, "hour": basis.hour,
    "doy": basis.dayofyear})
basis_df = basis_df.merge(profil, on=["month","day","hour"], how="left")

# Beregning
dynamisk = basis_df["Utslippsintensitet"].values
statisk  = float(np.mean(dynamisk))
avvik_pct = (statisk - dynamisk) / dynamisk * 100.0

# Matrise: time × dag-i-året
mat = np.full((24, 365), np.nan)
for h, d, v in zip(basis_df["hour"], basis_df["doy"], avvik_pct):
    mat[h, d-1] = v

#Plotter
vmax = float(np.nanmax(np.abs(mat)))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(mat, aspect="auto", origin="lower",
               cmap="RdBu_r", norm=norm,
               extent=[1, 365, 0, 24], interpolation="nearest")

maaned_start = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
maaned_midt  = [(maaned_start[i] + maaned_start[i+1]) / 2 for i in range(12)]
maaned_navn  = ["Jan","Feb","Mar","Apr","Mai","Jun",
                "Jul","Aug","Sep","Okt","Nov","Des"]
ax.set_xticks(maaned_midt); ax.set_xticklabels(maaned_navn)
ax.set_yticks(np.arange(0, 25, 3))
ax.set_xlabel("Måned")
ax.set_ylabel("Time på døgnet")
ax.set_title(f"Avvik mellom statisk og dynamisk utslippsintensitet "
             f"({start_year}-{end_year}, snitt = {statisk:.0f} gCO$_2$-eq/kWh)")

cb = plt.colorbar(im, ax=ax, pad=0.02)
cb.set_label("(Statisk − dynamisk) / dynamisk  [%]")

plt.tight_layout()
plt.savefig("heatmap_avvik.png", dpi=200, bbox_inches="tight")
plt.show()


