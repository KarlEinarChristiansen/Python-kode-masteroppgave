import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#Last inn data
utslipps_csv = r"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/Raadata/Alle_timer_2015-2024_utslippsintensitet.csv"
df = pd.read_csv(utslipps_csv)

df["Dato"] = pd.to_datetime({
    "year":  df["Aar"],
    "month": df["Maaned"],
    "day":   df["Dag"],
    "hour":  df["time"]
})
df = df.set_index("Dato").sort_index()

#Typisk døgnprofil per måned
maaneder = ["Jan","Feb","Mar","Apr","Mai","Jun",
            "Jul","Aug","Sep","Okt","Nov","Des"]

fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)

for m, ax in zip(range(1, 13), axes.flat):
    serie = df.loc[df.index.month == m, "Utslippsintensitet"]
    dekomp = seasonal_decompose(serie, model="additive", period=24)
    doegnprofil = dekomp.seasonal.groupby(dekomp.seasonal.index.hour).mean()

    ax.plot(doegnprofil.index, doegnprofil.values)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(maaneder[m-1])
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Time")

axes[1, 0].set_ylabel("g CO₂-eq/kWh avvik")
fig.suptitle("Typiske døgnprofiler per måned (2015–2024)", fontsize=16)
plt.tight_layout()
plt.show()