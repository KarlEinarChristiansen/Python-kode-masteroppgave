# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- INNDATA ----------------------------------------------------------
utslipps_csv = r"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/Raadata/Alle_timer_2015-2024_utslippsintensitet.csv"
start_year, end_year = 2015, 2024

# ---- LAST INN ---------------------------------------------------------
df = pd.read_csv(utslipps_csv)
df["Dato"] = pd.to_datetime({
    "year": df["Aar"], "month": df["Maaned"],
    "day":  df["Dag"], "hour":  df["time"]})
df = df[(df["Aar"] >= start_year) & (df["Aar"] <= end_year)].copy()
df = df.set_index("Dato").sort_index()

df["month"] = df.index.month
df["day"]   = df.index.day
df["hour"]  = df.index.hour

profil = (df.groupby(["month", "day", "hour"], as_index=False)
            ["Utslippsintensitet"].mean())

basis = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
basis_df = pd.DataFrame({
    "month": basis.month, "day": basis.day, "hour": basis.hour})
basis_df = basis_df.merge(profil, on=["month","day","hour"], how="left")
basis_df.index = basis

statisk = float(basis_df["Utslippsintensitet"].mean())

# Splitt i tre tertialer
del_1 = basis_df.loc["2023-01-01":"2023-04-30 23:00"]
del_2 = basis_df.loc["2023-05-01":"2023-08-31 23:00"]
del_3 = basis_df.loc["2023-09-01":"2023-12-31 23:00"]

# ---- PLOT ------------------------------------------------------------
maaned_navn = ["Jan","Feb","Mar","Apr","Mai","Jun",
               "Jul","Aug","Sep","Okt","Nov","Des"]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharey=True)

for ax, halv, tittel in [
    (axes[0], del_1, "Januar – April"),
    (axes[1], del_2, "Mai – August"),
    (axes[2], del_3, "September – Desember"),
]:
    ax.plot(halv.index, halv["Utslippsintensitet"],
            color="tab:blue", lw=0.8,
            label="Dynamisk (timesoppløst, snitt 2015–2024)")
    ax.axhline(statisk, color="tab:red", ls="--", lw=1.5,
               label=f"Statisk = {statisk:.0f} gCO$_2$-eq/kWh")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(""))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(plt.FuncFormatter(
        lambda x, pos=None: maaned_navn[mdates.num2date(x).month - 1]))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", length=4)

    ax.set_ylabel("gCO$_2$-eq/kWh")
    ax.set_title(tittel)
    ax.grid(alpha=0.3)
    ax.margins(x=0.01)

# Felles y-grense med litt luft til legenden
y_top = max(del_1["Utslippsintensitet"].max(),
            del_2["Utslippsintensitet"].max(),
            del_3["Utslippsintensitet"].max())
y_bot = min(del_1["Utslippsintensitet"].min(),
            del_2["Utslippsintensitet"].min(),
            del_3["Utslippsintensitet"].min())
for ax in axes:
    ax.set_ylim(y_bot - 5, y_top + 30)

axes[0].legend(loc="upper right", framealpha=0.9)
axes[2].set_xlabel("Måned")

fig.suptitle(f"Europeisk utslippsintensitet, gjennomsnitt "
             f"({start_year}-{end_year})", y=0.995)

plt.tight_layout()
plt.savefig("aarsprofil_tre_deler.png", dpi=200, bbox_inches="tight")
plt.show()