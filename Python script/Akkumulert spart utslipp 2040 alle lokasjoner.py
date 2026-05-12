# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.iotools import read_epw
from pvlib.iam import martin_ruiz, martin_ruiz_diffuse

t0 = time.perf_counter()

#==============================================================================
#INNSTILLINGER

#Lokasjoner med optimale orienteringer fra kapittel 3 i rapporten
LOCATIONS = [
    {"navn": "Oslo",      "tilt": 49, "azimuth": 181},
    {"navn": "Trondheim", "tilt": 46, "azimuth": 179},
    {"navn": "Tromsø",    "tilt": 50, "azimuth": 176},
    {"navn": "Paris",     "tilt": 39, "azimuth": 177},
    {"navn": "Rome",      "tilt": 38, "azimuth": 169},
]

#PV-parametere (PVWatts DC-modell)
pdc0 = 200            # Nominell effekt under STC [Wp]
gamma_pdc = -0.004    # Temperaturkoeffisient [1/°C]
temp_ref = 25.0       # Referansetemperatur [°C]
a_r = 0.16            # IAM-koeffisient for standard antirefleks-glass

#True for Perez beregning, False for default
perez = True
tz = "Europe/Oslo"

#Analyseperiode
start_year = 2025
end_year = 2039

#Filstier
forecast_csv = r'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\forecast_hourly_2040.csv'
epw_template = r"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/raadata/Klimadata for bygninger/TMYNO_{sted}_CERRA_1991-2020.epw"


#==============================================================================
#LAST INN FORECAST

forecast_df_raw = pd.read_csv(forecast_csv, index_col=0, parse_dates=True)
forecast_df_raw.columns = ['utslipp']

#Lokaliser eller konverter tidssone
if forecast_df_raw.index.tz is None:
    forecast_df_raw.index = forecast_df_raw.index.tz_localize(
        tz, nonexistent='shift_forward', ambiguous='NaT')
else:
    forecast_df_raw.index = forecast_df_raw.index.tz_convert(tz)
forecast_df_raw = forecast_df_raw[forecast_df_raw.index.notna()]

#Fjern duplikater fra sommertidsovergangen
n_dupes = forecast_df_raw.index.duplicated().sum()
if n_dupes > 0:
    print(f"Fjernet {n_dupes} duplikate tidsstempler (DST-overgang)")
    forecast_df_raw = forecast_df_raw[~forecast_df_raw.index.duplicated(keep='first')]

print(f"Forecast: {len(forecast_df_raw)} timer")
print(f"Range: {forecast_df_raw.index[0]} to {forecast_df_raw.index[-1]}")

#Bruk forecast slik den er, trenden er allerede satt 
forecast_df = forecast_df_raw.copy()

#Skriv ut årsmidler for verifikasjon av at definerte trenden er tisltede
print("\nÅrsmidler i forecast:")
annual_means_fc = forecast_df.groupby(forecast_df.index.year)['utslipp'].mean()
for y in sorted(annual_means_fc.index):
    print(f"  {y}: {annual_means_fc.loc[y]:.1f} gCO2/kWh")


#==============================================================================
#PV-BEREGNING FOR EN LOKASJON

def beregn_pv_lokasjon(sted, tilt_opt, azimuth_opt):
    """Kjør PV-CO2 beregning 2025-2039 for én lokasjon. Returnerer DataFrame."""
    print(f"\n--- {sted} (tilt={tilt_opt}°, az={azimuth_opt}°) ---")

    #Last EPW-værdata
    epw_file = epw_template.format(sted=sted)
    try:
        weather_data, meta = read_epw(epw_file, coerce_year=None)
    except UnicodeDecodeError:
        #Fallback til cp1252-encoding for EPW-filer med Windows-tegnsett
        from pvlib.iotools.epw import _parse_epw
        with open(epw_file, 'r', encoding='cp1252') as f:
            weather_data, meta = _parse_epw(f, coerce_year=None)
        print(f"  Note: leste {sted} EPW med cp1252-encoding")

    #Lokaliser eller konverter tidssone
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        weather_data.index = weather_data.index.tz_convert(tz)
    weather_data = weather_data[weather_data.index.notna()]

    #Stats fra lokasjon
    location = pvlib.location.Location(
        latitude=float(meta["latitude"]),
        longitude=float(meta["longitude"]),
        tz=tz,
        altitude=float(meta["altitude"]))

    #Termiske parametere for SAPM-modellen, frittstående glass-glass-paneler
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    yearly_results = []

    #Hovedløkke, kjøres for hvert år 2025-2039
    for year in range(start_year, end_year + 1):

        #Remap TMY-vær til dette året
        idx = weather_data.index
        new_naive = pd.to_datetime({
            "year": np.full(len(idx), year),
            "month": idx.month,
            "day": idx.day,
            "hour": idx.hour,
            "minute": idx.minute,
            "second": idx.second,
        }, errors="coerce")

        mask = ~pd.isna(new_naive.to_numpy())
        weather_year = weather_data.iloc[mask].copy()
        weather_year.index = pd.DatetimeIndex(new_naive[mask]).tz_localize(
            tz, nonexistent="shift_forward", ambiguous="NaT")
        weather_year = weather_year[weather_year.index.notna()]

        #Begrens til ett kalenderår
        start_ts = pd.Timestamp(f"{year}-01-01", tz=tz)
        slutt_ts = pd.Timestamp(f"{year+1}-01-01", tz=tz)
        weather_year = weather_year[(weather_year.index >= start_ts) & (weather_year.index < slutt_ts)]

        #Utslippsverdi for hver time, fyll evt. NaN med årssnittet
        utslipp = forecast_df['utslipp'].reindex(weather_year.index)
        utslipp = utslipp.fillna(annual_means_fc.loc[year])

        #Setter statisk utslippsfaktor lik årsgjennomsnittet av den samme dynamiske profilen
        utslipp_static = utslipp.mean()

        #Solposisjon og hjelpestørrelser
        solpos = location.get_solarposition(weather_year.index)
        dni_extra = pvlib.irradiance.get_extra_radiation(weather_year.index)
        airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

        if perez:
            total_irr = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt_opt, surface_azimuth=azimuth_opt,
                dni=weather_year["dni"], ghi=weather_year["ghi"], dhi=weather_year["dhi"],
                solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
                dni_extra=dni_extra, airmass=airmass,
                model="perez", model_perez="allsitescomposite1990")
        else:
            total_irr = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt_opt, surface_azimuth=azimuth_opt,
                dni=weather_year["dni"], ghi=weather_year["ghi"], dhi=weather_year["dhi"],
                solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"])

        #Innfallsvinkel mellom direkte stråling og panel
        aoi = pvlib.irradiance.aoi(tilt_opt, azimuth_opt, solpos["apparent_zenith"], solpos["azimuth"])

        #Incidence Angle Modifier
        iam_beam = martin_ruiz(aoi, a_r=a_r)
        iam_sky, iam_gnd = martin_ruiz_diffuse(tilt_opt, a_r=a_r)

        #Effektiv innstråling
        effective_irr = (
            iam_beam * total_irr["poa_direct"].clip(lower=0) +
            iam_sky * total_irr["poa_sky_diffuse"].clip(lower=0) +
            iam_gnd * total_irr["poa_ground_diffuse"].clip(lower=0))

        #Temperatur på panel
        temp_cell = pvlib.temperature.sapm_cell(
            poa_global=total_irr["poa_global"],
            temp_air=weather_year["temp_air"],
            wind_speed=weather_year["wind_speed"],
            **temp_params)

        #Produsert DC
        #Tap fra vekselretter er ikke modellert
        pdc = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_irr, temp_cell=temp_cell,
            pdc0=pdc0, gamma_pdc=gamma_pdc, temp_ref=temp_ref)

        #Normaliser til per kWp installert effekt
        prod_per_wp = pdc / pdc0

        #Spart utslipp. PV-produksjon*utslippsintensitet, summert over året
        co2_dyn = (prod_per_wp * utslipp / 1000).sum()
        co2_stat = (prod_per_wp * utslipp_static / 1000).sum()
        prod_total = prod_per_wp.sum()

        yearly_results.append({
            'year': year,
            'prod_kWh_per_kWp': prod_total,
            'co2_dyn_kg_per_kWp': co2_dyn,
            'co2_stat_kg_per_kWp': co2_stat,
            'mean_utslipp': utslipp_static
        })

    #DataFrame for alle resultatene + kumulativ og prosentvis forskjell
    df = pd.DataFrame(yearly_results).set_index('year')
    df['cum_co2_dyn'] = df['co2_dyn_kg_per_kWp'].cumsum()
    df['cum_co2_stat'] = df['co2_stat_kg_per_kWp'].cumsum()
    df['cum_diff'] = df['cum_co2_stat'] - df['cum_co2_dyn']
    df['yearly_diff_pct'] = np.where(
        df['co2_dyn_kg_per_kWp'] > 0.01,
        (df['co2_stat_kg_per_kWp'] - df['co2_dyn_kg_per_kWp']) / df['co2_dyn_kg_per_kWp'] * 100,
        np.nan
    )

    print(f"  Sum dyn: {df['co2_dyn_kg_per_kWp'].sum():.1f} kg | "
          f"Sum stat: {df['co2_stat_kg_per_kWp'].sum():.1f} kg | "
          f"Diff: {df['cum_diff'].iloc[-1]:.1f} kg")

    return df


#==============================================================================
#KJØR FOR ALLE LOKASJONER


all_results = {}
for loc in LOCATIONS:
    all_results[loc["navn"]] = beregn_pv_lokasjon(
        loc["navn"], loc["tilt"], loc["azimuth"]
    )

#Logger kjøretid
elapsed = time.perf_counter() - t0
minutes = int(elapsed // 60)
seconds = elapsed % 60
print(f"\nTotal kjøretid: {minutes}min {seconds:.0f}s")

#Lagre resultater per lokasjon
for navn, df in all_results.items():
    df.to_csv(
        rf'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\pv_co2_results_2025_2039_{navn}.csv'
    )



#==============================================================================
#FELLES AKSESKALA

#Finn felles y-grenser på tvers av alle lokasjoner så figurene blir direkte sammenlignbare
all_dfs = list(all_results.values())

ymax_yearly = max(
    max(df['co2_dyn_kg_per_kWp'].max(), df['co2_stat_kg_per_kWp'].max())
    for df in all_dfs
) * 1.10

ymax_cum = max(
    max(df['cum_co2_dyn'].max(), df['cum_co2_stat'].max())
    for df in all_dfs
) * 1.05

ymax_diff = max(df['yearly_diff_pct'].max() for df in all_dfs) * 1.10
ymin_diff = min(0, min(df['yearly_diff_pct'].min() for df in all_dfs) * 1.10)

print("\n=== Felles akseskala ===")
print(f"Plot 1 (årlig):     0 → {ymax_yearly:.0f} kgCO2/kWp")
print(f"Plot 2 (kumulativ): 0 → {ymax_cum:.0f} kgCO2/kWp")
print(f"Plot 3 (diff %):    {ymin_diff:.1f} → {ymax_diff:.1f} %")


#==============================================================================
#PLOT EN FIGUR PER LOKASJON MED FELLES AKSESKALA

for navn, df in all_results.items():
    loc_settings = next(loc for loc in LOCATIONS if loc["navn"] == navn)
    tilt_opt = loc_settings["tilt"]
    azimuth_opt = loc_settings["azimuth"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    #Plot 1: Årlig CO2-offset (statisk vs dynamisk side om side)
    axes[0].bar(df.index - 0.2, df['co2_dyn_kg_per_kWp'],
                width=0.4, label='Dynamisk', color='#3b82f6')
    axes[0].bar(df.index + 0.2, df['co2_stat_kg_per_kWp'],
                width=0.4, label='Statisk', color='red', alpha=0.6)
    axes[0].set_ylabel('kgCO₂/kWp·år')
    axes[0].set_title(f'Årlig CO₂-offset – {navn} (tilt={tilt_opt}°, azimuth={azimuth_opt}°)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, ymax_yearly)

    #Plot 2: Kumulativ CO2-offset
    axes[1].plot(df.index, df['cum_co2_dyn'],
                 'o-', label='Kumulativ dynamisk', color='#3b82f6', linewidth=2)
    axes[1].plot(df.index, df['cum_co2_stat'],
                 's--', label='Kumulativ statisk', color='red', alpha=0.6, linewidth=2)
    axes[1].set_ylabel('Kumulativ kgCO₂/kWp')
    axes[1].set_title(f'Kumulativ CO₂-offset 2025–2039 – {navn}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, ymax_cum)
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(500))

    #Plot 3: Årlig prosentvis forskjell (positiv = statisk overestimerer)
    axes[2].bar(df.index, df['yearly_diff_pct'], color='#10b981')
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].set_ylabel('Forskjell (stat−dyn) / dyn [%]')
    axes[2].set_xlabel('År')
    axes[2].set_title(f'Årlig forskjell mellom statisk og dynamisk – {navn}')
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(ymin_diff, ymax_diff)
    axes[2].yaxis.set_major_locator(plt.MultipleLocator(5))

    axes[2].set_xlim(start_year - 0.5, end_year + 0.5)

    plt.tight_layout()
    plt.savefig(
        rf'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\pv_co2_{navn}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()

print("\nFerdig!")

#==============================================================================
#INTERAKTIVE PLOTLY-FIGURER (én HTML per lokasjon)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

for navn, df in all_results.items():
    loc_settings = next(loc for loc in LOCATIONS if loc["navn"] == navn)
    tilt_opt = loc_settings["tilt"]
    azimuth_opt = loc_settings["azimuth"]

    fig_pv = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            f'Årlig CO₂-offset – {navn} (tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
            f'Kumulativ CO₂-offset {start_year}–{end_year} – {navn}',
            f'Årlig forskjell mellom statisk og dynamisk – {navn}'
        ],
        vertical_spacing=0.08
    )

    #Rad 1: Årlig CO2-offset (statisk vs dynamisk)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_dyn_kg_per_kWp'],
        name='Dynamisk', marker_color='#3b82f6'
    ), row=1, col=1)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_stat_kg_per_kWp'],
        name='Statisk', marker_color='red', opacity=0.6
    ), row=1, col=1)

    #Rad 2: Kumulativ CO2-offset
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_dyn'],
        name='Kumulativ dynamisk',
        line=dict(color='#3b82f6', width=2),
        mode='lines+markers'
    ), row=2, col=1)
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_stat'],
        name='Kumulativ statisk',
        line=dict(color='red', width=2, dash='dash'),
        mode='lines+markers', opacity=0.6
    ), row=2, col=1)

    #Rad 3: Årlig prosentvis forskjell
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['yearly_diff_pct'],
        name='Forskjell %', marker_color='#10b981',
        showlegend=False
    ), row=3, col=1)

    fig_pv.update_layout(
        title=f'PV CO₂-offset {start_year}–{end_year} – {navn} '
              f'(tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
        template='plotly_white',
        height=1000,
        barmode='group',
        hovermode='x unified',
        showlegend=True
    )

    #Bruker felles y-akser på tvers av lokasjoner for direkte sammenligning
    fig_pv.update_yaxes(title_text='kgCO₂/kWp·år',
                        range=[0, ymax_yearly], row=1, col=1)
    fig_pv.update_yaxes(title_text='Kumulativ kgCO₂/kWp',
                        range=[0, ymax_cum], row=2, col=1)
    fig_pv.update_yaxes(title_text='Forskjell (stat−dyn) / dyn [%]',
                        range=[ymin_diff, ymax_diff], row=3, col=1)
    fig_pv.update_xaxes(title_text='År', row=3, col=1)

    html_path = rf'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\pv_co2_{navn}.html'
    fig_pv.write_html(html_path, auto_open=False)
    print(f"  HTML lagret: {html_path}")

print("\nInteraktive HTML-figurer ferdig!")#==============================================================================
#INTERAKTIVE PLOTLY-FIGURER (én HTML per lokasjon)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

for navn, df in all_results.items():
    loc_settings = next(loc for loc in LOCATIONS if loc["navn"] == navn)
    tilt_opt = loc_settings["tilt"]
    azimuth_opt = loc_settings["azimuth"]

    fig_pv = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            f'Årlig CO₂-offset – {navn} (tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
            f'Kumulativ CO₂-offset {start_year}–{end_year} – {navn}',
            f'Årlig forskjell mellom statisk og dynamisk – {navn}'
        ],
        vertical_spacing=0.08
    )

    #Rad 1: Årlig CO2-offset (statisk vs dynamisk)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_dyn_kg_per_kWp'],
        name='Dynamisk', marker_color='#3b82f6'
    ), row=1, col=1)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_stat_kg_per_kWp'],
        name='Statisk', marker_color='red', opacity=0.6
    ), row=1, col=1)

    #Rad 2: Kumulativ CO2-offset
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_dyn'],
        name='Kumulativ dynamisk',
        line=dict(color='#3b82f6', width=2),
        mode='lines+markers'
    ), row=2, col=1)
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_stat'],
        name='Kumulativ statisk',
        line=dict(color='red', width=2, dash='dash'),
        mode='lines+markers', opacity=0.6
    ), row=2, col=1)

    #Rad 3: Årlig prosentvis forskjell
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['yearly_diff_pct'],
        name='Forskjell %', marker_color='#10b981',
        showlegend=False
    ), row=3, col=1)

    fig_pv.update_layout(
        title=f'PV CO₂-offset {start_year}–{end_year} – {navn} '
              f'(tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
        template='plotly_white',
        height=1000,
        barmode='group',
        hovermode='x unified',
        showlegend=True
    )

    #Bruker felles y-akser på tvers av lokasjoner for direkte sammenligning
    fig_pv.update_yaxes(title_text='kgCO₂/kWp·år',
                        range=[0, ymax_yearly], row=1, col=1)
    fig_pv.update_yaxes(title_text='Kumulativ kgCO₂/kWp',
                        range=[0, ymax_cum], row=2, col=1)
    fig_pv.update_yaxes(title_text='Forskjell (stat−dyn) / dyn [%]',
                        range=[ymin_diff, ymax_diff], row=3, col=1)
    fig_pv.update_xaxes(title_text='År', row=3, col=1)

    html_path = rf'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\pv_co2_{navn}.html'
    fig_pv.write_html(html_path, auto_open=False)
    print(f"  HTML lagret: {html_path}")

print("\nInteraktive HTML-figurer ferdig!")

#%%
#==============================================================================
#INTERAKTIVE PLOTLY-FIGURER (én HTML per lokasjon)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

for navn, df in all_results.items():
    loc_settings = next(loc for loc in LOCATIONS if loc["navn"] == navn)
    tilt_opt = loc_settings["tilt"]
    azimuth_opt = loc_settings["azimuth"]

    fig_pv = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            f'Årlig CO₂-offset – {navn} (tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
            f'Kumulativ CO₂-offset {start_year}–{end_year} – {navn}',
            f'Årlig forskjell mellom statisk og dynamisk – {navn}'
        ],
        vertical_spacing=0.08
    )

    #Rad 1: Årlig CO2-offset (statisk vs dynamisk)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_dyn_kg_per_kWp'],
        name='Dynamisk', marker_color='#3b82f6'
    ), row=1, col=1)
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['co2_stat_kg_per_kWp'],
        name='Statisk', marker_color='red', opacity=0.6
    ), row=1, col=1)

    #Rad 2: Kumulativ CO2-offset
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_dyn'],
        name='Kumulativ dynamisk',
        line=dict(color='#3b82f6', width=2),
        mode='lines+markers'
    ), row=2, col=1)
    fig_pv.add_trace(go.Scatter(
        x=df.index, y=df['cum_co2_stat'],
        name='Kumulativ statisk',
        line=dict(color='red', width=2, dash='dash'),
        mode='lines+markers', opacity=0.6
    ), row=2, col=1)

    #Rad 3: Årlig prosentvis forskjell
    fig_pv.add_trace(go.Bar(
        x=df.index, y=df['yearly_diff_pct'],
        name='Forskjell %', marker_color='#10b981',
        showlegend=False
    ), row=3, col=1)

    fig_pv.update_layout(
        title=f'PV CO₂-offset {start_year}–{end_year} – {navn} '
              f'(tilt={tilt_opt}°, azimuth={azimuth_opt}°)',
        template='plotly_white',
        height=1000,
        barmode='group',
        hovermode='x unified',
        showlegend=True
    )

    #Bruker felles y-akser på tvers av lokasjoner for direkte sammenligning
    fig_pv.update_yaxes(title_text='kgCO₂/kWp·år',
                        range=[0, ymax_yearly], row=1, col=1)
    fig_pv.update_yaxes(title_text='Kumulativ kgCO₂/kWp',
                        range=[0, ymax_cum], row=2, col=1)
    fig_pv.update_yaxes(title_text='Forskjell (stat−dyn) / dyn [%]',
                        range=[ymin_diff, ymax_diff], row=3, col=1)
    fig_pv.update_xaxes(title_text='År', row=3, col=1)

    html_path = rf'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\Interaktiv\pv_co2_{navn}.html'
    fig_pv.write_html(html_path, auto_open=False)
    print(f"  HTML lagret: {html_path}")

print("\nInteraktive HTML-figurer ferdig!")