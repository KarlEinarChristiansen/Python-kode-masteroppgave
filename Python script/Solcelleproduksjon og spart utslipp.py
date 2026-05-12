# -*- coding: utf-8 -*-
import pvlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time; t0 = time.perf_counter()

from pvlib.iam import martin_ruiz, martin_ruiz_diffuse
from matplotlib.ticker import FormatStrFormatter  
from pvlib.iotools import read_epw


#Epw filer til lokasjonene må være lastet opp
Steder = ["Oslo", "Trondheim", "Tromsø", "Paris", "Rome"]


for sted in Steder:
    """
    Kjører for alle lokasjoner definert over
    """
    
    Sted = sted
    print(f"{Sted}...")
    
    #Brukes kun for å mappe TMY-år til et år
    År = 2015 
    
    #Hvilke orientering og helningsvinkler som skal kjøres
    stepsize = 10
    azimuths = np.arange(0, 360, stepsize) 
    tilts = np.arange(0, 90, stepsize)
    
    #True for Perez beregning, False for default
    perez = True 
    
    #Inndata
    #-------------------------------------------------
    pdc0 = 200           # Nominell effekt [Wp]
    gamma_pdc = -0.004   # Temperaturkoeffisient [1/°C]
    temp_ref = 25.0      # Referansetemperatur [°C]
    a_r = 0.16           # Koeffisient for vinkeltap
    
    start_year = 2015
    end_year   = 2024
    År_label = f"{start_year}-{end_year}"
    #-------------------------------------------------
    
    #Hvor figurer skal lagres
    save_dir = rf"C:\Users\Eier\OneDrive\Skrivebord\master 2026\SEBORTIFRA\{Sted}"
    os.makedirs(save_dir, exist_ok=True)
    
    #TMY filnavn må hvertfall inneholde {Sted}, anbefalt: TMYNO_{Sted}_CERRA_1991-2020.epw
    epw_file = rf"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/raadata/Klimadata for bygninger/TMYNO_{Sted}_CERRA_1991-2020.epw"
    utslipps_csv = r"C:/Users/Eier/OneDrive/Skrivebord/master 2026/Python/Data/Raadata/Alle_timer_2015-2024_utslippsintensitet.csv"
    utslippsfil = pd.read_csv(utslipps_csv)
    
    if not os.path.exists(epw_file):
        raise FileNotFoundError(f"EPW file not found: {epw_file}")
    
    #latin-1 er tilstrekkelig for de fleste EPW-filer
    with open(epw_file, "r", encoding="latin-1") as f:
        weather_data, meta = read_epw(f, coerce_year=None)
    
    tz = "Europe/Oslo"
    
    # Lokaliser eller konverter tidssone
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        weather_data.index = weather_data.index.tz_convert(tz)
    weather_data = weather_data[weather_data.index.notna()]
    
    #Alle til samme år i TMY
    idx = weather_data.index
    new_naive = pd.to_datetime(
        {
            "year":   np.full(len(idx), År),
            "month":  idx.month,
            "day":    idx.day,
            "hour":   idx.hour,
            "minute": idx.minute,
            "second": idx.second,
        }, errors="coerce")
    mask = ~pd.isna(new_naive.to_numpy())
    weather_data = weather_data.iloc[mask].copy()
    weather_data.index = pd.DatetimeIndex(new_naive[mask]).tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    weather_data = weather_data[weather_data.index.notna()]

    #Begrens til det nye året        
    start = pd.Timestamp(f"{År}-01-01", tz=tz)
    slutt = pd.Timestamp(f"{År+1}-01-01", tz=tz)
    
    weather_data_lim = weather_data[(weather_data.index >= start) & (weather_data.index < slutt)]
    
    utslippsfil["month"] = utslippsfil["Maaned"]
    utslippsfil["day"]   = utslippsfil["Dag"]
    utslippsfil["hour"]  = utslippsfil["time"]
    
    utslipp_years = utslippsfil[(utslippsfil["Aar"] >= start_year) & (utslippsfil["Aar"] <= end_year)].copy()
    utslipp_profile = (utslipp_years.groupby(["month", "day", "hour"], as_index=False)["Utslippsintensitet"].mean()) 
    utslipp_profile = utslipp_profile.set_index(["month", "day", "hour"])["Utslippsintensitet"]
    
    #Stats fra lokasjon
    location = pvlib.location.Location(
        latitude=float(meta["latitude"]),
        longitude=float(meta["longitude"]),
        tz=tz,
        altitude=float(meta["altitude"]))
    
    results = []
    ratio_ts = {}
   
    weather_year = weather_data_lim.copy()
    
    #solposisjon
    solpos = location.get_solarposition(weather_year.index)
    
    #Slå opp utslippsverdi for hver time 
    keys = list(zip(weather_year.index.month, weather_year.index.day, weather_year.index.hour))
    utslipp = pd.Series(utslipp_profile.reindex(keys).to_numpy(), index=weather_year.index)
    
    #returnerer NaN der det ikke finnes match, fylles med årssnittet som fallback
    if utslipp.isna().any():
        utslipp = utslipp.fillna(utslipp_profile.mean())
    
    #Termiske parametere for SAPM-modellen, frittstående glass-glass-paneler
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    dni_extra = pvlib.irradiance.get_extra_radiation(weather_year.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    
    #Hovedløkke, kjøres for alle tilts og asiumer definert over
    for tilt in tilts:
        for azimuth in azimuths:
            if perez:
                total_irradiance = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=azimuth,
                    dni=weather_year["dni"],
                    ghi=weather_year["ghi"],
                    dhi=weather_year["dhi"],
                    solar_zenith=solpos["apparent_zenith"],
                    solar_azimuth=solpos["azimuth"],
                    dni_extra=dni_extra,
                    airmass=airmass,
                    model="perez",
                    model_perez="allsitescomposite1990")
            else:
                total_irradiance = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=azimuth,
                    dni=weather_year["dni"],
                    ghi=weather_year["ghi"],
                    dhi=weather_year["dhi"],
                    solar_zenith=solpos["apparent_zenith"],
                    solar_azimuth=solpos["azimuth"])
                
            #Innfallsvinkel mellom direkte stråling og panel
            aoi = pvlib.irradiance.aoi(
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                solar_zenith=solpos["apparent_zenith"],
                solar_azimuth=solpos["azimuth"])
            
            #Incidence Angle Modifier
            iam_beam = martin_ruiz(aoi, a_r=a_r)
            iam_sky, iam_gnd = martin_ruiz_diffuse(tilt, a_r=a_r)
            
            #Effektiv innstråling
            effective_irradiance = (
                iam_beam * total_irradiance["poa_direct"].clip(lower=0) +
                iam_sky  * total_irradiance["poa_sky_diffuse"].clip(lower=0) +
                iam_gnd  * total_irradiance["poa_ground_diffuse"].clip(lower=0))
            
            #Temperatur på panel
            temp_cell = pvlib.temperature.sapm_cell(
                poa_global=total_irradiance["poa_global"],
                temp_air=weather_year["temp_air"],
                wind_speed=weather_year["wind_speed"],
                **temp_params)
            
            #Produsert DC 
            #Tap fra vekselretter er ikke modellert
            pdc = pvlib.pvsystem.pvwatts_dc(
                effective_irradiance=effective_irradiance,
                temp_cell=temp_cell,
                pdc0=pdc0,
                gamma_pdc=gamma_pdc,
                temp_ref=temp_ref)
    
    
            #Konverter til kWh og normaliser til per kWp installert effekt
            total_irradiance["Prod"] = pdc / 1000
            total_irradiance["Prod_per_Wp"] = total_irradiance["Prod"] / (pdc0/1000)
    
    
            #Setter statisk utslippsfaktor lik årsgjennomsnittet av den samme dynamiske profilen
            total_irradiance["Utslippsdata [kg CO2 / MWh]"] = utslipp
            total_irradiance["Utslippsdata statisk"] = utslipp.mean()
            
            #Spart utslipp. PV-produksjon*utslippsintensitet, per time 
            total_irradiance["CO2_POA_DYN"] = total_irradiance["Prod_per_Wp"] * total_irradiance["Utslippsdata [kg CO2 / MWh]"] / 1000
            total_irradiance["CO2_POA_STAT"] = total_irradiance["Prod_per_Wp"] * total_irradiance["Utslippsdata statisk"] / 1000
            
            #Time for time differanse mellom spart utslipp, brukes til varmekartet
            total_irradiance["DYN_STAT_ratio"] = total_irradiance["CO2_POA_STAT"] - total_irradiance["CO2_POA_DYN"]
            
            #Summerer opp årlig
            sum_poa = total_irradiance["poa_global"].sum()
            sum_poa_kWh_per_m2 = sum_poa / 1000.0
            
            sum_prod = total_irradiance["Prod_per_Wp"].sum()
            sum_prod_utsl_dyn = total_irradiance["CO2_POA_DYN"].sum()
            sum_prod_utsl_stat = total_irradiance["CO2_POA_STAT"].sum()
            
            results.append((tilt, azimuth, sum_poa_kWh_per_m2, sum_prod, sum_prod_utsl_dyn, sum_prod_utsl_stat))
            ratio_ts[(tilt, azimuth)] = total_irradiance["DYN_STAT_ratio"].copy()

    #Logger kjøretid
    elapsed = time.perf_counter() - t0
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Total runtime: {minutes}min {seconds:.0f}s")
    
    
    #DataFrame for alle resulatene
    results_df = pd.DataFrame(results,
        columns=["Tilt", "Azimuth", "Sum_POA", "Sum_Prod","sum_CO2POA_Dyn","sum_CO2POA_Stat"]
    )
    
    #Lager pivot tabeller for alle verdiene, asimut og helningsvinkel som kolonner og rader
    pivot_poa = results_df.pivot(index="Tilt", columns="Azimuth", values="Sum_POA")
    pivot_prod = results_df.pivot(index="Tilt", columns="Azimuth", values="Sum_Prod")
    pivot_dynamic = results_df.pivot(index="Tilt", columns="Azimuth", values="sum_CO2POA_Dyn")
    pivot_static = results_df.pivot(index="Tilt", columns="Azimuth", values="sum_CO2POA_Stat")
    
    Z0 = pivot_poa.values
    Z1 = pivot_prod.values
    Z2 = pivot_dynamic.values
    Z3 = pivot_static.values
    
    
    
    
    #Innstråling (POA)
    def plot_POA(pivot_poa, Z0, År_label, Sted, perez, save_dir):
        
        """
        Konturplott av årlig innstråling [kWh/m² per år].

        X-akse: asimut (0–360°), Y-akse: helningsvinkel (0–90°).
        Fargeskala viser akkumulert POA-energi for hver kombinasjon.
        """

    
        #Lager rutenett for konturplottet
        tilts_plot = pivot_poa.index.values
        azis_plot = pivot_poa.columns.values
        X, Y = np.meshgrid(azis_plot, tilts_plot)

        
        plt.figure(figsize=(14, 8))
        
        #Finn høyeste og laveste POA-verdi i matrisen (nanmin/nanmax ignorerer NaN)
        zmin = np.nanmin(Z0)
        zmax = np.nanmax(Z0)
        
        #Steglengde mellom konturlinjer
        step = 100
        
        #Rund minimum nedover til nærmeste hele 100
        vmin = np.floor(zmin/step)*step
        
        #Rund maksimum oppover til nærmeste hele 100
        vmax = np.ceil(zmax/step)*step
        
        ticks = np.arange(vmin,vmax+step,step)
        levels_cf = np.linspace(vmin, vmax, 100)
        
        #Kontur med valgt fargegradient
        cf2 = plt.contourf(X, Y, Z0, levels=levels_cf, cmap="cividis")
        
        #Hvite konturlinjer, med tallverdi langs hver linje
        cs2 = plt.contour(X, Y, Z0,levels=ticks, colors="white", linewidths=2)
        plt.clabel(cs2, inline=True, fontsize=10, fmt="%.0f")
        
        #Fargestolpe med samme ticks som konturlinjene, formatert uten desimaler
        cbar2 = plt.colorbar(cf2, ticks=ticks)
        cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar2.set_label("Årlig POA-energi [kWh/m²·år]")
        
        #Aksetitler
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        #Tittel
        plt.title(
            f"Innstrålt energi på flate (GPOA) i {Sted} "
            f"\n"
            f"POA max: {pivot_poa.stack().max():.0f} kWh/m\u00B2 ved helningsvinkel {pivot_poa.stack().idxmax()[0]}° "
            f"og asimut {pivot_poa.stack().idxmax()[1]}°",
            fontsize=15
        )
        
        #x-akse ticks og lagre figur 
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"POA_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_POA(pivot_poa, Z0, År_label, Sted, perez,save_dir)
    
    
    
    
    #Produksjon DC
    def plot_PROD(pivot_prod, Z1, År_label, Sted, perez, save_dir):
        """
        Konturplott av årlig produsert årlig i alle asimuter og helnigsvinkler
        
        """
        
        #Lager rutenett for konturplottet
        tilts_plot = pivot_prod.index.values
        azis_plot = pivot_prod.columns.values
        X, Y = np.meshgrid(azis_plot, tilts_plot)
    
        
        plt.figure(figsize=(14, 8))
    
        #Høyeste produksjon i datasettet (nanmax ignorerer NaN)
        zmax = np.nanmax(Z1)
        
        #Steglengde mellom konturlinjer
        step = 100
        
        #Rund maksimum oppover til nærmeste hele 100
        vmax = np.ceil(zmax/step)*step
        
        levels_lines = np.arange(0, vmax + step, step)
        levels_cf = np.linspace(0, vmax, 100)
        
        #Kontur med valgt fargegradient
        cf2 = plt.contourf(X, Y, Z1, levels=levels_cf, cmap="cividis")
        
        #Hvite konturlinjer, med tallverdi langs hver linje
        cs2 = plt.contour(X, Y, Z1, levels=levels_lines, colors="white", linewidths=2)
        plt.clabel(cs2, inline=True, fontsize=10, fmt="%.0f")
        
        #Fargestolpe med samme inndeling som konturlinjene
        cbar2 = plt.colorbar(cf2, ticks=levels_lines)
        cbar2.set_label("Årlig produsert [kWh / Wp·år]")
    
        #Aksetittel
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
        
        #Tittel
        plt.title(
            f"P-DC i {Sted} "
            f"\n"
            f"Maks: {pivot_prod.stack().max():.0f} kWh/kWp ved helningsvinkel {pivot_prod.stack().idxmax()[0]}° "
            f"og asimut {pivot_prod.stack().idxmax()[1]}°",
            fontsize=15
        )
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"P_DC_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_PROD(pivot_prod, Z1, År_label, Sted, perez,save_dir)
    
  
    
  
    #C02DC kgCO2/kWp Statisk
    def plot_CO2POA_static(pivot_static, Z2, Z3, År, Sted, perez, save_dir):
        """Konturplott av unngåtte utslipp [kgCO₂-eq/kWp per år] med statisk metode.
    
        Z2 = dynamisk verdi, Z3 = statisk verdi. Begge sendes inn for å bygge
        en felles fargeskala.
        """
    
        #Lager rutenett for konturplottet
        X, Y = np.meshgrid(pivot_static.columns.values, pivot_static.index.values)
    
        plt.figure(figsize=(14, 8))
    
        #Steglengde mellom konturlinjer
        step = 25
    
        #Min/maks finnes på tvers av både statisk og dynamisk datasett,
        #slik får begge figurene samme fargeskala og lar seg sammenligne visuelt
        data_min = np.nanmin([Z2.min(), Z3.min()])
        data_max = np.nanmax([Z2.max(), Z3.max()])
    
        #Rund minimum nedover til nærmeste hele 25
        vmin = np.floor(data_min / step) * step
        #Rund maksimum oppover til nærmeste hele 25
        vmax = np.ceil(data_max / step) * step
    
        ticks = np.arange(vmin, vmax + step/2, step)
        full_levels = np.linspace(vmin, vmax, 100)
    
        #Kontur med valgt fargegradient
        cf = plt.contourf(X, Y, Z3, levels=full_levels, cmap="cividis")
    
        #Hvite konturlinjer, med tallverdi langs hver linje
        cs = plt.contour(X, Y, Z3, levels=ticks, colors="white", linewidths=2)
        plt.clabel(cs, inline=True, fontsize=10, fmt="%.0f")
    
        #Fargestolpe med samme ticks som konturlinjene, formatert uten desimaler
        cbar = plt.colorbar(cf, ticks=ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
        #Hvit prikk i maksimum
        i, j = np.unravel_index(np.nanargmax(Z3), Z3.shape)
        plt.scatter(X[i, j], Y[i, j], color="white", s=50, edgecolor="white")
    
        cbar.set_label("[kgCO₂-eq /kWp·år]")
    
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{UNNGÅTT\ UTSLIPP \ STATISK\ METODE}}$" "\n"
            f"Snitt {År} i {Sted}\n"
            f"Optimum: {pivot_static.stack().max():.0f} kgCO\u2082-eq/kWp ved helningsvinkel {pivot_static.stack().idxmax()[0]}° "
            f"og asimut {pivot_static.stack().idxmax()[1]}°",
            fontsize=15
            )
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CO2-DC_STATIC_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_CO2POA_static(pivot_static, Z2, Z3, År_label, Sted, perez, save_dir)
    
    
    
    
    #C02DC kgCO2/kWp Dynamisk
    def plot_CO2POA_dynamic(pivot_dynamic, Z2, Z3, År, Sted, perez, save_dir):
        """Konturplott av unngåtte utslipp [kgCO₂-eq/kWp per år] med dynamisk metode.
    
        Z2 = dynamisk verdi, Z3 = statisk verdi. Begge sendes inn for å bygge
        en felles fargeskala.
        """
    
        #Lager rutenett for konturplottet
        X, Y = np.meshgrid(pivot_dynamic.columns.values, pivot_dynamic.index.values)
    
        plt.figure(figsize=(14, 8))
    
        #Steglengde mellom konturlinjer
        step = 25
    
        #Min/maks finnes på tvers av både statisk og dynamisk datasett,
        #slik får begge figurene samme fargeskala og lar seg sammenligne visuelt
        data_min = np.nanmin([Z2.min(), Z3.min()])
        data_max = np.nanmax([Z2.max(), Z3.max()])
    
        #Rund minimum nedover til nærmeste hele 25
        vmin = np.floor(data_min / step) * step
        #Rund maksimum oppover til nærmeste hele 25
        vmax = np.ceil(data_max / step) * step

        ticks = np.arange(vmin, vmax + step/2, step)
        full_levels = np.linspace(vmin, vmax, 100)
    
        #Kontur med valgt fargegradient
        cf = plt.contourf(X, Y, Z2, levels=full_levels, cmap="cividis")
    
        #Hvite konturlinjer, med tallverdi langs hver linje
        cs = plt.contour(X, Y, Z2, levels=ticks, colors="white", linewidths=2)
        plt.clabel(cs, inline=True, fontsize=10, fmt="%.0f")
    
        #Fargestolpe med samme ticks som konturlinjene, formatert uten desimaler
        cbar = plt.colorbar(cf, ticks=ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
        #Hvit prikk i maksimum
        i, j = np.unravel_index(np.nanargmax(Z2), Z2.shape)
        plt.scatter(X[i, j], Y[i, j], color="white", s=50, edgecolor="white")
    
        cbar.set_label("[kgCO₂-eq /kWp·år]")
    
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{UNNGÅTT\ UTSLIPP \ DYNAMISK\ METODE}}$" "\n"
            f"Snitt {År} i {Sted}\n"
            f"Optimum: {pivot_dynamic.stack().max():.0f} kgCO\u2082-eq/kWp ved helningsvinkel {pivot_dynamic.stack().idxmax()[0]}° "
            f"og asimut {pivot_dynamic.stack().idxmax()[1]}°",
            fontsize=15
            )
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CO2-DC_DYNAMIC_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_CO2POA_dynamic(pivot_dynamic, Z2, Z3, År_label, Sted, perez, save_dir)




    #Statisk vs dynamisk forskjell
    def plot_stat_vs_dyn(pivot_dynamic, Z2, Z3, År, Sted, save_dir):
        """Konturplott av relativ forskjell mellom statisk og dynamisk metode [%].

        Viser hvor mye den statiske metoden misrepresenterer
        unngåtte utslipp sammenlignet med den dynamiske, beregnet som
        (Zstat - Zdyn) / Zdyn * 100 for alle orienteringer og helningsvinkler.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    
        X, Y = np.meshgrid(pivot_dynamic.columns.values, pivot_dynamic.index.values)
    
        #Klarere navngivning for resten av funksjonen
        Zdyn = Z2
        Zsta = Z3
    
        #Håndterer deling på null
        eps = 1e-12
        Zdyn1 = np.where(np.abs(Zdyn) < eps, np.nan, Zdyn)
    
        #Hvor mye statisk overestimerer ifht. dynamisk [%]
        Zpct = (Zsta - Zdyn) / Zdyn1 * 100.0
        
        #Finn størst og minst overestimering
        imax, jmax = np.unravel_index(np.nanargmax(Zpct), Zpct.shape)
        imin, jmin = np.unravel_index(np.nanargmin(Zpct), Zpct.shape)
    
        tilts = pivot_dynamic.index.values
        azis  = pivot_dynamic.columns.values
    
        tilt_max, azi_max = tilts[imax], azis[jmax]
        tilt_min, azi_min = tilts[imin], azis[jmin]
    
        val_max = Zpct[imax, jmax]
        val_min = Zpct[imin, jmin]
    
        plt.figure(figsize=(14, 8))
    
        #Høyeste og laveste prosentverdi i datasettet
        rminP = np.nanmin(Zpct)
        rmaxP = np.nanmax(Zpct)
        
        vmin = np.floor(rminP)
        vmax = np.ceil(rmaxP)
        
        ticks = np.arange(vmin, vmax + 1, 1)      
        levels = np.linspace(vmin, vmax, 100)     
        
        #Konturplot og konturlinjer
        cf = plt.contourf(X, Y, Zpct, levels=levels, cmap="cividis")
        cs = plt.contour(X, Y, Zpct, levels=ticks, colors="white", linewidths=1.5)
        
        plt.clabel(cs, inline=True, fontsize=10, fmt="%.0f%%")
        
        cbar = plt.colorbar(cf, ticks=ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        cbar.set_label("Overestimering av statisk metode [%]")
        
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{Forskjell\ på\ statisk\ og\ dynamisk\ metode\ i\ {Sted}}}$" "\n"
            f"Størst: {val_max:.1f}% ved helningsvinkel {tilt_max}°, asimut {azi_max}°\n"
            f"Minst: {val_min:.1f}% ved helningsvinkel {tilt_min}°, asimut {azi_min}°",
            fontsize=15
        )
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CO2_statVSdyn_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_stat_vs_dyn(pivot_dynamic, Z2, Z3, År_label, Sted, save_dir)
    
    
    
    
    #Normaliserte statisk og dynamisk, prosent verdier 0-100
    def plot_CO2POA_dynamic_percent(pivot_dynamic, Z2, Z3, År_label, Sted, perez,save_dir):
        """
        Normalisert (0-100 %) konturplott av dynamisk unngåtte utslipp.

        """
        
        X, Y = np.meshgrid(pivot_dynamic.columns.values, pivot_dynamic.index.values)
    
        plt.figure(figsize=(14, 8))
        
        #Laveste og høyeste verdi i datasettet
        vmin = np.nanmin(Z2)
        vmax = np.nanmax(Z2)
        
        #Håndterer deling på null
        den = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        
        #Normalisering
        Z2n = (Z2 - vmin) / den * 100.0
        
        #plotting
        full_levels = np.linspace(0, 100, 101)
        cf = plt.contourf(X, Y, Z2n, levels=full_levels, cmap="cividis")
    
        ticks = np.linspace(0, 100, 11)
        cs = plt.contour(X, Y, Z2n, levels=ticks, colors="white", linewidths=2)
        plt.clabel(cs, inline=True, fontsize=10, fmt="%.0f%%")
    
        i, j = np.unravel_index(np.nanargmax(Z2n), Z2n.shape)
        plt.scatter(X[i, j], Y[i, j], color="white", s=50, edgecolor="white")
    
        cbar = plt.colorbar(cf, ticks=ticks)
        cbar.set_label("Normalisert CO2-DC [0–100 %]")
    
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{CO_{{2}}-DC \ DYNAMISK \ NORMALISERT\ {Sted}}}$",
            fontsize=15
        )
    
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CO2-DC_dynamic_norm(percent)_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    plot_CO2POA_dynamic_percent(pivot_dynamic, Z2, Z3, År_label, Sted, perez,save_dir)

    
    
    
    
    def plot_CO2POA_static_percent(pivot_static, Z2, Z3, År_label, Sted, perez,save_dir):
        """
        Normalisert (0-100 %) konturplott av statisk unngåtte utslipp.

        """
        X, Y = np.meshgrid(pivot_static.columns.values, pivot_static.index.values)
    
        plt.figure(figsize=(14, 8))
        
        #Laveste og høyeste verdi i datasettet
        vmin = np.nanmin(Z3)
        vmax = np.nanmax(Z3)
        
        #Håndterer deling på null
        den = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        
        #Normalisering
        Z3n = (Z3 - vmin) / den * 100.0
        
        #plotting
        full_levels = np.linspace(0, 100, 101)
        cf = plt.contourf(X, Y, Z3n, levels=full_levels, cmap="cividis")
    
        ticks = np.linspace(0, 100, 11)
        cs = plt.contour(X, Y, Z3n, levels=ticks, colors="white", linewidths=2)
        plt.clabel(cs, inline=True, fontsize=10, fmt="%.0f%%")
    
        cbar = plt.colorbar(cf, ticks=ticks)
    
        i, j = np.unravel_index(np.nanargmax(Z3n), Z3n.shape)
        plt.scatter(X[i, j], Y[i, j], color="white", s=50, edgecolor="white")
    
        cbar.set_label("Normalisert CO2-DC [0–100 %]")
    
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{CO_{{2}}-DC \ STATISK \ NORMALISERT\ {Sted}}}$",
            fontsize=15
        )
    
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CO2-DC_static_norm(percent)_{Sted}.png"), dpi=300, bbox_inches="tight")
        plt.show()
        
    plot_CO2POA_static_percent(pivot_static, Z2, Z3, År_label, Sted, perez, save_dir)
    
    


    #Diff i normaliserte verdier, prosentpoeng
    def plot_CO2POA_percent_difference(pivot_dynamic, pivot_static, Z2, Z3, År_label, Sted, perez,save_dir):
        """
        Differanse mellom de normaliserte 
        """
        X, Y = np.meshgrid(pivot_dynamic.columns.values, pivot_dynamic.index.values)
    
        plt.figure(figsize=(14, 8))
    
        vmin_dyn = np.nanmin(Z2)
        vmax_dyn = np.nanmax(Z2)
        den_dyn = (vmax_dyn - vmin_dyn) if (vmax_dyn - vmin_dyn) != 0 else 1.0
        Z2n = (Z2 - vmin_dyn) / den_dyn * 100.0
    
        vmin_sta = np.nanmin(Z3)
        vmax_sta = np.nanmax(Z3)
        den_sta = (vmax_sta - vmin_sta) if (vmax_sta - vmin_sta) != 0 else 1.0
        Z3n = (Z3 - vmin_sta) / den_sta * 100.0
        
        #Diff
        Zdiff = Z3n - Z2n
    
        absmax = np.nanmax(np.abs(Zdiff))
        L = np.ceil(absmax*2)/2
        
        vmin = -L
        vmax = L
        
        levels_cf = np.linspace(vmin, vmax, 101)
        ticks = np.arange(vmin, vmax + 1, 0.5)
        
        cf = plt.contourf(X, Y, Zdiff, levels=levels_cf, cmap="bwr")
        cs = plt.contour(X, Y, Zdiff, levels=ticks, colors="black", linewidths=1.2)
        
        plt.clabel(cs, inline=True, fontsize=9, fmt="%.1f")
        
        cbar = plt.colorbar(cf, ticks=ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        cbar.set_label("Differanse Statisk normalisert[%] og dynamisk normalisert[%]")
        
        i_dyn, j_dyn = np.unravel_index(np.nanargmax(Z2), Z2.shape)
        i_sta, j_sta = np.unravel_index(np.nanargmax(Z3), Z3.shape)
    
        tilt_dyn = pivot_dynamic.index.values[i_dyn]
        azi_dyn  = pivot_dynamic.columns.values[j_dyn]
    
        tilt_sta = pivot_static.index.values[i_sta]
        azi_sta  = pivot_static.columns.values[j_sta]
    
        plt.scatter(azi_dyn, tilt_dyn, marker="o", s=40, edgecolor="black", facecolor="powderblue",
                    label=f"Dynamisk optimum ({tilt_dyn}°, {azi_dyn}°)", zorder=10)
        plt.scatter(azi_sta, tilt_sta, marker="o", s=40, edgecolor="black", facecolor="pink",
                    label=f"Statisk optimum ({tilt_sta}°, {azi_sta}°)", zorder=10)
    
        plt.xlabel("Asimut")
        plt.ylabel("Helningsvinkel")
    
        plt.title(
            rf"$\bf{{Forskjell\ mellom\ normaliserte\ CO2-DC\ grafer\ ({Sted}\ {start_year}-{end_year})}}$",
            fontsize=15
        )
    
        plt.legend()
        plt.grid(alpha=0.2)
        plt.xticks(np.arange(0, 361, 30))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"norm percent diff stat dyn co2-dc)_{Sted}.png"), dpi=300, bbox_inches="tight")
    
        plt.show()
    
    plot_CO2POA_percent_difference(pivot_dynamic, pivot_static, Z2, Z3, År_label, Sted, perez,save_dir)
    
    
    
    #Varmekart
    def plot_ratio_heatmaps(ratio_ts, cases, År, save_dir):
        """
        Plot varmekart for differansen hver time hele året, for valgt asimut og helningsvinkel
        """
        
        
        #Hopp over hvis kombinasjonen ikke finnes i resultatordboka,
        #kan skje dersoms steglengde ikke samsvarer med valgt orienering og helningsvinkel
        for key in cases:
            if key not in ratio_ts:
                print(f"Fant ikke {key} i ratio_ts (ikke beregnet / ikke på grid).")
                continue
    
            tilt, az = key
            s = ratio_ts[key]
    
            #Matrise med time på døgnet og dag på året
            df = s.to_frame("ratio")
            df["doy"] = df.index.dayofyear
            df["hour"] = df.index.hour
    
            #Pivot å time og dag
            H = df.groupby(["hour", "doy"])["ratio"].mean().unstack("doy").fillna(0)
    
            #Normalisering
            max_abs = np.nanmax(np.abs(H.values))
            H_norm = H / max_abs if max_abs != 0 else H * 0
            
            #Plot utseende
            month_starts = pd.date_range(
                start=f"{År}-01-01",
                end=f"{År}-12-31",
                freq="MS",
                tz=s.index.tz
            )
            month_doys = month_starts.dayofyear
            norske_maaneder = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun",
                       "Jul", "Aug", "Sep", "Okt", "Nov", "Des"]
            month_labels = [norske_maaneder[m - 1] for m in month_starts.month]
    
            
            plt.figure(figsize=(16, 6))
            im = plt.imshow(
                H_norm.values,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                extent=[H.columns.min(), H.columns.max(), 0, 23]
            )
    
            plt.xlabel("Måned")
            plt.ylabel("Time")
            plt.yticks(np.arange(0, 24, 2))
            plt.xticks(month_doys, month_labels)
    
            plt.title(f"Differanse på statisk og dynamisk per time for Tilt={tilt}° og Azimuth={az}°")
    
            cbar = plt.colorbar(im)
            cbar.set_label("Normalisert differanse [%]")
    
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"ratio_heatmap_{tilt}_{az}_{Sted}.png"), dpi=300, bbox_inches="tight")
            plt.show()
            
    
    #Finn optimum fra dynamisk metode
    dyn_opt = pivot_dynamic.stack().idxmax()
    
    cases = {dyn_opt}
    
    plot_ratio_heatmaps(ratio_ts, cases, År, save_dir)
                                 
