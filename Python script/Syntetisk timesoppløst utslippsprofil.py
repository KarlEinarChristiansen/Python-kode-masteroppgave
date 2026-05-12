# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skforecast.stats import Arima
from skforecast.recursive import ForecasterStats
from skforecast.metrics import mean_absolute_scaled_error


#==============================================================================
#MÅNEDSSNITT

# 1M. Laste inn data
raw_M = pd.read_csv(
    r'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\Raadata\Alle_timer_2015-2024_utslippsintensitet.csv',
    sep=','
)

#Bygg datetime-kolonne fra År/Måned/Dag/Time
raw_M['date'] = pd.to_datetime(
    raw_M['Aar'].astype(str) + '-' + raw_M['Maaned'].astype(str) + '-' + raw_M['Dag'].astype(str)
    + ' ' + raw_M['time'].astype(str) + ':00:00'
)

#Aggreger til månedssnitt
data_M = (raw_M.groupby(raw_M['date'].dt.to_period('M'))
          .agg({'Utslippsintensitet': 'mean'})
          .rename(columns={'Utslippsintensitet': 'y'}))
data_M.index = data_M.index.to_timestamp()
data_M.index.name = 'date'
data_M = data_M.asfreq('MS').sort_index()

#Førstedifferanse for å gjøre serien stasjonær
data_M['y_original'] = data_M['y'].copy()
data_M['y'] = data_M['y'].diff()
data_M = data_M.dropna()


#2M. Utforske dataen
print(data_M.describe())
print("\nMissing values:")
print(data_M.isna().sum())

#Historisk månedsgjennomsnitt plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_M.index, data_M['y_original'], label='Utslippsintensitet', color='#F7931E', linewidth=1.5)
ax.legend()
ax.set_title('Månedsgjennomsnitt')
ax.set_xlabel('dato')
ax.set_ylabel('gCO₂/kWh')
plt.tight_layout()
plt.show()

#Plot av Førstedifferansen
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_M.index, data_M['y'], label='Utslippsintensitet', color='#F7931E', linewidth=1.5)
ax.legend()
ax.set_title('Månedsgjennomsnitt førstedifferansen')
ax.set_xlabel('dato')
ax.set_ylabel('gCO₂/kWh')
plt.tight_layout()
plt.show()

# ACF og PACF 
maxLags_M = 40
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(data_M['y'].dropna(), ax=axes[0], lags=maxLags_M, title='ACF - Monthly')
plot_pacf(data_M['y'].dropna(), ax=axes[1], lags=maxLags_M, title='PACF - Monthly')
plt.tight_layout()
plt.show()



#4M. SARIMA

#Auto-SARIMA: lar AICc velge (p,q,P,Q). d=D=0 siden vi allerede har differensiert.
estimator_params_M = {
    'order': None,
    'seasonal_order': None,
    'm': 12,
    'max_p': 5,
    'max_q': 5,
    'max_P': 2,
    'max_Q': 2,
    'max_d': 0,
    'max_D': 0,
    'ic': 'aicc',
    'seasonal': True
}

forecaster_M = ForecasterStats(
    estimator        = Arima(**estimator_params_M),
    transformer_y    = None,
    transformer_exog = None
)


#5M. Fit & Predict

#Trene og test periode 
end_train_M = '2022-12-01'
data_train_M = data_M.loc[:end_train_M]
data_test_M = data_M.loc[data_M.index > end_train_M]

#Tilpasser SARIMA og printer modellsammendrag
forecaster_M.fit(y=data_train_M['y'])
print("Monthly forecaster fitted successfully!")
print(forecaster_M.estimators_[0].summary())

#Hvor mange måneder skal predikeres
steps_M = 24
predictions_M = forecaster_M.predict(steps=steps_M)
print("Monthly Predictions:")
print(predictions_M.head(5))

# Evalueringsmetrikker: MAE, MSE, MASE, og Ljung-Box for residualautokorrelasjon
actual_M = data_test_M['y'].iloc[:steps_M]
mae_M = mean_absolute_error(actual_M, predictions_M)
mse_M = mean_squared_error(actual_M, predictions_M)
mase_M = mean_absolute_scaled_error(y_true=actual_M, y_pred=predictions_M, y_train=data_train_M['y'])
lb_pvalue = acorr_ljungbox(actual_M - predictions_M, lags=12, return_df=True)['lb_pvalue'].iloc[-1]
print(f"MAE  : {mae_M:.4f}")
print(f"MSE  : {mse_M:.4f}")
print(f"MASE : {mase_M:.4f}")
print(f"Ljung-Box p (lag 12): {lb_pvalue:.4f}")

# Plot: train + test + SARIMA-prediksjon på førstedifferansenivå
fig, ax = plt.subplots(figsize=(12, 6))
data_train_M['y'].plot(ax=ax, label='Trening', color='#F7931E', linewidth=1)
data_test_M['y'].iloc[:steps_M].plot(ax=ax, label='Test', color='#F7931E', linewidth=2, alpha=0.5)
predictions_M.plot(ax=ax, label='Forecast', color='red', linewidth=2)
ax.set_title('Månedsgjennomsnitt forecast av førstedifferansen')
ax.set_xlabel('Dato')
ax.set_ylabel('gCO2/kWh')
ax.legend()
plt.tight_layout()
plt.show()

importance_M = forecaster_M.get_feature_importances()
print("Monthly Model Coefficients:\n")
print(importance_M)


#6M. profil mot 2040 med å kombinere SARIMA og egendefinert trend

# 1. Konverter SARIMA-førstedifferanser til nivåer (kumulativ sum av første 12 mnd)
last_value_M = data_M['y_original'].iloc[-1]
sarima_year_levels_M = last_value_M + predictions_M.iloc[:12].cumsum()

# 2. Henter ut sesongform = avvik fra årsmiddel (12 verdier, 1 per måned)
sarima_mean_M = sarima_year_levels_M.mean()
sarima_profile_M = sarima_year_levels_M - sarima_mean_M

print("SARIMA monthly profile (12 deviations):")
print(sarima_profile_M)

# 3. Fremtidig månedsindeks
last_date_M = data_M.index[-1]
future_index_M = pd.date_range(start=last_date_M + pd.DateOffset(months=1), end='2039-12-01', freq='MS')
months_to_2040 = len(future_index_M)


#Trend mot 0 i 2050
TREND_START_YEAR = 2025
TREND_END_YEAR   = 2050
TREND_END_VALUE  = 0.0

# Startverdi er treårssnitt av siste 3 historiske år, fra NS3720:2018
last_3y = data_M['y_original'].loc['2022':'2024']
TREND_START_VALUE = float(last_3y.mean())

print("\nNS 3720-trend:")
print(f"  Startverdi (3-årssnitt 2022-2024) i {TREND_START_YEAR}: {TREND_START_VALUE:.1f} gCO2/kWh")
print(f"  Sluttverdi i {TREND_END_YEAR}: {TREND_END_VALUE:.1f} gCO2/kWh")
slope_per_year = (TREND_END_VALUE - TREND_START_VALUE) / (TREND_END_YEAR - TREND_START_YEAR)
print(f"  Stigningstall: {slope_per_year:.2f} gCO2/kWh per år")

# Lineær interpolering for hvert månedlig tidspunkt i future_index_M
future_year_decimal = future_index_M.year + (future_index_M.month - 1) / 12
progress = (future_year_decimal - TREND_START_YEAR) / (TREND_END_YEAR - TREND_START_YEAR)
custom_trend_M = TREND_START_VALUE + progress * (TREND_END_VALUE - TREND_START_VALUE)

# 5. Map sesongprofilen til fremtidige måneder
sarima_seasonal_M = dict(zip(sarima_profile_M.index.month, sarima_profile_M.values))
seasonal_future_M = future_index_M.month.map(sarima_seasonal_M)

# 6. Kombiner trend + sesong
forecast_2040_M = custom_trend_M + seasonal_future_M.values
forecast_series_M = pd.Series(forecast_2040_M, index=future_index_M)

# Plot: historisk + månedlig forecast mot 2040
fig, ax = plt.subplots(figsize=(14, 6))
data_M['y_original'].plot(ax=ax, label='Historisk data', color='#F7931E')
ax.set_title('Månedsgjennomsnitt forecast mot 2040')
forecast_series_M.plot(ax=ax, label='Månedsgjennomsnitt forecast', color='red')
ax.set_ylabel('gCO2/kWh')
ax.set_xlabel('Dato')
ax.legend()
plt.tight_layout()
plt.show()




#==============================================================================
#UKESNITT

# 1W. Laste inn data
raw_W = pd.read_csv(
    r'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\Raadata\Alle_timer_2015-2024_utslippsintensitet.csv',
    sep=','
)

#Bygg datetime-kolonne fra År/Måned/Dag/Time
raw_W['date'] = pd.to_datetime(
    raw_W['Aar'].astype(str) + '-' + raw_W['Maaned'].astype(str) + '-' + raw_W['Dag'].astype(str)
    + ' ' + raw_W['time'].astype(str) + ':00:00'
)

#Aggreger til ukesnitt (uker starter på mandag)
data_W = (raw_W.set_index('date')
          .resample('W-MON')
          .agg({'Utslippsintensitet': 'mean'})
          .rename(columns={'Utslippsintensitet': 'y'}))
data_W = data_W.sort_index()

#Førstedifferanse for å gjøre serien stasjonær
data_W['y_original'] = data_W['y'].copy()
data_W['y'] = data_W['y'].diff()
data_W = data_W.dropna()


#2W. Utforske dataen
print(data_W.describe())
print("\nMissing values:")
print(data_W.isna().sum())

#Historisk ukesgjennomsnitt plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_W.index, data_W['y_original'], label='Utslippsintensitet', color='#F7931E', linewidth=1.5)
ax.legend()
ax.set_title('Ukesgjennomsnitt')
ax.set_xlabel('dato')
ax.set_ylabel('gCO₂/kWh')
plt.tight_layout()
plt.show()

#Plot av Førstedifferansen
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_W.index, data_W['y'], label='Utslippsintensitet', color='#F7931E', linewidth=1.5)
ax.legend()
ax.set_title('Ukesgjennomsnitt førstedifferansen')
ax.set_xlabel('dato')
ax.set_ylabel('gCO₂/kWh')
plt.tight_layout()
plt.show()

# ACF og PACF
maxLags_W = 40
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(data_W['y'].dropna(), ax=axes[0], lags=maxLags_W, title='ACF - Weekly')
plot_pacf(data_W['y'].dropna(), ax=axes[1], lags=maxLags_W, title='PACF - Weekly')
plt.tight_layout()
plt.show()



#4W. SARIMA

#Auto-SARIMA: lar AICc velge (p,q,P,Q). d=D=0 siden vi allerede har differensiert.
#m=52 fanger årlig syklus i ukentlige data
estimator_params_W = {
    'order': None,
    'seasonal_order': None,
    'm': 52,
    'max_p': 5,
    'max_q': 5,
    'max_P': 2,
    'max_Q': 2,
    'max_d': 0,
    'max_D': 0,
    'ic': 'aicc',
    'seasonal': True
}

forecaster_W = ForecasterStats(
    estimator        = Arima(**estimator_params_W),
    transformer_y    = None,
    transformer_exog = None
)


#5W. Fit & Predict

#Trene og test periode
end_train_W = '2023-01-01'
data_train_W = data_W.loc[:end_train_W]
data_test_W = data_W.loc[data_W.index > end_train_W]

#Tilpasser SARIMA og printer modellsammendrag
forecaster_W.fit(y=data_train_W['y'])
print("Weekly forecaster fitted successfully!")
print(forecaster_W.estimators_[0].summary())

#Hvor mange uker skal predikeres
steps_W = 52
predictions_W = forecaster_W.predict(steps=steps_W)
print("Weekly Predictions:")
print(predictions_W.head(5))

# Evalueringsmetrikker: MAE, MSE, MASE, og Ljung-Box for residualautokorrelasjon
actual_W = data_test_W['y'].iloc[:steps_W]
mae_W = mean_absolute_error(actual_W, predictions_W)
mse_W = mean_squared_error(actual_W, predictions_W)
mase_W = mean_absolute_scaled_error(y_true=actual_W, y_pred=predictions_W, y_train=data_train_W['y'])
lb_pvalue_W = acorr_ljungbox(actual_W - predictions_W, lags=12, return_df=True)['lb_pvalue'].iloc[-1]
print(f"MAE  : {mae_W:.4f}")
print(f"MSE  : {mse_W:.4f}")
print(f"MASE : {mase_W:.4f}")
print(f"Ljung-Box p (lag 12): {lb_pvalue_W:.4f}")

# Plot: train + test + SARIMA-prediksjon på førstedifferansenivå
fig, ax = plt.subplots(figsize=(12, 6))
data_train_W['y'].plot(ax=ax, label='Trening', color='#F7931E', linewidth=1)
data_test_W['y'].iloc[:steps_W].plot(ax=ax, label='Test', color='#F7931E', linewidth=2, alpha=0.5)
predictions_W.plot(ax=ax, label='Forecast', color='#3b82f6', linewidth=2)
ax.set_title('Ukesgjennomsnitt forecast av førstedifferansen')
ax.set_xlabel('Dato')
ax.set_ylabel('gCO2/kWh')
ax.legend()
plt.tight_layout()
plt.show()

importance_W = forecaster_W.get_feature_importances()
print("Weekly Model Coefficients:\n")
print(importance_W)


#6W. profil mot 2040 med å kombinere månedlig trend og SARIMA ukesprofil

# 1. Konverter SARIMA-førstedifferanser til nivåer (kumulativ sum av første 52 uker)
last_value_W = data_train_W['y_original'].iloc[-1]
sarima_year_levels_W = last_value_W + predictions_W.iloc[:52].cumsum()

# 2. Henter ut sesongform = avvik fra årsmiddel (52 verdier, 1 per uke)
sarima_mean_W = sarima_year_levels_W.mean()
sarima_profile_W = sarima_year_levels_W - sarima_mean_W

print("SARIMA weekly profile (52 deviations):")
print(sarima_profile_W.head(10))

# 3. Fremtidig ukesindeks - start på mandag i uke 1, 2025
future_start_W = pd.Timestamp('2024-12-30')
future_index_W = pd.date_range(
    start=future_start_W,
    end='2039-12-31',
    freq='W-MON'
)

# 4. Interpoler månedlig forecast til ukentlig 
monthly_trend_daily = forecast_series_M.resample('D').interpolate(method='linear')
monthly_trend_weekly = monthly_trend_daily.resample('W-MON').mean()
monthly_trend_weekly = monthly_trend_weekly.reindex(future_index_W)
monthly_trend_weekly = monthly_trend_weekly.interpolate(method='linear')
monthly_trend_weekly = monthly_trend_weekly.bfill().ffill()

# 5. Map ukentlig sesongprofil etter uke i året
sarima_seasonal_W = dict(zip(
    sarima_profile_W.index.isocalendar().week.astype(int),
    sarima_profile_W.values
))

# Uke 53 forekommer i enkelte år, bruk gjennomsnitt av uke 52 og uke 1
if 53 not in sarima_seasonal_W:
    sarima_seasonal_W[53] = (sarima_seasonal_W.get(52, 0) + sarima_seasonal_W.get(1, 0)) / 2

future_weeks = future_index_W.isocalendar().week.astype(int)
seasonal_future_W = future_weeks.map(sarima_seasonal_W)

# Fallback for evt. uker uten profil
n_nan = seasonal_future_W.isna().sum()
if n_nan > 0:
    print(f"ADVARSEL: {n_nan} uker uten profil, fyller med 0")
    seasonal_future_W = seasonal_future_W.fillna(0)

# 6. Kombiner månedlig trend (interpolert) + ukentlig sesongavvik
forecast_weekly_2040 = monthly_trend_weekly.values + seasonal_future_W.values
forecast_weekly_series = pd.Series(forecast_weekly_2040, index=future_index_W)

# Plot: historisk + ukentlig forecast + månedlig forecast (referanse)
fig, ax = plt.subplots(figsize=(14, 6))
data_W['y_original'].plot(ax=ax, label='Historisk data', color='#F7931E')
forecast_weekly_series.plot(ax=ax, label='Ukentlig forecast', color='#3b82f6')
forecast_series_M.plot(ax=ax, label='Månedlig forecast', color='red', linewidth=2, alpha=0.4)
ax.set_title('Ukesgjennomsnitt forecast mot 2040')
ax.set_ylabel('gCO2/kWh')
ax.set_xlabel('Dato')
ax.legend()
plt.tight_layout()
plt.show()





#==============================================================================
#TIMESOPPLØSNING
 
#7H. Timesprofiler per måned og forecast mot 2040
# 1. Bygg månedsspesifikke timesprofiler via STL (period=24)
raw_H = raw_W.copy()
hourly = raw_H.set_index('date')['Utslippsintensitet']

monthly_hourly_profiles = {}
for month in range(1, 13):
    month_data = hourly[hourly.index.month == month]
    decomp = seasonal_decompose(month_data, model='additive', period=24)
    # Gjennomsnittlig døgnprofil (avvik fra månedsmiddel)
    profile = decomp.seasonal.groupby(decomp.seasonal.index.hour).mean()
    monthly_hourly_profiles[month] = profile
    print(f"Month {month:2d}: range {profile.min():.2f} to {profile.max():.2f}")

# Plot alle 12 månedsprofiler
fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i, (month, profile) in enumerate(monthly_hourly_profiles.items()):
    ax = axes[i // 4, i % 4]
    ax.plot(profile.index, profile.values, color='#0ea5e9', linewidth=2)
    ax.set_title(month_names[i])
    ax.set_xlabel('Hour')
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_ylabel('gCO2/kWh deviation')
plt.suptitle('Times profiler hver måned', fontsize=14)
plt.tight_layout()
plt.show()

# 2. Fremtidig timesindeks
future_index_H = pd.date_range(
    start=future_index_W[0],
    end=future_index_W[-1] + pd.DateOffset(days=6, hours=23),
    freq='h'
)

# 3. Interpoler ukentlig forecast til timesoppløsning
forecast_hourly_base = forecast_weekly_series.resample('h').interpolate(method='linear')
forecast_hourly_base = forecast_hourly_base.reindex(future_index_H, method='ffill')

# 4. Legg på månedsspesifikk døgnprofil
hourly_seasonal_future = pd.Series(
    [monthly_hourly_profiles[m][h] for m, h in zip(future_index_H.month, future_index_H.hour)],
    index=future_index_H
)

# 5. Kombiner base + døgnprofil
forecast_hourly_2040 = forecast_hourly_base.values + hourly_seasonal_future.values
forecast_hourly_series = pd.Series(forecast_hourly_2040, index=future_index_H)

#Datasettet dekker eksakt 2025-2039
forecast_hourly_series = forecast_hourly_series[
    (forecast_hourly_series.index.year >= 2025) &
    (forecast_hourly_series.index.year <= 2039)
]

print(f"\nHourly forecast shape: {forecast_hourly_series.shape}")
print(f"Date range: {forecast_hourly_series.index[0]} to {forecast_hourly_series.index[-1]}")

# Plot: historisk timesoppløsning + alle tre forecast-nivåer
fig, ax = plt.subplots(figsize=(14, 6))
hourly.plot(ax=ax, label='Historisk (timesoppløsning)', color='#F7931E', linewidth=0.5, alpha=0.7)
forecast_hourly_series.plot(ax=ax, label='Times forecast', color='green', linewidth=0.5, alpha=0.5)
forecast_weekly_series.plot(ax=ax, label='Ukesmiddel forecast', color='blue', linewidth=1.5, alpha=0.3)
forecast_series_M.plot(ax=ax, label='Månedsmiddel forecast', color='red', linewidth=2, alpha=0.3)
ax.set_title('Times forecast mot 2040')
ax.set_xlabel('Dato')
ax.set_ylabel('gCO2/kWh')
ax.legend()
plt.tight_layout()
plt.show()


# Lagre timesoppløst forecast til CSV
forecast_hourly_series.to_frame('gCO2_per_kWh').to_csv(
    r'C:\Users\Eier\OneDrive\Skrivebord\master 2026\Python\Data\forecast_hourly_2040.csv'
)
print(f"Lagret {len(forecast_hourly_series)} timesverdier til CSV")

#%%
# =============================================================================
#Interaktiv figur

import plotly.graph_objects as go

fig = go.Figure()

# Historisk ukentlig
fig.add_trace(go.Scatter(
    x=data_W['y_original'].index, y=data_W['y_original'].values,
    name='Historical (weekly)', line=dict(color='#F7931E', width=1)
))

# Månedlig forecast
fig.add_trace(go.Scatter(
    x=forecast_series_M.index, y=forecast_series_M.values,
    name='Monthly forecast', mode='lines+markers',
    line=dict(color='red', width=2), marker=dict(size=4),
    opacity=0.5
))

# Ukentlig forecast
fig.add_trace(go.Scatter(
    x=forecast_weekly_series.index, y=forecast_weekly_series.values,
    name='Weekly forecast', mode='lines+markers',
    line=dict(color='#3b82f6', width=1.5), marker=dict(size=3)
))

# Timesoppløst forecast - kun de første 19 år synlig for ytelse
sample_end_H = future_index_H[0] + pd.DateOffset(years=19)
hourly_sample = forecast_hourly_series.loc[:sample_end_H]

fig.add_trace(go.Scatter(
    x=hourly_sample.index, y=hourly_sample.values,
    name='Hourly forecast', mode='lines',
    line=dict(color='green', width=0.5), opacity=0.6,
    visible='legendonly'
))

fig.update_layout(
    title='Emission Intensity Forecast to 2040 (SARIMA-based profiles)',
    xaxis_title='Date', yaxis_title='gCO2/kWh',
    hovermode='x unified', template='plotly_white',
    legend=dict(x=0.01, y=0.99),
    xaxis=dict(
        rangeselector=dict(buttons=[
            dict(count=1, label='1M', step='month', stepmode='backward'),
            dict(count=3, label='3M', step='month', stepmode='backward'),
            dict(count=1, label='1Y', step='year', stepmode='backward'),
            dict(count=5, label='5Y', step='year', stepmode='backward'),
            dict(step='all', label='All')
        ]),
        rangeslider=dict(visible=True),
    )
)
fig.write_html('forecast_all_sarima.html', auto_open=True)


