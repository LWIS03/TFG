import openmeteo_requests
import pandas as pd
import requests_cache
import datetime
from retry_requests import retry
from simulacion_riego import CalculadoraCultivo, calcular_gdu, calcular_altura_por_gdu

# Configuración del cliente (es buena idea mantener el caché para Streamlit)
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def obtener_datos_meteorologicos():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 42.4413,
        "longitude": -5.7928,
        "daily": ["temperature_2m_max", "temperature_2m_min", "et0_fao_evapotranspiration", "wind_speed_10m_mean", "relative_humidity_2m_mean", "precipitation_sum"],
        "hourly": "soil_temperature_6cm",
        "timezone": "Europe/Madrid", 
        "wind_speed_unit": "ms",
        "forecast_days": 2 
    }
    
    # 1. Llamada a la API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # 2. Procesar datos horarios (Temperatura del suelo)
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "soil_temperature_6cm": hourly.Variables(0).ValuesAsNumpy()
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Extraemos solo el día de la fecha (ignorando la hora)
    hourly_dataframe['dia'] = hourly_dataframe['date'].dt.date
    # Agrupamos por día y calculamos la media
    media_suelo_diaria = hourly_dataframe.groupby('dia')['soil_temperature_6cm'].mean().values

    # 3. Procesar datos diarios
    daily = response.Daily()
    daily_dataframe = pd.DataFrame({
        "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(2).ValuesAsNumpy(),
        "wind_speed_10m_mean": daily.Variables(3).ValuesAsNumpy(),
        "relative_humidity_2m_mean": daily.Variables(4).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(5).ValuesAsNumpy(),
        "soil_temperature_mean": media_suelo_diaria
    })

    # 4. Extraer variables para la IA
    # El índice 0 es HOY, el índice 1 es MAÑANA
    clima_hoy = {
        "t_max": float(daily_dataframe['temperature_2m_max'][0]),
        "t_min": float(daily_dataframe['temperature_2m_min'][0]),
        "t_suelo": float(daily_dataframe['soil_temperature_mean'][0]),
        "humedad": float(daily_dataframe['relative_humidity_2m_mean'][0]),
        "viento": float(daily_dataframe['wind_speed_10m_mean'][0]),
        "precip_hoy": float(daily_dataframe['precipitation_sum'][0]),
        "eto_hoy": float(daily_dataframe['et0_fao_evapotranspiration'][0])
    }

    clima_manana = {
        "precip_manana": float(daily_dataframe['precipitation_sum'][1]),
        "eto_manana": float(daily_dataframe['et0_fao_evapotranspiration'][1])
    }

    return clima_hoy, clima_manana

def get_past_months_data(fecha_plantacion):
    """
    Calcula el GDU acumulado y el agotamiento del suelo (Dr) iterando 
    el clima real desde la fecha de plantación hasta hoy.
    """
    hoy = datetime.date.today()
    dias_desde_plantacion = (hoy - fecha_plantacion).days

    if dias_desde_plantacion <= 0:
        return 0.0, 0.0 # Si no se ha plantado aún o se planta hoy

    # Open-Meteo permite hasta 92 días en el endpoint de forecast gratuito
    dias_a_pedir = min(dias_desde_plantacion, 60)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 42.4413,
        "longitude": -5.7928,
        "daily": ["temperature_2m_max", "temperature_2m_min", "et0_fao_evapotranspiration", "wind_speed_10m_mean", "relative_humidity_2m_mean", "precipitation_sum"],
        "past_days": dias_a_pedir,
        "forecast_days": 0, # Solo queremos el pasado, el presente/futuro ya lo saca tu otra función
        "wind_speed_unit": "ms",
        "timezone": "Europe/Madrid"
    }

    # Llamada a la API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_dataframe = pd.DataFrame({
        "t_max": daily.Variables(0).ValuesAsNumpy(),
        "t_min": daily.Variables(1).ValuesAsNumpy(),
        "eto": daily.Variables(2).ValuesAsNumpy(),
        "v_viento": daily.Variables(3).ValuesAsNumpy(),
        "humedad": daily.Variables(4).ValuesAsNumpy(),
        "precipitacion": daily.Variables(5).ValuesAsNumpy(),
    })

    # Inicializamos el gemelo digital
    calculadora = CalculadoraCultivo()
    gdu_acumulado = 0.0

    # Iteramos día a día desde la plantación hasta ayer
    for i in range(len(daily_dataframe)):
        t_max = daily_dataframe['t_max'].iloc[i]
        t_min = daily_dataframe['t_min'].iloc[i]
        eto = daily_dataframe['eto'].iloc[i]
        v_viento = daily_dataframe['v_viento'].iloc[i]
        h_humedad = daily_dataframe['humedad'].iloc[i]
        precipitacion = daily_dataframe['precipitacion'].iloc[i]


        print("temeratura max:", t_max)
        print("temeratura min:", t_min)
        print("eto:", eto)
        print("viento:", v_viento)
        print("humedad:", h_humedad)
        print("precipitacion:", precipitacion)
        print("gdu_acumulado:", gdu_acumulado)
        print("Dr actual:", calculadora.Dr)
        print("-" * 40)

        # 1. Crecimiento fisiológico
        gdu_acumulado += calcular_gdu(t_max, t_min)
        altura = calcular_altura_por_gdu(gdu_acumulado)
        dia_plantacion_actual = i + 1

        # NOTA: Aquí asumimos que no se ha aplicado riego en el pasado (secano). 
        # Si tienes un registro de riegos pasados, deberías inyectarlo aquí en lugar de 0.0
        riego_historico = 0.0 

        # 2. Cálculo de Evapotranspiración y coeficiente Ke
        etc, kc_calculada = calculadora.Calcular_ETc(
            eto=eto, 
            dias_desde_plantacion=dia_plantacion_actual, 
            viento=v_viento, 
            precipitacion=precipitacion, 
            riego=riego_historico, 
            humedad=h_humedad, 
            altura=altura, 
            gdu_acumulado=gdu_acumulado
        )

        # 3. Aplicar estrés hídrico si la planta se está secando
        coeficiente_ks = calculadora.calcular_estres_ks()
        etc_real = etc * coeficiente_ks

        # 4. Actualizar el balance profundo de la raíz (esto modifica internamente calculadora.Dr)
        calculadora.actualizar_balance_radicular(
            precipitacion_real=precipitacion, 
            riego_ejecutado=riego_historico, 
            etc_real=etc_real
        )

    # Devolvemos el GDU alcanzado y el Dr con el que amanecemos hoy
    return float(calculadora.Dr), float(gdu_acumulado)

if __name__ == "__main__":

    get_past_months_data(datetime.date(2026, 3, 4))

