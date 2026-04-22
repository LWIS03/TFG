import pandas as pd
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement

def main():
    print("1. Cargando y limpiando los datasets originales...")
    
    # --- 1. LIMPIEZA DE DATOS ---
    cols_to_use_1 = range(18)
    cols_to_use_2 = range(10)  

    dades_absolutes = pd.read_csv("../DATASET_BUSTILLO/aquacrop.csv", sep=";", decimal=",", encoding="latin-1", usecols=cols_to_use_1)
    dades_calculades = pd.read_csv("../DATASET_BUSTILLO/dades_calculades.csv", sep=";", decimal=",", encoding="latin-1", usecols=cols_to_use_2)

    dades_absolutes = dades_absolutes.drop(0).reset_index(drop=True)
    dades_absolutes.columns = dades_absolutes.columns.str.strip()

    dades_calculades = dades_calculades.drop(0).reset_index(drop=True)
    dades_calculades.columns = dades_calculades.columns.str.strip()

    for col in dades_absolutes.columns:
        if col != 'Fecha' and 'Hora' not in col:
            dades_absolutes[col] = pd.to_numeric(dades_absolutes[col].astype(str).str.replace(',', '.'), errors='coerce')

    for col in dades_calculades.columns:
        if col != 'Fecha' and 'Hora' not in col:
            dades_calculades[col] = pd.to_numeric(dades_calculades[col].astype(str).str.replace(',', '.'), errors='coerce')

    df_aquacrop = dades_absolutes[['Fecha', 'Temp. max', 'Temp. min', 'Precipitacion']].copy()
    df_aquacrop["ReferenceET"] = dades_calculades["ETo(P.MON.)"]

    # Renombramos al inglés y ordenamos para AquaCrop
    df_aquacrop = df_aquacrop.rename(columns={
        "Fecha": "Date", 
        "Temp. max": "MaxTemp", 
        "Temp. min": "MinTemp", 
        "Precipitacion": "Precipitation"
    })
    
    df_aquacrop["Date"] = pd.to_datetime(df_aquacrop["Date"], format="%d/%m/%Y")
    df_aquacrop = df_aquacrop[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET', 'Date']]

    # --- 2. CÁLCULOS TÉRMICOS PREVIOS ---
    # Pre-calculamos la temperatura estimada del suelo para decidir cuándo sembrar
    df_aquacrop['Temp_Media'] = (df_aquacrop['MaxTemp'] + df_aquacrop['MinTemp']) / 2.0
    df_aquacrop['Temp_Suelo_Est'] = df_aquacrop['Temp_Media'].rolling(window=4, min_periods=1).mean()

    # --- 3. CONFIGURACIÓN BASE DEL ENTORNO FAO ---
    print("\n2. Configurando el entorno del Maíz y la estrategia del agricultor...")
    suelo = Soil('Loam')
    # El suelo empieza más seco (Punto de marchitez en la capa superior) para simular realidad
    agua_inicial = InitialWaterContent(value=['FW']) 
    # Estrategia realista: Regar cuando se agota el 35% (Mantiene la humedad alta, gasta más agua pero produce más)
    estrategia_riego = IrrigationManagement(irrigation_method=1, SMT=[100]*4) 
    #estrategia_riego = IrrigationManagement(irrigation_method=0) 

    # Listas para guardar resultados de todos los años
    lista_resultados_comparacion = []
    lista_auditoria = []
    lista_rendimiento_anual = []   # Yield por año — necesario para la evaluación comparativa
    
    años_simulados = df_aquacrop['Date'].dt.year.unique()

    print("\n3. Iniciando simulaciones dinámicas año a año...")
    
    for año in años_simulados:
        df_año = df_aquacrop[df_aquacrop['Date'].dt.year == año].copy()
        
        # Omitimos años que no tengan datos completos (ej. si el año 2026 termina en marzo)
        if len(df_año) < 300: 
            print(f"[{año}] Saltando año por falta de datos climáticos completos.")
            continue
            
        # --- LÓGICA DE SIEMBRA DINÁMICA ---
        condicion_fecha = ((df_año['Date'].dt.month == 4) & (df_año['Date'].dt.day >= 15)) | (df_año['Date'].dt.month >= 5)
        condicion_temp = df_año['Temp_Suelo_Est'] >= 9.0
        
        dias_validos = df_año[condicion_fecha & condicion_temp]
        
        if dias_validos.empty:
            fecha_siembra_str = "05/15" 
            print(f"   🌱 [{año}] Clima frío. Siembra forzada el: 15/05/{año}")
        else:
            dia_siembra = dias_validos.iloc[0]['Date']
            fecha_siembra_str = dia_siembra.strftime('%m/%d')
            print(f"   🌱 [{año}] Clima óptimo. Siembra el: {dia_siembra.strftime('%d/%m/%Y')}")

        # --- EJECUCIÓN DE AQUACROP PARA ESTE AÑO ---
        cultivo = Crop('Maize', planting_date=fecha_siembra_str)
        
        inicio_año = df_año['Date'].min().strftime('%Y/%m/%d')
        fin_año = df_año['Date'].max().strftime('%Y/%m/%d')
        
        modelo = AquaCropModel(
            sim_start_time=inicio_año,
            sim_end_time=fin_año,
            weather_df=df_año, 
            soil=suelo,
            crop=cultivo,
            irrigation_management=estrategia_riego,
            initial_water_content=agua_inicial
        )
        
        modelo.run_model(till_termination=True)
        estadisticas = modelo._outputs.final_stats


        columna_yield = [col for col in estadisticas.columns if 'Yield' in col]
        
        if len(columna_yield) > 0:
            nombre_col = columna_yield[0]
            rendimiento_kg_ha = estadisticas[nombre_col].iloc[0] * 1000
            print(f"   🌾 Rendimiento AquaCrop: {rendimiento_kg_ha:.2f} kg/ha")
        else:
            rendimiento_kg_ha = 0.0
            print(f"   ⚠️ No se encontró la columna 'Yield'. Columnas disponibles: {estadisticas.columns.tolist()}")

        # Guardamos el rendimiento anual para exportarlo (necesario para comparación justa)
        lista_rendimiento_anual.append({'Año': año, 'Rendimiento_AquaCrop_kg_ha': rendimiento_kg_ha})


        # --- EXTRACCIÓN DE DATOS DE ESTE AÑO ---
        flujo_agua = modelo._outputs.water_flux.copy()
        crecimiento = modelo._outputs.crop_growth.copy()
        fechas_año = pd.date_range(start=inicio_año, end=fin_año, freq='D')

        # ── MEJORA: filtrar por fechas REALES de siembra y cosecha ───────────
        # Antes: between(5, 9) → siempre 01/05-30/09 sin importar el año.
        # Ahora: usamos la fecha de siembra dinámica calculada arriba y la fecha
        # de cosecha que AquaCrop reporta en final_stats.
        # Esto refleja la campaña real simulada (puede variar ±2-3 semanas/año).
        fecha_siembra_real = dia_siembra if not dias_validos.empty \
                             else pd.to_datetime(f"{año}-05-15")
        try:
            fecha_cosecha_real = pd.to_datetime(
                estadisticas['Harvest Date (YYYY/MM/DD)'].iloc[0]
            )
        except Exception:
            # Fallback: si AquaCrop no reporta la columna, usamos fin de año
            fecha_cosecha_real = fechas_año[-1]

        print(f"   📅 Siembra: {fecha_siembra_real.strftime('%d/%m/%Y')} | "
              f"Cosecha: {fecha_cosecha_real.strftime('%d/%m/%Y')} | "
              f"Duración: {(fecha_cosecha_real - fecha_siembra_real).days} días")
        # ─────────────────────────────────────────────────────────────────────

        # 1. Dataset para comparar (Solo Riego)
        df_temp = pd.DataFrame({
            'Año': año,
            'Fecha': fechas_año.strftime('%d/%m/%Y'),
            'Fecha_dt': fechas_año,
            'Riego_Optimo_FAO_mm': flujo_agua['IrrDay'].values
        })
        
        # 2. Dataset de Auditoría
        df_audi_temp = pd.DataFrame({
            'Fecha': fechas_año.strftime('%d/%m/%Y'),
            'Fecha_dt': fechas_año,
            'Riego_Decidido_mm': flujo_agua['IrrDay'].values,
            'Lluvia_mm': df_año['Precipitation'].values
        })
        
        df_audi_temp['Evaporacion_Suelo_mm'] = flujo_agua.get('Es', pd.Series([0]*len(flujo_agua))).values
        df_audi_temp['Transpiracion_Planta_mm'] = flujo_agua.get('Tr', pd.Series([0]*len(flujo_agua))).values
        
        tr = flujo_agua.get('Tr', pd.Series([0.0] * len(flujo_agua)))
        tr_pot = flujo_agua.get('TrPot', pd.Series([0.0] * len(flujo_agua)))
        ks_calculado = tr / tr_pot.replace(0.0, 1.0)
        ks_calculado = ks_calculado.where(tr_pot > 0.0, 1.0)
        df_audi_temp['Estrés_Hídrico_pct'] = (1.0 - ks_calculado) * 100
        
        # Filtramos por siembra→cosecha reales y eliminamos columna auxiliar
        mask = (df_temp['Fecha_dt'] >= pd.to_datetime(fecha_siembra_real)) & \
               (df_temp['Fecha_dt'] <= pd.to_datetime(fecha_cosecha_real))
        df_temp      = df_temp[mask].drop(columns=['Fecha_dt'])
        df_audi_temp = df_audi_temp[mask].drop(columns=['Fecha_dt'])
        
        lista_resultados_comparacion.append(df_temp)
        lista_auditoria.append(df_audi_temp)

    # --- 4. ENSAMBLAJE FINAL Y EXPORTACIÓN ---
    print("\n4. Ensamblando y exportando resultados finales...")
    
    df_resultados_finales = pd.concat(lista_resultados_comparacion, ignore_index=True)
    df_auditoria_final = pd.concat(lista_auditoria, ignore_index=True)
    df_rendimiento_anual = pd.DataFrame(lista_rendimiento_anual)
    
    df_resultados_finales.to_csv("riego_optimo_aquacrop_dinamico.csv", index=False, sep=";", decimal=",")
    df_auditoria_final.to_csv("auditoria_calculos_fao.csv", index=False, sep=";", decimal=",")
    df_rendimiento_anual.to_csv("rendimiento_anual_aquacrop.csv", index=False, sep=";", decimal=",")
    
    agua_total = df_resultados_finales['Riego_Optimo_FAO_mm'].sum()
    dias_regados = (df_resultados_finales['Riego_Optimo_FAO_mm'] > 0).sum()
    
    print(f"✅ ÉXITO. Simulación completada.")
    print(f"📊 Archivos generados: 'riego_optimo_aquacrop_dinamico.csv' y 'auditoria_calculos_fao.csv'")
    print(f"💧 Agua neta total utilizada en todos los años: {agua_total:.2f} mm")
    print(f"📅 Días totales con riego aplicado: {dias_regados} días")

    # --- 5. RESUMEN DE AGUA POR TEMPORADA ---
    print("\n5. Calculando el gasto de agua por temporada...")
    
    # Convertimos la columna Fecha a formato datetime para poder extraer el año
    df_resultados_finales['Fecha_dt'] = pd.to_datetime(df_resultados_finales['Fecha'], format='%d/%m/%Y')
    df_resultados_finales['Año'] = df_resultados_finales['Fecha_dt'].dt.year
    
    # Agrupamos por año y sumamos los milímetros de riego
    resumen_temporadas = df_resultados_finales.groupby('Año')['Riego_Optimo_FAO_mm'].sum().reset_index()
    resumen_temporadas['Riego_Optimo_FAO_mm'] = resumen_temporadas['Riego_Optimo_FAO_mm'].round(2)
    
    # Guardamos esta tabla tan útil en un nuevo archivo
    resumen_temporadas.to_csv("resumen_agua_por_temporada.csv", index=False, sep=";", decimal=",")
    
    # Limpiamos la columna auxiliar
    df_resultados_finales = df_resultados_finales.drop(columns=['Fecha_dt', 'Año'])
    
    print("\n📊 Resumen anual generado ('resumen_agua_por_temporada.csv'):")
    print(resumen_temporadas.to_string(index=False))

if __name__ == "__main__":
    main()