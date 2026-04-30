import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Noise variance per forecast day: increases with distance (less certain = more noise)
VARIANZAS_PRECIP = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
VARIANZAS_ETO    = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# Exponential decay weights for the irrigation decision function (alpha=0.85)
ALPHA = 0.85
PESOS_7D = np.array([ALPHA**d for d in range(7)])  # [1.0, 0.85, 0.72, 0.61, 0.52, 0.44, 0.38]

class CalculadoraCultivo:
    def __init__(self, fw = 1.0):
        # Parámetros capa superficial (Ke)
        self.AET = 1000 * (0.25 - 0.5 * 0.12) * 0.10
        self.afe = 9.0
        self.fw = fw
        self.De = 0.0
        self.Kc_min = 0.15

        # Parámetros zona radicular (Dr) para el balance hídrico profundo del maíz
        self.fc = 0.25
        self.wp = 0.12
        self.z_r = 0.60
        self.p_base = 0.55
        self.TAW = 1000 * (self.fc - self.wp) * self.z_r
        self.RAW = self.p_base * self.TAW
        self.Dr = 0.0
        self.factor_rendimiento_diario = 1.0

    def actualizar_penalizacion_rendimiento(self, etc_hoy, etc_real_hoy, fase_nombre):
        ky_hoy = self.obtener_ky_fase(fase_nombre)
        if etc_hoy > 0:
            deficit_diario = 1.0 - (etc_real_hoy / etc_hoy)
            penalizacion_diaria = ky_hoy * deficit_diario * (etc_hoy / 800.0)
            self.factor_rendimiento_diario *= (1.0 - penalizacion_diaria)
            self.factor_rendimiento_diario = max(0.0, self.factor_rendimiento_diario)

    def actualizar_taw_y_raw_dinamico(self, etc_hoy, profundidad_raiz_hoy):
        self.z_r = profundidad_raiz_hoy
        self.TAW = 1000 * (self.fc - self.wp) * self.z_r
        p_dinamico = self.p_base + 0.04 * (5.0 - etc_hoy)
        p_dinamico = max(0.1, min(p_dinamico, 0.8))
        self.RAW = p_dinamico * self.TAW

    def obtener_ky_fase(self, fase_nombre):
        if "Inicial" in fase_nombre or "Desarrollo" in fase_nombre:
            return 0.40
        elif "Madurez" in fase_nombre:
            return 1.30
        elif "Final" in fase_nombre:
            return 0.50
        else:
            return 0.0

    def calcular_estres_ks(self):
        if self.Dr <= self.RAW:
            return 1.0
        else:
            ks = (self.TAW - self.Dr) / (self.TAW - self.RAW)
            return max(0.0, min(ks, 1.0))

    def calcular_perdida_rendimiento(self, sum_ET_real, sum_ET_max, fase_nombre):
        ky = self.obtener_ky_fase(fase_nombre)
        if sum_ET_max == 0:
            return 0.0
        deficit_evapotranspiracion = 1 - (sum_ET_real / sum_ET_max)
        return ky * deficit_evapotranspiracion

    def ecuacion_kcbMax(self, viento, humedad, altura):
        return 1.2 + (0.04 * (viento - 2) - 0.004 * (humedad - 45)) * pow((altura / 3), 0.3)

    def ajustar_kcb_por_clima(self, kcb_tabla, u2, hrmin, altura):
        if kcb_tabla <= 0.45:
            return kcb_tabla
        u2    = max(1.0, min(u2,    6.0))
        hrmin = max(20.0, min(hrmin, 80.0))
        altura = max(0.1,  min(altura, 10.0))
        ajuste = (0.04 * (u2 - 2) - 0.004 * (hrmin - 45)) * pow(altura / 3, 0.3)
        return kcb_tabla + ajuste

    def calcular_balance_diario_humedad_suelo(self, ke, eto, precipitacion, riego, few):
        RO = 0.0
        T_ew = 0.0
        lluvia_efectiva = precipitacion - RO
        riego_efectivo = riego / self.fw if self.fw > 0.0 else 0.0
        Ei = ke * eto
        evaporacion_concentrada = Ei / max(few, 0.01)
        entradas_totales = lluvia_efectiva + riego_efectivo
        DPe = max(0.0, entradas_totales - self.De)
        self.De = self.De - lluvia_efectiva - riego_efectivo + evaporacion_concentrada + T_ew + DPe
        self.De = max(0.0, min(self.De, self.AET))

    def calcular_kcb_por_gdu(self, gdu_acumulado):
        kcb_ini = 0.15
        kcb_mid = 1.15
        kcb_fin = 0.15
        gdu_ini  = 250
        gdu_dev  = 600
        gdu_mid  = 1100
        gdu_late = 1275

        if gdu_acumulado <= gdu_ini:
            return kcb_ini
        elif gdu_acumulado <= gdu_dev:
            progreso = (gdu_acumulado - gdu_ini) / (gdu_dev - gdu_ini)
            return kcb_ini + progreso * (kcb_mid - kcb_ini)
        elif gdu_acumulado <= gdu_mid:
            return kcb_mid
        elif gdu_acumulado <= gdu_late:
            progreso = (gdu_acumulado - gdu_mid) / (gdu_late - gdu_mid)
            return kcb_mid - progreso * (kcb_mid - kcb_fin)
        else:
            return kcb_fin

    def calcular_ke(self, viento, precipitacion, riego, humedad, altura, Kcb, eto):
        KcAux  = self.ecuacion_kcbMax(viento, humedad, altura)
        KcAux2 = Kcb + 0.05
        Kc_max = max(KcAux, KcAux2)

        fc = pow(((Kcb - self.Kc_min) / (Kc_max - self.Kc_min)), 1 + 0.5 * altura)
        fc = max(0.0, min(fc, 0.99))

        fw_actual = 1.0 if precipitacion > 3.0 else self.fw
        self.fw = fw_actual
        few = min(1 - fc, fw_actual)

        if self.De <= self.afe:
            kr = 1.0
        else:
            kr = (self.AET - self.De) / (self.AET - self.afe)
            kr = max(0.0, min(kr, 1.0))

        ke = kr * (Kc_max - Kcb)
        ke = max(0.0, min(ke, few * Kc_max))

        self.calcular_balance_diario_humedad_suelo(ke, eto, precipitacion, riego, few)
        return ke

    def Calcular_ETc(self, eto, dias_desde_plantacion, viento, precipitacion, riego, humedad, altura, gdu_acumulado):
        kcb_tabla = self.calcular_kcb_por_gdu(gdu_acumulado)
        kcb = self.ajustar_kcb_por_clima(kcb_tabla, viento, humedad, altura)
        ke = self.calcular_ke(viento, precipitacion, riego, humedad, altura, kcb, eto)
        return (kcb + ke) * eto, kcb + ke

    def inyectar_ruido_estocastico_meteorologico(self, pronostico_precipitacion, coeficiente_varianza=0.30):
        if pronostico_precipitacion <= 0.5:
            vector_ruido = np.random.normal(loc=0.0, scale=0.5)
        else:
            vector_ruido = np.random.normal(loc=0.0, scale=(pronostico_precipitacion * coeficiente_varianza))
        return round(max(0.0, pronostico_precipitacion + vector_ruido), 3)

    def evaluar_accion_riego_predictivo_7d(self, etc_estimada_hoy, precip_fcst_7d, etc_fcst_7d, gdu_acumulado):
        """
        Decides optimal irrigation using a 7-day weighted forecast.
        Each future day is discounted by alpha=0.85 per day (day 1 = full trust,
        day 7 = 38% trust), reflecting increasing forecast uncertainty.
        """
        if gdu_acumulado >= 1150:
            return 0.0

        n = len(precip_fcst_7d)
        pesos = PESOS_7D[:n]

        precip_ponderada = float(np.dot(pesos, precip_fcst_7d))
        etc_ponderada    = float(np.dot(pesos, etc_fcst_7d))

        agotamiento_futuro = self.Dr + etc_estimada_hoy + etc_ponderada - precip_ponderada

        volumen_riego_optimo = 0.0
        umbral_disparo = self.RAW * 0.95

        if agotamiento_futuro > umbral_disparo:
            margen_seguridad_lluvia = self.TAW * 0.35
            volumen_riego_optimo = agotamiento_futuro - margen_seguridad_lluvia
            volumen_riego_optimo = max(0.0, min(volumen_riego_optimo, self.TAW))

        return round(volumen_riego_optimo, 3)

    def actualizar_balance_radicular(self, precipitacion_real, riego_ejecutado, etc_real):
        self.Dr = self.Dr - precipitacion_real - riego_ejecutado + etc_real
        percolacion_profunda = 0.0
        if self.Dr < 0:
            percolacion_profunda = abs(self.Dr)
            self.Dr = 0.0
        self.Dr = min(self.Dr, self.TAW)
        return round(percolacion_profunda, 3)


def calcular_columna_temperatura_suelo(df, dias_media=3):
    temp_aire = pd.to_numeric(df['Temp. media'], errors='coerce')
    return temp_aire.rolling(window=dias_media, min_periods=4).mean()


def calcular_raiz_por_gdu(gdu_acumulado):
    zr_min = 0.10
    zr_max = 1.20
    if gdu_acumulado < 100:
        return zr_min
    elif gdu_acumulado < 350:
        progreso = (gdu_acumulado - 100) / (350 - 100)
        return zr_min + progreso * (zr_max - zr_min) * 0.3
    elif gdu_acumulado < 700:
        progreso = (gdu_acumulado - 350) / (700 - 350)
        return zr_min + (zr_max - zr_min) * 0.3 + progreso * (zr_max - zr_min) * 0.7
    else:
        return zr_max


def calcular_altura_por_gdu(gdu_acumulado):
    h_max = 2.60
    h_min = 0.05
    h_v4  = 0.40
    if gdu_acumulado < 100:
        return h_min
    elif gdu_acumulado < 350:
        progreso = (gdu_acumulado - 100) / (350 - 100)
        return h_min + progreso * (h_v4 - h_min)
    elif gdu_acumulado < 700:
        progreso = (gdu_acumulado - 350) / (700 - 350)
        return h_v4 + progreso * (h_max - h_v4)
    else:
        return h_max


def calcular_gdu(t_max, t_min):
    try:
        t_max = float(t_max)
        t_min = float(t_min)
    except (ValueError, TypeError):
        return 0.0
    t_max_adj = min(max(t_max, 10.0), 30.0)
    t_min_adj = min(max(t_min, 10.0), 30.0)
    return max(((t_max_adj + t_min_adj) / 2.0) - 10.0, 0.0)


def calcular_fase_y_kc(gdu_acumulado):
    if gdu_acumulado < 250:
        Kc = 0.30 + 0.20 * (gdu_acumulado / 250)
        return Kc, "Fase Inicial (VE-V4)"
    elif gdu_acumulado < 600:
        Kc = 0.50 + 0.55 * ((gdu_acumulado - 250) / 350)
        return Kc, "Fase de Desarrollo (V5-V12)"
    elif gdu_acumulado < 1100:
        Kc = 1.05 + 0.15 * ((gdu_acumulado - 600) / 500)
        return Kc, "Fase de Madurez (V13-R3)"
    elif gdu_acumulado < 1275:
        Kc = 1.20 - 0.40 * ((gdu_acumulado - 1100) / 175)
        return Kc, "Fase Final (R4-R5)"
    else:
        return 0.55, "Fase de Cosecha (R6)"


def _get_forecast_day(dades_absolutes, dades_calculades, index, d, fallback_eto):
    """Returns (precip_raw, eto_raw) for index + d+1, using fallbacks at boundaries."""
    future_idx = index + d + 1
    if future_idx < len(dades_absolutes):
        p = dades_absolutes.at[future_idx, 'Precipitacion']
        e = dades_calculades.at[future_idx, 'ETo(P.MON.)']
        p = 0.0 if pd.isna(p) else float(p)
        e = fallback_eto if pd.isna(e) else float(e)
    else:
        p = 0.0
        e = fallback_eto
    return p, e


def main():
    print("Iniciando proceso con modelo predictivo estocástico (horizonte 7 días)...")

    cols_to_use_1 = range(18)
    cols_to_use_2 = range(10)
    try:
        dades_absolutes = pd.read_csv("../../DATASET_BUSTILLO/dades_absolutes_registrades.csv",
                                      sep=";", encoding="latin-1", usecols=cols_to_use_1)
        dades_calculades = pd.read_csv("../../DATASET_BUSTILLO/dades_calculades.csv",
                                       sep=";", encoding="latin-1", usecols=cols_to_use_2)

        dades_absolutes = dades_absolutes.drop(0).reset_index(drop=True)
        dades_absolutes.columns = dades_absolutes.columns.str.strip()
        dades_calculades = dades_calculades.drop(0).reset_index(drop=True)
        dades_calculades.columns = dades_calculades.columns.str.strip()

        for col in dades_absolutes.columns:
            if col != 'Fecha' and 'Hora' not in col:
                dades_absolutes[col] = pd.to_numeric(
                    dades_absolutes[col].astype(str).str.replace(',', '.'), errors='coerce')
        for col in dades_calculades.columns:
            if col != 'Fecha' and 'Hora' not in col:
                dades_calculades[col] = pd.to_numeric(
                    dades_calculades[col].astype(str).str.replace(',', '.'), errors='coerce')

        dades_calculades['Temp_suelo_est'] = calcular_columna_temperatura_suelo(dades_absolutes, dias_media=4)

        print(f"Procesando {len(dades_absolutes)} filas...")

        # --- Base state lists ---
        lista_fecha            = []
        lista_dias_plantacion  = []
        lista_gdu_acumulado    = []
        lista_temp_max         = []
        lista_temp_min         = []
        lista_temp_suelo       = []
        lista_humedad          = []
        lista_viento           = []
        lista_precip_hoy       = []
        lista_eto              = []
        lista_riego_recomendado = []

        # --- 7-day forecast lists (index 0 = D+1, index 6 = D+7) ---
        lista_precip_fcst = [[] for _ in range(7)]
        lista_eto_fcst    = [[] for _ in range(7)]

        en_temporada          = False
        gdu_acumulado         = 0.0
        dias_desde_plantacion = 0
        calc                  = None

        for index, row in dades_absolutes.iterrows():

            fecha = row['Fecha']
            fecha_dt = pd.to_datetime(fecha, format='%d/%m/%Y', errors='coerce')
            if pd.isna(fecha_dt):
                continue

            mes = fecha_dt.month
            dia = fecha_dt.day

            t_max                  = row['Temp. max']
            t_min                  = row['Temp. min']
            v_viento               = row['Vel. viento']
            h_humedad              = row['Hum. med.']
            precipitacion_pronostico = row['Precipitacion']
            precipitacion_efectiva = dades_calculades.at[index, 'Precipit. efectiva(P.MON.)']
            eto                    = dades_calculades.at[index, 'ETo(P.MON.)']
            temp_suelo             = dades_calculades.at[index, 'Temp_suelo_est']

            # ── LÓGICA DE SIEMBRA ──────────────────────────────────────────
            if not en_temporada:
                ventana_siembra = (mes == 4 and dia >= 15) or (mes == 5)
                temp_adecuada   = not pd.isna(temp_suelo) and temp_suelo >= 9.0

                if ventana_siembra and temp_adecuada:
                    en_temporada          = True
                    gdu_acumulado         = 0.0
                    dias_desde_plantacion = 1
                    calc = CalculadoraCultivo()

                    sum_etc_max_temporada  = 0.0
                    sum_etc_real_temporada = 0.0
                    total_riego_mm_temporada = 0.0
                    print(f"🌱 SIEMBRA: {fecha} (Temp. Suelo: {temp_suelo:.2f}ºC)")
                else:
                    # Barbecho: store base variables, zero-out forecasts
                    lista_fecha.append(fecha)
                    lista_dias_plantacion.append(0)
                    lista_gdu_acumulado.append(0.0)
                    lista_temp_max.append(t_max)
                    lista_temp_min.append(t_min)
                    lista_viento.append(v_viento)
                    lista_humedad.append(h_humedad)
                    lista_precip_hoy.append(precipitacion_efectiva)
                    lista_eto.append(eto)
                    lista_temp_suelo.append(temp_suelo)
                    lista_riego_recomendado.append(0.0)
                    for d in range(7):
                        lista_precip_fcst[d].append(0.0)
                        lista_eto_fcst[d].append(0.0)
                    continue

            # ── CRECIMIENTO Y FISIOLOGÍA DE HOY ───────────────────────────
            calcular_gdu_diario = calcular_gdu(t_max, t_min)
            gdu_acumulado      += calcular_gdu_diario
            kc_gdu, fase        = calcular_fase_y_kc(gdu_acumulado)
            altura_planta       = calcular_altura_por_gdu(gdu_acumulado)
            raiz_planta         = calcular_raiz_por_gdu(gdu_acumulado)

            # ── HORIZONTE DE 7 DÍAS CON RUIDO CRECIENTE ───────────────────
            eto_base = eto if not pd.isna(eto) else 0.0
            precip_fcst_7d = []
            eto_fcst_7d    = []

            for d in range(7):
                p_raw, e_raw = _get_forecast_day(dades_absolutes, dades_calculades, index, d, eto_base)

                p_noisy = calc.inyectar_ruido_estocastico_meteorologico(p_raw, coeficiente_varianza=VARIANZAS_PRECIP[d])
                ruido_e = np.clip(np.random.normal(loc=1.0, scale=VARIANZAS_ETO[d]), 0.3, 2.0)
                e_noisy = max(0.0, e_raw * ruido_e)

                precip_fcst_7d.append(p_noisy)
                eto_fcst_7d.append(e_noisy)

                lista_precip_fcst[d].append(p_noisy)
                lista_eto_fcst[d].append(e_noisy)

            # ETc forecast list for the decision function
            etc_fcst_7d = [e * kc_gdu for e in eto_fcst_7d]

            # ── DECISIÓN DE RIEGO ──────────────────────────────────────────
            etc_estimada_hoy = eto_base * kc_gdu
            riego_aplicable  = calc.evaluar_accion_riego_predictivo_7d(
                etc_estimada_hoy, precip_fcst_7d, etc_fcst_7d, gdu_acumulado)

            # ── RUIDO SOBRE LA LLUVIA DE HOY (realidad del terreno) ────────
            precip_base = precipitacion_efectiva if not pd.isna(precipitacion_efectiva) else precipitacion_pronostico
            precipitacion_efectiva_real = calc.inyectar_ruido_estocastico_meteorologico(precip_base)

            # ── ETc Y ESTRÉS ───────────────────────────────────────────────
            etc, kc_calculada = calc.Calcular_ETc(
                eto_base, dias_desde_plantacion, v_viento, precipitacion_efectiva_real,
                riego_aplicable, h_humedad, altura_planta, gdu_acumulado)

            calc.actualizar_taw_y_raw_dinamico(etc, raiz_planta)
            coeficiente_ks = calc.calcular_estres_ks()
            etc_real       = etc * coeficiente_ks

            calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)

            sum_etc_max_temporada  += etc
            sum_etc_real_temporada += etc_real
            total_riego_mm_temporada += riego_aplicable

            percolacion_prof = calc.actualizar_balance_radicular(
                precipitacion_efectiva_real, riego_aplicable, etc_real)

            # ── ALMACENAMIENTO ─────────────────────────────────────────────
            lista_fecha.append(fecha)
            lista_dias_plantacion.append(dias_desde_plantacion)
            lista_gdu_acumulado.append(gdu_acumulado)
            lista_temp_max.append(t_max)
            lista_temp_min.append(t_min)
            lista_viento.append(v_viento)
            lista_humedad.append(h_humedad)
            lista_precip_hoy.append(precipitacion_efectiva_real)
            lista_eto.append(eto_base)
            lista_temp_suelo.append(temp_suelo)
            lista_riego_recomendado.append(riego_aplicable)

            dias_desde_plantacion += 1

            # ── FIN DE TEMPORADA ───────────────────────────────────────────
            if (gdu_acumulado >= 1275) or (mes >= 11 and dia >= 15):
                rendimiento_max_kg_ha = 15000.0
                precio_maiz_kg        = 0.23
                coste_agua_m3         = 0.04
                coste_fijo_boe        = 32.18

                coste_agua_ha   = total_riego_mm_temporada * 10 * coste_agua_m3 + coste_fijo_boe
                rendimiento_real = rendimiento_max_kg_ha * calc.factor_rendimiento_diario
                ingresos_cosecha = rendimiento_real * precio_maiz_kg
                recompensa_final = ingresos_cosecha - coste_agua_ha

                print(f"🌾 COSECHA: {fecha} (GDU: {gdu_acumulado:.2f})")
                print(f"   -> Riego gastado: {total_riego_mm_temporada:.2f} mm (-{coste_agua_ha:.2f} €/ha)")
                print(f"   -> Rendimiento:   {rendimiento_real:.2f} kg/ha (+{ingresos_cosecha:.2f} €/ha)")
                print(f"   -> RECOMPENSA:    {recompensa_final:.2f} €/ha de Beneficio Bruto")
                en_temporada = False

        # ── CONSTRUCCIÓN DEL DATASET ───────────────────────────────────────
        base_cols = {
            'Fecha':                lista_fecha,
            'Dias_Plantacion':      lista_dias_plantacion,
            'GDU_Acumulado':        lista_gdu_acumulado,
            'Temp_Max_C':           lista_temp_max,
            'Temp_Min_C':           lista_temp_min,
            'Temp_Suelo_C':         lista_temp_suelo,
            'Humedad_Relativa_pct': lista_humedad,
            'Velocidad_Viento_ms':  lista_viento,
            'Precipitacion_Hoy_mm': lista_precip_hoy,
            'ETo_mm':               lista_eto,
        }

        resultados = pd.DataFrame(base_cols)
        resultados_regresion = pd.DataFrame({**base_cols, 'Riego_Recomendado_mm': lista_riego_recomendado})

        # Add 7-day per-day forecast columns (D+1 … D+7)
        for d in range(7):
            col_p = f'Precip_D{d+1}_mm'
            col_e = f'ETo_D{d+1}_mm'
            resultados[col_p]          = lista_precip_fcst[d]
            resultados[col_e]          = lista_eto_fcst[d]
            resultados_regresion[col_p] = lista_precip_fcst[d]
            resultados_regresion[col_e] = lista_eto_fcst[d]

        # Simulated water price (random, no fixed seed — model learns the principle, not the values)
        precios_simulados = np.random.normal(loc=0.04, scale=0.015, size=len(resultados))
        resultados['Precio_Agua_Hoy']          = np.clip(precios_simulados, 0.02, 0.08)
        resultados_regresion['Precio_Agua_Hoy'] = np.clip(precios_simulados, 0.02, 0.08)

        fcst_cols = [f'Precip_D{d+1}_mm' for d in range(7)] + [f'ETo_D{d+1}_mm' for d in range(7)]
        columnas_estado = [
            'GDU_Acumulado', 'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C',
            'Humedad_Relativa_pct', 'Velocidad_Viento_ms', 'Precipitacion_Hoy_mm',
            'ETo_mm', 'Precio_Agua_Hoy',
        ] + fcst_cols

        resultados[columnas_estado]          = resultados[columnas_estado].fillna(0.0)
        resultados_regresion[columnas_estado] = resultados_regresion[columnas_estado].fillna(0.0)

        # ── EXPORTACIÓN SIN ESCALAR ────────────────────────────────────────
        resultados.to_csv(
            "../../DATASET_IA/dataset_entrenamiento_sin_escalar_7d.csv",
            sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset sin escalar (7d) preparado exitosamente.")

        resultados_regresion.to_csv(
            "../../DATASET_IA/dataset_entrenamiento_regresion_7d.csv",
            sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset regresión (7d) preparado exitosamente.")

        # ── ESCALADO Y EXPORTACIÓN RL ──────────────────────────────────────
        scaler = MinMaxScaler()
        resultados_escalados = resultados.copy()
        resultados_escalados[columnas_estado] = scaler.fit_transform(resultados[columnas_estado])

        columnas_finales = ['Fecha'] + columnas_estado
        resultados_escalados[columnas_finales].to_csv(
            "../../DATASET_IA/dataset_entrenamiento_rl_7d.csv",
            sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset RL escalado (7d) preparado exitosamente.")

    except FileNotFoundError:
        print("Error: No se encontró el archivo csv.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
