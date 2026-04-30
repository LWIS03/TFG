import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CalculadoraCultivo:
    def __init__(self, fw = 1.0):
        # Parámetros capa superficial (Ke)
        self.AET = 1000 * (0.25 - 0.5 * 0.12) * 0.10
        self.afe = 9.0
        self.fw = fw
        self.De = 0.0
        self.Kc_min = 0.15

        # Parámetros zona radicular (Dr) para el balance hídrico profundo del maíz
        self.fc = 0.25         # Capacidad de Campo
        self.wp = 0.12         # Punto de Marchitez
        self.z_r = 0.60        # Profundidad de la raíz en metros
        self.p_base = 0.55   # Factor de agotamiento permisible para el maíz
        self.TAW = 1000 * (self.fc - self.wp) * self.z_r
        self.RAW = self.p_base * self.TAW
        self.Dr = 0.0          # Agotamiento radicular inicial (Suelo a capacidad de campo = 0.0)
        self.factor_rendimiento_diario = 1.0 # Starts at 1.0 (100% potential yield)

    def actualizar_penalizacion_rendimiento(self, etc_hoy, etc_real_hoy, fase_nombre):
        ky_hoy = self.obtener_ky_fase(fase_nombre)
        
        if etc_hoy > 0:
            deficit_diario = 1.0 - (etc_real_hoy / etc_hoy)
            # Convertimos el déficit diario a su peso dentro de la temporada completa.
            # Denominador = ETc estacional estimada para maíz en Bustillo del Páramo
            # (~180 días, climate-adjusted Kcb, condiciones semi-áridas): ≈ 800 mm.
            # Un día de ETc alta (p.ej. 7mm) tiene más peso que uno de ETc baja (1mm),
            # lo que es correcto: el estrés en floración con alta demanda daña más.
            penalizacion_diaria = ky_hoy * deficit_diario * (etc_hoy / 800.0)
            self.factor_rendimiento_diario *= (1.0 - penalizacion_diaria)
            self.factor_rendimiento_diario = max(0.0, self.factor_rendimiento_diario)

    def actualizar_taw_y_raw_dinamico(self, etc_hoy, profundidad_raiz_hoy):
        """Ajusta el TAW según la raíz y el RAW (p) según la demanda (FAO-56)."""
        # 1. Actualizar el tamaño de la maceta (TAW)
        self.z_r = profundidad_raiz_hoy
        self.TAW = 1000 * (self.fc - self.wp) * self.z_r
        
        # 2. Ecuación FAO-56 para el factor p dinámico
        p_dinamico = self.p_base + 0.04 * (5.0 - etc_hoy)
        p_dinamico = max(0.1, min(p_dinamico, 0.8))
        
        # 3. Recalculamos el umbral de estrés (RAW) para HOY
        self.RAW = p_dinamico * self.TAW

    def obtener_ky_fase(self, fase_nombre):
        """Devuelve el factor de respuesta del rendimiento (Ky) según la fase del maíz."""
        # Valores de Ky para maíz basados en la metodología FAO-33
        if "Inicial" in fase_nombre or "Desarrollo" in fase_nombre:
            return 0.40
        elif "Madurez" in fase_nombre: # Fase de floración/formación de rendimiento (Muy sensible)
            return 1.30
        elif "Final" in fase_nombre: # Fase de maduración
            return 0.50
        else:
            return 0.0

    def calcular_estres_ks(self):
        """
        Calcula el coeficiente de estrés (Ks). 
        Si el agotamiento (Dr) supera el límite de agua fácilmente disponible (RAW), la planta sufre.
        """
        if self.Dr <= self.RAW:
            return 1.0 # Sin estrés, la planta transpira al 100%
        else:
            # Fórmula de reducción lineal del estrés de FAO-56
            ks = (self.TAW - self.Dr) / (self.TAW - self.RAW)
            return max(0.0, min(ks, 1.0))

    def calcular_perdida_rendimiento(self, sum_ET_real, sum_ET_max, fase_nombre):
        """
        Aplica la Ecuación FAO-33 para calcular el % de pérdida de cosecha en una fase dada.
        """
        ky = self.obtener_ky_fase(fase_nombre)
        
        if sum_ET_max == 0:
            return 0.0
            
        deficit_evapotranspiracion = 1 - (sum_ET_real / sum_ET_max)
        fraccion_perdida_rendimiento = ky * deficit_evapotranspiracion
        
        return fraccion_perdida_rendimiento

    def ecuacion_kcbMax(self, viento, humedad, altura):
        Kcb = 1.2 + (0.04 * (viento - 2) - 0.004 * (humedad - 45)) * pow((altura / 3), 0.3)
        return Kcb

    def ajustar_kcb_por_clima(self, kcb_tabla, u2, hrmin, altura):
        """
        FAO-56 Ecuación 70: Ajusta Kcb por condiciones climáticas locales.
        Solo se aplica cuando Kcb > 0.45 (valores bajos no son sensibles al clima).
        
        Para Bustillo del Páramo (León) en verano:
          - u2 ≈ 3-4 m/s (meseta ventosa) → incrementa Kcb
          - HRmin ≈ 30-35% (semi-árido) → incrementa Kcb
        Efecto neto: Kcb_mid sube de 1.15 (tabla) a ~1.22-1.27 (ajustado).
        
        Args:
            kcb_tabla: Valor de Kcb de la tabla FAO (sin ajuste climático)
            u2:        Velocidad del viento a 2m [m/s], limitado a [1, 6]
            hrmin:     Humedad relativa mínima [%], limitado a [20, 80]
            altura:    Altura media de la planta [m], limitado a [0.1, 10]
        Returns:
            Kcb ajustado al clima local
        """
        if kcb_tabla <= 0.45:
            return kcb_tabla  # Sin ajuste para valores basales bajos
        
        # Aplicamos los límites que exige la FAO-56
        u2    = max(1.0, min(u2,    6.0))
        hrmin = max(20.0, min(hrmin, 80.0))
        altura = max(0.1,  min(altura, 10.0))
        
        ajuste = (0.04 * (u2 - 2) - 0.004 * (hrmin - 45)) * pow(altura / 3, 0.3)
        return kcb_tabla + ajuste

    def calcular_balance_diario_humedad_suelo(self, ke, eto, precipitacion, riego, few):
        # Asumimos que la escorrentía superficial (RO) es 0 salvo lluvia torrencial
        # y la transpiración superficial (T_ew) es 0 para el maíz (raíces profundas).
        RO = 0.0 
        T_ew = 0.0

        # Lluvia efectiva (asumiendo que la lluvia moja toda la superficie uniformemente)
        lluvia_efectiva = precipitacion - RO
        # Riego efectivo concentrado en la fracción humedecida (I / fw)
        riego_efectivo = riego / self.fw if self.fw > 0.0 else 0.0

        # Evaporación concentrada en la porción expuesta y mojada (E / few)
        Ei = ke * eto  # Evaporación real en mm
        evaporacion_concentrada = Ei / max(few, 0.01)

        # Cálculo de la Percolación Profunda (DPe) de la capa superficial (Ec. 79)
        entradas_totales = lluvia_efectiva + riego_efectivo
        if entradas_totales > self.De:
            DPe = entradas_totales - self.De
        else:
            DPe = 0.0
            
        # Aplicación estricta de la Ecuación 77 de la FAO-56
        self.De = self.De - lluvia_efectiva - riego_efectivo + evaporacion_concentrada + T_ew + DPe
        self.De = max(0.0, min(self.De, self.AET))

    def calcular_kcb_por_gdu(self, gdu_acumulado):
        kcb_ini = 0.15
        kcb_mid = 1.15
        kcb_fin = 0.15 
        
        gdu_ini = 250
        gdu_dev = 600
        gdu_mid = 1100
        gdu_late = 1275
        
        if gdu_acumulado <= gdu_ini:
            return kcb_ini
        elif gdu_acumulado <= gdu_dev:
            # Fase de desarrollo (Sube de 0.15 a 1.15)
            progreso = (gdu_acumulado - gdu_ini) / (gdu_dev - gdu_ini)
            return kcb_ini + progreso * (kcb_mid - kcb_ini)
        elif gdu_acumulado <= gdu_mid:
            # Fase media (Consumo máximo constante en 1.15)
            return kcb_mid
        elif gdu_acumulado <= gdu_late:
            # Fase de maduración (Baja de 1.15 a 0.15)
            progreso = (gdu_acumulado - gdu_mid) / (gdu_late - gdu_mid)
            return kcb_mid - progreso * (kcb_mid - kcb_fin)
        else:
            return kcb_fin    

    def calcular_ke(self, viento, precipitacion, riego, humedad, altura, Kcb, eto):
        KcAux = self.ecuacion_kcbMax(viento, humedad, altura)
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
        limit_superior = few * Kc_max
        ke = max(0.0, min(ke, limit_superior))

        self.calcular_balance_diario_humedad_suelo(ke, eto, precipitacion, riego, few)
        return ke

    def Calcular_ETc(self, eto, dias_desde_plantacion, viento, precipitacion, riego, humedad, altura, gdu_acumulado):
        # 1. Kcb de tabla (valores FAO para clima estándar sub-húmedo)
        kcb_tabla = self.calcular_kcb_por_gdu(gdu_acumulado)
        # 2. Kcb ajustado al clima real de Bustillo (FAO-56 Eq.70)
        #    Usa el viento y humedad del DÍA para reflejar las condiciones actuales
        kcb = self.ajustar_kcb_por_clima(kcb_tabla, viento, humedad, altura)
        # 3. Ke: componente de evaporación superficial (depende del balance de De)
        ke = self.calcular_ke(viento, precipitacion, riego, humedad, altura, kcb, eto)
        return (kcb + ke) * eto, kcb + ke

    # ------------------ NUEVAS FUNCIONES ESTOCÁSTICAS Y PREDICTIVAS ------------------

    def inyectar_ruido_estocastico_meteorologico(self, pronostico_precipitacion, coeficiente_varianza=0.30):
        """Añade ruido gaussiano al pronóstico para simular la realidad del terreno."""
        if pronostico_precipitacion <= 0.5:
            vector_ruido = np.random.normal(loc=0.0, scale=0.5)
        else:
            vector_ruido = np.random.normal(loc=0.0, scale=(pronostico_precipitacion * coeficiente_varianza))
        
        precipitacion_real = max(0.0, pronostico_precipitacion + vector_ruido)
        return round(precipitacion_real, 3)

    def evaluar_accion_riego_predictivo(self, etc_estimada_hoy, precip_fcst_manana, etc_fcst_manana, gdu_acumulado):
        """
        Evalúa el estado proyectado a mañana para decidir la lámina de riego óptima,
        dejando un margen de seguridad para aprovechar posibles lluvias imprevistas.
        """
        # 1. Proyectamos el balance asumiendo que NO aplicamos ningún riego
        if gdu_acumulado >= 1150:
            return 0.0
        agotamiento_futuro = self.Dr + etc_estimada_hoy + etc_fcst_manana - precip_fcst_manana
        
        volumen_riego_optimo = 0.0
        
        # 2. Umbral de disparo: Aguantamos hasta consumir el 80% del agua fácilmente disponible (RAW)
        umbral_disparo = self.RAW * 0.95
        
        if agotamiento_futuro > umbral_disparo:
            # 3. EL TRUCO AGRONÓMICO: El colchón de lluvia (Rain Buffer)
            # En lugar de llenar el suelo a tope (Dr = 0), dejamos un espacio vacío.
            # Por ejemplo, dejamos un 15% del TAW libre para capturar lluvia gratis.
            margen_seguridad_lluvia = self.TAW * 0.35
            
            # Calculamos el riego necesario para dejar el suelo exactamente en ese margen seguro
            volumen_riego_optimo = agotamiento_futuro - margen_seguridad_lluvia
            
            # 4. Límites de seguridad física
            # Evitamos regar en negativo y nunca regamos más de lo que cabe en el suelo entero
            volumen_riego_optimo = max(0.0, min(volumen_riego_optimo, self.TAW))
            
        return round(volumen_riego_optimo, 3)

    def actualizar_balance_radicular(self, precipitacion_real, riego_ejecutado, etc_real):
        """Cierra el balance diario calculando percolación y agotamiento."""
        self.Dr = self.Dr - precipitacion_real - riego_ejecutado + etc_real
        
        percolacion_profunda = 0.0
        # Si baja de 0 (se satura más de la capacidad de campo), percola
        if self.Dr < 0:
            percolacion_profunda = abs(self.Dr)
            self.Dr = 0.0  
            
        # El límite fisiológico natural en contra de la fuerza matricial
        self.Dr = min(self.Dr, self.TAW)
        
        return round(percolacion_profunda, 3)

# Funciones auxiliares originales mantenidas igual
def calcular_columna_temperatura_suelo(df, dias_media=3):
    """
    Calcula la temperatura estimada del suelo a 5 cm basada en la 
    media móvil de la temperatura del aire de los últimos días.
    """
    # 1. Nos aseguramos de que la columna es numérica
    temp_aire = pd.to_numeric(df['Temp. media'], errors='coerce')
    
    # 2. Calculamos la media móvil (min_periods=1 evita que los primeros días salgan en blanco)
    temp_suelo = temp_aire.rolling(window=dias_media, min_periods=4).mean()
    
    return temp_suelo


def calcular_raiz_por_gdu(gdu_acumulado):
    # Parámetros Altura Planta
    zr_min = 0.10 # La semilla nace a 10cm
    zr_max = 1.80 # La raíz del maíz puede llegar a 1
    
    # Parámetros Profundidad Raíz (Z_r)
    zr_min = 0.10 # La semilla nace a 10cm
    zr_max = 1.20 # La raíz del maíz puede llegar a 1.20m
    
    if gdu_acumulado < 100:
        raiz = zr_min
    elif 100 <= gdu_acumulado < 350:
        progreso = (gdu_acumulado - 100) / (350 - 100)
        raiz = zr_min + progreso * (zr_max - zr_min) * 0.3 # Crece lento al principio
    elif 350 <= gdu_acumulado < 700:
        progreso = (gdu_acumulado - 350) / (700 - 350)
        raiz = zr_min + (zr_max - zr_min) * 0.3 + progreso * (zr_max - zr_min) * 0.7 # Crece rápido aquí
    else:
        raiz = zr_max

    return raiz

def calcular_altura_por_gdu(gdu_acumulado):
    h_max = 2.60 
    h_min = 0.05 
    h_v4 = 0.40  
    
    if gdu_acumulado < 100:
        return h_min
    elif 100 <= gdu_acumulado < 350:
        progreso = (gdu_acumulado - 100) / (350 - 100)
        return h_min + progreso * (h_v4 - h_min)
    elif 350 <= gdu_acumulado < 700:
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
    gdu = ((t_max_adj + t_min_adj) / 2.0) - 10.0
    return max(gdu, 0.0)

def calcular_fase_y_kc(gdu_acumulado):
    # Tramos re-escalados para una meta de ~1275 GDU (FAO 400)
    if gdu_acumulado < 250:
        min_kc, max_kc = 0.30, 0.50
        Kc = min_kc + (max_kc - min_kc) * (gdu_acumulado / 250)
        return Kc, "Fase Inicial (VE-V4)"
    elif 250 <= gdu_acumulado < 600:
        min_kc, max_kc = 0.50, 1.05
        Kc = min_kc + (max_kc - min_kc) * ((gdu_acumulado - 250) / (600 - 250))
        return Kc, "Fase de Desarrollo (V5-V12)"
    elif 600 <= gdu_acumulado < 1100:
        min_kc, max_kc = 1.05, 1.20
        Kc = min_kc + (max_kc - min_kc) * ((gdu_acumulado - 600) / (1100 - 600))
        return Kc, "Fase de Madurez (V13-R3)"
    elif 1100 <= gdu_acumulado < 1275:
        ini, fin = 1.20, 0.80
        Kc = ini - (ini - fin) * ((gdu_acumulado - 1100) / (1275 - 1100))
        return Kc, "Fase Final (R4-R5)"
    else:
        return 0.55, "Fase de Cosecha (R6)"

def main():
    print("Iniciando proceso con modelo predictivo estocástico...")

    cols_to_use_1 = range(18)
    cols_to_use_2 = range(10)  
    try:
        dades_absolutes = pd.read_csv("../../DATASET_BUSTILLO/dades_absolutes_registrades.csv", sep=";", encoding="latin-1", usecols=cols_to_use_1)
        dades_calculades = pd.read_csv("../../DATASET_BUSTILLO/dades_calculades.csv", sep=";", encoding="latin-1", usecols= cols_to_use_2)
        
        dades_absolutes = dades_absolutes.drop(0).reset_index(drop=True)
        dades_absolutes.columns = dades_absolutes.columns.str.strip()

        dades_calculades = dades_calculades.drop(0).reset_index(drop=True)
        dades_calculades.columns = dades_calculades.columns.str.strip()

        for col in dades_absolutes.columns:
            if col!= 'Fecha' and 'Hora' not in col:
                dades_absolutes[col] = pd.to_numeric(dades_absolutes[col].astype(str).str.replace(',', '.'), errors='coerce')

        for col in dades_calculades.columns:
            if col!= 'Fecha' and 'Hora' not in col:
                dades_calculades[col] = pd.to_numeric(dades_calculades[col].astype(str).str.replace(',', '.'), errors='coerce')

        dades_calculades['Temp_suelo_est'] = calcular_columna_temperatura_suelo(dades_absolutes, dias_media=4)

        print(f"Procesando {len(dades_absolutes)} filas...")

        #lista_fecha, lista_fase = [], []
        lista_dias_plantacion = []
        #lista_gdu_acumulado, lista_altura = [], [], []
        #lista_tmax, lista_tmin, lista_viento, lista_humedad = [], [], [], []
        #lista_precip_hoy, lista_eto, lista_temp_suelo, lista_agotamiento_dr = [], [], [], []

        lista_temp_suelo = []
        lista_gdu_acumulado = []
        lista_fecha = []
        lista_dias_plantacion = []
        lista_temp_max = []
        lista_temp_min = []
        lista_viento = []
        lista_humedad = []
        lista_precip_hoy = []
        lista_eto = []
        lista_riego_recomendado = []
        #lista_ks_gemelo = []
        
        # Nuevas listas para almacenar la auditoría del riego
        lista_precip_observada = []
        lista_riego_recomendado = []
        lista_agotamiento_dr = []
        lista_percolacion = []

        en_temporada = False
        gdu_acumulado = 0.0
        dias_desde_plantacion = 0
        calc = None 

        for index, row in dades_absolutes.iterrows():

            fecha = row['Fecha']
            fecha_dt = pd.to_datetime(fecha, format='%d/%m/%Y', errors='coerce')
            if pd.isna(fecha_dt):
                continue
            
            mes = fecha_dt.month
            dia = fecha_dt.day

            t_max = row['Temp. max']
            t_min = row['Temp. min']
            v_viento = row['Vel. viento']
            h_humedad = row['Hum. med.']
            precipitacion_pronostico = row['Precipitacion']
            precipitacion_efectiva = dades_calculades.at[index, 'Precipit. efectiva(P.MON.)']
            eto = dades_calculades.at[index, 'ETo(P.MON.)']
            temp_suelo = dades_calculades.at[index, 'Temp_suelo_est'] 

            # LÓGICA DE SIEMBRA
            if not en_temporada:
                ventana_siembra = (mes == 4 and dia >= 15) or (mes == 5)
                temp_adecuada = not pd.isna(temp_suelo) and temp_suelo >= 9.0
                
                if ventana_siembra and temp_adecuada:
                    en_temporada = True
                    gdu_acumulado = 0.0
                    dias_desde_plantacion = 1
                    calc = CalculadoraCultivo() # Instanciamos el gemelo digital

                    sum_etc_max_temporada = 0.0
                    sum_etc_real_temporada = 0.0
                    total_riego_mm_temporada = 0.0

                    print(f"🌱 SIEMBRA: {fecha} (Temp. Suelo: {temp_suelo:.2f}ºC)")
                else:
                    # Guardamos el barbecho con el clima natural sin alteraciones
                    lista_fecha.append(fecha)
                    lista_dias_plantacion.append(0)
                    #lista_fase.append("Barbecho / Sin cultivo")
                    lista_gdu_acumulado.append(0.0)
                    #lista_altura.append(0.0)
                    lista_temp_max.append(t_max)
                    lista_temp_min.append(t_min)
                    lista_viento.append(v_viento)
                    lista_humedad.append(h_humedad)
                    lista_precip_hoy.append(precipitacion_efectiva)
                    lista_eto.append(eto)
                    lista_temp_suelo.append(temp_suelo)
                    lista_riego_recomendado.append(0.0)  # Sin riego en barbecho
                    #lista_ks_gemelo.append(1.0)  # Sin estrés en barbecho
                    #lista_agotamiento_dr.append(0.0)
                    continue 

            # Crecimiento y cálculos fisiológicos de HOY
            calcular_gdu_diario = calcular_gdu(t_max, t_min)
            gdu_acumulado += calcular_gdu_diario
            kc_gdu, fase = calcular_fase_y_kc(gdu_acumulado)
            altura_planta = calcular_altura_por_gdu(gdu_acumulado)
            raiz_planta = calcular_raiz_por_gdu(gdu_acumulado)

            # ------------- LÓGICA DE PREDICCIÓN Y ESTOCÁSTICA ------------- #
            # 1. Miramos el horizonte de previsión a mañana (T+1)
            if index + 1 < len(dades_absolutes):
                precip_fcst_manana = dades_absolutes.at[index + 1, 'Precipitacion']
                eto_manana = dades_calculades.at[index + 1, 'ETo(P.MON.)']
            else:
                precip_fcst_manana = 0.0
                eto_manana = eto
            
            # 2. Estimamos las demandas
            etc_estimada_hoy = eto * kc_gdu
            etc_fcst_manana = eto_manana * kc_gdu
            
            # 3. Calculamos cuánto riego recomendamos para neutralizar un posible estrés futuro
            riego_aplicable = calc.evaluar_accion_riego_predictivo(etc_estimada_hoy, precip_fcst_manana, etc_fcst_manana, gdu_acumulado)
            #riego_aplicable = 0.0
            # 4. Inyectamos incertidumbre a la lluvia que nos pronosticaban que caería HOY
            precip_base = precipitacion_efectiva if not pd.isna(precipitacion_efectiva) else precipitacion_pronostico
            precipitacion_efectiva_real = calc.inyectar_ruido_estocastico_meteorologico(precip_base)

            # 5. Ejecutamos el cálculo térmico real y actualizamos la superficie (De)
            etc, kc_calculada = calc.Calcular_ETc(
                eto, dias_desde_plantacion, v_viento, precipitacion_efectiva_real, 
                riego_aplicable, h_humedad, altura_planta, gdu_acumulado
            )

            calc.actualizar_taw_y_raw_dinamico(etc, raiz_planta)
            coeficiente_ks = calc.calcular_estres_ks()
            etc_real = etc * coeficiente_ks

            calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)

            sum_etc_max_temporada += etc
            sum_etc_real_temporada += etc_real
            total_riego_mm_temporada += riego_aplicable

            # 6. Actualizamos y cuadramos el balance radicular profundo (Dr)
            percolacion_prof = calc.actualizar_balance_radicular(precipitacion_efectiva_real, riego_aplicable, etc_real)
            # -------------------------------------------------------------- #

            # Almacenamiento de variables
            lista_fecha.append(fecha)
            lista_dias_plantacion.append(dias_desde_plantacion)
            #lista_fase.append(fase)
            lista_gdu_acumulado.append(gdu_acumulado)
            #lista_altura.append(altura_planta)
            lista_temp_max.append(t_max)
            lista_temp_min.append(t_min)
            lista_viento.append(v_viento)
            lista_humedad.append(h_humedad)
            lista_precip_hoy.append(precipitacion_efectiva_real)
            lista_eto.append(eto)
            lista_temp_suelo.append(temp_suelo)
            lista_riego_recomendado.append(riego_aplicable)
            #lista_ks_gemelo.append(coeficiente_ks)

            #lista_agotamiento_dr.append(calc.Dr)

            condicion_fecha = (mes >= 11 and dia >= 15)
            condicion_madurez = (gdu_acumulado >= 1275)
            
            dias_desde_plantacion += 1


            if condicion_madurez or condicion_fecha:

                rendimiento_max_kg_ha = 15000.0  # Yx: Producción ideal en regadío
                precio_maiz_kg = 0.23            # Precio de venta
                coste_agua_m3 = 0.04             # Precio del agua
                coste_fijo_boe = 32.18           # Cuota fija anual BOE 2026

                coste_agua_ha = total_riego_mm_temporada * 10 * coste_agua_m3
                coste_agua_ha += coste_fijo_boe

                rendimiento_real = rendimiento_max_kg_ha * calc.factor_rendimiento_diario

                ingresos_cosecha = rendimiento_real * precio_maiz_kg
                recompensa_final = ingresos_cosecha - coste_agua_ha

                print(f"🌾 COSECHA: {fecha} (GDU alcanzado: {gdu_acumulado:.2f})")
                print(f"   -> Riego gastado: {total_riego_mm_temporada:.2f} mm (-{coste_agua_ha:.2f} €/ha)")
                print(f"   -> Rendimiento:   {rendimiento_real:.2f} kg/ha (+{ingresos_cosecha:.2f} €/ha)")
                print(f"   -> RECOMPENSA:    {recompensa_final:.2f} €/ha de Beneficio Bruto")
                en_temporada = False 


        
        # Guardado del conjunto
        resultados_limpios = pd.DataFrame({
            'Fecha': lista_fecha,
            'Dias_Plantacion': lista_dias_plantacion,
            'GDU_Acumulado': lista_gdu_acumulado,
            'Temp_Max_C': lista_temp_max,
            'Temp_Min_C': lista_temp_min,
            'Temp_Suelo_C': lista_temp_suelo,
            'Humedad_Relativa_pct': lista_humedad,
            'Velocidad_Viento_ms': lista_viento,
            'Precipitacion_Hoy_mm': lista_precip_hoy,
            'ETo_mm': lista_eto
            #'Ks_Gemelo': lista_ks_gemelo
        })

        resultados_limpios_regresion = pd.DataFrame({
            'Fecha': lista_fecha,
            'Dias_Plantacion': lista_dias_plantacion,
            'GDU_Acumulado': lista_gdu_acumulado,
            'Temp_Max_C': lista_temp_max,
            'Temp_Min_C': lista_temp_min,
            'Temp_Suelo_C': lista_temp_suelo,
            'Humedad_Relativa_pct': lista_humedad,
            'Velocidad_Viento_ms': lista_viento,
            'Precipitacion_Hoy_mm': lista_precip_hoy,
            'ETo_mm': lista_eto,
            'Riego_Recomendado_mm': lista_riego_recomendado
        })

        mapeo_fases = {
            "Barbecho / Sin cultivo": 0,
            "Fase Inicial (VE-V4)": 1,
            "Fase de Desarrollo (V5-V12)": 2,
            "Fase de Madurez (V13-R3)": 3,
            "Fase Final (R4-R5)": 4,
            "Fase de Cosecha (R6)": 5
        }
        
        #resultados_limpios['Fase_Num'] = resultados_limpios['Fase'].map(mapeo_fases).fillna(0)

        # --- PRONÓSTICO REALISTA CON RUIDO ---
        # Antes: shift(-1) daba el valor exacto de mañana (trampa perfecta).
        # Ahora: añadimos ruido gaussiano para simular la incertidumbre de un pronóstico real.
        # La IA aprenderá a tomar decisiones robustas bajo incertidumbre, no a memorizar el futuro.

        # Precipitación: alta varianza (±30%) porque la lluvia es muy difícil de predecir
        precip_real_manana = resultados_limpios['Precipitacion_Hoy_mm'].shift(-1).fillna(0.0)
        calc_ruido = CalculadoraCultivo()

        resultados_limpios['Precip_Manana_mm'] = precip_real_manana.apply(
            lambda p: calc_ruido.inyectar_ruido_estocastico_meteorologico(p, coeficiente_varianza=0.30)
        )

        resultados_limpios_regresion['Precip_Manana_mm'] = precip_real_manana.apply(
            lambda p: calc_ruido.inyectar_ruido_estocastico_meteorologico(p, coeficiente_varianza=0.30)
        )

        # ETo: menor varianza (±15%) porque temperatura y radiación son más predecibles
        eto_real_manana = resultados_limpios['ETo_mm'].shift(-1).fillna(0.0)
        ruido_eto = np.random.normal(loc=1.0, scale=0.15, size=len(eto_real_manana))
        ruido_eto = np.clip(ruido_eto, 0.5, 1.5)  # Nunca más del doble ni menos de la mitad
        resultados_limpios['ETo_Manana_mm'] = np.maximum(0.0, eto_real_manana * ruido_eto)
        resultados_limpios_regresion['ETo_Manana_mm'] = np.maximum(0.0, eto_real_manana * ruido_eto)

        # --- SIMULAMOS PRECIO DEL AGUA ---
        # Usamos ruido SIN semilla fija para que cada regeneración del dataset
        # produzca una secuencia de precios diferente. Así el agente aprende el
        # principio económico (regar más barato, menos cuando es caro) en vez de
        # memorizar los precios exactos de cada día del calendario.
        precios_simulados = np.random.normal(loc=0.04, scale=0.015, size=len(resultados_limpios))
        resultados_limpios['Precio_Agua_Hoy'] = np.clip(precios_simulados, 0.02, 0.08)
        resultados_limpios_regresion['Precio_Agua_Hoy'] = np.clip(precios_simulados, 0.02, 0.08)

        columnas_estado = [
            'GDU_Acumulado', 'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C', 
            'Humedad_Relativa_pct', 'Velocidad_Viento_ms', 'Precipitacion_Hoy_mm', 
            'ETo_mm', 'Precip_Manana_mm', 'ETo_Manana_mm', 'Precio_Agua_Hoy'
        ]
        
        resultados_limpios[columnas_estado] = resultados_limpios[columnas_estado].fillna(0.0)
        resultados_limpios.to_csv("../../DATASET_IA/dataset_entrenamiento_sin_escalar.csv", sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset de entrenamiento sin escalar preparado exitosamente.")

        resultados_limpios_regresion[columnas_estado] = resultados_limpios_regresion[columnas_estado].fillna(0.0)
        resultados_limpios_regresion.to_csv("../../DATASET_IA/dataset_entrenamiento_regresion.csv", sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset de entrenamiento para regresión preparado exitosamente.")

        scaler = MinMaxScaler()
        resultados_limpios_escalados = resultados_limpios.copy()
        resultados_limpios_escalados[columnas_estado] = scaler.fit_transform(resultados_limpios[columnas_estado])

        # 5. Exportar solo Fecha + Columnas de estado escaladas
        columnas_finales = ['Fecha'] + columnas_estado
        dataset_final = resultados_limpios_escalados[columnas_finales]

        dataset_final.to_csv("../../DATASET_IA/dataset_entrenamiento_rl.csv", sep=";", decimal=",", index=False, float_format="%.4f")
        print("Dataset de entrenamiento preparado y escalado exitosamente.")

    except FileNotFoundError:
        print("Error: No se encontró el archivo csv.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()