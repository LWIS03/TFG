"""
evaluar_modelo.py
=================
Compara 5 estrategias de riego sobre los 20 años de datos reales de Bustillo del Páramo.

Estrategias evaluadas:
  1. Sin riego              — baseline A (secano)
  2. Semanal fija 20mm      — baseline B (agricultor no experto)
  3. FAO-56 regla experta   — evaluar_accion_riego_predictivo() de simulacion_riego.py
  4. AquaCrop (SMT=100%)    — modelo FAO peer-reviewed (desde CSV pre-generado)
  5. Agente SAC             — tu modelo entrenado

PRINCIPIO DE COMPARACIÓN JUSTA:
  Todas las estrategias se evalúan a través del MISMO motor físico FAO-56
  (CalculadoraCultivo de simulacion_riego.py). Lo único que cambia es la
  decisión de riego en cada día. Así las diferencias reflejan la calidad
  de cada política, no diferencias en la física subyacente.

Uso:
  python evaluar_modelo.py

Archivos requeridos:
  - ../DATASET_IA/dataset_entrenamiento_sin_escalar.csv
  - modelos/best_model.zip
  - modelos/vecnormalize_mejor.pkl
  - riego_optimo_aquacrop_dinamico.csv   (generado por aquacrop_script.py)

Salidas:
  - resultados_evaluacion.csv      — métricas por estrategia y año
  - resumen_evaluacion.csv         — tabla resumen (lista para la memoria TFG)
  - figura_comparacion.png         — figura de barras para la memoria TFG
  - figura_variabilidad.png        — boxplot de variabilidad interanual
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(__file__))
from simulacion_riego import (
    CalculadoraCultivo,
    calcular_altura_por_gdu,
    calcular_raiz_por_gdu,
    calcular_fase_y_kc,
    calcular_gdu,
)
from entrenar_ia import EntornoRiegoMaiz

# ══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════

RUTA_DATASET    = "../DATASET_IA/dataset_entrenamiento_sin_escalar.csv"
RUTA_MODELO     = "modelos/best_model.zip"
RUTA_VECNORM    = "modelos/vecnormalize_mejor.pkl"
RUTA_AQUACROP   = "../Aquacrop/riego_optimo_aquacrop_dinamico.csv"

RENDIMIENTO_MAX = 15000.0   # kg/ha potencial
PRECIO_MAIZ     = 0.23      # €/kg
PRECIO_AGUA     = 0.04      # €/m³ (precio medio del dataset)
COSTE_FIJO      = 32.18     # €/ha (cuota BOE 2026)


# ══════════════════════════════════════════════════════════
# MOTOR DE SIMULACIÓN — igual para todas las estrategias
# ══════════════════════════════════════════════════════════

def simular_temporada(df_season, fn_riego):
    """
    Ejecuta una temporada completa a través del motor FAO-56.

    Args:
        df_season : DataFrame con las filas del dataset para esta temporada
                    (desde Dias_Plantacion==1 hasta el fin)
        fn_riego  : función (calc, row, dia_num) → mm de riego a aplicar hoy

    Returns:
        dict con métricas de la temporada
    """
    calc = CalculadoraCultivo()
    gdu_acumulado   = 0.0
    ks_actual       = 1.0
    total_riego_mm  = 0.0
    total_precip_mm = 0.0
    dias_estres     = 0
    suma_ks         = 0.0
    coste_agua_acum = 0.0

    for dia_num, (_, row) in enumerate(df_season.iterrows(), start=1):
        t_max   = float(row['Temp_Max_C'])
        t_min   = float(row['Temp_Min_C'])
        eto     = float(row['ETo_mm'])
        precip  = float(row['Precipitacion_Hoy_mm'])
        viento  = float(row['Velocidad_Viento_ms'])
        humedad = float(row['Humedad_Relativa_pct'])
        gdu     = float(row['GDU_Acumulado'])

        altura  = calcular_altura_por_gdu(gdu)
        raiz    = calcular_raiz_por_gdu(gdu)
        _, fase = calcular_fase_y_kc(gdu)

        # Decisión de riego según la estrategia
        riego = float(fn_riego(calc, row, dia_num))
        riego = max(0.0, min(riego, 30.0))   # mismos límites que el agente

        # Física
        etc, _ = calc.Calcular_ETc(
            eto, dia_num, viento, precip, riego, humedad, altura, gdu
        )
        calc.actualizar_taw_y_raw_dinamico(etc, raiz)
        ks = calc.calcular_estres_ks()
        etc_real = etc * ks

        calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)
        calc.actualizar_balance_radicular(precip, riego, etc_real)

        # Acumuladores
        total_riego_mm  += riego
        total_precip_mm += precip
        coste_agua_acum += riego * 10 * PRECIO_AGUA
        if ks < 1.0:
            dias_estres += 1
        suma_ks += ks

    # Métricas finales
    n_dias           = len(df_season)
    rendimiento_real = RENDIMIENTO_MAX * calc.factor_rendimiento_diario
    ingresos         = rendimiento_real * PRECIO_MAIZ
    beneficio_neto   = ingresos - coste_agua_acum - COSTE_FIJO
    agua_total       = total_riego_mm + total_precip_mm
    wue              = rendimiento_real / max(agua_total, 1.0)

    return {
        'rendimiento_kg_ha':    round(rendimiento_real, 1),
        'factor_rendimiento':   round(calc.factor_rendimiento_diario, 4),
        'riego_mm':             round(total_riego_mm, 1),
        'precip_mm':            round(total_precip_mm, 1),
        'agua_total_mm':        round(agua_total, 1),
        'beneficio_neto_eur':   round(beneficio_neto, 2),
        'ingresos_eur':         round(ingresos, 2),
        'coste_agua_eur':       round(coste_agua_acum, 2),
        'dias_estres':          dias_estres,
        'ks_medio':             round(suma_ks / max(n_dias, 1), 4),
        'wue_kg_mm':            round(wue, 3),
        'n_dias':               n_dias,
    }


# ══════════════════════════════════════════════════════════
# ESTRATEGIAS
# ══════════════════════════════════════════════════════════

def estrategia_sin_riego(calc, row, dia_num):
    return 0.0


def estrategia_semanal_20mm(calc, row, dia_num):
    """Riego fijo de 20mm cada 7 días sin importar el estado del suelo."""
    return 20.0 if dia_num % 7 == 0 else 0.0


def estrategia_fao_experta(calc, row, dia_num):
    """
    Estrategia regla-experta basada en evaluar_accion_riego_predictivo().
    Replica exactamente la lógica del gemelo digital de simulacion_riego.py.
    """
    eto            = float(row['ETo_mm'])
    precip_manana  = float(row.get('Precip_Manana_mm', 0.0))
    eto_manana     = float(row.get('ETo_Manana_mm', eto))
    gdu            = float(row['GDU_Acumulado'])

    _, fase = calcular_fase_y_kc(gdu)
    altura  = calcular_altura_por_gdu(gdu)
    viento  = float(row['Velocidad_Viento_ms'])
    humedad = float(row['Humedad_Relativa_pct'])

    # Estimamos ETc para hoy y mañana con el Kc actual (sin riego aún)
    kcb_tabla = calc.calcular_kcb_por_gdu(gdu)
    kcb       = calc.ajustar_kcb_por_clima(kcb_tabla, viento, humedad, altura)
    kc_hoy    = kcb  # aproximación sin Ke para el pronóstico

    etc_hoy    = eto * kc_hoy
    etc_manana = eto_manana * kc_hoy

    return calc.evaluar_accion_riego_predictivo(etc_hoy, precip_manana, etc_manana, gdu)


def construir_estrategia_aquacrop(df_aquacrop):
    """
    Devuelve una función de riego que consulta el CSV de AquaCrop por fecha.
    Si la fecha no está en el CSV (año no simulado), devuelve 0.
    """
    # Índice fecha → mm de riego
    lookup = {}
    if df_aquacrop is not None:
        df_aquacrop = df_aquacrop.copy()
        df_aquacrop['Fecha_dt'] = pd.to_datetime(
            df_aquacrop['Fecha'], format='%d/%m/%Y', errors='coerce'
        )
        for _, fila in df_aquacrop.iterrows():
            if pd.notna(fila['Fecha_dt']):
                lookup[fila['Fecha_dt'].date()] = float(fila['Riego_Optimo_FAO_mm'])

    def fn(calc, row, dia_num):
        fecha_str = str(row.get('Fecha', ''))
        try:
            from datetime import datetime
            fecha = datetime.strptime(fecha_str, '%d/%m/%Y').date()
            return lookup.get(fecha, 0.0)
        except Exception:
            return 0.0

    return fn


# ══════════════════════════════════════════════════════════
# INFERENCIA DEL AGENTE SAC
# ══════════════════════════════════════════════════════════

class AgenteInferencia:
    """
    Wrapper de inferencia para el modelo SAC entrenado.

    Reconstruye la observación de 17 variables en cada paso,
    la normaliza con las stats de VecNormalize y obtiene la
    acción determinista del modelo.
    """
    def __init__(self, ruta_modelo, ruta_vecnorm, ruta_dataset):
        # Cargamos el entorno solo para que VecNormalize tenga referencia
        env_base = DummyVecEnv([lambda: EntornoRiegoMaiz(ruta_dataset)])
        self.vn  = VecNormalize.load(ruta_vecnorm, env_base)
        self.vn.training    = False   # No actualizar stats durante inferencia
        self.vn.norm_reward = False   # Solo normalizar observaciones

        self.modelo = SAC.load(ruta_modelo, env=self.vn)
        print(f"  Modelo cargado: {ruta_modelo}")
        print(f"  VecNormalize:   {ruta_vecnorm}")

    def construir_obs(self, row, calc, ks_actual, factor_rendimiento_prev):
        """Replica _obtener_estado_actual() de EntornoRiegoMaiz."""
        columnas_hoy = [
            'GDU_Acumulado', 'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C',
            'Humedad_Relativa_pct', 'Velocidad_Viento_ms', 'Precipitacion_Hoy_mm',
            'ETo_mm', 'Precio_Agua_Hoy'
        ]
        estado = [float(row[c]) for c in columnas_hoy]

        # Pronóstico mañana (sin ruido extra en evaluación — usamos el valor del CSV)
        estado.append(float(row.get('Precip_Manana_mm', 0.0)))
        estado.append(float(row.get('ETo_Manana_mm', float(row['ETo_mm']))))

        # Variables dinámicas del cultivo
        gdu           = float(row['GDU_Acumulado'])
        altura_planta = calcular_altura_por_gdu(gdu)
        estado.append(altura_planta)

        dr_taw = calc.Dr / max(calc.TAW, 1.0)
        estado.append(dr_taw)

        estado.append(calc.De)
        estado.append(ks_actual)

        stress_margin = (calc.RAW - calc.Dr) / max(calc.TAW, 1.0)
        estado.append(stress_margin)

        factor_rend = calc.factor_rendimiento_diario
        estado.append(factor_rend)

        return np.array(estado, dtype=np.float32)

    def predecir_riego(self, obs_raw):
        """Normaliza la observación y devuelve mm de riego (acción determinista)."""
        obs_norm = self.vn.normalize_obs(obs_raw.reshape(1, -1))
        accion, _ = self.modelo.predict(obs_norm, deterministic=True)
        return float(np.clip(accion[0], 0.0, 30.0))

    def simular_temporada_ia(self, df_season):
        """
        Versión especial de simular_temporada para el agente SAC.
        Necesita acceso a calc en cada paso para construir la observación.
        """
        calc = CalculadoraCultivo()
        ks_actual             = 1.0
        factor_rendimiento_prev = 1.0
        total_riego_mm  = 0.0
        total_precip_mm = 0.0
        dias_estres     = 0
        suma_ks         = 0.0
        coste_agua_acum = 0.0

        for dia_num, (_, row) in enumerate(df_season.iterrows(), start=1):
            eto     = float(row['ETo_mm'])
            precip  = float(row['Precipitacion_Hoy_mm'])
            viento  = float(row['Velocidad_Viento_ms'])
            humedad = float(row['Humedad_Relativa_pct'])
            gdu     = float(row['GDU_Acumulado'])

            altura  = calcular_altura_por_gdu(gdu)
            raiz    = calcular_raiz_por_gdu(gdu)
            _, fase = calcular_fase_y_kc(gdu)

            # Construye observación y obtiene acción del modelo
            obs = self.construir_obs(row, calc, ks_actual, factor_rendimiento_prev)
            riego = self.predecir_riego(obs)

            # Física (misma que todas las demás estrategias)
            etc, _ = calc.Calcular_ETc(
                eto, dia_num, viento, precip, riego, humedad, altura, gdu
            )
            calc.actualizar_taw_y_raw_dinamico(etc, raiz)
            ks = calc.calcular_estres_ks()
            etc_real = etc * ks

            factor_rendimiento_prev = calc.factor_rendimiento_diario
            calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)
            calc.actualizar_balance_radicular(precip, riego, etc_real)

            ks_actual = ks
            total_riego_mm  += riego
            total_precip_mm += precip
            coste_agua_acum += riego * 10 * PRECIO_AGUA
            if ks < 1.0:
                dias_estres += 1
            suma_ks += ks

        n_dias           = len(df_season)
        rendimiento_real = RENDIMIENTO_MAX * calc.factor_rendimiento_diario
        ingresos         = rendimiento_real * PRECIO_MAIZ
        beneficio_neto   = ingresos - coste_agua_acum - COSTE_FIJO
        agua_total       = total_riego_mm + total_precip_mm

        return {
            'rendimiento_kg_ha':    round(rendimiento_real, 1),
            'factor_rendimiento':   round(calc.factor_rendimiento_diario, 4),
            'riego_mm':             round(total_riego_mm, 1),
            'precip_mm':            round(total_precip_mm, 1),
            'agua_total_mm':        round(agua_total, 1),
            'beneficio_neto_eur':   round(beneficio_neto, 2),
            'ingresos_eur':         round(ingresos, 2),
            'coste_agua_eur':       round(coste_agua_acum, 2),
            'dias_estres':          dias_estres,
            'ks_medio':             round(suma_ks / max(n_dias, 1), 4),
            'wue_kg_mm':            round(rendimiento_real / max(agua_total, 1.0), 3),
            'n_dias':               n_dias,
        }


# ══════════════════════════════════════════════════════════
# LOOP PRINCIPAL DE EVALUACIÓN
# ══════════════════════════════════════════════════════════

def identificar_temporadas(df):
    """
    Devuelve lista de (año_aprox, slice_inicio, slice_fin) para cada temporada.
    Una temporada empieza en Dias_Plantacion==1 y termina justo antes de la siguiente.
    """
    inicios = df.index[df['Dias_Plantacion'] == 1].tolist()
    temporadas = []
    for i, idx_ini in enumerate(inicios):
        idx_fin = inicios[i + 1] if i + 1 < len(inicios) else len(df)
        # Estimamos el año a partir de la primera fila si hay columna Fecha
        año = None
        if 'Fecha' in df.columns:
            try:
                año = pd.to_datetime(df.iloc[idx_ini]['Fecha'],
                                     format='%d/%m/%Y').year
            except Exception:
                año = 2000 + i
        else:
            año = 2000 + i
        temporadas.append((año, idx_ini, idx_fin))
    return temporadas


def evaluar_todas_estrategias(df, agente, df_aquacrop):
    """Evalúa las 5 estrategias sobre todas las temporadas. Devuelve DataFrame."""

    temporadas = identificar_temporadas(df)
    print(f"\n  Temporadas detectadas: {len(temporadas)}")

    fn_aquacrop = construir_estrategia_aquacrop(df_aquacrop)

    estrategias_simples = [
        ("Sin riego",           estrategia_sin_riego),
        ("Semanal 20mm",        estrategia_semanal_20mm),
        ("FAO-56 experta",      estrategia_fao_experta),
        ("AquaCrop SMT=100%",   fn_aquacrop),
    ]

    todos_resultados = []

    for año, idx_ini, idx_fin in temporadas:
        df_season = df.iloc[idx_ini:idx_fin].copy()

        if len(df_season) < 30:
            print(f"  [{año}] Temporada muy corta ({len(df_season)} días), saltando.")
            continue

        print(f"\n  [{año}] {len(df_season)} días — evaluando estrategias...")

        # Estrategias simples
        for nombre, fn in estrategias_simples:
            metricas = simular_temporada(df_season, fn)
            metricas['año']       = año
            metricas['estrategia'] = nombre
            todos_resultados.append(metricas)
            print(f"    {nombre:<22} → {metricas['rendimiento_kg_ha']:>7.0f} kg/ha | "
                  f"{metricas['riego_mm']:>6.1f} mm | "
                  f"{metricas['beneficio_neto_eur']:>8.2f} €/ha")

        # Agente SAC (inferencia especial)
        metricas_ia = agente.simular_temporada_ia(df_season)
        metricas_ia['año']        = año
        metricas_ia['estrategia'] = "Agente SAC"
        todos_resultados.append(metricas_ia)
        print(f"    {'Agente SAC':<22} → {metricas_ia['rendimiento_kg_ha']:>7.0f} kg/ha | "
              f"{metricas_ia['riego_mm']:>6.1f} mm | "
              f"{metricas_ia['beneficio_neto_eur']:>8.2f} €/ha")

    return pd.DataFrame(todos_resultados)


# ══════════════════════════════════════════════════════════
# TABLAS DE RESULTADOS
# ══════════════════════════════════════════════════════════

def generar_resumen(df_resultados):
    """Genera tabla resumen agregada (media ± std) por estrategia."""
    metricas = {
        'Rendimiento (kg/ha)':      'rendimiento_kg_ha',
        'Riego total (mm)':         'riego_mm',
        'Beneficio neto (€/ha)':    'beneficio_neto_eur',
        'Días de estrés':           'dias_estres',
        'Ks medio':                 'ks_medio',
        'WUE (kg/mm)':              'wue_kg_mm',
    }

    orden = ["Sin riego", "Semanal 20mm", "FAO-56 experta",
             "AquaCrop SMT=100%", "Agente SAC"]

    filas = []
    for estrategia in orden:
        sub = df_resultados[df_resultados['estrategia'] == estrategia]
        if sub.empty:
            continue
        fila = {'Estrategia': estrategia}
        for label, col in metricas.items():
            media = sub[col].mean()
            std   = sub[col].std()
            fila[label] = f"{media:.1f} ± {std:.1f}"
        filas.append(fila)

    return pd.DataFrame(filas).set_index('Estrategia')


# ══════════════════════════════════════════════════════════
# FIGURAS PARA LA MEMORIA
# ══════════════════════════════════════════════════════════

COLORES = {
    "Sin riego":         "#d62728",
    "Semanal 20mm":      "#ff7f0e",
    "FAO-56 experta":    "#2ca02c",
    "AquaCrop SMT=100%": "#1f77b4",
    "Agente SAC":        "#9467bd",
}

def figura_comparacion(df_resultados, ruta_salida="figura_comparacion.png"):
    """
    4 paneles: rendimiento, riego, beneficio neto, días de estrés.
    Un grupo de barras por estrategia.
    """
    orden = ["Sin riego", "Semanal 20mm", "FAO-56 experta",
             "AquaCrop SMT=100%", "Agente SAC"]

    metricas_plot = [
        ('rendimiento_kg_ha',  'Rendimiento medio (kg/ha)',    'kg/ha'),
        ('riego_mm',           'Riego total medio (mm/temp.)', 'mm'),
        ('beneficio_neto_eur', 'Beneficio neto medio (€/ha)',  '€/ha'),
        ('dias_estres',        'Días de estrés medio',         'días'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, titulo, unidad) in zip(axes, metricas_plot):
        medias, stds, colores_bar, etiquetas = [], [], [], []
        for est in orden:
            sub = df_resultados[df_resultados['estrategia'] == est]
            if sub.empty:
                continue
            medias.append(sub[col].mean())
            stds.append(sub[col].std())
            colores_bar.append(COLORES.get(est, 'gray'))
            etiquetas.append(est.replace(' ', '\n'))

        x = np.arange(len(etiquetas))
        bars = ax.bar(x, medias, yerr=stds, capsize=4,
                      color=colores_bar, edgecolor='white', linewidth=0.8,
                      error_kw={'elinewidth': 1.2, 'ecolor': 'black'})

        # Etiquetar barras con el valor numérico
        for bar, val in zip(bars, medias):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8.5)

        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel(unidad, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Comparación de estrategias de riego — Bustillo del Páramo\n'
        f'(Media ± σ sobre {df_resultados["año"].nunique()} años)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figura guardada: '{ruta_salida}'")


def figura_variabilidad(df_resultados, ruta_salida="figura_variabilidad.png"):
    """
    Boxplot del beneficio neto por estrategia — muestra variabilidad interanual.
    Fundamental para el TFG: un modelo robusto tiene caja pequeña.
    """
    orden = ["Sin riego", "Semanal 20mm", "FAO-56 experta",
             "AquaCrop SMT=100%", "Agente SAC"]
    orden_presentes = [e for e in orden
                       if e in df_resultados['estrategia'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))

    datos_box = [
        df_resultados[df_resultados['estrategia'] == est]['beneficio_neto_eur'].values
        for est in orden_presentes
    ]
    colores_box = [COLORES.get(est, 'gray') for est in orden_presentes]

    bp = ax.boxplot(datos_box, patch_artist=True, notch=False,
                    medianprops={'color': 'black', 'linewidth': 2})

    for patch, color in zip(bp['boxes'], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(orden_presentes) + 1))
    ax.set_xticklabels(orden_presentes, fontsize=10)
    ax.set_ylabel('Beneficio neto (€/ha)', fontsize=11)
    ax.set_title(
        'Variabilidad interanual del beneficio neto por estrategia\n'
        f'(Bustillo del Páramo, {df_resultados["año"].nunique()} años)',
        fontsize=12, fontweight='bold'
    )
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Leyenda de colores
    parches = [mpatches.Patch(color=COLORES.get(e, 'gray'), label=e, alpha=0.75)
               for e in orden_presentes]
    ax.legend(handles=parches, loc='upper left', fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figura guardada: '{ruta_salida}'")


def figura_evolucion_anual(df_resultados, ruta_salida="figura_evolucion_anual.png"):
    """
    Líneas de beneficio neto año a año para las 3 estrategias clave:
    FAO-56 experta, AquaCrop, Agente SAC.
    Muestra en qué años el AI supera o pierde frente a los benchmarks.
    """
    estrategias_clave = ["FAO-56 experta", "AquaCrop SMT=100%", "Agente SAC"]
    años = sorted(df_resultados['año'].unique())

    fig, ax = plt.subplots(figsize=(13, 5))

    for est in estrategias_clave:
        sub = df_resultados[df_resultados['estrategia'] == est].sort_values('año')
        if sub.empty:
            continue
        ax.plot(sub['año'], sub['beneficio_neto_eur'],
                marker='o', markersize=5, linewidth=1.8,
                color=COLORES.get(est, 'gray'), label=est)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Año', fontsize=11)
    ax.set_ylabel('Beneficio neto (€/ha)', fontsize=11)
    ax.set_title(
        'Evolución anual del beneficio neto — Estrategias clave\n'
        '(Agente SAC vs. benchmarks científicos)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(años)
    ax.set_xticklabels([str(a) for a in años], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figura guardada: '{ruta_salida}'")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def separador(char='═', n=65):
    print(char * n)

if __name__ == "__main__":

    separador()
    print("  EVALUACIÓN COMPARATIVA — Riego Maíz Bustillo del Páramo")
    separador()

    # ── 1. Cargar dataset ─────────────────────────────────
    print(f"\n[1] Cargando dataset: {RUTA_DATASET}")
    df = pd.read_csv(RUTA_DATASET, sep=";", decimal=",")
    n_temporadas = len(df[df['Dias_Plantacion'] == 1])
    print(f"    Filas totales: {len(df):,} | Temporadas: {n_temporadas}")

    # ── 2. Cargar datos AquaCrop ──────────────────────────
    print(f"\n[2] Cargando datos AquaCrop: {RUTA_AQUACROP}")
    try:
        df_aquacrop = pd.read_csv(RUTA_AQUACROP, sep=";", decimal=",")
        print(f"    Filas: {len(df_aquacrop):,} | "
              f"Riego total: {df_aquacrop['Riego_Optimo_FAO_mm'].sum():.1f} mm")
    except FileNotFoundError:
        print(f"    ⚠️  No encontrado. La estrategia AquaCrop usará 0mm.")
        df_aquacrop = None

    # ── 3. Cargar agente SAC ──────────────────────────────
    print(f"\n[3] Cargando agente SAC...")
    agente = AgenteInferencia(RUTA_MODELO, RUTA_VECNORM, RUTA_DATASET)

    # ── 4. Evaluación ─────────────────────────────────────
    separador('─')
    print("\n[4] Evaluando estrategias año a año...")
    df_resultados = evaluar_todas_estrategias(df, agente, df_aquacrop)

    # ── 5. Tabla resumen ──────────────────────────────────
    separador('─')
    print("\n[5] Tabla resumen (media ± σ sobre todos los años):\n")
    df_resumen = generar_resumen(df_resultados)
    print(df_resumen.to_string())

    # ── 6. Exportar CSVs ──────────────────────────────────
    df_resultados.to_csv("resultados_evaluacion.csv", index=False, sep=";", decimal=",")
    df_resumen.to_csv("resumen_evaluacion.csv", sep=";", decimal=",")
    print(f"\n  CSV guardados: 'resultados_evaluacion.csv', 'resumen_evaluacion.csv'")

    # ── 7. Generar figuras ────────────────────────────────
    separador('─')
    print("\n[6] Generando figuras para la memoria TFG...")
    figura_comparacion(df_resultados)
    figura_variabilidad(df_resultados)
    figura_evolucion_anual(df_resultados)

    # ── 8. Diagnóstico rápido ─────────────────────────────
    separador('═')
    print("\n  DIAGNÓSTICO RÁPIDO\n")
    orden = ["Sin riego", "Semanal 20mm", "FAO-56 experta",
             "AquaCrop SMT=100%", "Agente SAC"]
    for est in orden:
        sub = df_resultados[df_resultados['estrategia'] == est]
        if sub.empty:
            continue
        rend  = sub['rendimiento_kg_ha'].mean()
        riego = sub['riego_mm'].mean()
        benef = sub['beneficio_neto_eur'].mean()
        wue   = sub['wue_kg_mm'].mean()
        print(f"  {est:<22} | {rend:>6.0f} kg/ha | {riego:>6.1f} mm | "
              f"{benef:>8.2f} €/ha | {wue:.3f} kg/mm")

    # Comparación AI vs benchmarks
    sub_ia  = df_resultados[df_resultados['estrategia'] == 'Agente SAC']
    sub_fao = df_resultados[df_resultados['estrategia'] == 'FAO-56 experta']
    sub_aqc = df_resultados[df_resultados['estrategia'] == 'AquaCrop SMT=100%']

    if not sub_ia.empty and not sub_fao.empty:
        diff_fao = sub_ia['beneficio_neto_eur'].mean() - sub_fao['beneficio_neto_eur'].mean()
        diff_riego_fao = sub_ia['riego_mm'].mean() - sub_fao['riego_mm'].mean()
        separador('─')
        print(f"\n  AI vs FAO-56 experta:")
        print(f"    Diferencia beneficio : {diff_fao:+.2f} €/ha")
        print(f"    Diferencia riego     : {diff_riego_fao:+.1f} mm/temporada")

    if not sub_ia.empty and not sub_aqc.empty:
        diff_aqc = sub_ia['beneficio_neto_eur'].mean() - sub_aqc['beneficio_neto_eur'].mean()
        diff_riego_aqc = sub_ia['riego_mm'].mean() - sub_aqc['riego_mm'].mean()
        print(f"\n  AI vs AquaCrop SMT=100%:")
        print(f"    Diferencia beneficio : {diff_aqc:+.2f} €/ha")
        print(f"    Diferencia riego     : {diff_riego_aqc:+.1f} mm/temporada")

    separador('═')
    print("\n  ✅ Evaluación completada.")
    print("  Para la memoria: usa 'resumen_evaluacion.csv' y las figuras PNG.")