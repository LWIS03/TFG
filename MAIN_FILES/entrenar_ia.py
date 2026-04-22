import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy

from simulacion_riego import CalculadoraCultivo, calcular_altura_por_gdu, calcular_raiz_por_gdu, calcular_fase_y_kc

class EntornoRiegoMaiz(gym.Env):
    def __init__(self, ruta_csv_sin_escalar):
        super(EntornoRiegoMaiz, self).__init__()
        
        # 1. Cargamos el dataset NORMAL (sin escalar) para que las físicas funcionen bien
        self.df = pd.read_csv(ruta_csv_sin_escalar, sep=";", decimal=",")
        self.max_dias = len(self.df) - 1
        
        # 2. ESPACIO DE ACCIÓN:
        self.action_space = spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)
        
        # 3. ESPACIO DE OBSERVACIÓN: 16 variables en total:
        #
        #  [0-8]  CLIMA HOY (9 vars):
        #         GDU_Acumulado, Temp_Max, Temp_Min, Temp_Suelo,
        #         Humedad, Viento, Precipitacion, ETo, Precio_Agua
        #
        #  [9-10] PRONOSTICO MANANA con ruido (2 vars):
        #         Precip_Manana, ETo_Manana
        #
        #  [11-16] ESTADO DINAMICO DEL CULTIVO (6 vars):
        #   11: Altura_Planta          -> tamano actual del cultivo
        #   12: Dr/TAW ratio [0-1]     -> fraccion de agotamiento radicular
        #                                 (mejor que Dr crudo porque TAW cambia con la raiz)
        #   13: De_Superficie          -> agotamiento capa superficial (dual Kc)
        #   14: Ks_actual [0-1]        -> estres hidrico HOY en tiempo real
        #   15: Stress_margin          -> (RAW - Dr) / TAW
        #                                 positivo = hay buffer antes del estres
        #                                 negativo = ya estamos en estres
        #                                 el agente puede actuar ANTES de que Ks baje
        #   16: Factor_Rendimiento     -> dano acumulado en rendimiento [0-1]
        #
        # Limites amplios porque VecNormalize normaliza todo durante el entrenamiento.
        self.observation_space = spaces.Box(low=-50.0, high=5000.0, shape=(17,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Buscamos en el CSV los días donde empieza la temporada (Dias_Plantacion == 1)
        # y elegimos uno al azar para que la IA entrene en años diferentes cada partida.
        dias_siembra = self.df.index[self.df['Dias_Plantacion'] == 1].tolist()
        if len(dias_siembra) > 0:
            self.dia_actual = int(np.random.choice(dias_siembra))
        else:
            self.dia_actual = 0
            
        self.dias_desde_plantacion = 1

        # Contadores para evaluar a la IA al final de la temporada
        self.sum_etc_max_temporada = 0.0
        self.sum_etc_real_temporada = 0.0

        # Acumulador del coste del agua durante la temporada (se descuenta todo en la cosecha)
        self.coste_agua_acumulado = 0.0

        # Reiniciamos tu gemelo digital PARA QUE CALCULE EN VIVO
        self.calc = CalculadoraCultivo()

        # --- VARIABLES DE ESTADO PARA LA OBSERVACIÓN ---
        # Ks del día actual (se actualiza en step() tras calcular el estrés)
        self.ks_actual = 1.0
        # Factor de rendimiento del día anterior (para calcular la caída diaria)
        self.factor_rendimiento_prev = 1.0
        
        # Devolvemos el estado del día 0
        return self._obtener_estado_actual(), {}

    def step(self, action):
        # 1. TRADUCIR ACCIÓN DE LA IA A MILÍMETROS DE RIEGO
        opciones_riego = float(action[0])  # La IA nos da un número entre 0 y 30
        riego_ia = opciones_riego

        # 2. OBTENER EL CLIMA DE "HOY" DEL CSV
        row_hoy = self.df.iloc[self.dia_actual]
        gdu = float(row_hoy['GDU_Acumulado'])
        eto = float(row_hoy['ETo_mm'])
        precip = float(row_hoy['Precipitacion_Hoy_mm'])
        viento = float(row_hoy['Velocidad_Viento_ms'])
        humedad = float(row_hoy['Humedad_Relativa_pct'])
        # Ruido en el precio del agua: ±20% sobre el valor del CSV.
        # Cada episodio el agente ve precios ligeramente distintos,
        # forzándole a reaccionar al nivel de precio y no a memorizar la secuencia.
        precio_base = float(row_hoy['Precio_Agua_Hoy'])
        ruido_precio = np.clip(np.random.normal(1.0, 0.20), 0.6, 1.4)
        precio_agua_hoy = float(np.clip(precio_base * ruido_precio, 0.02, 0.08))
        
        # 3. AVANZAR LAS FÍSICAS DE LA SIMULACIÓN
        altura = calcular_altura_por_gdu(gdu)
        raiz = calcular_raiz_por_gdu(gdu)
        _, fase = calcular_fase_y_kc(gdu)  # Solo necesitamos 'fase' para el Ky de rendimiento

        etc, kc_calculada = self.calc.Calcular_ETc(
            eto, self.dias_desde_plantacion, viento, precip, 
            riego_ia, humedad, altura, gdu
        )

        self.calc.actualizar_taw_y_raw_dinamico(etc, raiz)

        coeficiente_ks = self.calc.calcular_estres_ks()
        etc_real = etc * coeficiente_ks

        # Guardamos Ks para que _obtener_estado_actual() lo incluya en la observación
        self.ks_actual = coeficiente_ks

        # Guardamos el factor de rendimiento ANTES de que se actualice,
        # para poder medir la caída que ocurre HOY
        self.factor_rendimiento_prev = self.calc.factor_rendimiento_diario

        self.calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)
        self.calc.actualizar_balance_radicular(precip, riego_ia, etc_real)

        # Guardamos la ETc para la fórmula de pérdida de rendimiento final
        self.sum_etc_max_temporada += etc
        self.sum_etc_real_temporada += etc_real

        # 4. LÓGICA DE RECOMPENSA (REWARD SHAPING)
        reward = 0.0

        # Acumulamos el coste real del agua.
        # El 80% se descontará en la cosecha; el 20% se señaliza hoy (ver abajo).
        self.coste_agua_acumulado += riego_ia * 10 * precio_agua_hoy  # mm -> m3/ha

        # --- FIX 0: Señal diaria de coste del agua (20% inmediato) ──────────
        # PROBLEMA DETECTADO en evaluación: el agente usaba 25% más agua que FAO-56
        # porque el coste del agua solo se sentía al final (terminal), mientras que
        # las penalizaciones por estrés se sentían cada día.
        # Con gamma=0.995, un reward a 150 días vale 0.47 veces el mismo reward hoy.
        # Solución: señalizar el 20% del coste del agua HOY (al tomar la decisión)
        # para que sea comparable con las penalizaciones diarias de estrés.
        # El 80% restante sigue en el terminal para mantener la escala económica.
        # IMPORTANTE: el total de penalización por agua es IDÉNTICO al anterior
        # (20% diario + 80% terminal = 100%), solo cambia la distribución temporal.
        coste_agua_hoy = riego_ia * 10 * precio_agua_hoy
        reward -= (coste_agua_hoy * 0.2) / 10.0

        # --- FIX 1: Penalización por regar con el suelo saturado ---
        # Dr/TAW < 0.10 → suelo más del 90% lleno, el agua percolará sin beneficio.
        taw_actual = max(self.calc.TAW, 1.0)
        dr_taw_ratio = self.calc.Dr / taw_actual
        if dr_taw_ratio < 0.10 and riego_ia > 0:
            reward -= (riego_ia * 0.2)

        # --- FIX 2: Penalización por estrés hídrico en tiempo real ---
        # Señal diaria inmediata. Escala × 2 (calibrada para ~180 días).
        if coeficiente_ks < 1.0:
            reward -= (1.0 - coeficiente_ks) * 2

        # --- FIX 3: Penalización por caída del rendimiento HOY ---
        # Escala × 40 (calibrada para que FIX2+FIX3 ≈ 1.3× terminal en caso sin riego).
        caida_rendimiento_hoy = self.factor_rendimiento_prev - self.calc.factor_rendimiento_diario
        if caida_rendimiento_hoy > 0:
            reward -= caida_rendimiento_hoy * 40
            
        # 5. PASAR DE DÍA Y COMPROBAR FIN DE TEMPORADA
        self.dia_actual += 1
        self.dias_desde_plantacion += 1
        
        if self.dia_actual < self.max_dias:
            gdu_actual = float(self.df.iloc[self.dia_actual]['GDU_Acumulado'])
        else:
            gdu_actual = 0.0

        terminated = False
        
        if self.dia_actual >= self.max_dias:
            terminated = True 
        elif self.df.iloc[self.dia_actual]['Dias_Plantacion'] == 0:
            terminated = True 
        elif gdu_actual >= 1275.0:  # GDU típico para maíz en madurez fisiológica
            terminated = True 
        elif self.dias_desde_plantacion > 190:
            terminated = True

        # 6. EL GRAN PREMIO FINAL (Día de la cosecha)
        if terminated:
            rendimiento_max_kg_ha = 15000.0
            precio_maiz_kg = 0.23
            coste_fijo_boe = 32.18           # Cuota fija anual BOE 2026

            # Rendimiento real según el estrés acumulado día a día
            rendimiento_real = rendimiento_max_kg_ha * self.calc.factor_rendimiento_diario

            ingresos_cosecha = rendimiento_real * precio_maiz_kg

            # Beneficio neto: ingresos - 80% del coste del agua - cuota fija.
            # El 20% restante ya fue señalizado diariamente (FIX 0).
            # Total coste agua = 20% diario + 80% terminal = 100% (sin doble conteo).
            # Dividido por 10 para no saturar la red neuronal.
            beneficio_neto = (ingresos_cosecha - self.coste_agua_acumulado * 0.8 - coste_fijo_boe) / 10.0
            reward += beneficio_neto

        # 7. OBTENER NUEVA OBSERVACIÓN PARA LA IA
        obs = self._obtener_estado_actual()
        
        return obs, reward, terminated, False, {}

    def _obtener_estado_actual(self):
        # Lee la fila del día actual
        row = self.df.iloc[self.dia_actual]

        # Variables de clima actual (observadas, sin ruido — ya pasaron)
        columnas_hoy = [
            'GDU_Acumulado', 'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C',
            'Humedad_Relativa_pct', 'Velocidad_Viento_ms', 'Precipitacion_Hoy_mm', 'ETo_mm',
            'Precio_Agua_Hoy'
        ]
        estado_base = list(row[columnas_hoy].values.astype(np.float32))

        # --- PRONÓSTICO DE MAÑANA CON RUIDO (segunda capa de incertidumbre) ---
        # El CSV ya tiene ruido baked-in desde la generación del dataset.
        # Aquí añadimos una segunda capa para que cada episodio sea ligeramente distinto,
        # forzando al agente a aprender políticas robustas en vez de memorizar secuencias.

        # Precipitación mañana: muy incierta, ruido gaussiano ±30%
        precip_fcst = float(row['Precip_Manana_mm'])
        if precip_fcst <= 0.5:
            precip_fcst_ruidosa = max(0.0, precip_fcst + np.random.normal(0.0, 0.5))
        else:
            precip_fcst_ruidosa = max(0.0, precip_fcst + np.random.normal(0.0, precip_fcst * 0.30))
        estado_base.append(np.float32(precip_fcst_ruidosa))

        # ETo mañana: más predecible, ruido multiplicativo ±15%
        eto_fcst = float(row['ETo_Manana_mm'])
        ruido_eto = np.clip(np.random.normal(1.0, 0.15), 0.5, 1.5)
        estado_base.append(np.float32(max(0.0, eto_fcst * ruido_eto)))

        # ¡INYECCIÓN DINÁMICA! Variables de estado del cultivo calculadas en tiempo real

        # [11] Altura de planta
        gdu = row['GDU_Acumulado']
        altura_planta = calcular_altura_por_gdu(gdu)
        estado_base.append(np.float32(altura_planta))

        # [12] Dr/TAW: fracción de agotamiento radicular [0=lleno, 1=marchitez]
        # Mejor que Dr crudo porque TAW cambia cada día con la profundidad de la raíz.
        # Un Dr=30mm significa cosas muy distintas con TAW=40mm (75% agotado, peligro)
        # que con TAW=78mm (38% agotado, sin problema).
        dr_actual = getattr(self.calc, 'Dr', 0.0)
        taw_actual = getattr(self.calc, 'TAW', 1.0)
        dr_taw_ratio = dr_actual / max(taw_actual, 1.0)  # [0, 1]
        estado_base.append(np.float32(dr_taw_ratio))

        # [13] De: agotamiento de la capa superficial (dual Kc, controla Ke)
        de_actual = getattr(self.calc, 'De', 0.0)
        estado_base.append(np.float32(de_actual))

        # [14] Ks: coeficiente de estrés hídrico HOY [0=estrés total, 1=sin estrés]
        # Sin esto el agente no sabe cuán estresada está la planta en tiempo real;
        # solo lo sabría al final de la temporada cuando el rendimiento ya está dañado.
        estado_base.append(np.float32(self.ks_actual))

        # [15] Stress margin: (RAW - Dr) / TAW
        # A diferencia de Ks (que es 1.0 hasta que el estrés ya ha comenzado),
        # este valor empieza a bajar ANTES de que haya estrés real.
        # Positivo → hay buffer todavía, el agente puede esperar
        # Negativo → Dr ya superó RAW, el estrés ya está activo
        # Esto permite al agente anticiparse en lugar de reaccionar tarde.
        raw_actual = getattr(self.calc, 'RAW', 0.0)
        taw_obs    = max(getattr(self.calc, 'TAW', 1.0), 1.0)
        dr_obs     = getattr(self.calc, 'Dr', 0.0)
        stress_margin = (raw_actual - dr_obs) / taw_obs
        estado_base.append(np.float32(stress_margin))

        # [16] Factor de rendimiento acumulado [0-1]
        # Permite al agente ver el daño acumulado por estrés pasado EN TIEMPO REAL,
        # no solo al final. Así puede corregir si está perdiendo demasiado rendimiento.
        factor_rend = getattr(self.calc, 'factor_rendimiento_diario', 1.0)
        estado_base.append(np.float32(factor_rend))

        return np.array(estado_base, dtype=np.float32)




# ══════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════

class GuardarVecNormalizeCallback(BaseCallback):
    """
    Se dispara cada vez que EvalCallback encuentra un nuevo mejor modelo.
    Guarda las estadísticas de VecNormalize junto al modelo para que sean
    consistentes — sin ellas, el modelo guardado no puede hacer inferencia.
    """
    def __init__(self, ruta_stats: str, env_train: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.ruta_stats = ruta_stats
        self.env_train  = env_train

    def _on_step(self) -> bool:
        self.env_train.save(self.ruta_stats)
        if self.verbose:
            print(f"    → VecNormalize stats guardadas: '{self.ruta_stats}'")
        return True


class ProgressCallback(BaseCallback):
    """
    Imprime un resumen legible cada `print_freq` pasos:
      · Progreso (%)
      · Tiempo transcurrido
      · ep_rew_mean de los últimos episodios
    """
    def __init__(self, print_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"\n{'─'*65}")
        print(f"  Inicio del entrenamiento — {time.strftime('%H:%M:%S')}")
        print(f"{'─'*65}")

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0 and self.n_calls > 0:
            elapsed   = time.time() - self.start_time
            progress  = self.num_timesteps / self.locals.get('total_timesteps', 1) * 100
            velocidad = self.num_timesteps / elapsed  # pasos/seg

            # ep_rew_mean de los últimos episodios (del buffer del Monitor)
            if len(self.model.ep_info_buffer) > 0:
                rew_mean = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                rew_str  = f"ep_rew_mean={rew_mean:>8.1f}"
            else:
                rew_str = "ep_rew_mean=        —"

            eta_seg = (self.locals.get('total_timesteps', 1) - self.num_timesteps) / max(velocidad, 1)
            eta_min = eta_seg / 60

            print(f"  [{self.num_timesteps:>8} pasos | {progress:>4.0f}% | "
                  f"{elapsed/60:>5.1f}min | ETA {eta_min:>4.0f}min] {rew_str}")
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        print(f"\n{'─'*65}")
        print(f"  Entrenamiento finalizado en {elapsed/60:.1f} min — {time.strftime('%H:%M:%S')}")
        print(f"{'─'*65}\n")


# ══════════════════════════════════════════════════════════════════
# CURVA DE APRENDIZAJE
# ══════════════════════════════════════════════════════════════════

def plotear_curva_aprendizaje(log_dir: str, ruta_salida: str = "curva_aprendizaje.png") -> None:
    """
    Lee los logs del Monitor y genera la curva de aprendizaje suavizada.
    Guarda la figura como PNG (útil para la memoria del TFG).
    """
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) == 0:
            print("  Sin datos suficientes para plotear la curva.")
            return

        # Suavizado con media móvil de 50 episodios
        ventana = min(50, len(y))
        y_smooth = np.convolve(y, np.ones(ventana) / ventana, mode='valid')
        x_smooth = x[ventana - 1:]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, alpha=0.25, color='steelblue', linewidth=0.8, label='Episodios raw')
        ax.plot(x_smooth, y_smooth, color='steelblue', linewidth=2.0,
                label=f'Media móvil ({ventana} ep)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Pasos de entrenamiento', fontsize=12)
        ax.set_ylabel('Reward por episodio (pts)', fontsize=12)
        ax.set_title('Curva de aprendizaje — Agente SAC (riego maíz Bustillo del Páramo)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=150)
        plt.close()
        print(f"  Curva guardada en: '{ruta_salida}'")
    except Exception as e:
        print(f"  No se pudo plotear la curva: {e}")


# ══════════════════════════════════════════════════════════════════
# ENTRENAMIENTO PRINCIPAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Rutas ────────────────────────────────────────────────────
    RUTA_DATASET   = "../DATASET_IA/dataset_entrenamiento_sin_escalar.csv"
    DIR_LOGS       = "./logs_monitor/"        # Monitor logs (para curva de aprendizaje)
    DIR_MODELOS    = "./modelos/"             # Mejor modelo durante entrenamiento
    RUTA_MODELO    = "ia_riego_maiz_sac"      # Modelo final
    RUTA_VN_MEJOR  = "./modelos/vecnormalize_mejor.pkl"   # Stats del mejor modelo
    RUTA_VN_FINAL  = "vecnormalize_stats.pkl"             # Stats del modelo final
    RUTA_CURVA     = "curva_aprendizaje.png"

    os.makedirs(DIR_LOGS, exist_ok=True)
    os.makedirs(DIR_MODELOS, exist_ok=True)

    # ── 1. Validar entorno ───────────────────────────────────────
    print("Validando entorno con Gymnasium checker...")
    env_check = EntornoRiegoMaiz(RUTA_DATASET)
    check_env(env_check)
    del env_check
    print("Entorno validado correctamente.\n")

    # ── 2. Entorno de entrenamiento ──────────────────────────────
    # Monitor guarda los logs de reward por episodio en DIR_LOGS
    env_train = DummyVecEnv([lambda: Monitor(EntornoRiegoMaiz(RUTA_DATASET), DIR_LOGS)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── 3. Entorno de evaluación (separado) ──────────────────────
    # norm_reward=False: queremos ver rewards REALES durante la evaluación,
    # no normalizados, para poder interpretarlos económicamente.
    # EvalCallback sincroniza automáticamente las estadísticas de obs de
    # env_train → env_eval en cada evaluación.
    env_eval = DummyVecEnv([lambda: Monitor(EntornoRiegoMaiz(RUTA_DATASET))])
    env_eval = VecNormalize(env_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── 4. Callbacks ─────────────────────────────────────────────
    # 4a. Guarda VecNormalize stats cuando EvalCallback encuentra mejor modelo
    cb_guardar_vn = GuardarVecNormalizeCallback(
        ruta_stats = RUTA_VN_MEJOR,
        env_train  = env_train,
        verbose    = 1,
    )

    # 4b. Evalúa cada 10k pasos sobre 15 episodios con política determinista.
    #     Guarda el mejor modelo automáticamente en DIR_MODELOS.
    #     n_eval_episodes=15 cubre varios años del dataset para un resultado estable.
    cb_eval = EvalCallback(
        eval_env              = env_eval,
        best_model_save_path  = DIR_MODELOS,
        log_path              = DIR_LOGS,
        eval_freq             = 10_000,
        n_eval_episodes       = 15,
        deterministic         = True,
        render                = False,
        verbose               = 1,
        callback_on_new_best  = cb_guardar_vn,
    )

    # 4c. Imprime progreso legible cada 50k pasos
    cb_progreso = ProgressCallback(print_freq=50_000, verbose=0)

    callbacks = CallbackList([cb_eval, cb_progreso])

    # ── 5. Modelo SAC ────────────────────────────────────────────
    print("Inicializando agente SAC...\n")
    modelo = SAC(
        policy        = "MlpPolicy",
        env           = env_train,
        verbose       = 0,               # Silenciamos SB3 — usamos nuestro ProgressCallback
        learning_rate = 3e-4,
        buffer_size   = 300_000,         # +200k vs antes: más experiencia diversa disponible
        batch_size    = 256,
        gamma         = 0.995,           # Horizonte largo (temporada completa)
        ent_coef      = "auto_0.1",      # Entropía automática, inicial 0.1
        policy_kwargs = dict(net_arch=[256, 256]),
        tensorboard_log = DIR_LOGS,      # Opcional: ver en TensorBoard con `tensorboard --logdir logs_monitor`
    )

    # ── 6. Entrenamiento ─────────────────────────────────────────
    # 1.5M pasos ≈ ~8.300 temporadas de siembra completas
    # Con ~180 días/temporada: ~8.300 episodios de experiencia real
    PASOS = 1_500_000
    print(f"Comenzando entrenamiento ({PASOS:,} pasos)...")

    modelo.learn(
        total_timesteps = PASOS,
        callback        = callbacks,
        progress_bar    = False,         # Usamos nuestro propio callback de progreso
    )

    # ── 7. Guardar modelo final ───────────────────────────────────
    modelo.save(RUTA_MODELO)
    env_train.save(RUTA_VN_FINAL)

    print(f"\n  Modelo final guardado  : '{RUTA_MODELO}.zip'")
    print(f"  VecNormalize final     : '{RUTA_VN_FINAL}'")
    print(f"  Mejor modelo (eval)    : '{DIR_MODELOS}best_model.zip'")
    print(f"  VecNormalize del mejor : '{RUTA_VN_MEJOR}'")

    # ── 8. Curva de aprendizaje ───────────────────────────────────
    print("\nGenerando curva de aprendizaje...")
    plotear_curva_aprendizaje(DIR_LOGS, RUTA_CURVA)

    print("\n🎉 ¡Entrenamiento completado!")
    print("   Para la evaluación usa: 'best_model.zip' + 'vecnormalize_mejor.pkl'")
    print("   (son los que maximizan el reward en validación, no el último checkpoint)")