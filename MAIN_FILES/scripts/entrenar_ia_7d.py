import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy

from simulacion_riego_7d import (
    CalculadoraCultivo,
    calcular_altura_por_gdu,
    calcular_raiz_por_gdu,
    calcular_fase_y_kc,
    VARIANZAS_PRECIP,
    VARIANZAS_ETO,
)

# ══════════════════════════════════════════════════════════════════
# ENTORNO
# ══════════════════════════════════════════════════════════════════

class EntornoRiegoMaiz7d(gym.Env):
    def __init__(self, ruta_csv_sin_escalar):
        super().__init__()

        self.df = pd.read_csv(ruta_csv_sin_escalar, sep=";", decimal=",")
        self.max_dias = len(self.df) - 1

        # Acción continua: mm de riego aplicados hoy
        self.action_space = spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)

        # Observación: 29 variables
        #
        #  [0-8]   CLIMA HOY (9 vars):
        #          GDU_Acumulado, Temp_Max, Temp_Min, Temp_Suelo,
        #          Humedad, Viento, Precipitacion, ETo, Precio_Agua
        #
        #  [9-22]  PRONÓSTICO 7 DÍAS con ruido creciente (14 vars):
        #          Precip_D1..D7  (7 vars) — varianza aumenta por día
        #          ETo_D1..D7     (7 vars) — varianza aumenta por día
        #
        #  [23-28] ESTADO DINÁMICO DEL CULTIVO (6 vars):
        #          Altura_Planta, Dr/TAW, De_Superficie,
        #          Ks_actual, Stress_margin, Factor_Rendimiento
        #
        # Límites amplios — VecNormalize normaliza en tiempo de entrenamiento.
        self.observation_space = spaces.Box(low=-50.0, high=5000.0, shape=(29,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        dias_siembra = self.df.index[self.df['Dias_Plantacion'] == 1].tolist()
        self.dia_actual = int(np.random.choice(dias_siembra)) if dias_siembra else 0

        self.dias_desde_plantacion = 1
        self.sum_etc_max_temporada  = 0.0
        self.sum_etc_real_temporada = 0.0
        self.coste_agua_acumulado   = 0.0
        self.calc = CalculadoraCultivo()
        self.ks_actual = 1.0
        self.factor_rendimiento_prev = 1.0

        return self._obtener_estado_actual(), {}

    def step(self, action):
        riego_ia = float(action[0])

        row_hoy  = self.df.iloc[self.dia_actual]
        gdu      = float(row_hoy['GDU_Acumulado'])
        eto      = float(row_hoy['ETo_mm'])
        precip   = float(row_hoy['Precipitacion_Hoy_mm'])
        viento   = float(row_hoy['Velocidad_Viento_ms'])
        humedad  = float(row_hoy['Humedad_Relativa_pct'])

        precio_base  = float(row_hoy['Precio_Agua_Hoy'])
        ruido_precio = np.clip(np.random.normal(1.0, 0.20), 0.6, 1.4)
        precio_agua_hoy = float(np.clip(precio_base * ruido_precio, 0.02, 0.08))

        altura = calcular_altura_por_gdu(gdu)
        raiz   = calcular_raiz_por_gdu(gdu)
        _, fase = calcular_fase_y_kc(gdu)

        etc, _ = self.calc.Calcular_ETc(
            eto, self.dias_desde_plantacion, viento, precip,
            riego_ia, humedad, altura, gdu
        )

        self.calc.actualizar_taw_y_raw_dinamico(etc, raiz)
        coeficiente_ks = self.calc.calcular_estres_ks()
        etc_real = etc * coeficiente_ks

        self.ks_actual = coeficiente_ks
        self.factor_rendimiento_prev = self.calc.factor_rendimiento_diario

        self.calc.actualizar_penalizacion_rendimiento(etc, etc_real, fase)
        self.calc.actualizar_balance_radicular(precip, riego_ia, etc_real)

        self.sum_etc_max_temporada  += etc
        self.sum_etc_real_temporada += etc_real

        # ── RECOMPENSA ────────────────────────────────────────────────────

        reward = 0.0

        # Penalización por ineficiencia hídrica: regar suelo ya húmedo es desperdicio
        if riego_ia > 0:
            taw_actual = max(self.calc.TAW, 1.0)
            humedad_relativa_suelo = 1.0 - (self.calc.Dr / taw_actual)  # 0=seco, 1=lleno
            reward -= riego_ia * humedad_relativa_suelo * 0.025

        self.coste_agua_acumulado += riego_ia * 10 * precio_agua_hoy

        # Penalización por estrés hídrico en tiempo real
        if coeficiente_ks < 1.0:
            reward -= (1.0 - coeficiente_ks) * 2

        # Penalización por caída de rendimiento hoy
        caida_rendimiento_hoy = self.factor_rendimiento_prev - self.calc.factor_rendimiento_diario
        if caida_rendimiento_hoy > 0:
            reward -= caida_rendimiento_hoy * 40

        # ── AVANCE DE DÍA ─────────────────────────────────────────────────

        self.dia_actual += 1
        self.dias_desde_plantacion += 1

        gdu_siguiente = float(self.df.iloc[self.dia_actual]['GDU_Acumulado']) \
            if self.dia_actual < self.max_dias else 0.0

        terminated = (
            self.dia_actual >= self.max_dias
            or self.df.iloc[self.dia_actual]['Dias_Plantacion'] == 0
            or gdu_siguiente >= 1275.0
            or self.dias_desde_plantacion > 190
        )

        # Premio final de cosecha
        if terminated:
            rendimiento_max_kg_ha = 15000.0
            precio_maiz_kg = 0.23
            coste_fijo_boe = 32.18

            rendimiento_real = rendimiento_max_kg_ha * self.calc.factor_rendimiento_diario
            ingresos_cosecha = rendimiento_real * precio_maiz_kg
            beneficio_neto   = (ingresos_cosecha - self.coste_agua_acumulado - coste_fijo_boe) / 10.0
            reward += beneficio_neto

        obs = self._obtener_estado_actual()
        return obs, reward, terminated, False, {}

    def _obtener_estado_actual(self):
        row = self.df.iloc[self.dia_actual]

        # [0-8] Clima hoy
        columnas_hoy = [
            'GDU_Acumulado', 'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C',
            'Humedad_Relativa_pct', 'Velocidad_Viento_ms',
            'Precipitacion_Hoy_mm', 'ETo_mm', 'Precio_Agua_Hoy',
        ]
        estado = list(row[columnas_hoy].values.astype(np.float32))

        # [9-22] Pronóstico 7 días con segunda capa de ruido creciente
        # El CSV ya tiene ruido baked-in desde simulacion_riego_7d.
        # Añadimos una segunda capa para que cada episodio sea diferente,
        # forzando al agente a aprender políticas robustas.
        for d in range(7):
            p_csv = float(row[f'Precip_D{d+1}_mm'])
            if p_csv <= 0.5:
                p_ruidosa = max(0.0, p_csv + np.random.normal(0.0, 0.5))
            else:
                p_ruidosa = max(0.0, p_csv + np.random.normal(0.0, p_csv * VARIANZAS_PRECIP[d]))
            estado.append(np.float32(p_ruidosa))

        for d in range(7):
            e_csv   = float(row[f'ETo_D{d+1}_mm'])
            ruido_e = np.clip(np.random.normal(1.0, VARIANZAS_ETO[d]), 0.3, 2.0)
            estado.append(np.float32(max(0.0, e_csv * ruido_e)))

        # [23] Altura de planta
        gdu = float(row['GDU_Acumulado'])
        estado.append(np.float32(calcular_altura_por_gdu(gdu)))

        # [24] Dr/TAW — fracción de agotamiento radicular [0=lleno, 1=marchitez]
        dr_actual  = getattr(self.calc, 'Dr',  0.0)
        taw_actual = max(getattr(self.calc, 'TAW', 1.0), 1.0)
        estado.append(np.float32(dr_actual / taw_actual))

        # [25] De — agotamiento capa superficial (dual Kc)
        estado.append(np.float32(getattr(self.calc, 'De', 0.0)))

        # [26] Ks — coeficiente de estrés hídrico HOY [0=estrés total, 1=sin estrés]
        estado.append(np.float32(self.ks_actual))

        # [27] Stress margin: (RAW - Dr) / TAW
        # Positivo → buffer disponible; negativo → estrés ya activo
        raw_actual    = getattr(self.calc, 'RAW', 0.0)
        stress_margin = (raw_actual - dr_actual) / taw_actual
        estado.append(np.float32(stress_margin))

        # [28] Factor de rendimiento acumulado [0-1]
        estado.append(np.float32(getattr(self.calc, 'factor_rendimiento_diario', 1.0)))

        return np.array(estado, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════

class GuardarVecNormalizeCallback(BaseCallback):
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
            elapsed  = time.time() - self.start_time
            progress = self.num_timesteps / self.locals.get('total_timesteps', 1) * 100
            velocidad = self.num_timesteps / elapsed

            if len(self.model.ep_info_buffer) > 0:
                rew_mean = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                rew_str  = f"ep_rew_mean={rew_mean:>8.1f}"
            else:
                rew_str = "ep_rew_mean=        —"

            eta_min = ((self.locals.get('total_timesteps', 1) - self.num_timesteps) / max(velocidad, 1)) / 60
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

def plotear_curva_aprendizaje(log_dir: str, ruta_salida: str) -> None:
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) == 0:
            print("  Sin datos suficientes para plotear la curva.")
            return

        ventana  = min(50, len(y))
        y_smooth = np.convolve(y, np.ones(ventana) / ventana, mode='valid')
        x_smooth = x[ventana - 1:]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, alpha=0.25, color='steelblue', linewidth=0.8, label='Episodios raw')
        ax.plot(x_smooth, y_smooth, color='steelblue', linewidth=2.0,
                label=f'Media móvil ({ventana} ep)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Pasos de entrenamiento', fontsize=12)
        ax.set_ylabel('Reward per episodi (pts)', fontsize=12)
        ax.set_title('Curva de aprenentatge — SAC 7d (reg blat de moro a Bustillo del Páramo)', fontsize=13)
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
    RUTA_DATASET  = "../../DATASET_IA/dataset_entrenamiento_sin_escalar_7d.csv"
    DIR_LOGS      = "../logs_monitor/SAC_7d/"
    DIR_MODELOS   = "../modelos/7d/"
    RUTA_MODELO   = "../modelos/7d/ia_riego_maiz_sac_7d"
    RUTA_VN_MEJOR = "../modelos/7d/vecnormalize_mejor_7d.pkl"
    RUTA_VN_FINAL = "../modelos/7d/vecnormalize_stats_7d.pkl"
    RUTA_CURVA    = "../resultados/figuras/curva_aprendizaje_7d.png"

    os.makedirs(DIR_LOGS,    exist_ok=True)
    os.makedirs(DIR_MODELOS, exist_ok=True)
    os.makedirs(os.path.dirname(RUTA_CURVA), exist_ok=True)

    # ── 1. Validar entorno ───────────────────────────────────────
    print("Validando entorno con Gymnasium checker...")
    env_check = EntornoRiegoMaiz7d(RUTA_DATASET)
    check_env(env_check)
    del env_check
    print("Entorno validado correctamente.\n")

    # ── 2. Entorno de entrenamiento ──────────────────────────────
    env_train = DummyVecEnv([lambda: Monitor(EntornoRiegoMaiz7d(RUTA_DATASET), DIR_LOGS)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── 3. Entorno de evaluación ─────────────────────────────────
    env_eval = DummyVecEnv([lambda: Monitor(EntornoRiegoMaiz7d(RUTA_DATASET))])
    env_eval = VecNormalize(env_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── 4. Callbacks ─────────────────────────────────────────────
    cb_guardar_vn = GuardarVecNormalizeCallback(
        ruta_stats = RUTA_VN_MEJOR,
        env_train  = env_train,
        verbose    = 1,
    )

    cb_eval = EvalCallback(
        eval_env             = env_eval,
        best_model_save_path = DIR_MODELOS,
        log_path             = DIR_LOGS,
        eval_freq            = 10_000,
        n_eval_episodes      = 15,
        deterministic        = True,
        render               = False,
        verbose              = 1,
        callback_on_new_best = cb_guardar_vn,
    )

    cb_progreso = ProgressCallback(print_freq=50_000)

    callbacks = CallbackList([cb_eval, cb_progreso])

    # ── 5. Modelo SAC ────────────────────────────────────────────
    # Red ligeramente más grande que la versión 1d (256,256,256) para
    # procesar el horizonte extendido de 14 variables de pronóstico.
    print("Inicializando agente SAC (horizonte 7 días)...\n")
    modelo = SAC(
        policy          = "MlpPolicy",
        env             = env_train,
        verbose         = 0,
        learning_rate   = 3e-4,
        buffer_size     = 300_000,
        batch_size      = 256,
        gamma           = 0.995,
        ent_coef        = "auto_0.1",
        policy_kwargs   = dict(net_arch=[256, 256, 256]),
        tensorboard_log = DIR_LOGS,
    )

    # ── 6. Entrenamiento ─────────────────────────────────────────
    PASOS = 1_500_000
    print(f"Comenzando entrenamiento ({PASOS:,} pasos)...")

    modelo.learn(
        total_timesteps = PASOS,
        callback        = callbacks,
        progress_bar    = False,
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

    print("\n Entrenamiento completado!")
    print("   Para la evaluación usa: 'best_model.zip' + 'vecnormalize_mejor_7d.pkl'")
