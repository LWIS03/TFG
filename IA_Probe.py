import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from MAIN_FILES.simulacion_riego import calcular_altura_por_gdu, calcular_raiz_por_gdu, calcular_fase_y_kc, CalculadoraCultivo

class MockEnv(gym.Env):
    def __init__(self):
        super(MockEnv, self).__init__()
        # AHORA SON 17 VARIABLES, igual que en entrenar_ia.py
        self.observation_space = spaces.Box(low=-50.0, high=5000.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        return np.zeros(17, dtype=np.float32), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(17, dtype=np.float32), {}
    
def cargar_modelo():
    try:
        # 1. Creamos el entorno falso (ahora con 17 vars)
        env = DummyVecEnv([lambda: MockEnv()])
        # 2. Cargamos las estadísticas de normalización del entrenamiento
        env = VecNormalize.load("./MAIN_FILES/vecnormalize_stats.pkl", env)
        env.training = False # IMPORTANTE: Evita que se modifiquen los pesos
        env.norm_reward = False
        
        # 3. Cargamos el cerebro 
        modelo = SAC.load("./MAIN_FILES/ia_riego_maiz_sac", env=env)
        return modelo, env
    except Exception as e:
        print(f"Error al cargar: {e}")
        return None, None
    
def calcular_aigua():
    ruta_dataset = "./DATASET_IA/dataset_entrenamiento_sin_escalar.csv"
    df = pd.read_csv(ruta_dataset, sep=";", decimal=",")

    modelo, env_normalizado = cargar_modelo()
    if modelo is None:
        return

    print("Iniciando predicción... (Esto puede tardar unos segundos)")

    resumen_temporadas = {}
    registro_diario = [] 
    
    # Variables de seguimiento entre días
    calculadora = None
    ks_actual = 1.0
    dias_sin_riego = 0

    for index, row in df.iterrows():
        dias_plantacion = row['Dias_Plantacion']

        # Si es el primer día de la temporada, reiniciamos el gemelo digital y los contadores
        if dias_plantacion == 1:
            calculadora = CalculadoraCultivo()
            ks_actual = 1.0
            dias_sin_riego = 0

        if dias_plantacion > 0:
            fecha = row['Fecha']
            año = str(fecha).split('/')[-1] 
            
            # 1. Extraer clima
            gdu_acumulado = row['GDU_Acumulado']
            t_max = row['Temp_Max_C']
            t_min = row['Temp_Min_C']
            t_suelo = row['Temp_Suelo_C']
            humitat = row['Humedad_Relativa_pct']
            viento = row['Velocidad_Viento_ms']
            precipitacion_hoy = row['Precipitacion_Hoy_mm']
            eto_hoy = row['ETo_mm']
            precipitacion_mañana = row['Precip_Manana_mm']
            eto_mañana = row['ETo_Manana_mm']
            precio_agua = row['Precio_Agua_Hoy'] 
            
            # 2. Físicas de la planta de hoy
            altura = calcular_altura_por_gdu(gdu_acumulado)
            raiz = calcular_raiz_por_gdu(gdu_acumulado)
            _, fase = calcular_fase_y_kc(gdu_acumulado)

            # 3. Extraer el estado interno del gemelo digital (NUEVAS VARIABLES)
            dr_actual = getattr(calculadora, 'Dr', 0.0)
            taw_actual = getattr(calculadora, 'TAW', 1.0)
            dr_taw_ratio = dr_actual / max(taw_actual, 1.0)
            de_actual = getattr(calculadora, 'De', 0.0)
            factor_rend = getattr(calculadora, 'factor_rendimiento_diario', 1.0)

            # 4. Formar el estado EXACTAMENTE en el mismo orden que _obtener_estado_actual()
            estado_actual = np.array([
                gdu_acumulado, t_max, t_min, t_suelo,
                humitat, viento, precipitacion_hoy, eto_hoy,
                precio_agua, precipitacion_mañana, eto_mañana,
                altura, dr_taw_ratio, de_actual, ks_actual,
                float(dias_sin_riego), factor_rend
            ], dtype=np.float32)

            # 5. La IA predice cuánto regar
            obs_normalizada = env_normalizado.normalize_obs(estado_actual)
            accion, _states = modelo.predict(obs_normalizada, deterministic=True)
            
            riego_recomendado = accion[0][0] if isinstance(accion[0], np.ndarray) else accion[0]
            riego_recomendado = max(0.0, float(riego_recomendado)) 

            # --- GUARDAR DATOS DEL DÍA A DÍA ---
            registro_diario.append({
                "Fecha": fecha,
                "Temporada": año,
                "Dias_Plantacion": dias_plantacion,
                "Dr_TAW_Ratio": round(dr_taw_ratio, 3), # Guardamos el ratio para ver qué pensaba la IA
                "Lluvia_Hoy_mm": precipitacion_hoy,
                "ETo_Hoy_mm": eto_hoy,
                "Riego_Decidido_IA_mm": round(riego_recomendado, 2)
            })

            # --- SUMAR DATOS A LA TEMPORADA ---
            if año not in resumen_temporadas:
                resumen_temporadas[año] = 0.0
            resumen_temporadas[año] += riego_recomendado

            # 6. ACTUALIZAR LAS FÍSICAS AL FINAL DEL DÍA (Como hace el step() en el entreno)
            etc, kc_calculada = calculadora.Calcular_ETc(
                eto_hoy, dias_plantacion, viento, precipitacion_hoy, 
                riego_recomendado, humitat, altura, gdu_acumulado
            )

            calculadora.actualizar_taw_y_raw_dinamico(etc, raiz)
            
            ks_actual = calculadora.calcular_estres_ks() # Guardamos para el state de mañana
            etc_real = etc * ks_actual

            # Actualizar días sin riego
            if riego_recomendado > 0.5:
                dias_sin_riego = 0
            else:
                dias_sin_riego += 1

            calculadora.actualizar_penalizacion_rendimiento(etc, etc_real, fase)
            calculadora.actualizar_balance_radicular(precipitacion_hoy, riego_recomendado, etc_real)

    # ==========================================
    # GUARDAR Y EXPORTAR
    # ==========================================
    df_diario = pd.DataFrame(registro_diario)
    df_diario.to_csv("historial_diario_ia.csv", index=False, sep=";", decimal=",")
    
    df_resumen = pd.DataFrame(list(resumen_temporadas.items()), columns=['Temporada_Año', 'Total_Riego_IA_mm'])
    df_resumen['Total_Riego_IA_mm'] = df_resumen['Total_Riego_IA_mm'].round(2)
    df_resumen.to_csv("resumen_riego_por_temporada.csv", index=False, sep=";", decimal=",")
    
    print("\n✅ Proceso completado con éxito. Se han generado dos archivos:")
    print("  1. historial_diario_ia.csv")
    print("  2. resumen_riego_por_temporada.csv")

if __name__ == "__main__":
    calcular_aigua()