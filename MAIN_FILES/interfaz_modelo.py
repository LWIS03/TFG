import streamlit as st
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from api_request import obtener_datos_meteorologicos, get_past_months_data

lat = 42.4411
lon = -5.7912

# ---------------------------------------------------------
# 1. FUNCIONS AUXILIARS FÍSIQUES
# ---------------------------------------------------------
def calcular_altura_por_gdu(gdu_acumulado):
    """Calcula l'alçada estimada del blat de moro en funció dels graus-dia."""
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

# ---------------------------------------------------------
# 2. ENTORN MOCK (FALS) PER CARREGAR LA NORMALITZACIÓ
# ---------------------------------------------------------
# Streamlit necessita aquest entorn "cascaró" per poder aplicar el VecNormalize
class MockEnv(gym.Env):
    def __init__(self):
        super(MockEnv, self).__init__()
        # Mateix espai que a entrenar_ia.py (13 variables)
        self.observation_space = spaces.Box(low=-50.0, high=5000.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        return np.zeros(13, dtype=np.float32), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(13, dtype=np.float32), {}

# ---------------------------------------------------------
# 3. INTERFÍCIE DE STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="IA de Reg - Blat de Moro", page_icon="🌽", layout="wide")

st.title("Assistent de Reg Agrícola amb IA")
st.markdown("Introdueix les condicions actuals de la parcel·la i la previsió meteorològica per obtenir la recomanació de reg d'avui.")

# Càrrega del model (A la memòria cau perquè no es recarregui a cada clic)
@st.cache_resource
def cargar_modelo():
    try:
        # 1. Creem l'entorn fals
        env = DummyVecEnv([lambda: MockEnv()])
        # 2. Carreguem les estadístiques de normalització de l'entrenament
        env = VecNormalize.load("vecnormalize_stats.pkl", env)
        env.training = False # IMPORTANT: Evita que la IA continuï aprenent/modificant els pesos
        env.norm_reward = False
        
        # 3. Carreguem el cervell (Segons els logs, vas utilitzar SAC)
        modelo = SAC.load("ia_riego_maiz_ppo") 
        return modelo, env
    except Exception as e:
        st.error(f"Error en carregar el model o les estadístiques: {e}")
        return None, None
        
modelo, env_normalizado = cargar_modelo()

if modelo is not None:
    # --- FORMULARI D'ENTRADA ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Dades de la Parcel·la")
        d = st.date_input("Selecciona la data de plantació", datetime.date.today())
        
        # Cridem a la funció millorada
        dr_calculado, gdu_calculado = get_past_months_data(d)
        
    with col2:
        st.subheader("Estat del Cultiu")
        # Emplenem els valors amb els càlculs del bessó digital
        gdu_calculado = st.number_input("GDU Acumulat (Graus-Dia)", min_value=0.0, max_value=2000.0, value=gdu_calculado, step=10.0)
        
        dr_actual = st.number_input("Esgotament del sòl (Dr en mm)", min_value=0.0, max_value=200.0, value=dr_calculado, 
                                      help="0 = Capacitat de Camp. Auto-calculat des de la sembra.")
        precio_agua = st.number_input("Preu de l'Aigua Avui (€/m³)", min_value=0.0, max_value=1.0, value=0.04, step=0.01)
        
    clima_hoy, clima_manana = obtener_datos_meteorologicos()

    with col3:
        st.subheader("Clima d'Avui")
        t_max = st.number_input("Temperatura Màx. (ºC)", value=clima_hoy["t_max"])
        t_min = st.number_input("Temperatura Mín. (ºC)", value=clima_hoy["t_min"])
        t_suelo = st.number_input("Temp. del Sòl (ºC)", value=clima_hoy["t_suelo"])
        humedad = st.number_input("Humitat Relativa (%)", value=clima_hoy["humedad"])
        viento = st.number_input("Velocitat del Vent (m/s)", value=clima_hoy["viento"])
        precip_hoy = st.number_input("Precipitació Avui (mm)", min_value=0.0, max_value=200.0, value=clima_hoy["precip_hoy"])
        eto_hoy = st.number_input("ETo Avui (mm)", min_value=0.0, max_value=15.0, value=clima_hoy["eto_hoy"])

    with col4:
        st.subheader("Previsió per Demà")
        precip_manana = st.number_input("Precipitació Demà (mm)", min_value=0.0, max_value=200.0, value=clima_manana["precip_manana"])
        eto_manana = st.number_input("ETo Demà (mm)", min_value=0.0, max_value=15.0, value=clima_manana["eto_manana"])
        
        st.write("---")
        # Calculem l'alçada en base al GDU introduït
        altura_estimada = calcular_altura_por_gdu(gdu_calculado)
        st.info(f"Alçada de la planta estimada: **{altura_estimada:.2f} m**")

    # --- BOTÓ DE PREDICCIÓ ---
    if st.button("Calcular Reg Òptim", use_container_width=True, type="primary"):
        # 1. Construïm el vector d'estat EXACTAMENT en l'ordre de l'entrenament
        estado_crudo = np.array([
            gdu_calculado,
            t_max,
            t_min,
            t_suelo,
            humedad,
            viento,
            precip_hoy,
            eto_hoy,
            precip_manana,
            eto_manana,
            precio_agua,
            altura_estimada,
            dr_actual
        ], dtype=np.float32)

        # 2. Normalitzem l'observació fent servir el nostre entorn embolicat
        obs_normalizada = env_normalizado.normalize_obs(estado_crudo)

        # 3. La IA decideix
        accion, _states = modelo.predict(obs_normalizada, deterministic=True)
        riego_recomendado = accion[0]

        # --- RESULTAT VISUAL ---
        st.write("---")
        st.header("Decisió de la IA")
        
        if riego_recomendado < 0.5:
            st.success(f"La IA recomana **NO REGAR** avui ({riego_recomendado:.2f} mm).")
            st.caption("El sòl té prou humitat o s'esperen pluges imminents que cobriran la demanda hídrica.")
        else:
            st.warning(f"La IA recomana aplicar una làmina de **{riego_recomendado:.2f} mm** de reg.")
            volumen_ha = riego_recomendado * 10
            coste_estimado = volumen_ha * precio_agua
            st.caption(f"Això equival a {volumen_ha:.0f} m³ per hectàrea. Cost estimat: {coste_estimado:.2f} €/ha.")