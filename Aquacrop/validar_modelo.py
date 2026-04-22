import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
def main():
    print("Iniciando validación: Gemelo Digital vs AquaCrop-OS...")

    # 1. Cargar los datos
    # Asumimos que has exportado los resultados de simulacion_riego.py a un CSV
    df_gemelo = pd.read_csv("../DATASET_IA/dataset_entrenamiento_sin_escalar.csv", sep=";", decimal=",")
    df_aquacrop = pd.read_csv("auditoria_calculos_fao.csv", sep=";", decimal=",")

    # 2. Preparar los datos
    # El Ks del gemelo va de 0 a 1 (1 es sin estrés).
    # AquaCrop exporta 'Estrés_Hídrico_pct' (0 es sin estrés, 100 es máximo).
    # Vamos a convertir el Ks de tu gemelo al mismo formato que AquaCrop para poder compararlos:
    df_gemelo['Estres_Gemelo_pct'] = (1.0 - df_gemelo['Ks_Gemelo']) * 100

    # Unimos ambos datasets usando la Fecha
    df_gemelo['Fecha'] = pd.to_datetime(df_gemelo['Fecha'], format='%d/%m/%Y')
    df_aquacrop['Fecha'] = pd.to_datetime(df_aquacrop['Fecha'], format='%d/%m/%Y')
    
    df_comparativa = pd.merge(df_gemelo, df_aquacrop, on='Fecha', how='inner')

    # 3. Métricas de Error
    rmse = np.sqrt(mean_squared_error(df_comparativa['Estrés_Hídrico_pct'], df_comparativa['Estres_Gemelo_pct']))
    print(f"\n📊 RMSE (Error Cuadrático Medio) del Estrés: {rmse:.2f}%")
    if rmse < 15.0:
        print("✅ Tu Gemelo Digital se comporta de forma muy similar a AquaCrop.")
    else:
        print("⚠️ Hay diferencias notables. Revisa los parámetros 'TAW' y 'RAW' en simulacion_riego.py.")

    # 4. Gráfico Comparativo de la Curva de Estrés
    plt.figure(figsize=(12, 6))
    plt.plot(df_comparativa['Fecha'], df_comparativa['Estrés_Hídrico_pct'], label='Estrés AquaCrop-OS', color='blue', alpha=0.7)
    plt.plot(df_comparativa['Fecha'], df_comparativa['Estres_Gemelo_pct'], label='Estrés Gemelo Digital', color='orange', linestyle='dashed')
    
    plt.title('Comparativa de Estrés Hídrico: Gemelo vs AquaCrop')
    plt.ylabel('Nivel de Estrés (%)')
    plt.xlabel('Fecha')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()