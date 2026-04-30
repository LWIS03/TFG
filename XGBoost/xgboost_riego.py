import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
import joblib

# ─────────────────────────────────────────────
# 1. CARGA Y PREPROCESADO
# ─────────────────────────────────────────────
print("=" * 55)
print("  XGBOOST PARA PREDICCION DE RIEGO - TFG")
print("=" * 55)

df = pd.read_csv('../DATASET_IA/dataset_entrenamiento_regresion.csv',
                 sep=';', decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)

# Features de entrada (quitamos Fecha y el target)
FEATURES = [
    'Dias_Plantacion', 'GDU_Acumulado',
    'Temp_Max_C', 'Temp_Min_C', 'Temp_Suelo_C',
    'Humedad_Relativa_pct', 'Velocidad_Viento_ms',
    'Precipitacion_Hoy_mm', 'ETo_mm',
    'Precip_Manana_mm', 'ETo_Manana_mm', 'Precio_Agua_Hoy'
]
TARGET = 'Riego_Recomendado_mm'

X = df[FEATURES]
y_reg = df[TARGET]                          # Regresión: mm de riego
y_cls = (df[TARGET] > 0).astype(int)        # Clasificación: 0=No regar, 1=Regar

print(f"\n📊 Dataset: {len(df)} filas | {len(FEATURES)} features")
print(f"   Días con riego:    {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
print(f"   Días sin riego:    {(y_cls==0).sum()} ({(y_cls==0).mean()*100:.1f}%)")
print(f"   Riego medio (cuando riega): {y_reg[y_reg>0].mean():.2f} mm")

# Split temporal (80/20) — respetamos el orden cronológico
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_cls_train, y_cls_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

print(f"\n📅 Split temporal (80/20):")
print(f"   Train: {df['Fecha'].iloc[0].date()} → {df['Fecha'].iloc[split_idx-1].date()} ({split_idx} días)")
print(f"   Test:  {df['Fecha'].iloc[split_idx].date()} → {df['Fecha'].iloc[-1].date()} ({len(df)-split_idx} días)")

# ─────────────────────────────────────────────
# 2. MODELO 1 — CLASIFICADOR (¿Regar o no?)
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  MODELO 1: CLASIFICADOR (¿Regar hoy?)")
print("─" * 55)

scale_pos = (y_cls_train == 0).sum() / (y_cls_train == 1).sum()

clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,   # Compensa el desbalance
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_cls_train,
        eval_set=[(X_test, y_cls_test)],
        verbose=False)

y_cls_pred = clf.predict(X_test)
print("\n📋 Classification Report:")
print(classification_report(y_cls_test, y_cls_pred,
                            target_names=['No regar', 'Regar']))

# ─────────────────────────────────────────────
# 3. MODELO 2 — REGRESOR (¿Cuántos mm?)
# ─────────────────────────────────────────────
print("─" * 55)
print("  MODELO 2: REGRESOR (¿Cuántos mm de riego?)")
print("─" * 55)

# Entrenamos solo con los días que SÍ se riega
mask_train = y_reg_train > 0
mask_test  = y_reg_test > 0

reg = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)
reg.fit(X_train[mask_train], y_reg_train[mask_train],
        eval_set=[(X_test[mask_test], y_reg_test[mask_test])],
        verbose=False)

y_reg_pred_raw = reg.predict(X_test[mask_test])
y_reg_pred_raw = np.clip(y_reg_pred_raw, 0, None)

mae  = mean_absolute_error(y_reg_test[mask_test], y_reg_pred_raw)
rmse = np.sqrt(mean_squared_error(y_reg_test[mask_test], y_reg_pred_raw))
r2   = r2_score(y_reg_test[mask_test], y_reg_pred_raw)

print(f"\n📋 Métricas del regresor (solo días de riego):")
print(f"   MAE  = {mae:.3f} mm")
print(f"   RMSE = {rmse:.3f} mm")
print(f"   R²   = {r2:.4f}")

# ─────────────────────────────────────────────
# 4. PIPELINE COMBINADO — evaluación final
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  PIPELINE COMBINADO (Clasificador → Regresor)")
print("─" * 55)

y_final_pred = np.zeros(len(X_test))
regar_mask = y_cls_pred == 1
if regar_mask.sum() > 0:
    y_final_pred[regar_mask] = np.clip(
        reg.predict(X_test[regar_mask]), 0, None)

mae_final  = mean_absolute_error(y_reg_test, y_final_pred)
rmse_final = np.sqrt(mean_squared_error(y_reg_test, y_final_pred))
r2_final   = r2_score(y_reg_test, y_final_pred)

print(f"\n📋 Métricas pipeline completo (todos los días):")
print(f"   MAE  = {mae_final:.3f} mm")
print(f"   RMSE = {rmse_final:.3f} mm")
print(f"   R²   = {r2_final:.4f}")
print(f"\n   Días donde el pipeline predice riego: {regar_mask.sum()}")
print(f"   Días donde realmente se riega:        {mask_test.sum()}")

# ─────────────────────────────────────────────
# 5. VISUALIZACIONES
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

DARK_BG   = "#ffffff"
CARD_BG   = "#d2d2d2"
ACCENT    = "#da0404"
ACCENT2   = '#ff6b6b'
ACCENT3   = "#000000"
TEXT      = "#000000"
GRID      = '#2a2d3e'

PREDICT   = "#0026ff"

plt.rcParams.update({
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
})

# ── A) Feature importance CLASIFICADOR ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD_BG)
fi_cls = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
colors = [ACCENT if v >= fi_cls.quantile(0.7) else '#4a5568' for v in fi_cls]
bars = ax1.barh(fi_cls.index, fi_cls.values, color=colors, height=0.7)
ax1.set_title('Feature Importance — Clasificador\n(¿Regar hoy?)', color=TEXT, fontsize=11, pad=10)
ax1.set_xlabel('Importancia', color=TEXT)
ax1.tick_params(colors=TEXT, labelsize=8)
ax1.spines[:].set_color(GRID)
ax1.grid(axis='x', color=GRID, alpha=0.5)

# ── B) Feature importance REGRESOR ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_BG)
fi_reg = pd.Series(reg.feature_importances_, index=FEATURES).sort_values()
colors2 = [ACCENT if v >= fi_reg.quantile(0.7) else '#4a5568' for v in fi_reg]
ax2.barh(fi_reg.index, fi_reg.values, color=colors2, height=0.7)
ax2.set_title('Feature Importance — Regresor\n(¿Cuántos mm?)', color=TEXT, fontsize=11, pad=10)
ax2.set_xlabel('Importancia', color=TEXT)
ax2.tick_params(colors=TEXT, labelsize=8)
ax2.spines[:].set_color(GRID)
ax2.grid(axis='x', color=GRID, alpha=0.5)

# ── C) Confusion Matrix ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD_BG)
cm = confusion_matrix(y_cls_test, y_cls_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=ax3,
            cmap=sns.light_palette(ACCENT, as_cmap=True),
            xticklabels=['No regar', 'Regar'],
            yticklabels=['No regar', 'Regar'],
            cbar=False, linewidths=1, linecolor=DARK_BG,
            annot_kws={'size': 13, 'color': TEXT, 'weight': 'bold'})
ax3.set_title('Matriu de confusió\nClassificador', color=TEXT, fontsize=11, pad=10)
ax3.set_xlabel('Predit', color=TEXT)
ax3.set_ylabel('Real', color=TEXT)
ax3.tick_params(colors=TEXT)

# ── D) Scatter Real vs Predicho (regresor) ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)
real_vals = y_reg_test[mask_test].values
pred_vals = y_reg_pred_raw
ax4.scatter(real_vals, pred_vals, alpha=0.4, color=ACCENT3, s=18, edgecolors='none')
max_val = max(real_vals.max(), pred_vals.max())
ax4.plot([0, max_val], [0, max_val], '--', color=ACCENT, lw=1.5, label='Predicción perfecta')
ax4.set_title(f'Real vs Predicho — Regresor\nR² = {r2:.4f}', color=TEXT, fontsize=11, pad=10)
ax4.set_xlabel('Riego Real (mm)', color=TEXT)
ax4.set_ylabel('Riego Predicho (mm)', color=TEXT)
ax4.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
ax4.tick_params(colors=TEXT)
ax4.spines[:].set_color(GRID)
ax4.grid(color=GRID, alpha=0.4)

# ── E) Serie temporal — Solo temporadas de cultivo ──
ax5 = fig.add_subplot(gs[2, :])
ax5.set_facecolor(CARD_BG)

test_dates       = df['Fecha'].iloc[split_idx:].reset_index(drop=True)
dias_plantacion  = df['Dias_Plantacion'].iloc[split_idx:].reset_index(drop=True)
y_reg_test_reset = y_reg_test.reset_index(drop=True)
y_final_series   = pd.Series(y_final_pred)

# Filtrar solo días con cultivo activo
mask_cultivo = dias_plantacion > 0
idx_cultivo  = mask_cultivo[mask_cultivo].index

real_show  = y_reg_test_reset.iloc[idx_cultivo].values
pred_show  = y_final_series.iloc[idx_cultivo].values
dates_show = test_dates.iloc[idx_cultivo].values
n_show     = len(real_show)

ax5.fill_between(range(n_show), real_show, alpha=0.35, color=ACCENT, label='Real')
ax5.fill_between(range(n_show), pred_show, alpha=0.35, color=PREDICT, label='Predicho')
ax5.plot(range(n_show), real_show, color=ACCENT, lw=1.2)
ax5.plot(range(n_show), pred_show, color=PREDICT, lw=1.2, linestyle='--')

# Etiqueta cada ~30 días de cultivo
step = 30
ax5.set_xticks(range(0, n_show, step))
ax5.set_xticklabels([pd.Timestamp(dates_show[i]).strftime('%d %b %Y')
                     for i in range(0, n_show, step)], rotation=30, fontsize=8)
ax5.set_title('Serie temporal — Solo temporadas de cultivo\n(Pipeline completo)', color=TEXT, fontsize=11, pad=10)
ax5.set_ylabel('Riego (mm)', color=TEXT)
ax5.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=9)
ax5.tick_params(colors=TEXT)
ax5.spines[:].set_color(GRID)
ax5.grid(color=GRID, alpha=0.3)

fig.suptitle('XGBoost — Modelo Predictivo de Riego  |  TFG',
             color=TEXT, fontsize=14, fontweight='bold', y=0.98)

plt.savefig('./outputs/xgboost_riego_resultados.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("\n✅ Gráfica guardada.")

# ─────────────────────────────────────────────
# 6. GUARDAR MODELOS
# ─────────────────────────────────────────────
joblib.dump(clf, './outputs/modelo_clasificador_riego.pkl')
joblib.dump(reg, './outputs/modelo_regresor_riego.pkl')
print("✅ Modelos guardados (.pkl)")
print("\n¡Entrenamiento completado! 🌱")
