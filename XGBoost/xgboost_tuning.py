import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import joblib

print("Cargando datos...")
df = pd.read_csv('../DATASET_IA/dataset_entrenamiento_regresion.csv', sep=';', decimal=',')
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)

FEATURES = ['Dias_Plantacion','GDU_Acumulado','Temp_Max_C','Temp_Min_C','Temp_Suelo_C',
            'Humedad_Relativa_pct','Velocidad_Viento_ms','Precipitacion_Hoy_mm','ETo_mm',
            'Precip_Manana_mm','ETo_Manana_mm','Precio_Agua_Hoy']
X = df[FEATURES]
y_cls = (df['Riego_Recomendado_mm'] > 0).astype(int)
y_reg = df['Riego_Recomendado_mm']
split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_cls_train, y_cls_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
mask_train = y_reg_train > 0
mask_test  = y_reg_test > 0
scale_pos = (y_cls_train==0).sum()/(y_cls_train==1).sum()
tscv = TimeSeriesSplit(n_splits=3)

# ── TUNING CLASIFICADOR ──
param_combos_clf = [
    {'n_estimators':200,'max_depth':4,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,'gamma':0.1,'reg_alpha':0.1,'reg_lambda':2.0},
    {'n_estimators':300,'max_depth':5,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0,'reg_alpha':0,'reg_lambda':1.0},
    {'n_estimators':200,'max_depth':4,'learning_rate':0.1,'subsample':0.9,'colsample_bytree':0.9,'min_child_weight':3,'gamma':0,'reg_alpha':0.1,'reg_lambda':1.0},
    {'n_estimators':300,'max_depth':6,'learning_rate':0.05,'subsample':0.7,'colsample_bytree':0.8,'min_child_weight':5,'gamma':0.3,'reg_alpha':0,'reg_lambda':2.0},
    {'n_estimators':200,'max_depth':3,'learning_rate':0.1,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':1,'gamma':0,'reg_alpha':0,'reg_lambda':1.0},
    {'n_estimators':500,'max_depth':4,'learning_rate':0.01,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,'gamma':0.1,'reg_alpha':0.1,'reg_lambda':2.0},
]

print("Tuning clasificador (6 combos x 3 folds)...")
best_score_clf = -1
best_params_clf = None
for i, params in enumerate(param_combos_clf):
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_cls_train.iloc[train_idx], y_cls_train.iloc[val_idx]
        m = XGBClassifier(**params, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr, verbose=False)
        scores.append(f1_score(yval, m.predict(Xval)))
    s = np.mean(scores)
    print(f"  Combo {i+1}: F1={s:.4f} | depth={params['max_depth']} lr={params['learning_rate']} n={params['n_estimators']}")
    if s > best_score_clf:
        best_score_clf = s
        best_params_clf = params

print(f"\nMejor F1 CV: {best_score_clf:.4f}")
best_clf = XGBClassifier(**best_params_clf, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42, n_jobs=-1)
best_clf.fit(X_train, y_cls_train, verbose=False)
y_cls_pred = best_clf.predict(X_test)
f1_final = f1_score(y_cls_test, y_cls_pred)
print(f"F1 test: {f1_final:.4f}")

# ── TUNING REGRESOR ──
param_combos_reg = [
    {'n_estimators':200,'max_depth':4,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,'gamma':0.1,'reg_alpha':0.1,'reg_lambda':2.0},
    {'n_estimators':300,'max_depth':5,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0,'reg_alpha':0,'reg_lambda':1.0},
    {'n_estimators':200,'max_depth':6,'learning_rate':0.05,'subsample':0.7,'colsample_bytree':0.9,'min_child_weight':5,'gamma':0.3,'reg_alpha':0,'reg_lambda':2.0},
    {'n_estimators':300,'max_depth':4,'learning_rate':0.1,'subsample':0.9,'colsample_bytree':0.8,'min_child_weight':3,'gamma':0,'reg_alpha':0.1,'reg_lambda':1.0},
    {'n_estimators':500,'max_depth':5,'learning_rate':0.01,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0.1,'reg_alpha':0.1,'reg_lambda':2.0},
    {'n_estimators':200,'max_depth':3,'learning_rate':0.1,'subsample':0.9,'colsample_bytree':0.7,'min_child_weight':3,'gamma':0,'reg_alpha':0,'reg_lambda':1.0},
]

print("\nTuning regresor (6 combos x 3 folds)...")
best_score_reg = 9999
best_params_reg = None
Xtr_reg = X_train[mask_train]
ytr_reg = y_reg_train[mask_train]
for i, params in enumerate(param_combos_reg):
    scores = []
    for train_idx, val_idx in tscv.split(Xtr_reg):
        Xtr2, Xval2 = Xtr_reg.iloc[train_idx], Xtr_reg.iloc[val_idx]
        ytr2, yval2 = ytr_reg.iloc[train_idx], ytr_reg.iloc[val_idx]
        m = XGBRegressor(**params, eval_metric='rmse', random_state=42, n_jobs=-1)
        m.fit(Xtr2, ytr2, verbose=False)
        scores.append(mean_absolute_error(yval2, np.clip(m.predict(Xval2),0,None)))
    s = np.mean(scores)
    print(f"  Combo {i+1}: MAE={s:.4f}mm | depth={params['max_depth']} lr={params['learning_rate']} n={params['n_estimators']}")
    if s < best_score_reg:
        best_score_reg = s
        best_params_reg = params

print(f"\nMejor MAE CV: {best_score_reg:.4f} mm")
best_reg = XGBRegressor(**best_params_reg, eval_metric='rmse', random_state=42, n_jobs=-1)
best_reg.fit(Xtr_reg, ytr_reg, verbose=False)
y_reg_pred_raw = np.clip(best_reg.predict(X_test[mask_test]), 0, None)
mae  = mean_absolute_error(y_reg_test[mask_test], y_reg_pred_raw)
rmse = np.sqrt(mean_squared_error(y_reg_test[mask_test], y_reg_pred_raw))
r2   = r2_score(y_reg_test[mask_test], y_reg_pred_raw)
print(f"MAE test: {mae:.3f} mm | RMSE: {rmse:.3f} mm | R2: {r2:.4f}")

# ── PIPELINE + COMPARATIVA ──
regar_mask = y_cls_pred == 1
y_final = np.zeros(len(X_test))
if regar_mask.sum() > 0:
    y_final[regar_mask] = np.clip(best_reg.predict(X_test[regar_mask]), 0, None)

clf_old = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
reg_old = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, n_jobs=-1)
clf_old.fit(X_train, y_cls_train, verbose=False)
reg_old.fit(Xtr_reg, ytr_reg, verbose=False)
y_cls_old = clf_old.predict(X_test)
y_old = np.zeros(len(X_test))
mk = y_cls_old==1
if mk.sum()>0:
    y_old[mk] = np.clip(reg_old.predict(X_test[mk]),0,None)

mae_old   = mean_absolute_error(y_reg_test, y_old)
rmse_old  = np.sqrt(mean_squared_error(y_reg_test, y_old))
f1_old    = f1_score(y_cls_test, y_cls_old)
mae_final = mean_absolute_error(y_reg_test, y_final)
rmse_final= np.sqrt(mean_squared_error(y_reg_test, y_final))

print("\n" + "="*55)
print("  COMPARATIVA FINAL")
print("="*55)
print(f"  {'Métrica':<15} {'Antes':>10} {'Después':>10} {'Mejora':>10}")
print(f"  {'MAE (mm)':<15} {mae_old:>10.3f} {mae_final:>10.3f} {mae_old-mae_final:>+10.3f}")
print(f"  {'RMSE (mm)':<15} {rmse_old:>10.3f} {rmse_final:>10.3f} {rmse_old-rmse_final:>+10.3f}")
print(f"  {'F1 Regar':<15} {f1_old:>10.4f} {f1_final:>10.4f} {f1_final-f1_old:>+10.4f}")

# ── GUARDAR ──
joblib.dump(best_clf, './outputs/modelo_clasificador_tuned.pkl')
joblib.dump(best_reg, './outputs/modelo_regresor_tuned.pkl')
with open('./outputs/mejores_parametros.txt', 'w') as f:
    f.write("=== MEJORES PARAMETROS XGBOOST - TFG RIEGO ===\n\n")
    f.write("CLASIFICADOR:\n")
    for k,v in best_params_clf.items(): f.write(f"  {k}: {v}\n")
    f.write(f"  F1 CV: {best_score_clf:.4f}\n  F1 test: {f1_final:.4f}\n\n")
    f.write("REGRESOR:\n")
    for k,v in best_params_reg.items(): f.write(f"  {k}: {v}\n")
    f.write(f"  MAE CV: {best_score_reg:.4f} mm\n  MAE test: {mae:.3f} mm\n  R2 test: {r2:.4f}\n")

# ── GRAFICAS ──
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

DARK_BG = '#ffffff'; CARD_BG = '#d2d2d2'
ACCENT  = '#da0404'; ACCENT2 = '#ff6b6b'; ACCENT3 = '#000000'; TEXT = '#000000'; GRID = '#2a2d3e'
PREDICT = '#0026ff'
plt.rcParams.update({'text.color':TEXT,'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT})

fig = plt.figure(figsize=(18,12))
fig.patch.set_facecolor(DARK_BG)
gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor(CARD_BG)
metricas=['MAE (mm)','RMSE (mm)','F1 Regar']
antes_v=[mae_old,rmse_old,f1_old]; despues_v=[mae_final,rmse_final,f1_final]
x=np.arange(3); w=0.35
ax1.bar(x-w/2,antes_v,w,label='Antes',color='#4a5568')
ax1.bar(x+w/2,despues_v,w,label='Después',color=ACCENT)
ax1.set_xticks(x); ax1.set_xticklabels(metricas,fontsize=9)
ax1.set_title('Antes vs Después del Tuning',color=TEXT,fontsize=11,pad=10)
ax1.legend(facecolor=CARD_BG,labelcolor=TEXT,fontsize=9)
ax1.spines[:].set_color(GRID); ax1.grid(axis='y',color=GRID,alpha=0.5)

ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor(CARD_BG)
fi=pd.Series(best_clf.feature_importances_,index=FEATURES).sort_values()
colors=[ACCENT if v>=fi.quantile(0.7) else '#4a5568' for v in fi]
ax2.barh(fi.index,fi.values,color=colors,height=0.7)
ax2.set_title('Feature Importance\nClasificador (tuned)',color=TEXT,fontsize=10,pad=10)
ax2.tick_params(labelsize=8); ax2.spines[:].set_color(GRID); ax2.grid(axis='x',color=GRID,alpha=0.5)

ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor(CARD_BG)
fi2=pd.Series(best_reg.feature_importances_,index=FEATURES).sort_values()
colors2=[ACCENT if v>=fi2.quantile(0.7) else '#4a5568' for v in fi2]
ax3.barh(fi2.index,fi2.values,color=colors2,height=0.7)
ax3.set_title('Feature Importance\nRegresor (tuned)',color=TEXT,fontsize=10,pad=10)
ax3.tick_params(labelsize=8); ax3.spines[:].set_color(GRID); ax3.grid(axis='x',color=GRID,alpha=0.5)

ax4 = fig.add_subplot(gs[1,0]); ax4.set_facecolor(CARD_BG)
cm = confusion_matrix(y_cls_test, y_cls_pred)
sns.heatmap(cm,annot=True,fmt='d',ax=ax4,cmap=sns.light_palette(ACCENT,as_cmap=True),
            xticklabels=['No regar','Regar'],yticklabels=['No regar','Regar'],
            cbar=False,linewidths=1,linecolor=DARK_BG,annot_kws={'size':13,'color':TEXT,'weight':'bold'})
ax4.set_title('Matriz de Confusión\nClasificador (tuned)',color=TEXT,fontsize=10,pad=10)
ax4.set_xlabel('Predicho'); ax4.set_ylabel('Real')

ax5 = fig.add_subplot(gs[1,1]); ax5.set_facecolor(CARD_BG)
rv=y_reg_test[mask_test].values
ax5.scatter(rv,y_reg_pred_raw,alpha=0.4,color=ACCENT3,s=18,edgecolors='none')
mv=max(rv.max(),y_reg_pred_raw.max())
ax5.plot([0,mv],[0,mv],'--',color=ACCENT,lw=1.5,label='Predicción perfecta')
ax5.set_title(f'Real vs Predicho (tuned)\nR² = {r2:.4f}',color=TEXT,fontsize=10,pad=10)
ax5.set_xlabel('Riego Real (mm)'); ax5.set_ylabel('Riego Predicho (mm)')
ax5.legend(facecolor=CARD_BG,labelcolor=TEXT,fontsize=8)
ax5.spines[:].set_color(GRID); ax5.grid(color=GRID,alpha=0.4)

# ── F) Serie temporal — Solo temporadas de cultivo ──
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(CARD_BG)

test_dates       = df['Fecha'].iloc[split_idx:].reset_index(drop=True)
dias_plantacion  = df['Dias_Plantacion'].iloc[split_idx:].reset_index(drop=True)
y_reg_test_reset = y_reg_test.reset_index(drop=True)
y_final_series = pd.Series(y_final)

mask_cultivo = dias_plantacion > 0
idx_cultivo  = mask_cultivo[mask_cultivo].index

real_show  = y_reg_test_reset.iloc[idx_cultivo].values
pred_show  = y_final_series.iloc[idx_cultivo].values
dates_show = test_dates.iloc[idx_cultivo].values
n_show     = len(real_show)

ax6.fill_between(range(n_show), real_show, alpha=0.35, color=ACCENT, label='Real')
ax6.fill_between(range(n_show), pred_show, alpha=0.35, color=PREDICT, label='Predicho')
ax6.plot(range(n_show), real_show, color=ACCENT, lw=1.2)
ax6.plot(range(n_show), pred_show, color=PREDICT, lw=1.2, linestyle='--')

step = 30
ax6.set_xticks(range(0, n_show, step))
ax6.set_xticklabels([pd.Timestamp(dates_show[i]).strftime('%d %b %Y')
                     for i in range(0, n_show, step)], rotation=30, fontsize=7)
ax6.set_title('Serie temporal — Solo temporadas de cultivo\n(Pipeline tuned)', color=TEXT, fontsize=10, pad=10)
ax6.set_ylabel('Riego (mm)')
ax6.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
ax6.spines[:].set_color(GRID); ax6.grid(color=GRID, alpha=0.3)

fig.suptitle('Hyperparameter Tuning — XGBoost Riego  |  TFG',color=TEXT,fontsize=14,fontweight='bold',y=0.98)
plt.savefig('./outputs/xgboost_tuning_resultados.png',dpi=150,bbox_inches='tight',facecolor=DARK_BG)
print("Grafica guardada!")
print("Todo completado!")
