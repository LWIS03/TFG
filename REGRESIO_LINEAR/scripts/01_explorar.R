# ══════════════════════════════════════════════════════════
# REGRESSIÓ LINEAL PER A RECOMANACIONS DE REGADIU
# ══════════════════════════════════════════════════════════
#' @title Script d'exploració i modelatge de regressió lineal
#'
#' @description
#' Script d'anàlisi exploratòria per a models de regressió lineal
#' que prediuen les recomanacions de regadiu basades en dades meteorològiques.
#' Inclou múltiples aproachaments: regressió lineal simple, regressió
#' ponderada, classificador logistic i pipeline hurdle.
#'
#' @details
#' El script realitza les següents etapes:
#' \enumerate{
#'   \item Carrega de dades i exploració inicial
#'   \item Divisió train/test per any (2005-2020 train, 2021-2024 test)
#'   \item Ajust de múltiples models de regressió lineal
#'   \item Ajust de model logistic per predir quan regar
#'   \item Pipeline hurdle (classificador + regressor)
#'   \item Generació de gràfics de diagnòstic
#' }
#'
#' @author
#' Lluid LLUIS \email{lluis@@example.com}
#'
#' @references
#' - FAO-56 Penman-Monteith per a ETo
#' - Script original per a recomanacions d'expert
#'
#' @note
#' Resultats principals del model final (model_final):
#' - Adjusted R-squared: 0.8986 (sobre dades d'entrenament)
#' - R² test: 0.8702 (sobre dades de test 2021-2024)
#' - RMSE test: 3.104 mm
#' - MAE test: 2.273 mm
#'
#' Variables seleccionades per al model_final:
#' - GDU_Acumulat (coeficient: 0.0314, p < 0.001)
#' - Temp_Max_C (coeficient: 0.205, p < 0.001)
#' - Temp_Suelo_C (coeficient: -0.539, p < 0.001)
#' - Humedad_Relativa_pct (coeficient: 0.060, p < 0.001)
#' - ETo_mm (coeficient: -0.559, p = 0.002)
#'
#' @examples
#' \dontrun{
#' # Execució completa del script
#' source("01_explorar.R")
#' }
#'
#' @rawNamespace
library(tidyverse)
library(car)
# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
#' @section 1. Carrega de Dades
#'
#' Carrega el dataset de recomanacions de regadiu des d'un fitxer CSV.
#' El fitxer utilitza ";" com a delimitador i "," com a decimal.
#'
#' @return
#' Objecte tibble amb 7.760 files i 14 columnes:
#' \describe{
#'   \item{Fecha}{Data en format dd/mm/aaaa}
#'   \item{Dias_Plantacio}{Dies des de la plantació (0-215)}
#'   \item{GDU_AcumulatGraus Dia Unit}{GDU acumulats (0-1281.5)}
#'   \item{Temp_Max_C}{Temperatura màxima (-3.85 a 35.98)}
#'   \item{Temp_Min_C}{Temperatura mínima (-18.45 a 18.11)}
#'   \item{Temp_Suelo_C}{Temperatura del sòl (-6.77 a 26.12)}
#'   \item{Humedad_Relativa_pct}{Humitat relativa (0-100)}
#'   \item{Velocidad_Viento_ms}{Velocitat del vent (0.18-7.4 m/s)}
#'   \item{Precipitacion_Hoy_mm}{Precipitació avui (0-33.61 mm)}
#'   \item{ETo_mm}{Evapotranspiració de referencia (0-7.62 mm)}
#'   \item{Riego_Recomendado_mm}{Recomanació expert (0-50.8 mm)}
#'   \item{Precip_Manana_mm}{Precipitació prevista demà (0-36.09 mm)}
#'   \item{ETo_Manana_mm}{ETo previst demà (0-9.9 mm)}
#'   \item{Precio_Agua_Hoy}{Preu de l'aigua (0.02-0.08 €/m3)}
#' }
#'
#' @examples
#' \dontrun{
#' df <- read_delim("data/dataset_entrenamiento_regresion.csv", delim = ";")
#' #> Rows: 7760 Columns: 14
#' #> Fecha: 01/01/2005 - 31/12/2024
#' #> Mitjana Riego_Recomendado_mm: 1.723 mm
#' }
#'
df <- read_delim("/home/lluis/Documents/TFG/REGRESIO_LINEAR/data/dataset_entrenamiento_regresion.csv",
                 delim = ";",
                 locale = locale(decimal_mark = ","))

summary(df)
glimpse(df)

# ══════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (by year, no data leakage)
# ══════════════════════════════════════════════════════════
#' @section 2. Divisió Train/Test
#'
#' Divideix les dades en conjunts d'entrenament i test basant-se en l'any
#' per evitar data leakage. Els anys 2005-2020 s'utilitzen per entrenar
#' i 2021-2024 per test.
#'
#' @details
#' Es creen 4 datasets:
#' \describe{
#'   \item{df_train}{Totes les dades 2005-2020 (5.844 files)}
#'   \item{df_test}{Totes les dades 2021-2024 (1.461 files)}
#'   \item{df_train_plantacio}{Dies amb planta plantada (Dias_Plantacio > 0)}
#'   \item{df_train_reg}{Dies amb recomanació de regadiu (Riego_Recomendado_mm > 0)}
#' }
#'
#' @examples
#' \dontrun{
#' df_train <- df %>% filter(any %in% 2005:2020)
#' df_test  <- df %>% filter(any %in% 2021:2024)
#' #> Train rows: 5844
#' #> Test rows:  1461
#' }
#'
set.seed(42)
years      <- sort(unique(substr(df$Fecha, 7, 10)))
train_years <- years[1:16]   # 2005-2020
test_years  <- years[17:20]  # 2021-2024

df_train <- df %>% filter(substr(Fecha, 7, 10) %in% train_years)
df_test  <- df %>% filter(substr(Fecha, 7, 10) %in% test_years)
# Com el model el farem servir nomes quan estan plantades
df_train_plantacio <- df_train %>% filter(Dias_Plantacion > 0)
df_test_plantiacio  <- df_test  %>% filter(Dias_Plantacion > 0)

df_train_reg <- df_train %>% filter(Riego_Recomendado_mm > 0)
df_test_reg  <- df_test  %>% filter(Riego_Recomendado_mm > 0)

cat("Train rows:", nrow(df_train), "\n")
cat("Test rows: ", nrow(df_test),  "\n")

# ========================================================= #
# PROBEM AMB TOTS ELS DIES                                  #
#========================================================== #
#' @section 3. Regressió Lineal (Tots els Dies)
#'
#' Primer model de regressió lineal utilitzant totes les dades d'entrenament
#' (incloent dies sense regadiu).
#'
#' @details
#' Model: Riego_Recomendat_mm ~ Temp_Max_C + Temp_Min_C + Velocidad_Viento_ms +
#'        Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + ETo_Manana_mm
#'
#' @results
#' Resultats del model:
#' \describe{
#'   \item{Multiple R-squared}{0.1502}
#'   \item{Adjusted R-squared}{0.1492}
#'   \item{F-statistic}{147.3 on 7 DF, p-value < 2e-16}
#' }
#'
#' Variables significatives (p < 0.05):
#' \describe{
#'   \item{Temp_Max_C}{coef: -0.086, p = 0.001}
#'   \item{Temp_Min_C}{coef: 0.081, p = 0.0002}
#'   \item{Velocidad_Viento_ms}{coef: -0.315, p = 2.2e-05}
#'   \item{ETo_mm}{coef: 1.054, p < 2e-16 ***}
#'   \item{Precip_Manana_mm}{coef: -0.118, p = 0.0007}
#' }
#'
#' @note Aquest model no és gaire bo (R² = 0.15), és a dir, només explica
#' el 15% de la variància. La majoria de valors de Riego_Recomendat_mm són 0.
#'
#' @examples
#' \dontrun{
#' model <- lm(Riego_Recomendado_mm ~ ..., data = df_train)
#' summary(model)
#' #> Residual standard error: 4.922 on 5836 degrees of freedom
#' #> Multiple R-squared: 0.1502, Adjusted R-squared: 0.1492
#' }
#'
model <- lm(Riego_Recomendado_mm ~ Temp_Max_C + Temp_Min_C + 
               + Velocidad_Viento_ms + 
               Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + 
               ETo_Manana_mm,
             data = df_train)

summary(model)

# Es un mierdon te un Adjustes R-squared de 0.1492

# ========================================================= #
# PROBEM AMB ELS DIAS QUE NOMES ESTA LA PLANTA              #
#========================================================== #
#' @section 4. Regressió Lineal (Dies amb Planta)
#'
#' Model de regressió lineal però només amb dies on hi ha planta plantada
#' (Dias_Plantacio > 0).
#'
#' @details
#' Model: Riego_Recomendat_mm ~ GDU_Acumulat + Temp_Max_C + Temp_Min_C +
#'        Temp_Suelo_C + Humedad_Relativa_pct + Velocidad_Viento_ms +
#'        Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + ETo_Manana_mm + Precio_Agua
#'
#' @results
#' Resultats del model:
#' \describe{
#'   \item{Multiple R-squared}{0.0695}
#'   \item{Adjusted R-squared}{0.0661}
#'   \item{F-statistic}{20.27 on 11 DF, p-value < 2.2e-16}
#' }
#'
#' Variables significatives (p < 0.05):
#' \describe{
#'   \item{Precipitacion_Hoy_mm}{coef: 0.174, p = 0.026}
#'   \item{ETo_mm}{coef: 0.790, p = 0.008}
#'   \item{Precip_Manana_mm}{coef: -0.246, p = 0.0006}
#' }
#'
#' @note El model pitjora! R² = 0.0661 vs 0.1492 anterior.
#' EnIncluding GDU i altres variables no millora el model.
#'
#' @examples
#' \dontrun{
#' model <- lm(Riego_Recomendado_mm ~ ..., data = df_train_plantacio)
#' summary(model)
#' #> Residual standard error: 6.8 on 2985 degrees of freedom
#' #> Multiple R-squared: 0.0695, Adjusted R-squared: 0.0661
#' }
#'
model <- lm(Riego_Recomendado_mm ~ GDU_Acumulado + Temp_Max_C + Temp_Min_C + 
                     Temp_Suelo_C + Humedad_Relativa_pct + Velocidad_Viento_ms + 
                     Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + 
                     ETo_Manana_mm + Precio_Agua_Hoy,
                   data = df_train_plantacio)

summary(model)


# ES ENCARA PITJOR I SI AGAFEM NOMES LES SIGNIFICATIVES
#' @section 4b. Regressió Simplificada (Dies amb Planta)
#'
#' Model reduït només amb variables significatives del model anterior.
#'
#' @results
#' \describe{
#'   \item{Multiple R-squared}{0.0674}
#'   \item{Adjusted R-squared}{0.0661}
#' }
#'
#' @note No millora gaire. R² = 0.0661

model2 <- lm(Riego_Recomendado_mm ~
              Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + 
              ETo_Manana_mm,
            data = df_train_plantacio)

summary(model2)

# No millora gaire Adjusted R-Square de 0.062

# ===================================================================#
# PROBEM AMB ELS DIAS QUE NOMES ES REGA QUE REALMENT ES EL QUE VOLEM #
#====================================================================#
#' @section 5. Regressió Lineal (Dies de Regadiu)
#'
#' Model de regressió només amb dies on l'expert recomana regar
#' (Riego_Recomendado_mm > 0). Aquest és el model que realment ens interessa.
#'
#' @details
#' Primer model complet:
#' Model: Riego_Recomendat_mm ~ GDU_Acumulat + Temp_Max_C + Temp_Min_C +
#'        Temp_Suelo_C + Humedad_Relativa_pct + Velocidad_Viento_ms +
#'        Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + ETo_Manana_mm + Precio_Agua
#'
#' @results Model complet:
#' \describe{
#'   \item{Multiple R-squared}{0.9001}
#'   \item{Adjusted R-squared}{0.8990}
#'   \item{F-statistic}{789.8 on 11 DF, p-value < 2.2e-16}
#' }
#'
#' Variables significatives (p < 0.001):
#' \describe{
#'   \item{GDU_Acumulado}{coef: 0.0314, p < 2e-16 *** (molt significatiu!)}
#'   \item{Temp_Suelo_C}{coef: -0.479, p < 2e-16 ***}
#'   \item{Temp_Max_C}{coef: 0.227, p = 0.0001}
#'   \item{Humedad_Relativa_pct}{coef: 0.071, p = 2e-05}
#' }
#'
#' @note Increible! Quan filtrem només als dies que realment es rega,
#' el model millora dramatically: R² = 0.90 vs 0.15 anterior.
#' La variable GDU_Acumulado és molt important (coeficient 0.0314).
#'
#' @examples
#' \dontrun{
#' model <- lm(Riego_Recomendado_mm ~ ..., data = df_train_reg)
#' summary(model)
#' #> Multiple R-squared: 0.9001, Adjusted R-squared: 0.899
#' #> GDU_Acumulado: coef 0.0314, p < 2e-16 ***
#' }
#'
model <- lm(Riego_Recomendado_mm ~ GDU_Acumulado + Temp_Max_C + Temp_Min_C + 
              Temp_Suelo_C + Humedad_Relativa_pct + Velocidad_Viento_ms + 
              Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm + 
              ETo_Manana_mm + Precio_Agua_Hoy,
            data = df_train_reg)

summary(model)

# AGAFEM NOMES LES SIGNIGICATIVES 
#' @section 5b. Model Final (Regressió Lineal Escollida)
#'
#' Model final seleccionat amb només les variables significatives.
#' Aquest és el millor model de regressió lineal.
#'
#' @details
#' Model: Riego_Recomendat_mm ~ GDU_Acumulat + Temp_Max_C + Temp_Suelo_C +
#'        Humedad_Relativa_pct + ETo_mm
#'
#' @results
#' \describe{
#'   \item{Multiple R-squared}{0.8991}
#'   \item{Adjusted R-squared}{0.8986}
#'   \item{F-statistic}{1729 on 5 DF, p-value < 2.2e-16}
#'   \item{Residual standard error}{2.734 on 970 degrees of freedom}
#' }
#'
#' Coeficients finals (tots significatius p < 0.01):
#' \describe{
#'   \item{Intercept}{3.077, p = 0.025}
#'   \item{GDU_Acumulado}{0.0314, p < 2e-16 ***}
#'   \item{Temp_Max_C}{0.205, p = 1.7e-07 ***}
#'   \item{Temp_Suelo_C}{-0.539, p < 2e-16 ***}
#'   \item{Humedad_Relativa_pct}{0.060, p = 3.9e-05 ***}
#'   \item{ETo_mm}{-0.559, p = 0.002}
#' }
#'
#' @results_test Resultats sobre test (2021-2024):
#' \describe{
#'   \item{RMSE}{3.104 mm}
#'   \item{MAE}{2.273 mm}
#'   \item{R² test}{0.8702}
#' }
#'
#' @examples
#' \dontrun{
#' model_final <- lm(Riego_Recomendado_mm ~ ..., data = df_train_reg)
#' #> Multiple R-squared: 0.8991, Adjusted R-squared: 0.8986
#' #> RMSE test: 3.104 mm
#' #> R² test: 0.8702
#' }
#'
model_final <- lm(Riego_Recomendado_mm ~ GDU_Acumulado + Temp_Max_C + 
                     Temp_Suelo_C + Humedad_Relativa_pct + ETo_mm,
                   data = df_train_reg)

summary(model_final)
vif(model_final)

# GENEREM ELS PLOTS PER VEURE QUE TAL
png("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/diagnosticos_modelo.png", width=1200, height=1000, res=150)
par(mfrow = c(2, 2))
plot(model_final)
dev.off()

df_test_reg$prediccion <- pmax(0, predict(model_final, newdata = df_test_reg))

rmse    <- sqrt(mean((df_test_reg$Riego_Recomendado_mm - df_test_reg$prediccion)^2))
mae     <- mean(abs(df_test_reg$Riego_Recomendado_mm - df_test_reg$prediccion))
r2_test <- cor(df_test_reg$Riego_Recomendado_mm, df_test_reg$prediccion)^2

cat("RMSE:   ", round(rmse, 3), "mm\n")
cat("MAE:    ", round(mae,  3), "mm\n")
cat("R² test:", round(r2_test, 4), "\n")

ggplot(df_test_reg, aes(x = Riego_Recomendado_mm, y = prediccion)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  labs(title = "Predicted vs Actual — Linear Regression (irrigation days only)",
       x = "FAO expert recommendation (mm)",
       y = "Linear model prediction (mm)") +
  theme_minimal()

ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/predicted_vs_actual.png",
       width = 8, height = 6, dpi = 150)



# Add predictions to the full test set (crop days only)
df_test_reg$prediccion <- pmax(0, predict(model_final, newdata = df_test_reg))

# Convert Fecha to proper date
df_test_reg$Fecha_dt <- as.Date(df_test_reg$Fecha, format = "%d/%m/%Y")

# Plot time series: actual vs predicted
ggplot(df_test_reg, aes(x = Fecha_dt)) +
  geom_line(aes(y = Riego_Recomendado_mm, color = "FAO Expert"), linewidth = 0.8) +
  geom_line(aes(y = prediccion, color = "Linear Regression"), 
            linewidth = 0.8, linetype = "dashed") +
  scale_color_manual(values = c("FAO Expert" = "steelblue", 
                                "Linear Regression" = "red")) +
  labs(title = "Irrigation recommendation — FAO Expert vs Linear Regression (2021-2024)",
       x = "Date", y = "Irrigation (mm)", color = "Strategy") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/timeseries_vs_actual.png",
       width = 12, height = 5, dpi = 150)


# ══════════════════════════════════════════════════════════
# PART 1 — WEIGHTED LINEAR REGRESSION (all days)
# ══════════════════════════════════════════════════════════
#' @section 6. Regressió Lineal Ponderada
#'
#' Aproach de regressió ponderada per tractar el desbalanceig de classes
#' (molts zeros vs pocs valors > 0).
#'
#' @details
#' S'assigna pes 1/n_zero als dies sense regadiu i 1/n_nonzero als dies amb regadiu.
#' Així totes les observacions tenen el mateix pes total.
#'
#' @results
#' \describe{
#'   \item{n_zero}{4.880 dies sense reg}
#'   \item{n_nonzero}{964 dies amb reg}
#'   \item{Multiple R-squared (model 1)}{0.2267}
#'   \item{Multiple R-squared (model reduït)}{0.2264}
#' }
#'
#' Variables significatives (model reduït):
#' \describe{
#'   \item{Temp_Max_C}{coef: 0.139, p = 0.001}
#'   \item{Temp_Min_C}{coef: 0.251, p < 2e-16}
#'   \item{Velocidad_Viento_ms}{coef: -0.349, p = 0.004}
#'   \item{ETo_mm}{coef: 0.624, p < 0.001}
#'   \item{Precip_Manana_mm}{coef: -0.312, p < 0.001}
#'   \item{ETo_Manana_mm}{coef: 0.274, p = 0.007}
#' }
#'
#' @note NOT GOOD ENOUGH - El model ponderat no millora prou (R² = 0.23 vs 0.90 del model_final).
#'
#' @examples
#' \dontrun{
#' model_weighted <- lm(..., weights = weight, data = df_train)
#' summary(model_weighted)
#' #> Multiple R-squared: 0.2264, Adjusted R-squared: 0.2256
#' }
#'
n_zero    <- sum(df_train$Riego_Recomendado_mm == 0)
n_nonzero <- sum(df_train$Riego_Recomendado_mm > 0)

df_train$weight <- ifelse(df_train$Riego_Recomendado_mm == 0,
                          1 / n_zero,
                          1 / n_nonzero)

model_weighted <- lm(Riego_Recomendado_mm ~ Temp_Max_C + Temp_Min_C +
                       Velocidad_Viento_ms + Precipitacion_Hoy_mm +
                       ETo_mm + Precip_Manana_mm + ETo_Manana_mm,
                     data    = df_train,
                     weights = weight)

summary(model_weighted)

# AGAFEM NOMES LES SIGNIGICATIVES
model_weighted <- lm(Riego_Recomendado_mm ~ Temp_Max_C + Temp_Min_C +
                       Velocidad_Viento_ms +
                       ETo_mm + Precip_Manana_mm + ETo_Manana_mm,
                     data    = df_train,
                     weights = weight)

summary(model_weighted)
# NOT GOOD ENOUGH

# ══════════════════════════════════════════════════════════
# PART 2A — LOGISTIC CLASSIFIER (irrigate yes/no)
# ══════════════════════════════════════════════════════════
#' @section 7. Classificador Logistic
#'
#' Model logistic per predir si cal regar (1) o no (0).
#' Útil per al pipeline hurdle (classificador + regressor).
#'
#' @details
#' Model: irrigar ~ GDU_Acumulat + Temp_Max_C + Temp_Min_C + Temp_Suelo_C +
#'        Humedad_Relativa_pct + Velocidad_Viento_ms + Precipitacion_Hoy_mm +
#'        ETo_mm + Precip_Manana_mm + ETo_Manana_mm + Precio_Agua
#'
#' @results
#' \describe{
#'   \item{Null deviance}{5272.6 on 5843 DF}
#'   \item{Residual deviance}{3268.2 on 5832 DF}
#'   \item{AIC}{3292.2}
#' }
#'
#' Coeficients significatius (p < 0.05):
#' \describe{
#'   \item{GDU_Acumulado}{coef: -0.0029, p < 2e-16 ***}
#'   \item{Temp_Max_C}{coef: -0.136, p = 5.4e-08}
#'   \item{Temp_Suelo_C}{coef: 0.089, p = 0.001}
#'   \item{Velocidad_Viento_ms}{coef: -0.235, p = 0.0007}
#'   \item{Precipitacion_Hoy_mm}{coef: 0.124, p = 7.7e-07}
#'   \item{ETo_mm}{coef: 0.992, p < 2e-16}
#'   \item{Precip_Manana_mm}{coef: -0.358, p = 5e-07}
#'   \item{ETo_Manana_mm}{coef: 0.245, p = 3.5e-08}
#' }
#'
#' @results_confusion Matriu de confusió (threshold = 0.5):
#' \describe{
#'   \item{True Negatives (TN)}{1208}
#'   \item{False Positives (FP)}{54}
#'   \item{False Negatives (FN)}{124}
#'   \item{True Positives (TP)}{75}
#' }
#'
#' @examples
#' \dontrun{
#' model_clf <- glm(irrigar ~ ..., family = binomial, data = df_train)
#' prob_irrigar <- predict(model_clf, newdata = df_test, type = "response")
#' #> Confusion matrix:
#' #>          Actual
#' #> Predicted    0    1
#' #>         0 1208  124
#' #>         1   54   75
#' }
#'
df_train$irrigar <- as.factor(ifelse(df_train$Riego_Recomendado_mm > 0, 1, 0))
df_test$irrigar  <- as.factor(ifelse(df_test$Riego_Recomendado_mm  > 0, 1, 0))

model_clf <- glm(irrigar ~ GDU_Acumulado + Temp_Max_C + Temp_Min_C +
                   Temp_Suelo_C + Humedad_Relativa_pct + Velocidad_Viento_ms +
                   Precipitacion_Hoy_mm + ETo_mm + Precip_Manana_mm +
                   ETo_Manana_mm + Precio_Agua_Hoy,
                 data   = df_train,
                 family = binomial)

summary(model_clf)

# Evaluate classifier on test set
prob_irrigar <- predict(model_clf, newdata = df_test, type = "response")
pred_class   <- ifelse(prob_irrigar > 0.5, 1, 0)

# Confusion matrix
table(Predicted = pred_class, Actual = df_test$irrigar)

# ══════════════════════════════════════════════════════════
# PART 2B — COMBINED PIPELINE (classifier + regressor)
# ══════════════════════════════════════════════════════════
#' @section 8. Pipeline Hurdle (Classificador + Regressor)
#'
#' Pipeline combinat que primer classifica si cal regar (1/0) i llavors
#' aplica el model de regressió només als dies classificats com a "regar".
#'
#' @details
#' Procés en 2 etapas:
#' \enumerate{
#'   \item Classificador logistic prediu probabilitat de regar
#'   \item Si prob > 0.5, aplica model_final per predir quantitat mm
#'   \item Si prob <= 0.5, predicció = 0
#' }
#'
#' @results
#' \describe{
#'   \item{Pipeline RMSE}{5.387 mm}
#'   \item{Pipeline MAE}{1.638 mm}
#'   \item{R² (dies amb regadiu)}{0.0072}
#' }
#'
#' @note Els resultats del pipeline simple (thr=0.5) no són bons!
#' R² només 0.007 (gairebé no prediu bé). El classificador
#' no aconsegueix identificar bé els dies de regadiu.
#'
#' @examples
#' \dontrun{
#' # Pipeline: classifier + regressor
#' df_test$pred_mm <- 0
#' idx_yes <- df_test$pred_class == 1
#' df_test$pred_mm[idx_yes] <- predict(model_final, newdata = df_test[idx_yes,])
#' #> Pipeline RMSE: 5.387 mm
#' #> Pipeline MAE:  1.638 mm
#' #> R² (irrig. days only): 0.0072
#' }
#'
df_test$prob_irrigar <- prob_irrigar
df_test$pred_class   <- pred_class

# Only apply the regression model on days classified as "irrigate"
df_test$pred_mm <- 0  # default: no irrigation

idx_yes <- df_test$pred_class == 1
df_test$pred_mm[idx_yes] <- pmax(0, predict(model_final, newdata = df_test[idx_yes, ]))

# ── End-to-end evaluation ──
rmse_pipeline <- sqrt(mean((df_test$Riego_Recomendado_mm - df_test$pred_mm)^2))
mae_pipeline  <- mean(abs(df_test$Riego_Recomendado_mm  - df_test$pred_mm))

# Evaluate only on actual irrigation days (where ground truth > 0)
df_test_actual_yes <- df_test %>% filter(Riego_Recomendado_mm > 0)
r2_pipeline <- cor(df_test_actual_yes$Riego_Recomendado_mm,
                   df_test_actual_yes$pred_mm)^2

cat("Pipeline RMSE:", round(rmse_pipeline, 3), "mm\n")
cat("Pipeline MAE: ", round(mae_pipeline,  3), "mm\n")
cat("R² (irrig. days only):", round(r2_pipeline, 4), "\n")

# ══════════════════════════════════════════════════════════
# MIREM QUIN ES EL THRESHOLD QUE FUNCIONA MILLOR
# ══════════════════════════════════════════════════════════
#' @section 9. Optimització del Threshold
#'
#' Prova diferents thresholds per al classificador logistic per trobar
#' el millor equilibri entre recall i precision.
#'
#' @details
#' Es provem 4 thresholds: 0.40, 0.30, 0.20, 0.15
#'
#' @results
#' \describe{
#'   \item{threshold 0.40}{Recall: 0.55, Precision: 0.53}
#'   \item{threshold 0.30}{Recall: 0.69, Precision: 0.45}
#'   \item{threshold 0.20}{Recall: 0.84, Precision: 0.40}
#'   \item{threshold 0.15}{Recall: 0.87, Precision: 0.36}
#' }
#'
#' @note El threshold 0.20 és el millor equilibri: recall = 0.84,
#' és a dir, identificam el 84% dels dies que s'hauria de regar.
#' Amb threshold 0.5 només teníem recall = 0.55/(0.55+0.45) = 0.37!
#'
#' @examples
#' \dontrun{
#' for (thr in c(0.4, 0.3, 0.2, 0.15)) {
#'   pred <- ifelse(prob_irrigar > thr, 1, 0)
#'   cm   <- table(Predicted = pred, Actual = df_test$irrigar)
#'   recall    <- cm[2,2] / sum(cm[,2])
#'   precision <- cm[2,2] / sum(cm[2,])
#'   cat(sprintf("Threshold %.2f — Recall: %.2f  Precision: %.2f\n", thr, recall, precision))
#' }
#' #> Threshold 0.40 — Recall: 0.55  Precision: 0.53
#' #> Threshold 0.30 — Recall: 0.69  Precision: 0.45
#' #> Threshold 0.20 — Recall: 0.84  Precision: 0.40
#' #> Threshold 0.15 — Recall: 0.87  Precision: 0.36
#' }
#'
thresholds <- c(0.4, 0.3, 0.2, 0.15)

for (thr in thresholds) {
  pred <- ifelse(prob_irrigar > thr, 1, 0)
  cm   <- table(Predicted = pred, Actual = df_test$irrigar)
  recall    <- cm[2,2] / sum(cm[,2])
  precision <- cm[2,2] / sum(cm[2,])
  cat(sprintf("Threshold %.2f — Recall: %.2f  Precision: %.2f\n", 
              thr, recall, precision))
}

# 0.2 es el millor

# ══════════════════════════════════════════════════════════
# PROBEM LA PIPELINE UN ALTRE COP PERO AMB THRESHOLD
# ══════════════════════════════════════════════════════════
#' @section 10. Pipeline amb ThresholdOptimitzat (0.20)
#'
#' Pipeline hurdle amb threshold optimitzat a 0.20 per millorar el recall.
#'
#' @results
#' \describe{
#'   \item{RMSE}{7.131 mm}
#'   \item{MAE}{2.887 mm}
#'   \item{R²}{0.1581}
#' }
#'
#' @sanity_check Comparació d'aigua total:
#' \describe{
#'   \item{Expert (total)}{2225.3 mm}
#'   \item{Pipeline (total)}{4863.8 mm}
#' }
#'
#' @note El pipeline amb threshold 0.20 prediu MOLT MÉS aigua (4863 vs 2225 mm)!
#' Això és un problema agronòmic greu: el model sobre-prediu.
#' El R² também és baix (0.1581 vs 0.87 del model_final).
#' Conclusió: El pipeline hurdle NO és una bona estrategia.
#'
#' @examples
#' \dontrun{
#' threshold <- 0.20
#' df_test$pred_mm_opt <- ...
#' cat("RMSE:", round(rmse_opt, 3), "mm\n")
#' #> RMSE: 7.131 mm
#' #> MAE:  2.887 mm
#' #> R²  : 0.1581
#' #>
#' #> Total water — Expert:   2225.3 mm
#' #> Total water — Pipeline: 4863.8 mm
#' }
#'
threshold <- 0.20

df_test$pred_class_opt <- ifelse(prob_irrigar > threshold, 1, 0)
df_test$pred_mm_opt    <- 0

idx_yes_opt <- df_test$pred_class_opt == 1
df_test$pred_mm_opt[idx_yes_opt] <- pmax(0, predict(model_final, 
                                                    newdata = df_test[idx_yes_opt, ]))

# Metrics
rmse_opt <- sqrt(mean((df_test$Riego_Recomendado_mm - df_test$pred_mm_opt)^2))
mae_opt  <- mean(abs(df_test$Riego_Recomendado_mm  - df_test$pred_mm_opt))

df_test_actual_yes_opt <- df_test %>% filter(Riego_Recomendado_mm > 0)
r2_opt <- cor(df_test_actual_yes_opt$Riego_Recomendado_mm,
              df_test_actual_yes_opt$pred_mm_opt)^2

cat("── Threshold 0.20 pipeline ──\n")
cat("RMSE:", round(rmse_opt, 3), "mm\n")
cat("MAE: ", round(mae_opt,  3), "mm\n")
cat("R²  :", round(r2_opt,   4), "\n")

# Compare total water applied vs expert (agronomic sanity check)
cat("\nTotal water — Expert:  ", round(sum(df_test$Riego_Recomendado_mm), 1), "mm\n")
cat("Total water — Pipeline:", round(sum(df_test$pred_mm_opt), 1), "mm\n")


# ══════════════════════════════════════════════════════════
# PLOT 1 — Predicted vs Actual scatter (model_final vs pipeline)
# ══════════════════════════════════════════════════════════
#' @section 11. Gràfic scatter: Predicted vs Actual
#'
#' Gràfic de dispersió que compara les prediccions del model_final
#' i el pipeline hurdle amb els valors reals de l'expert.
#'
#' @details
#' Genera un gràfic facetat amb:
#' \itemize{
#'   \item Model: Linear Regression (model_final) - color blau
#'   \item Model: Hurdle Pipeline (thr=0.20) - color tomàquet
#' }
#'
#' @output
#' Fitxer: /outputs/scatter_comparison.png (10x5 polzades, 150 dpi)
#'
#' @note El gràfic mostra que el model_final s'acosta molt a la diagonal
#' (predicció = real), mentre el pipeline tiene molta dispersió.
#'
#' @examples
#' \dontrun{
#' ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/scatter_comparison.png")
#' #> Arguments: width=10, height=5, dpi=150
#' }
#'
# GENEREM ELS PLOTS PER VEURE QUE TAL
# pipeline predictions: need to extract irrigation days from df_test
df_test_irrig <- df_test %>% filter(Riego_Recomendado_mm > 0)

scatter_data <- bind_rows(
  df_test_reg %>% 
    transmute(actual = Riego_Recomendado_mm, 
              predicted = prediccion, 
              model = "Linear Regression (model_final)"),
  df_test_irrig %>% 
    transmute(actual = Riego_Recomendado_mm, 
              predicted = pred_mm_opt, 
              model = "Hurdle Pipeline (thr=0.20)")
)

ggplot(scatter_data, aes(x = actual, y = predicted, color = model)) +
  geom_point(alpha = 0.3, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.8, linetype = "dashed") +
  scale_color_manual(values = c("Linear Regression (model_final)" = "steelblue",
                                "Hurdle Pipeline (thr=0.20)"      = "tomato")) +
  facet_wrap(~model) +
  labs(title = "Predicted vs Actual — Irrigation days only (2021–2024)",
       x = "FAO Expert recommendation (mm)",
       y = "Model prediction (mm)") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/scatter_comparison.png",
       width = 10, height = 5, dpi = 150)


# ══════════════════════════════════════════════════════════
# PLOT 2 — Time series: FAO vs model_final vs pipeline
# (irrigation days only, same style as plot 1)
# ══════════════════════════════════════════════════════════
#' @section 12. Gràfic de sèrie temporal
#'
#' Gràfic de sèrie temporal que mostra l'evolució de les
#' recomanacions d'irrigació al llarg del temps (2021-2024).
#'
#' @details
#' Compara 3 estratègies:
#' \itemize{
#'   \item FAO Expert (línia sòlida, blau cel)
#'   \item Linear Regression (línia discontínua, verd)
#'   \item Hurdle Pipeline (línia puntets, tomàquet)
#' }
#'
#' @output
#' Fitxer: /outputs/timeseries_comparison.png (14x5 polzades, 150 dpi)
#'
#' @note Només es mostren els dies on l'expert recomana regar (Riego_Recomendado_mm > 0).
#' El model_final segueix bé la tendència de l'expert, mentre el pipeline
#' té pics molt més alts (sobre-prediu).
#'
#' @examples
#' \dontrun{
#' ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/timeseries_comparison.png")
#' #> Arguments: width=14, height=5, dpi=150
#' }
#'
df_test$Fecha_dt      <- as.Date(df_test$Fecha, format = "%d/%m/%Y")
df_test$pred_mm_final <- 0
df_test$pred_mm_final[df_test$Riego_Recomendado_mm > 0] <- df_test_reg$prediccion

# KEY CHANGE: filter to FAO irrigation days only
df_plot <- df_test %>%
  filter(Riego_Recomendado_mm > 0) %>%
  transmute(date     = Fecha_dt,
            FAO      = Riego_Recomendado_mm,
            Final    = pred_mm_final,
            Pipeline = pred_mm_opt) %>%
  pivot_longer(-date, names_to = "Strategy", values_to = "mm")

ggplot(df_plot, aes(x = date, y = mm, color = Strategy, linetype = Strategy)) +
  geom_line(linewidth = 0.8, alpha = 0.9) +
  scale_color_manual(
    values = c("FAO"      = "steelblue",
               "Final"    = "forestgreen",
               "Pipeline" = "tomato"),
    labels = c("FAO Expert",
               "Linear Regression (irrigation days only)",
               "Hurdle Pipeline (thr=0.20)")
  ) +
  scale_linetype_manual(
    values = c("FAO"      = "solid",
               "Final"    = "dashed",
               "Pipeline" = "dotdash"),
    labels = c("FAO Expert",
               "Linear Regression (irrigation days only)",
               "Hurdle Pipeline (thr=0.20)")
  ) +
  labs(title    = "Irrigation recommendation — FAO Expert vs Models (2021–2024)",
       subtitle = "Shown on irrigation days only (FAO > 0 mm)",
       x = "Date", y = "Irrigation (mm)",
       color    = "Strategy",
       linetype = "Strategy") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("/home/lluis/Documents/TFG/REGRESIO_LINEAR/outputs/timeseries_comparison.png",
       width = 14, height = 5, dpi = 150)