# Guía del Proyecto — Predicción de Churn en Telecomunicaciones
### TI-26 — Vargas García / Mendita Jiménez

Este documento explica en detalle **qué hicimos**, **por qué lo hicimos** y **qué contiene cada parte** del proyecto. Sirve como referencia para entender las decisiones tomadas antes de presentar o defender el trabajo.

---

## ¿De qué trata el proyecto?

Una empresa de telecomunicaciones quiere saber **qué clientes van a abandonar el servicio** (churn) antes de que lo hagan. Si podemos predecirlo, la empresa puede actuar: ofrecer descuentos, mejorar el servicio, contactar al cliente, etc.

Para eso usamos **Machine Learning supervisado**: le damos al modelo ejemplos de clientes pasados (con sus características y si se fueron o no) para que aprenda a predecir casos nuevos.

---

## El dataset

**Archivo:** `data/processed/Telco_customer_churn.csv`  
**Origen:** IBM Sample Dataset — Telco Customer Churn  
**Tamaño:** ~7,000 clientes, 33 columnas

### ¿Qué información contiene?

| Tipo de columna | Ejemplos | Para qué sirven |
|---|---|---|
| Identificación | `CustomerID` | Solo para identificar, no se usa en el modelo |
| Geografía | `City`, `Zip Code`, `Latitude` | No son predictivas, se eliminan |
| Datos del cliente | `Gender`, `Senior Citizen`, `Partner`, `Dependents` | Perfil demográfico |
| Servicios contratados | `Phone Service`, `Internet Service`, `Streaming TV`... | Qué tiene contratado |
| Tipo de contrato | `Contract`, `Payment Method`, `Paperless Billing` | Cómo paga y por cuánto tiempo |
| Cargos económicos | `Monthly Charges`, `Total Charges`, `Tenure Months` | Cuánto paga y hace cuánto es cliente |
| Variable objetivo | `Churn Label` (Yes/No) | Lo que queremos predecir |
| Variables derivadas | `Churn Score`, `CLTV`, `Churn Reason` | Calculadas a partir del churn, se eliminan para evitar trampa |

> **Nota importante:** `Churn Score` y `CLTV` son columnas que ya contienen información sobre el churn (fueron calculadas sabiendo quién se fue). Si las dejáramos en el modelo, estaríamos "haciendo trampa" — el modelo aprendería de la respuesta, no de las características. A esto se le llama **data leakage** y hace que el modelo parezca perfecto en entrenamiento pero falle en producción.

---

## Notebook 01 — Análisis Exploratorio de Datos (EDA)

**Archivo:** `notebooks/01_EDA.ipynb`

### ¿Qué es el EDA y para qué sirve?

Antes de modelar, necesitamos entender los datos. El EDA nos permite:
- Detectar errores o valores raros
- Entender la distribución de las variables
- Ver qué variables se relacionan con el churn
- Tomar decisiones informadas sobre el preprocesamiento

### ¿Qué hace cada parte?

**`.info()`** — Muestra el tipo de dato de cada columna y cuántos valores no nulos tiene. Con esto detectamos que `Total Charges` tiene valores vacíos (clientes con 0 meses, que nunca pagaron).

**`.describe()`** — Estadísticas de resumen: media, mediana, mínimo, máximo. Nos dice, por ejemplo, que el cargo mensual promedio es ~$65 USD.

**`.value_counts()`** — Cuenta cuántas veces aparece cada categoría. Útil para ver los tipos de contrato más comunes o el balance de churn/no-churn.

**Pie chart de Churn** — Muestra que ~73% de los clientes NO se fueron y ~27% SÍ. Esto es un **desbalance de clases**: hay muchos más "No Churn" que "Churn". Esto importa porque un modelo que siempre diga "No Churn" tendría 73% de accuracy pero sería inútil.

**Histogramas** — Distribución de las tres variables numéricas clave:
- `Monthly Charges`: distribución bimodal (picos en bajo ~$20 y alto ~$80)
- `Tenure Months`: muchos clientes nuevos y muchos muy antiguos
- `Total Charges`: sesgado a la derecha (clientes nuevos acumulan poco)

**Boxplots por tipo de contrato** — Los clientes con contrato mes a mes tienen mayor variabilidad en sus cargos mensuales y tienden a pagar más. Son el grupo de mayor riesgo de churn.

**Mapa de calor de correlaciones** — Correlaciones entre variables numéricas:
- `Tenure Months` y `Total Charges` tienen correlación alta (~0.83): lógico, más tiempo → más pagos acumulados
- `Churn Value` correlaciona negativamente con `Tenure Months`: clientes nuevos abandonan más
- `Churn Value` correlaciona positivamente con `Monthly Charges`: cargos altos → más churn

---

## Notebook 02 — Preprocesamiento

**Archivo:** `notebooks/02_preprocessing.ipynb`

### ¿Por qué es necesario el preprocesamiento?

Los algoritmos de Machine Learning solo entienden números. Necesitamos transformar el dataset para:
1. Eliminar lo que no sirve o contamina
2. Convertir texto a números
3. Poner todas las variables en la misma escala
4. Dividir en entrenamiento y prueba de forma correcta

### Paso a paso

**Eliminación de columnas:**
Quitamos `CustomerID` (identificador sin valor predictivo), columnas geográficas (`City`, `Lat Long`, etc.) y columnas de data leakage (`Churn Score`, `CLTV`, `Churn Reason`, `Churn Value`).

**Limpieza de `Total Charges`:**
Esta columna está guardada como texto porque los clientes con 0 meses de permanencia tienen el campo vacío. La convertimos a número y rellenamos los vacíos con la **mediana** (no la media, porque la media es más sensible a valores extremos y puede distorsionar la imputación).

**Label Encoding (variables binarias):**
Para variables con solo dos categorías (Yes/No) usamos un mapeo simple: `No → 0`, `Yes → 1`. Aplicado a: `Partner`, `Dependents`, `Phone Service`, `Paperless Billing`, todos los servicios opcionales, `Gender` y `Senior Citizen`.

**One-Hot Encoding (variables con más de 2 categorías):**
Si tuviéramos `Contract` con valores `Month-to-month=0`, `One year=1`, `Two year=2`, el modelo podría pensar que "Two year" es el doble de "One year", lo cual no tiene sentido. Para evitar esto, creamos una columna binaria separada para cada categoría. Aplicado a: `Contract`, `Internet Service`, `Payment Method`, `Multiple Lines`.

`drop_first=True` elimina una categoría redundante de cada variable (si sabemos que no es "Month-to-month" ni "One year", forzosamente es "Two year"), evitando multicolinealidad.

**División Train/Test (80/20):**
Separamos el 80% de los datos para entrenar el modelo y el 20% para evaluarlo. Usamos:
- `stratify=y` → garantiza que la proporción de churn sea igual en ambos conjuntos
- `random_state=42` → hace el experimento reproducible (siempre el mismo split)

**StandardScaler:**
Transforma las variables numéricas continuas para que tengan **media = 0** y **desviación estándar = 1**. Sin esto, `Total Charges` (en miles) dominaría sobre `Tenure Months` (en decenas) solo por su escala.

> **Regla crítica:** El scaler se ajusta **solo con datos de entrenamiento** y luego se aplica al test. Si ajustáramos con todo el dataset, estaríamos usando información del futuro (test) para escalar el entrenamiento — otro tipo de data leakage.

---

## Notebook 03 — Modelo Principal: Random Forest

**Archivo:** `notebooks/03_model_main.ipynb`

### ¿Qué es Bagging?

**Bagging** (Bootstrap Aggregating) es una técnica que combina múltiples modelos para obtener uno mejor. La idea es:

1. Crear varias versiones del dataset de entrenamiento tomando muestras aleatorias **con reemplazo** (bootstrap): cada muestra tiene el mismo tamaño que el original pero con repetición de algunos datos y ausencia de otros (~37% queda fuera en cada muestra)
2. Entrenar un modelo independiente sobre cada muestra
3. Combinar las predicciones: **votación** (clasificación) o **promedio** (regresión)

**¿Por qué funciona?** Los modelos individuales tienen errores, pero cuando son independientes entre sí, sus errores se cancelan al promediarlos. Es como preguntarle a 200 expertos y quedarse con la opinión mayoritaria en lugar de seguir a uno solo.

### ¿Qué es Random Forest?

Random Forest es Bagging aplicado a árboles de decisión, con un ingrediente extra: en cada nodo de cada árbol, en lugar de evaluar **todas** las variables para hacer el split, solo evalúa un **subconjunto aleatorio** (`max_features = sqrt(n_features)`).

Esto hace que los árboles sean menos correlacionados entre sí — si todos usaran las mismas variables importantes, todos cometerían errores similares y el promedio no mejoraría tanto.

### GridSearchCV — búsqueda de hiperparámetros

Los hiperparámetros son configuraciones del modelo que no se aprenden de los datos, sino que nosotros elegimos. Para encontrar la mejor combinación, `GridSearchCV` prueba **todas las combinaciones posibles** del grid:

| Hiperparámetro | Valores probados | Descripción |
|---|---|---|
| `n_estimators` | 100, 200, 300 | Cuántos árboles tiene el bosque |
| `max_depth` | None, 10, 20 | Qué tan profundo puede crecer cada árbol |
| `min_samples_split` | 2, 5, 10 | Mínimo de muestras para dividir un nodo |
| `max_features` | 'sqrt', 'log2' | Cuántas features se evalúan en cada nodo |

Usa **validación cruzada de 5 pliegues** (5-fold CV): divide el training en 5 partes, entrena en 4 y valida en 1, rotando 5 veces. Así evita que la elección de hiperparámetros dependa de una sola partición aleatoria.

### Métricas de evaluación

**Matriz de confusión:** Tabla 2x2 que muestra cuántas predicciones fueron correctas e incorrectas para cada clase. En churn, los **Falsos Negativos** (predijo "no se va" pero sí se fue) son los más costosos para el negocio.

**Curva ROC y AUC:** La curva ROC grafica la tasa de verdaderos positivos vs falsos positivos para todos los umbrales de decisión posibles. El área bajo la curva (**AUC**) mide en general qué tan bien discrimina el modelo: 0.5 = aleatorio, 1.0 = perfecto.

**Feature Importance:** Random Forest mide cuánto contribuye cada variable a reducir la impureza de los nodos (índice de Gini) en promedio a lo largo de todos los árboles. Las variables más importantes son las que más ayudan a separar churn de no-churn.

---

## Notebook 04 — Comparación: Decision Tree vs Random Forest

**Archivo:** `notebooks/04_model_comparison.ipynb`

### ¿Por qué comparar con un árbol simple?

El Árbol de Decisión es el "bloque base" del Random Forest. Comparar ambos nos permite demostrar empíricamente que el ensemble es mejor, y entender exactamente **qué aporta el Bagging**.

Un árbol de decisión profundo tiene **alta varianza**: si entrenamos en datos ligeramente diferentes, el árbol resultante puede ser muy distinto. Memoriza el ruido del entrenamiento y falla con datos nuevos (sobreajuste / overfitting).

### ¿Qué contiene el notebook?

1. **Entrenamiento del Decision Tree** con GridSearchCV (misma metodología, mismos datos)
2. **Re-entrenamiento del Random Forest** con los mejores parámetros del Notebook 03
3. **Tabla comparativa** con 5 métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
4. **Curvas ROC superpuestas** en el mismo gráfico — visualmente muestra qué modelo discrimina mejor
5. **Barras agrupadas** de métricas — comparación directa de cada métrica entre ambos modelos
6. **Conclusión** explicando el mecanismo por el que RF supera a DT

### Resultado esperado

| Métrica | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | ~79% | ~82% |
| F1-Score | ~0.78 | ~0.82 |
| ROC-AUC | ~0.72 | ~0.86 |

*Los valores exactos dependen de los hiperparámetros encontrados por GridSearchCV en tu ejecución.*

El Random Forest consistentemente supera al árbol simple porque:
- **Menor varianza:** Promedia 200+ árboles, cancelando errores individuales
- **Mejor generalización:** Cada árbol ve datos distintos (bootstrap), el ensemble ve el panorama completo
- **Features decorrelacionadas:** El subsampling de variables fuerza a descubrir patrones complementarios

---

## Flujo completo del proyecto

```
Dataset original (CSV)
        │
        ▼
01_EDA.ipynb          ← Explorar, visualizar, entender
        │
        ▼
02_preprocessing.ipynb ← Limpiar, codificar, escalar, dividir
        │
        ├── X_train.csv, X_test.csv
        └── y_train.csv, y_test.csv
               │
               ├──▶ 03_model_main.ipynb    ← Random Forest + GridSearchCV + evaluación
               │           │
               │           └── metricas_rf.json
               │
               └──▶ 04_model_comparison.ipynb ← DT vs RF → conclusión final
```

---

## Decisiones de diseño y por qué las tomamos

| Decisión | Alternativa | Por qué elegimos esta |
|---|---|---|
| Imputar `Total Charges` con mediana | Media o eliminar filas | La mediana es robusta ante outliers; eliminar filas reduce datos |
| `stratify=y` en el split | Sin stratify | Garantiza balance de clases en train y test |
| Ajustar scaler solo en train | Ajustar en todo el dataset | Evita data leakage del test al entrenamiento |
| `drop_first=True` en OHE | Conservar todas las dummies | Evita multicolinealidad (trampa de la variable ficticia) |
| F1-weighted como scoring en GridSearchCV | Accuracy | Accuracy es engañosa con desbalance de clases |
| `random_state=42` en todo | Sin semilla | Hace el experimento 100% reproducible |
