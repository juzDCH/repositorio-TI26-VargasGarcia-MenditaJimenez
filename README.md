# Predicción de Churn en Telecomunicaciones
### TI-26 — Vargas García / Mendita Jiménez

Proyecto de Machine Learning para predecir la deserción de clientes (*churn*) en una empresa de telecomunicaciones usando técnicas de clasificación supervisada, con énfasis en el método de **Bagging (Random Forest)**.

---

## Estructura del proyecto

```
repositorio-TI26/
├── README.md                        ← Este archivo (cómo correr el proyecto)
├── GUIA_DEL_PROYECTO.md             ← Explicación detallada de cada parte
├── requirements.txt                 ← Librerías necesarias
├── data/
│   ├── raw/                         ← Dataset original sin modificar
│   └── processed/                   ← Archivos generados por los notebooks
│       ├── Telco_customer_churn.csv ← Dataset principal
│       ├── X_train.csv              ← Features de entrenamiento (generado por NB02)
│       ├── X_test.csv               ← Features de prueba (generado por NB02)
│       ├── y_train.csv              ← Etiquetas de entrenamiento (generado por NB02)
│       ├── y_test.csv               ← Etiquetas de prueba (generado por NB02)
│       └── metricas_rf.json         ← Métricas del mejor modelo (generado por NB03)
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Análisis exploratorio de datos
│   ├── 02_preprocessing.ipynb       ← Limpieza, codificación y escalado
│   ├── 03_model_main.ipynb          ← Random Forest + GridSearchCV
│   └── 04_model_comparison.ipynb    ← Decision Tree vs Random Forest
└── docs/
    ├── informe_final.pdf
    └── presentacion.pdf
```

---

## Cómo ejecutar el proyecto (primera vez)

> En este equipo Python se usa con el comando `py` (no `python`).

### Paso 1 — Abrir PowerShell en la carpeta del proyecto

```powershell
cd "C:\Users\Sebas\Documents\Github\repositorio-TI26-VargasGarcia-MenditaJimenez"
```

### Paso 2 — Crear el entorno virtual

```powershell
py -m venv venv
```

> Esto crea una carpeta `venv/` con un Python aislado solo para este proyecto.

### Paso 3 — Instalar las librerías

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

> Descarga e instala pandas, scikit-learn, matplotlib, seaborn, jupyterlab y todas sus dependencias dentro del venv.

### Paso 4 — Registrar el kernel en Jupyter (solo la primera vez)

```powershell
.\venv\Scripts\python.exe -m ipykernel install --user --name=venv-churn --display-name "Python (venv-churn)"
```

> Esto le dice a Jupyter que existe un kernel llamado **Python (venv-churn)** que usa el Python del venv (con todas las librerías instaladas).

### Paso 5 — Lanzar Jupyter Lab

```powershell
.\venv\Scripts\python.exe -m jupyter lab
```

> Se abrirá el navegador automáticamente. Si no abre, copia la URL que aparece en la terminal (`http://localhost:8888/lab?token=...`).

---

## Cómo ejecutar el proyecto (desde la segunda vez)

No necesitas reinstalar nada. Solo lanza Jupyter:

```powershell
cd "C:\Users\Sebas\Documents\Github\repositorio-TI26-VargasGarcia-MenditaJimenez"
.\venv\Scripts\python.exe -m jupyter lab
```

---

## Seleccionar el kernel correcto en Jupyter

Cuando abras un notebook, debes asegurarte de que esté usando el kernel del venv:

1. Abre cualquier notebook (`.ipynb`)
2. En la esquina superior derecha verás el nombre del kernel activo
3. Si dice algo distinto de **Python (venv-churn)**, haz clic en ese nombre
4. Selecciona **Python (venv-churn)** de la lista

> Si usas el kernel equivocado, obtendrás `ModuleNotFoundError: No module named 'pandas'` aunque todo esté instalado.

---

## Ejecutar los notebooks en orden

Desde Jupyter, abre la carpeta `notebooks/` y ejecuta cada notebook **en este orden**:

| Orden | Archivo | Qué genera |
|---|---|---|
| 1° | `01_EDA.ipynb` | Solo visualizaciones (no genera archivos) |
| 2° | `02_preprocessing.ipynb` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` |
| 3° | `03_model_main.ipynb` | `metricas_rf.json` |
| 4° | `04_model_comparison.ipynb` | Tabla y gráficos comparativos |

Para ejecutar todas las celdas de un notebook de una vez:  
**Menú → Run → Run All Cells**

O celda por celda con **`Shift + Enter`**

---

## Descripción del problema

El churn (abandono de clientes) es uno de los problemas de mayor impacto económico en la industria de telecomunicaciones. Este proyecto desarrolla un modelo predictivo que permite identificar, con anticipación, qué clientes tienen mayor probabilidad de cancelar su servicio, habilitando estrategias de retención proactivas.

**Dataset:** Telco Customer Churn (~7,000 clientes, 33 variables)  
**Variable objetivo:** `Churn Label` (Yes = abandona, No = permanece)

Para una explicación detallada de la metodología, los modelos y las decisiones tomadas, consulta [GUIA_DEL_PROYECTO.md](GUIA_DEL_PROYECTO.md).

---

## Versiones instaladas

| Librería | Versión |
|---|---|
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.0 |
| jupyterlab | 4.5.8 |
