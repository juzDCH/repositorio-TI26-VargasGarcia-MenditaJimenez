# Guía de Inicio Rápido

---

## Primera vez que abres el proyecto

Abre **PowerShell** y ejecuta estos comandos en orden:

```powershell
# 1. Ir a la carpeta del proyecto
cd "C:\Users\Sebas\Documents\Github\repositorio-TI26-VargasGarcia-MenditaJimenez"

# 2. Crear el entorno virtual
py -m venv venv

# 3. Instalar todas las librerías
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Registrar el kernel en Jupyter (solo se hace una vez)
.\venv\Scripts\python.exe -m ipykernel install --user --name=venv-churn --display-name "Python (venv-churn)"

# 5. Abrir Jupyter
.\venv\Scripts\python.exe -m jupyter lab
```

---

## Desde la segunda vez en adelante

```powershell
cd "C:\Users\Sebas\Documents\Github\repositorio-TI26-VargasGarcia-MenditaJimenez"
.\venv\Scripts\python.exe -m jupyter lab
```

---

## Dentro de Jupyter

**Paso 1** — En la esquina superior derecha del notebook, verifica que el kernel sea **Python (venv-churn)**  
→ Si no lo es, haz clic en el nombre del kernel y cámbialo

**Paso 2** — Ejecuta los notebooks en este orden:

| # | Notebook | Acción |
|---|---|---|
| 1 | `01_EDA.ipynb` | Run All Cells |
| 2 | `02_preprocessing.ipynb` | Run All Cells → genera los CSV de train/test |
| 3 | `03_model_main.ipynb` | Run All Cells → puede tardar ~5 min (GridSearchCV) |
| 4 | `04_model_comparison.ipynb` | Run All Cells |

> **Run All Cells:** Menú → Run → Run All Cells  
> **Celda por celda:** Shift + Enter

---

## Si algo falla

| Error | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | El kernel no es **Python (venv-churn)** — cámbialo |
| `FileNotFoundError: X_train.csv` | Ejecuta primero el Notebook 02 |
| `FileNotFoundError: metricas_rf.json` | Ejecuta primero el Notebook 03 |
| Jupyter no abre el navegador | Copia la URL `http://localhost:8888/lab?token=...` que aparece en PowerShell |
