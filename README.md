# Titanic
 Kaggle competitivo 1

Para iniciar este contenedor, se deben de seguir los siguientes pasos. Esto mapeará la carpeta actual al directorio raiz del contenedor. Lo cual es una ventaja pues nos permite seguir manteniendo nuestro git sincronizado con github sin tener que estar autorizando el contenedor desde dentro.

Este contenedor usa Py Torch, la cual instala CUDA, se use o no, es util.
---

## Docker Deployment

To build and deploy the project with Docker, follow these steps:

Build and run the container:
```bash
docker-compose up --build
```

Despues de esto, podemos crear con visual code una conexion remota al contenedor y seleccionar la carpeta raíz.

```
Note: You can adjust the ports as needed; some antivirus systems may use port 5000.

Note: When using Mac, is important to turn-off Hand-off and AirDrop, because those tools use the port 5000, so it the code will not be able to run properly
---


Ahora, para volver este simple ejercicio un proyecto de MLops, vamos a tener que tener una estructura organizada (modulos) y escalable para poder realizar el desarrollo y despligue. En resumen, ya lo estamos haciendo desde el momento que iniciamos el contenedor.

Para esto, vamos a ir ejecutando el archivo ipynb del EDA, el cual nos van a ayudar a configurar el resto del proyecto.



---
---

1. **Exploración de datos (`eda.ipynb`)**
   - Este notebook es puramente **exploratorio**.
   - Se utiliza para analizar los datos crudos, identificar problemas como valores faltantes o distribuciones desbalanceadas, y generar ideas para el preprocesamiento.
   - Incluye:
     - Estadísticas descriptivas.
     - Visualización de datos.
     - Hipótesis iniciales para el modelado.


---
Más Adelante, si quisieramos crear el flujo completo:
**Diseño del pipeline (`pipeline_design.ipynb`)**
- Este notebook se usa como una **guía de implementación** para el pipeline.
- Ayuda a diseñar y validar las transformaciones, división de datos, entrenamiento del modelo y evaluación inicial.
- Incluye:
   - Preprocesamiento paso a paso.
   - Feature engineering.
   - Prototipos de modelos.
   - Visualización de métricas iniciales.

- Los notebooks NO forman parte del pipeline final de producción. Son herramientas para explorar y diseñar el flujo.
- El pipeline se implementa como scripts en `src/` (por ejemplo, `preprocess.py`, `train.py`) basados en lo que se validó en los notebooks.


### Estructura completa del proyecto: paso a paso**

#### **Fase 1: Configuración inicial**
1. **Crea la estructura del proyecto:**
   - Define carpetas como `datos/`, `src/`, `notebooks/`, y `tests/`.
2. **Inicializa Git y DVC:**
   ```bash
   git init
   dvc init
   ```
3. **Configura dependencias en `requirements.txt`:**
   ```plaintext
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   dvc
   mlflow
   kaggle
   pytest
   ```
   Instálalas:
   ```bash
   pip install -r requirements.txt
   ```

---

#### **Fase 2: Exploración de datos**
1. **Notebook `eda.ipynb`:**
   - Realiza análisis exploratorio de los datos crudos (`datos/raw.csv`).
   - Incluye:
     - Estadísticas descriptivas (`.describe()`, `.info()`).
     - Visualización de distribuciones y correlaciones (usando `matplotlib`/`seaborn`).
     - Identificación de valores faltantes o inconsistencias.
   - Guarda observaciones en el notebook como guía para el preprocesamiento.

---

#### **Fase 3: Diseño del pipeline**
1. **Notebook `pipeline_design.ipynb`:**
   - Prototipa el flujo de preprocesamiento y entrenamiento:
     - Limpieza de datos (manejo de valores nulos, transformación de variables categóricas).
     - Feature engineering (normalización, codificación).
     - División en train/val/test.
     - Entrenamiento inicial del modelo (con `scikit-learn`).
     - Evaluación básica (accuracy, F1, etc.).
   - Guarda el modelo como archivo `.pkl` para usarlo más adelante.

2. **Documenta en el notebook los pasos validados.**

---

#### **Fase 4: Implementación del pipeline**
1. **Preprocesamiento (`src/data/preprocess.py`):**
   - Implementa las transformaciones validadas en el notebook.
   - Usa DVC para conectar este paso al pipeline.

2. **División de datos (`src/data/split.py`):**
   - Divide los datos procesados en `train.csv`, `val.csv`, y `test.csv`.

3. **Entrenamiento del modelo (`src/models/train.py`):**
   - Implementa el código para entrenar el modelo y guardar los artefactos (`model.pkl` y `metrics.json`).
   - Registra el experimento con MLflow.

4. **Evaluación (`src/models/evaluate.py`):**
   - Calcula métricas en el conjunto de validación/test y guarda los resultados.

5. **Predicción (`src/models/predict.py`):**
   - Carga el modelo entrenado y genera predicciones en nuevos datos.

---

#### **Fase 5: Automatización del pipeline**
1. **Define el pipeline con DVC (`dvc.yaml`):**
   - Conecta los scripts con entradas y salidas:
     ```yaml
     stages:
       preprocess:
         cmd: python src/data/preprocess.py
         deps:
           - src/data/preprocess.py
           - datos/raw.csv
         outs:
           - datos/processed.csv

       train:
         cmd: python src/models/train.py
         deps:
           - src/models/train.py
           - datos/processed.csv
         outs:
           - models/model.pkl
           - models/metrics.json
     ```

2. **Reproduce el pipeline:**
   ```bash
   dvc repro
   ```

---

#### **Fase 6: Pruebas de unidad**
1. **Escribe pruebas para cada componente en `tests/`:**
   - **`test_data.py`:** Valida el formato y la integridad de los datos.
   - **`test_models.py`:** Asegúrate de que el modelo cumple con métricas mínimas.

2. **Ejecuta pruebas con `pytest`:**
   ```bash
   pytest src/tests/
   ```

---

#### **Fase 7: Gestión de experimentos**
1. **Configura MLflow para el seguimiento:**
   - Inicia MLflow en el contenedor:
     ```bash
     mlflow ui
     ```
   - Agrega registro en `train.py`:
     ```python
     import mlflow

     mlflow.set_experiment("Titanic-MLOps")
     with mlflow.start_run():
         mlflow.log_param("max_depth", 5)
         mlflow.log_metric("accuracy", 0.85)
         mlflow.sklearn.log_model(model, "model")
     ```

2. **Rastrea experimentos desde la interfaz web de MLflow.**

---

#### **Fase 8: CI/CD**
1. **Configura GitHub Actions para pruebas automáticas:**
   - Archivo `.github/workflows/ci.yml`:
     ```yaml
     name: CI Pipeline

     on: [push, pull_request]

     jobs:
       test:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v3
           - name: Set up Python
             uses: actions/setup-python@v3
             with:
               python-version: "3.10"
           - name: Install dependencies
             run: pip install -r requirements.txt
           - name: Run tests
             run: pytest src/tests/
     ```

2. **Verifica automáticamente que las pruebas pasen antes de realizar un merge.**

---

## ML Flow

En **MLflow**, un **run** es una ejecución específica de un experimento. Es decir, cada vez que realizas un entrenamiento de modelo, pruebas un conjunto de hiperparámetros o evalúas un enfoque diferente, se registra como un nuevo **run** dentro de un **experimento**.

---

### **¿Qué es un "run" en detalle?**
- Un **run** guarda toda la información de un experimento individual:
  - Hiperparámetros utilizados.
  - Métricas obtenidas (precisión, F1, etc.).
  - Artefactos generados (modelos, predicciones, gráficos, etc.).
  - El modelo resultante (si lo guardas).

- **Cuándo se crea un "run":**
  - Se crea automáticamente al llamar a `mlflow.start_run()` en tu código.
  - Si no lo llamas manualmente, MLflow creará un "run" por defecto en el experimento actual.

---

### **¿Cómo asignar un nombre a un "run"?**
Puedes asignar un nombre personalizado a cada "run" usando el parámetro `run_name` dentro de `mlflow.start_run()`. Esto es útil para identificar rápidamente cada ejecución en la interfaz de MLflow.

#### **Ejemplo básico:**
```python
import mlflow

# Configurar MLflow
mlflow.set_experiment("Titanic_Competition")

# Crear un run con un nombre personalizado
with mlflow.start_run(run_name="RandomForest_depth10"):
    # Registrar parámetros
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 100)

    # Registrar métricas
    mlflow.log_metric("accuracy", 0.85)

    # Mensaje final
    print("Run registrado con el nombre 'RandomForest_depth10'")
```

---

### **¿Qué sucede si no le asignas un nombre?**
Si no especificas `run_name`, MLflow asigna un identificador único al "run", que puedes ver en la interfaz web. Sin embargo, no será descriptivo.

---

### **Organización de runs y experimentos**
1. **Experimento:**
   - Un contenedor lógico para varios "runs" relacionados.
   - Ejemplo: Un experimento llamado `"Titanic_Competition"` puede tener múltiples "runs" para diferentes configuraciones de modelos.

2. **Run:**
   - Una instancia específica de un experimento.
   - Ejemplo: Dentro de `"Titanic_Competition"`, puedes tener un "run" llamado `"RandomForest_depth10"` y otro `"SVM_linear_kernel"`.

---

### **Asignar nombres dinámicamente**
Si quieres nombrar los "runs" basándote en parámetros o configuraciones, puedes hacerlo dinámicamente:

```python
run_name = f"RandomForest_depth{10}_estimators{100}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.85)

    print(f"Run registrado con el nombre '{run_name}'")
```

---

### **Ver tus "runs" en la interfaz de MLflow**
1. Abre la interfaz web de MLflow (`http://localhost:5000`).
2. Selecciona tu experimento (por ejemplo, `"Titanic_Competition"`).
3. Verás una tabla con todos los "runs" registrados. La columna **Run Name** mostrará el nombre que asignaste.

---

El comportamiento de **MLflow** con **Grid Search** y otras optimizaciones de hiperparámetros depende de cómo estructures tu código:


### **1. Con Grid Search**
Si usas herramientas como `GridSearchCV` de Scikit-Learn, **solo se registra el mejor modelo al final**, a menos que implementes manualmente un registro para cada combinación de hiperparámetros probados.

Esto sucede porque `GridSearchCV`:
- Prueba todas las combinaciones de hiperparámetros internamente.
- Solo expone el mejor modelo (`best_estimator_`) y los mejores hiperparámetros (`best_params_`) al final.

#### **Cómo registrar solo el mejor modelo con Grid Search**
Por defecto, puedes registrar el modelo que `GridSearchCV` selecciona como el mejor:
```python
with mlflow.start_run(run_name="Best_Model"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    print("Registrado solo el mejor modelo encontrado por Grid Search.")
```

---
---
