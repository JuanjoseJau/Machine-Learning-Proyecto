# Predicción de Viabilidad Comercial y Aceptación de Videojuegos en Steam mediante Machine Learning y Deep Learning

<p align="center">
  <img src="https://shared.fastly.steamstatic.com/store_item_assets/steam/clusters/sale_autumn2019_assets/54b5034d397baccb93181cc6/deck_banner_desktop_english.jpg" width="800" height="300" alt="Banner Proyecto Steam ML"/>
</p>

## Autores
Juan Jose Jauregui Mendoza (Código: 2250920)

---

## Objetivo
Predecir el nivel de aceptación del público (`pct_pos_total` y éxito comercial $\ge 70\%$) de un videojuego en la plataforma Steam a partir de sus características técnicas intrínsecas, alcance de localización y variables de diseño comercial (sin fuga de datos), integrando modelos de Machine Learning clásico y Redes Neuronales Densas (Deep Learning) en una herramienta funcional de simulación pre-lanzamiento.

---

## Dataset
El proyecto utiliza el conjunto de datos público **Steam Games Dataset (Marzo 2025)** almacenado localmente en el archivo `games_march2025_full.csv`:

* **Fuente Oficial y Descarga (Kaggle):** [Steam Games Dataset 2025 — Kaggle (artermiloff)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_cleaned.csv)
* **Volumen de Datos:** **89,618 videojuegos** registrados en el catálogo histórico de Steam.
* **Dimensiones y Limpieza Estructural:**
  * **47 variables originales**, depuradas a **33 columnas limpias** tras descartar metadatos web no predictivos (`header_image`, `screenshots`, `movies`, `website`, `support_url`, `support_email`, `user_score`) y atributos con valores nulos estructurales, preservando la totalidad de los **89,618 registros**.
  * **Eliminación de Fuga de Datos (*Data Leakage*):** Se excluyeron explícitamente de la matriz predictora $\mathbf{X}$ todas las variables de conteo directo de reseñas (`positive`, `negative`, `num_reviews_total`, `pct_pos_recent`, `num_reviews_recent`, `recommendations`) que componían matemática o directamente la variable objetivo.
  * **Ingeniería de Características (`features_supervisadas` — 17 predictores):** Extracción y transformación logarítmica ($\log(1+x)$) de variables de diseño, soporte técnico y mercado: `price`, `dlc_count`, `log_logros`, `num_idiomas`, `num_idiomas_audio`, `num_categorias`, `es_multijugador`, `es_coop`, `tiene_cloud`, `es_indie`, `es_accion`, `es_rpg`, `soporte_mac_linux`, `antiguedad_anios`, `metacritic_score`, `log_playtime` y `log_compradores`.
* **Variables Objetivo Evaluadas:**
  * **Clasificación Binaria (`clase_exito`):** `Clase 1` (Alta Aceptación: $\text{pct\_pos\_total} \ge 70\%$, 42.75% del catálogo) vs. `Clase 0` (Baja Aceptación / Sin Tracción: $\text{pct\_pos\_total} < 70\%$, 57.25% del catálogo).
  * **Clasificación Multi-Clase (`clase_multiclase`):** `Clase 0` ($<50\%$), `Clase 1` ($50\% - 79\%$) y `Clase 2` ($\ge 80\%$).
  * **Regresión Continua (`pct_pos_total`):** Estimación directa del porcentaje de reseñas positivas en el rango $[0, 100]$.

---

## Modelos
Clasificador Estadístico Gaussiano (PDF/CDF), Gaussian Naive Bayes (GNB), Decision Tree Classifier, Random Forest Classifier, Support Vector Machines (SVM - Kernel RBF), Linear Regression, Decision Tree Regressor, Support Vector Regression (SVR - Kernel RBF), Random Forest Regressor, Deep Learning, Multilayer Perceptron (MLP / Red Neuronal Densa), Ensamble Híbrido (RF + DNN).

---

## Enlaces y Estructura del Proyecto (Código, Presentaciones y Video)

### 1. Enlaces Principales
* **Repositorio Oficial (GitHub):** [https://github.com/JuanjoseJau/Machine-Learning-Proyecto](https://github.com/JuanjoseJau/Machine-Learning-Proyecto)
* **Notebook Principal y Definitivo (Entrega Final):** [`Proyecto_Avance2_JJJM final.ipynb`](./Proyecto_Avance2_JJJM%20final.ipynb)
* **Dataset en el Repositorio:** [`games_march2025_full.csv`](./games_march2025_full.csv)
* **Video de Sustentación Final:** *[Enlace al Video - Pendiente de grabación y publicación]*

### 2. Evolución de Notebooks de Código (`.ipynb`)
El repositorio conserva la trazabilidad completa de las tres entregas del curso de Inteligencia Artificial / Aprendizaje de Máquina (UIS - BIVL2AB):

1. **Primera Entrega (Avance 1 — EDA Inicial):** [`Proyecto_Avance1_JJJM.ipynb`](./Proyecto_Avance1_JJJM.ipynb)
   * Exploración inicial del catálogo de Steam, diagnóstico de valores faltantes, limpieza estructural de metadatos y análisis estadístico univariado.
2. **Segunda Entrega (Avance 2 — Enfoque Supervisado y Refinamiento EDA):** [`Proyecto_EDA_Steam_actualizado (3).ipynb`](./Proyecto_EDA_Steam_actualizado%20(3).ipynb)
   * Corrección de valores especiales y de fuga de datos (*Data Leakage*), ingeniería de las 17 características predictoras, análisis de distribuciones por clase (Notebook 05) y formulación del esquema supervisado (Notebooks 06 y 07).
3. **Entrega Final (Avance 3 — Proyecto Completo Consolidado y Funcional):** [`Proyecto_Avance2_JJJM final.ipynb`](./Proyecto_Avance2_JJJM%20final.ipynb)
   * **Notebook principal y definitivo del proyecto.** Integra todo el recorrido histórico de los Avances 1 y 2 con:
     * Análisis interpretativos detallados en Markdown debajo de cada experimento estadístico y gráfica de validación.
     * Modelado supervisado completo: Clasificación Binaria, Clasificación Multi-Clase (3 niveles) y Regresión Supervisada (`pct_pos_total`).
     * **Cierre Funcional (Celda 36):** Auditoría de errores en Test sobre 3,000 videojuegos reales (identificación de *Falsos Positivos* y *Joyas Ocultas / Falsos Negativos*) y curva de calibración del umbral de decisión (compromiso entre TPR, TNR, Precisión y F1-Score).
     * **Simulador Pre-Lanzamiento "¿Qué pasa si...?" (Celda 37):** Herramienta funcional que evalúa cómo evoluciona la probabilidad de éxito de un videojuego prototipo a medida que el estudio desarrollador añade logros, guardado en la nube (*Steam Cloud*), soporte de categorías, localización internacional de idiomas y ajuste competitivo de precio.
     * **Deep Learning — Red Neuronal Densa Multicapa MLP (Celdas 38 y 39, basadas en el Notebook 09):** Estandarización estricta (`StandardScaler`), arquitectura secuencial con capas ocultas `Dense(32, ReLU)` + `Dropout(0.20)` + `Dense(16, ReLU)` + `Dense(1, Sigmoid)`, control de sobreajuste con `EarlyStopping`, rescate de juegos exitosos (*Joyas Ocultas*) y diagnóstico pre-lanzamiento mediante **Consenso Híbrido (Random Forest + Red Neuronal Densa)**.

### 3. Presentaciones de Sustentación (`.pptx`)
* **Presentación Avance 1 (Actualizada):** [`Proyecto_Avance1_JJJM_Actualizado.pptx`](./Proyecto_Avance1_JJJM_Actualizado.pptx) *(Nota: Se conserva únicamente como referencia la versión actualizada de la primera entrega; el archivo histórico `Proyecto_Avance1_JJJM.pptx` permanece en la carpeta solo por respaldo).*
* **Presentación Avance 2 (Actualizada):** [`Proyecto_Avance2_JJJM_Actualizado.pptx`](./Proyecto_Avance2_JJJM_Actualizado.pptx)
* **Presentación 3 (Sustentación Final):** *[Pendiente por anexar / En preparación: `Proyecto_Avance3_JJJM_Final.pptx`]*

---

## Instrucciones de Ejecución Local y Comando `pip install` Completo

Los notebooks están diseñados para ejecutarse directamente en **Google Colab** o en un **entorno local** (VS Code, Jupyter Lab, Jupyter Notebook, Anaconda o Antigravity).

### 1. Ubicación del Dataset
Asegúrate de que el archivo **`games_march2025_full.csv`** se encuentre exactamente en la **misma carpeta raíz** que los notebooks (`entrega/`).

### 2. Creación de Entorno Virtual Limpio (Opcional)
Antes de instalar las librerías en tu máquina local, puedes crear y activar un entorno virtual limpio ejecutando:

```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activar en Windows (CMD):
venv\Scripts\activate.bat

# Activar en Linux / macOS:
source venv/bin/activate
```

### 3. Instalación de Librerías (`pip install`)
Si no ejecutas el proyecto en Google Colab sino en un entorno local, ejecuta el siguiente comando `pip` para instalar todas y cada una de las librerías necesarias:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow ipython jupyter notebook ipykernel
```

---

## Estructura de Archivos del Repositorio

```text
entrega/
├── games_march2025_full.csv                 # Dataset completo de Steam (89,618 videojuegos, 47 columnas)
├── Proyecto_Avance1_JJJM.ipynb              # Primera Entrega (Avance 1 - EDA Inicial)
├── Proyecto_EDA_Steam_actualizado (3).ipynb # Segunda Entrega (Avance 2 - Enfoque Supervisado)
├── Proyecto_Avance2_JJJM final.ipynb        # Entrega Final (Avance 3 - Notebook Principal Consolidado + Deep Learning)
├── Proyecto_Avance1_JJJM_Actualizado.pptx   # Presentación Avance 1 (Actualizada - Referencia oficial)
├── Proyecto_Avance1_JJJM.pptx               # Presentación Avance 1 (Versión histórica inicial)
├── Proyecto_Avance2_JJJM_Actualizado.pptx   # Presentación Avance 2 (Actualizada)
├── LICENSE                                  # Licencia del proyecto
└── README.md                                # Documentación general según rúbrica oficial UIS (BIVL2AB)
```
