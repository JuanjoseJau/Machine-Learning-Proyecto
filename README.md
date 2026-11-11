# Predicción de Viabilidad Comercial y Aceptación de Videojuegos en Steam mediante Machine Learning y Deep Learning

<p align="center">
  <img src="banner.jpeg" width="800" height="300" alt="Banner Proyecto Steam ML"/>
</p>

## Autores
Juan Jose Jauregui Mendoza (Código: 2250920)

---

## Objetivo
Predecir el éxito y nivel de aceptación del público (recepción positiva igual o superior al 70%) de un videojuego en la plataforma Steam a partir de sus características técnicas intrínsecas, alcance de localización y variables de diseño comercial (sin fuga de datos), integrando modelos de Machine Learning clásico y Redes Neuronales Densas (Deep Learning) en una herramienta funcional de simulación pre-lanzamiento.

---

## Dataset
El proyecto utiliza el conjunto de datos público **Steam Games Dataset (Marzo 2025)** almacenado en el archivo **games_march2025_full.csv**:

* **Fuente Oficial y Descarga (Kaggle):** [Steam Games Dataset 2025 — Kaggle (artermiloff)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_cleaned.csv)
* **Volumen Total:** **89,618 videojuegos** registrados en el catálogo de Steam.
* **Limpieza Estructural y Trazabilidad:**
  * Se partió de **47 columnas originales**, depuradas a **33 columnas limpias** tras descartar metadatos web no predictivos (enlaces de imágenes, capturas, videos, sitios web, correos de soporte) y columnas con valores nulos estructurales, conservando intactas las **89,618 filas** del catálogo.
  * **Eliminación de Fuga de Datos (Data Leakage):** Se excluyeron por completo de las variables predictoras todas las métricas de conteo directo de reseñas (**positive**, **negative**, **num_reviews_total**, **pct_pos_recent**, **num_reviews_recent** y **recommendations**), ya que a partir de ellas se calcula directamente el porcentaje de aceptación.

### Variable Objetivo Principal: Clasificación Binaria (clase_exito)
El enfoque central del proyecto modela el éxito comercial y de crítica de un videojuego como un problema de **Clasificación Binaria Supervisada**, dividiendo los 89,618 videojuegos a partir del umbral de aceptación del 70% sobre el porcentaje total de reseñas positivas (**pct_pos_total**):

| Clase | Definición en el Negocio | Condición sobre Aceptación | Cantidad de Juegos | Proporción del Catálogo |
| :--- | :--- | :--- | :---: | :---: |
| **Clase 1 (Éxito)** | Videojuego con **Alta Aceptación del Público** | Recepción positiva **≥ 70%** | 38,314 | **42.75%** |
| **Clase 0 (No Éxito)** | Videojuego con **Baja Aceptación o Sin Tracción** | Recepción positiva **< 70%** | 51,304 | **57.25%** |
| **Total** | Catálogo completo analizado | **0% a 100%** | **89,618** | **100.00%** |

> **Formulaciones Complementarias Exploradas en el Proyecto:**
> * **Clasificación Multi-Clase (3 niveles de recepción):** **Clase 0 — Baja** (menor al 50%, 45.28% del catálogo), **Clase 1 — Mixta/Moderada** (entre 50% y 79%, 22.71% del catálogo) y **Clase 2 — Aclamado** (igual o superior al 80%, 32.01% del catálogo).
> * **Regresión Supervisada Continua:** Predicción directa del porcentaje continuo de reseñas positivas (**pct_pos_total**, en la escala de 0% a 100%).

### Ingeniería de Características: Las 17 Variables Predictoras (Sin Fuga de Datos)
Para predecir la clase de éxito utilizando únicamente información de diseño, soporte técnico, localización y alcance comercial, se construyeron **17 características predictoras**:

| # | Variable Predictora | Categoría | Descripción y Transformación Aplicada |
| :-: | :--- | :--- | :--- |
| 1 | **price** | Comercial | Precio de venta del videojuego en dólares (USD). |
| 2 | **dlc_count** | Contenido | Cantidad de contenidos descargables (DLCs) publicados para el juego. |
| 3 | **log_logros** | Progresión | Transformación logarítmica log(1 + achievements) del número de logros en Steam. |
| 4 | **num_idiomas** | Localización | Número total de idiomas soportados en interfaz y subtítulos. |
| 5 | **num_idiomas_audio** | Localización | Número de idiomas con doblaje de voz completo (full audio). |
| 6 | **num_categorias** | Ecosistema Steam | Total de funcionalidades técnicas habilitadas en Steam (logros, mando, cromos, nube, etc.). |
| 7 | **es_multijugador** | Mecánica | Indicador binario (1 = Sí, 0 = No) si incluye modos Multi-player o PvP. |
| 8 | **es_coop** | Mecánica | Indicador binario (1 = Sí, 0 = No) si incluye modo Cooperativo. |
| 9 | **tiene_cloud** | Soporte Técnico | Indicador binario (1 = Sí, 0 = No) si cuenta con guardado en la nube (Steam Cloud). |
| 10 | **es_indie** | Género | Indicador binario (1 = Sí, 0 = No) si pertenece al género Indie. |
| 11 | **es_accion** | Género | Indicador binario (1 = Sí, 0 = No) si pertenece al género Acción (Action). |
| 12 | **es_rpg** | Género | Indicador binario (1 = Sí, 0 = No) si pertenece al género Juego de Rol (RPG). |
| 13 | **soporte_mac_linux** | Multiplataforma | Nivel de soporte en sistemas operativos adicionales (0 = Solo Windows, 1 = Mac o Linux, 2 = Ambos). |
| 14 | **antiguedad_anios** | Mercado | Antigüedad del videojuego en años cumplidos hasta 2025 desde su fecha de lanzamiento. |
| 15 | **metacritic_score** | Crítica Externa | Puntaje oficial de la prensa especializada en Metacritic (0 a 100). |
| 16 | **log_playtime** | Retención | Transformación logarítmica log(1 + average_playtime_forever) del tiempo promedio jugado. |
| 17 | **log_compradores** | Tracción | Transformación logarítmica log(1 + compradores_estimados) calculada desde el punto medio de propietarios. |

---

## Modelos
Clasificador Estadístico Gaussiano (PDF/CDF), Gaussian Naive Bayes (GNB), Decision Tree Classifier, Random Forest Classifier, Support Vector Machines (SVM - Kernel RBF), Linear Regression, Decision Tree Regressor, Support Vector Regression (SVR - Kernel RBF), Random Forest Regressor, Deep Learning, Multilayer Perceptron (MLP / Red Neuronal Densa), Ensamble Híbrido (RF + DNN).

---

## Resumen de Resultados Experimentales y Hallazgos Clave

### 1. Desempeño en Clasificación Binaria (Validación Cruzada 5-Folds y Test Final)
Sobre una muestra estratificada de **15,000 videojuegos** dividida estrictamente en **60% Entrenamiento (9,000)**, **20% Validación (3,000)** y **20% Prueba Final Reservada (3,000)**:

| Modelo Supervisado | CV Accuracy (5-Folds) | CV Sensibilidad (TPR) | CV Especificidad (TNR) | CV F1-Score | Rol y Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaussian Naive Bayes (GNB)** | 0.647 ± 0.013 | 0.356 ± 0.042 | 0.864 ± 0.020 | 0.462 ± 0.038 | Línea base; asunción normal e independencia limita la detección de éxitos. |
| **Decision Tree (max_depth = 4)** | 0.670 ± 0.012 | 0.478 ± 0.040 | 0.814 ± 0.036 | 0.552 ± 0.023 | Reglas interpretables sin sobreajuste (evita la memorización de depth = None). |
| **SVM (Kernel RBF Escalado, C = 1.0)** | 0.674 ± 0.009 | 0.492 ± 0.017 | 0.811 ± 0.006 | 0.563 ± 0.015 | Superior al kernel lineal al capturar fronteras no lineales tras estandarizar. |
| **Random Forest (100 Árboles)** | **0.686 ± 0.011** | **0.590 ± 0.020** | **0.757 ± 0.010** | **0.616 ± 0.016** | **Mejor modelo clásico**; alcanza en Test Final **Accuracy = 0.682** y **F1 = 0.613**. |

### 2. Importancia de Variables: ¿Qué factores determinan el éxito en Steam?
El análisis de reducción de impureza de Gini en el bosque aleatorio demuestra que el éxito comercial y de recepción está gobernado por 5 factores principales que acumulan más del **63%** del poder de decisión:
1. **Precio de venta (price):** 16.72%
2. **Cantidad de logros implementados (log_logros):** 14.96%
3. **Antigüedad en el mercado (antiguedad_anios):** 13.66%
4. **Funcionalidades del ecosistema Steam (num_categorias):** 9.37%
5. **Localización de idiomas de texto/interfaz (num_idiomas):** 8.41%

### 3. Cierre Funcional, Simulador Pre-Lanzamiento y Deep Learning (Notebook 09)
* **Auditoría de Errores y Calibración de Umbral (Celda 36):** Se cruzaron las predicciones del conjunto de prueba (3,000 juegos) con los títulos reales de Steam para caracterizar **Falsos Positivos** (proyectos con buen soporte técnico pero rechazados por problemas de jugabilidad o monetización) y **Falsos Negativos ("Joyas Ocultas")** (juegos independientes austeros que lograron alta aceptación). Además, se construyó la curva de calibración de umbral que permite adaptar el modelo a dos perfiles: **Inversionista Conservador (umbral = 0.65, Especificidad > 90%)** o **Cazatalentos Indie (umbral = 0.40, Sensibilidad > 75%)**.
* **Simulador Pre-Lanzamiento "¿Qué pasa si...?" (Celda 37):** Función interactiva que evalúa un proyecto prototipo antes de su lanzamiento y cuantifica cómo sube su probabilidad de éxito paso a paso al añadir 35 logros, habilitar Steam Cloud y 6 categorías técnicas, localizar el juego a 8 idiomas y ajustar su precio a $9.99 USD.
* **Red Neuronal Densa Multicapa (MLP) y Consenso Híbrido (Celdas 38 y 39):** Implementación de una arquitectura de Deep Learning en TensorFlow/Keras (Entrada de 17 variables estandarizadas con StandardScaler -> Capa Densa de 32 neuronas ReLU -> Dropout 20% -> Capa Densa de 16 neuronas ReLU -> Salida Sigmoid con EarlyStopping). Mientras Random Forest realiza cortes ortogonales estrictos por variable, la Red Neuronal pondera la sinergia continua entre las 17 características, logrando **rescatar en Test decenas de "Joyas Ocultas"** que los árboles descartaban y ofreciendo un **Consenso Híbrido (RF + DNN)** para decisiones de publicación.

---

## Enlaces del Código, Video y Estructura de Entregas

### 1. Enlaces Principales del Proyecto
* **Repositorio Oficial en GitHub:** [https://github.com/JuanjoseJau/Machine-Learning-Proyecto](https://github.com/JuanjoseJau/Machine-Learning-Proyecto)
* **Notebook Final Consolidado (Código Principal):** [Proyecto_Avance2_JJJM final.ipynb](./Proyecto_Avance2_JJJM%20final.ipynb)
* **Dataset Completo en el Repositorio:** [games_march2025_full.csv](./games_march2025_full.csv)
* **Video de Sustentación Final:** *[Enlace al Video - Pendiente de grabación y publicación]*

### 2. Notebooks de Código (.ipynb) por Entrega
1. **Primera Entrega (Avance 1 — EDA Inicial):** [Proyecto_Avance1_JJJM.ipynb](./Proyecto_Avance1_JJJM.ipynb)
   * Exploración inicial del catálogo de Steam, diagnóstico de valores faltantes, limpieza de metadatos y análisis estadístico univariado.
2. **Segunda Entrega (Avance 2 — Enfoque Supervisado):** [Proyecto_EDA_Steam_actualizado (3).ipynb](./Proyecto_EDA_Steam_actualizado%20(3).ipynb)
   * Corrección de fuga de datos (Data Leakage), ingeniería de las 17 características predictoras, pruebas estadísticas por clase (Notebook 05) y modelos supervisados (Notebooks 06 y 07).
3. **Entrega Final (Avance 3 — Proyecto Completo Consolidado):** [Proyecto_Avance2_JJJM final.ipynb](./Proyecto_Avance2_JJJM%20final.ipynb)
   * **Notebook principal y definitivo del proyecto.** Contiene todo el desarrollo histórico más los análisis interpretativos detallados debajo de cada gráfica, clasificación binaria, clasificación multiclase, regresión supervisada, auditoría funcional de errores con juegos reales de Steam, calibración de umbral de decisión, Simulador Pre-Lanzamiento ("¿Qué pasa si...?") e implementación de Deep Learning (Red Neuronal Densa Multicapa — Notebook 09) con consenso híbrido.

### 3. Presentaciones de Sustentación (.pptx)
* **Presentación Avance 1 (Actualizada):** [Proyecto_Avance1_JJJM_Actualizado.pptx](./Proyecto_Avance1_JJJM_Actualizado.pptx) *(Nota: Se conserva únicamente como referencia oficial la versión actualizada de la primera entrega; el archivo previo Proyecto_Avance1_JJJM.pptx permanece en la carpeta solo como respaldo histórico).*
* **Presentación Avance 2 (Actualizada):** [Proyecto_Avance2_JJJM_Actualizado.pptx](./Proyecto_Avance2_JJJM_Actualizado.pptx)
* **Presentación 3 (Sustentación Final):** *[Pendiente por anexar / En preparación: Proyecto_Avance3_JJJM_Final.pptx]*

---

## Instrucciones de Ejecución Local y Comando pip install Completo

El proyecto puede ejecutarse directamente en **Google Colab** o en cualquier **entorno local** (VS Code, Jupyter Lab, Jupyter Notebook, Anaconda o Antigravity).

### 1. Requisito del Dataset en la Carpeta Raíz
Verifica que el archivo **games_march2025_full.csv** se encuentre ubicado exactamente en la **misma carpeta raíz** que los notebooks del repositorio.

### 2. Creación de un Entorno Virtual Limpio (Opcional)
Si deseas ejecutar el proyecto en un entorno aislado en tu computador antes de instalar las dependencias, abre una terminal en la carpeta del proyecto y ejecuta:

```bash
# Crear entorno virtual limpio
python -m venv venv

# Activar en Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activar en Windows (CMD):
venv\Scripts\activate.bat

# Activar en Linux / macOS:
source venv/bin/activate
```

### 3. Comando pip install con Todas las Librerías Necesarias
Si no ejecutas los notebooks en Google Colab sino en tu entorno local, ejecuta el siguiente comando para instalar todas las librerías requeridas por el proyecto (procesamiento de datos, visualización estadística, Machine Learning clásico con Scikit-Learn y Deep Learning con TensorFlow/Keras):

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow ipython jupyter notebook ipykernel
```

---

## Estructura de Archivos del Repositorio

```text
entrega/
├── banner.jpeg                              # Imagen de banner principal del proyecto (800 x 300)
├── games_march2025_full.csv                 # Dataset completo de Steam (89,618 videojuegos, 47 columnas)
├── Proyecto_Avance1_JJJM.ipynb              # Primera Entrega (Avance 1 - EDA Inicial)
├── Proyecto_EDA_Steam_actualizado (3).ipynb # Segunda Entrega (Avance 2 - Enfoque Supervisado)
├── Proyecto_Avance2_JJJM final.ipynb        # Entrega Final (Avance 3 - Notebook Principal Consolidado + Deep Learning)
├── Proyecto_Avance1_JJJM_Actualizado.pptx   # Presentación Avance 1 (Actualizada - Referencia oficial)
├── Proyecto_Avance1_JJJM.pptx               # Presentación Avance 1 (Versión histórica inicial)
├── Proyecto_Avance2_JJJM_Actualizado.pptx   # Presentación Avance 2 (Actualizada)
├── LICENSE                                  # Licencia MIT del repositorio
└── README.md                                # Documentación general del proyecto (Rúbrica UIS - BIVL2AB)
```
