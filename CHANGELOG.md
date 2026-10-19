# Historial de cambios - Actividad5JCP

## [1.0.1] - Mejoras de portabilidad y reproducibilidad

### Corregido

- Celda 3: detección del repositorio sin depender de nombres
  específicos de carpetas.
- Celda 25: corrección de la variable `base` por `BASE`.
- Conservación de las 64 celdas del notebook original.

### Validación independiente

- Clonación del repositorio desde GitHub.
- Descarga y preparación del dataset desde Kaggle.
- Reproducción de 164,679 registros preparados.
- División de entrenamiento y prueba sin perfiles compartidos.
- Construcción de pipelines de ambos modelos.
- Validación cruzada con StratifiedGroupKFold.
- Optimización con RandomizedSearchCV.
- Reproducción de métricas finales a seis decimales.
- Reproducción exacta de la tabla de McNemar.
- Registro y recuperación de ambos modelos desde MLflow.
- Verificación de predicciones idénticas en 100 registros.

### Resultados reproducidos

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.740546 | 0.746591 |
| Precision | 0.742343 | 0.732839 |
| Recall | 0.867455 | 0.907203 |
| F1-score | 0.800037 | 0.810752 |
| ROC-AUC | 0.781445 | 0.794283 |

### Documentación

- Ampliación de la sección de reproducibilidad del README.
- Registro de las pruebas independientes realizadas.
- Documentación de las condiciones de ejecución del notebook.

### Pendiente

- Ejecución consecutiva completa de las 64 celdas
  desde una única sesión limpia.

---

## [1.0.0] - 2026-10-18

### Incorporado
- Estructura del proyecto de Machine Learning.
- Script de preparación `fuentes/datos_prep.py`.
- Script de entrenamiento `fuentes/train.py`.
- Notebook académico `fuentes/entrena.ipynb`.
- Archivo `requirements.txt`.
- Documentación del proyecto en `README.md`.
- Seguimiento de experimentos con MLflow.
- Registro y recuperación de modelos mediante skops.
- Resultados de validación cruzada y optimización.
- Matrices de confusión y curvas ROC.
- Comparación estadística mediante McNemar.

### Preparación de datos
- Análisis de 185,560 registros originales.
- Preparación de 164,679 registros.
- Construcción de la variable objetivo TARGET.
- Selección de 16 variables predictoras.
- Tratamiento de valores faltantes y anomalías.

### Modelado
- Regresión Logística.
- Random Forest.
- División por grupos con GroupShuffleSplit.
- Validación cruzada con StratifiedGroupKFold.
- Optimización mediante RandomizedSearchCV.

### Resultados finales

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.7405 | 0.7466 |
| Precision | 0.7423 | 0.7328 |
| Recall | 0.8675 | 0.9072 |
| F1-score | 0.8000 | 0.8108 |
| ROC-AUC | 0.7814 | 0.7943 |

### Mejoras de reproducibilidad
- Uso de random_state = 42.
- Centralización de rutas del proyecto.
- Registro de parámetros, métricas y artefactos.
- Verificación de recuperación de modelos.
- Exclusión de datasets y artefactos mediante .gitignore.
- Respaldos de código y notebook en Google Drive.

### Pendiente de verificación
- Ejecución completa del notebook desde un entorno limpio.
- Verificación final de la publicación en GitHub.
