# Historial de cambios - Actividad5JCP

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
