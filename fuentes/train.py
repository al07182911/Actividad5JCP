"""
Proyecto: Clasificacion de polizas de seguros de vida
Actividad 5 - Master en Inteligencia Artificial

Modelos:
- Regresion Logistica
- Random Forest

Seguimiento de experimentos:
- MLflow
"""

from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from datos_prep import preparar_datos


# Directorio principal del proyecto
BASE = Path(__file__).resolve().parent.parent

# Rutas de datos
RUTA_DATOS = BASE / "datos/datos_ini/Kaggle.csv"

# Base de datos MLflow
RUTA_MLFLOW = BASE / "mlflow.db"

RUTA_ARTEFACTOS = BASE / "mlartifacts"

# Configuracion general
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

print("Configuracion inicial cargada correctamente.")

# ==================================================
# CARGA Y SELECCION DE DATOS
# ==================================================

VARIABLES_PREDICTORAS = [
    "CHANNEL1",
    "CHANNEL2",
    "CHANNEL3",
    "ENTRY AGE",
    "SEX",
    "POLICY TYPE 1",
    "POLICY TYPE 2",
    "POLICY TYPE 3",
    "PAYMENT MODE",
    "BENEFIT",
    "SUBSTANDARD RISK",
    "INITIAL BENEFIT",
    "Full Benefit?",
    "Premium",
    "Issue Year",
    "Issue Month"
]


def cargar_datos():

    # Verificar disponibilidad del dataset
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {RUTA_DATOS}"
        )

    # Aplicar preparacion reproducible
    df = preparar_datos(RUTA_DATOS)

    # Seleccionar variables predictoras
    X = df[VARIABLES_PREDICTORAS].copy()

    # Seleccionar variable objetivo
    y = df["TARGET"].copy()

    # Validaciones
    assert len(X) == len(y)
    assert X.shape[1] == 16
    assert y.notna().all()
    assert set(y.unique()) == {0, 1}

    print("\nDatos cargados correctamente.")
    print("Dimensiones de X:", X.shape)
    print("Dimensiones de y:", y.shape)

    return X, y

# ==================================================
# AGRUPACION Y DIVISION DE DATOS
# ==================================================

def dividir_datos(X, y):

    # Identificar perfiles con caracteristicas identicas
    grupos = X.groupby(
        list(X.columns),
        dropna=False,
        sort=False
    ).ngroup()

    # Configurar division por grupos
    divisor = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    idx_train, idx_test = next(
        divisor.split(X, y, groups=grupos)
    )

    # Crear conjuntos
    X_train = X.iloc[idx_train].copy()
    X_test = X.iloc[idx_test].copy()

    y_train = y.iloc[idx_train].copy()
    y_test = y.iloc[idx_test].copy()

    grupos_train = grupos.iloc[idx_train].copy()
    grupos_test = grupos.iloc[idx_test].copy()

    # Validar separacion
    grupos_compartidos = (
        set(grupos_train) & set(grupos_test)
    )

    assert len(grupos_compartidos) == 0
    assert len(X_train) + len(X_test) == len(X)

    print("\nDivision completada.")
    print("Entrenamiento:", len(X_train))
    print("Prueba:", len(X_test))
    print("Perfiles compartidos:", len(grupos_compartidos))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        grupos_train,
        grupos_test
    )

# ==================================================
# PREPROCESAMIENTO DE VARIABLES
# ==================================================

VARIABLES_NUMERICAS = [
    "ENTRY AGE",
    "BENEFIT",
    "SUBSTANDARD RISK",
    "INITIAL BENEFIT",
    "Premium",
    "Issue Year",
    "Issue Month"
]

VARIABLES_CATEGORICAS = [
    "CHANNEL1",
    "CHANNEL2",
    "CHANNEL3",
    "SEX",
    "POLICY TYPE 1",
    "POLICY TYPE 2",
    "POLICY TYPE 3",
    "PAYMENT MODE",
    "Full Benefit?"
]


def crear_preprocesadores():

    pipeline_num_logistica = Pipeline([
        ("imputador", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("escalador", StandardScaler())
    ])

    pipeline_num_forest = Pipeline([
        ("imputador", SimpleImputer(
            strategy="median",
            add_indicator=True
        ))
    ])

    pipeline_categorico = Pipeline([
        ("imputador", SimpleImputer(
            strategy="most_frequent"
        )),
        ("codificador", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocesador_logistica = ColumnTransformer([
        (
            "numericas",
            pipeline_num_logistica,
            VARIABLES_NUMERICAS
        ),
        (
            "categoricas",
            pipeline_categorico,
            VARIABLES_CATEGORICAS
        )
    ])

    preprocesador_forest = ColumnTransformer([
        (
            "numericas",
            pipeline_num_forest,
            VARIABLES_NUMERICAS
        ),
        (
            "categoricas",
            pipeline_categorico,
            VARIABLES_CATEGORICAS
        )
    ])

    return preprocesador_logistica, preprocesador_forest

# ==================================================
# CONSTRUCCION DE MODELOS OPTIMIZADOS
# ==================================================

def crear_modelos():

    pre_logistica, pre_forest = crear_preprocesadores()

    modelo_logistica = Pipeline([
        ("preprocesamiento", pre_logistica),
        ("clasificador", LogisticRegression(
            C=0.1,
            class_weight=None,
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    modelo_forest = Pipeline([
        ("preprocesamiento", pre_forest),
        ("clasificador", RandomForestClassifier(
            n_estimators=30,
            max_depth=16,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return {
        "Regresion_Logistica": modelo_logistica,
        "Random_Forest": modelo_forest
    }

# ==================================================
# VALIDACION CRUZADA
# ==================================================

def validar_modelo(modelo, X_train, y_train, grupos_train):

    # Configurar validacion cruzada
    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Metricas estandarizadas
    metricas = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    # Ejecutar validacion cruzada
    resultados = cross_validate(
        modelo,
        X_train,
        y_train,
        groups=grupos_train,
        cv=cv,
        scoring=metricas,
        n_jobs=1,
        error_score="raise"
    )

    # Resultados por particion
    tabla = pd.DataFrame({
        metrica: resultados[f"test_{metrica}"]
        for metrica in metricas
    })

    tabla.index = [
        f"Fold_{i}"
        for i in range(1, CV_FOLDS + 1)
    ]

    # Estadisticas descriptivas
    promedios = tabla.mean()
    desviaciones = tabla.std(ddof=1)

    return tabla, promedios, desviaciones

# ==================================================
# EVALUACION FINAL DEL MODELO
# ==================================================

def evaluar_modelo(modelo, X_test, y_test):

    # Generar predicciones
    predicciones = modelo.predict(X_test)

    # Probabilidades de la clase positiva (Lapse)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    # Calcular metricas
    resultados = {
        "accuracy": accuracy_score(
            y_test, predicciones
        ),
        "precision": precision_score(
            y_test, predicciones,
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predicciones
        ),
        "f1": f1_score(
            y_test, predicciones
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilidades
        )
    }

    return resultados

# ==================================================
# REGISTRO AUTOMATICO EN MLFLOW
# ==================================================

def registrar_experimento(
    nombre,
    modelo,
    tabla_cv,
    promedios,
    desviaciones,
    metricas_test
):

    # Configurar almacenamiento SQLite
    # Configurar experimento MLflow
    configurar_mlflow()

    # Registrar ejecucion
    with mlflow.start_run(
        run_name=f"{nombre}_Reproducible"
    ) as run:

        # Configuracion general

        # Registrar pipeline completo en MLflow
        # Se autorizan tipos internos de modelos propios
        tipos_confiables = [
            "numpy.dtype",
            "sklearn.compose._column_transformer._RemainderColsList",
            "sklearn.tree._tree.Tree"
        ]

        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            serialization_format="skops",
            skops_trusted_types=tipos_confiables
        )

        mlflow.log_params({
            "algoritmo": nombre,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "test_size": TEST_SIZE,
            "metrica_principal": "f1"
        })

        # Hiperparametros del clasificador
        parametros = modelo.named_steps[
            "clasificador"
        ].get_params()

        for clave in [
            "C",
            "max_iter",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight"
        ]:
            if clave in parametros:
                mlflow.log_param(
                    clave,
                    str(parametros[clave])
                )

        # Metricas de validacion cruzada
        for metrica in promedios.index:

            mlflow.log_metric(
                f"cv_{metrica}_mean",
                float(promedios[metrica])
            )

            mlflow.log_metric(
                f"cv_{metrica}_std",
                float(desviaciones[metrica])
            )

        # Metricas de prueba
        for metrica, valor in metricas_test.items():
            mlflow.log_metric(
                f"test_{metrica}",
                float(valor)
            )

        # Guardar tabla de validacion
        carpeta_resultados = BASE / "resultados"
        carpeta_resultados.mkdir(
            parents=True,
            exist_ok=True
        )

        ruta_resultados = (
            carpeta_resultados /
            f"resultados_cv_{nombre}.csv"
        )

        tabla_cv.to_csv(ruta_resultados)

        mlflow.log_artifact(
            str(ruta_resultados),
            artifact_path="validacion_cruzada"
        )

        print(
            f"Experimento registrado: {nombre}"
        )

        print("Run ID:", run.info.run_id)

# ==================================================
# PROCEDIMIENTO PRINCIPAL
# ==================================================

def main():

    print("=" * 55)
    print("CLASIFICACION DE POLIZAS DE SEGUROS")
    print("=" * 55)

    # 1. Cargar datos
    X, y = cargar_datos()

    # 2. Dividir datos
    (
        X_train,
        X_test,
        y_train,
        y_test,
        grupos_train,
        grupos_test
    ) = dividir_datos(X, y)

    # 3. Crear modelos optimizados
    modelos = crear_modelos()

    # 4. Ejecutar experimentos
    for nombre, modelo in modelos.items():

        print("\n" + "=" * 55)
        print("MODELO:", nombre)

        # Validacion cruzada
        tabla_cv, promedios, desviaciones = validar_modelo(
            modelo,
            X_train,
            y_train,
            grupos_train
        )

        print("\nF1 promedio CV:")
        print(round(promedios["f1"], 4))

        # Entrenar con todos los datos de entrenamiento
        modelo.fit(X_train, y_train)

        # Evaluar sobre el conjunto de prueba
        metricas_test = evaluar_modelo(
            modelo,
            X_test,
            y_test
        )

        print("\nMetricas finales:")
        for metrica, valor in metricas_test.items():
            print(metrica, round(valor, 4))

        # Registrar resultados en MLflow
        registrar_experimento(
            nombre,
            modelo,
            tabla_cv,
            promedios,
            desviaciones,
            metricas_test
        )

    print("\nProceso completado correctamente.")




# ==================================================
# CONFIGURACION DE MLFLOW
# ==================================================

def configurar_mlflow():

    # Preparar directorio de artefactos
    RUTA_ARTEFACTOS.mkdir(
        parents=True,
        exist_ok=True
    )

    # Configurar almacenamiento SQLite
    mlflow.set_tracking_uri(
        f"sqlite:///{RUTA_MLFLOW}"
    )

    nombre_experimento = "Clasificacion_Polizas_Seguros"

    # Buscar experimento existente
    experimento = mlflow.get_experiment_by_name(
        nombre_experimento
    )

    # Crear experimento si no existe
    if experimento is None:
        mlflow.create_experiment(
            name=nombre_experimento,
            artifact_location=RUTA_ARTEFACTOS.as_uri()
        )

    mlflow.set_experiment(nombre_experimento)

    return nombre_experimento


if __name__ == "__main__":
    main()
