
import pandas as pd
from pathlib import Path

def preparar_datos(ruta_entrada):

    # Cargar datos originales
    df = pd.read_csv(ruta_entrada, sep=";")

    print("Registros originales:", len(df))

    # Eliminar columnas completamente vacías
    df = df.drop(
        columns=["Unnamed: 20", "Unnamed: 21"],
        errors="ignore"
    )

    # Seleccionar estados para clasificación binaria
    df = df[
        df["POLICY STATUS"].isin(["Lapse", "Inforce"])
    ].copy()

    # Crear variable objetivo
    df["TARGET"] = df["POLICY STATUS"].map({
        "Lapse": 1,
        "Inforce": 0
    })


    # Convertir variables monetarias a formato numerico
    for columna in ["BENEFIT", "Premium"]:

        df[columna] = pd.to_numeric(
            df[columna].astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .replace("-", pd.NA),
            errors="coerce"
        )

    # Tratar primas negativas como valores faltantes
    df.loc[df["Premium"] < 0, "Premium"] = float("nan")


    # Normalizar variables categoricas
    columnas_categoricas = [
        "SEX",
        "POLICY TYPE 3",
        "PAYMENT MODE",
        "NON LAPSE GUARANTEED",
        "Full Benefit?"
    ]

    for columna in columnas_categoricas:
        df[columna] = df[columna].str.strip()

    # Corregir nombre de modalidad de pago
    df["PAYMENT MODE"] = df["PAYMENT MODE"].replace({
        "Quaterly": "Quarterly"
    })


    # Transformar fecha de contratacion
    df["Issue Date"] = pd.to_datetime(
        df["Issue Date"].astype(str).str.strip(),
        format="%b-%y",
        errors="coerce"
    )

    # Crear variables temporales
    df["Issue Year"] = df["Issue Date"].dt.year
    df["Issue Month"] = df["Issue Date"].dt.month


    # Excluir variable temporal redundante
    # Se conserva Policy Year (Decimal)
    # por ofrecer mayor precision temporal
    df = df.drop(columns=["Policy Year"])

    return df
