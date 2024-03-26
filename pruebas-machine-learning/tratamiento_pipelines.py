import pandas as pd
import numpy as np



#funcion para cualquier columna

def tratar_columna_negativos(df, column):
    df[column] = df[column].apply(lambda x: np.nan if x < 0 else x) #reemplazar valores negativos por 0
    return df

def remove_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std() #calcular z-score
    df = df[(z_scores < threshold) & (z_scores > -threshold)] #eliminar outliers
    return df

#mapeos de datos

def map_column_values(df, column, mapping_dict):
    df[column].apply(lambda value: mapping_dict.get(value.lower().strip(), np.nan)if value is not np.nan else np.nan)  #mapear valores
    return df #get retorna valor dada una clave, strip elimina los espacios que hay en una cadena.

#rellenar valores vacíos

def llenar_na(df, column, fill_value):
    df[column].fillna(fill_value, inplace=True) #rellenar valores faltantes
    return df


def process_data(df):
    education_mapping = {
        "Bachelors": "Universitario",
        "matre":"Maestria",
        "pHd":"Doctorado",
        "no educatio":"Sin educacion"
    }

    gender_mapping = {
        "m": "Masculino",
        "f": "Femenino"
    }
    
    
    return (
        df.pipe(tratar_columna_negativos, "Edad")
        .pipe(tratar_columna_negativos, "Ingresos")
        .pipe(tratar_columna_negativos, "Hijos")
        .pipe(remove_outliers, "Edad")
        .pipe(remove_outliers, "Ingresos")
        .pipe(remove_outliers, "Hijos")
        .pipe(remove_outliers, "Altura")
        .pipe(map_column_values, "Nivel_Educación", education_mapping)
        .pipe(map_column_values, "Género", gender_mapping)
        .pipe(llenar_na, "Nivel_Educación", "Sin educacion")
        .pipe(llenar_na, "Ciudad", df["Ciudad"].mode())
        .pipe(llenar_na, "Género", "IDK")
    )
    
df = pd.read_csv('pruebas-machine-learning/tratamiento_datos.csv', index_col=0) #leer dataset
print(df) #imprimir dataset

df = process_data(df)
print(df) #imprimir dataset