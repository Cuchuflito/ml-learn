import pandas as pd
import numpy as np


df = pd.read_csv('pruebas-machine-learning/tratamiento_datos.csv', index_col=0) #leer dataset
print(df) #imprimir dataset

print(df.describe()) #imprimir estadisticas del dataset
print(df.info()) #imprimir informacion del dataset


set_genero= set(df['Género'].tolist()) #obtener los valores unicos de la columna genero
set_edu= set(df['Nivel_Educación'].tolist()) #obtener los valores unicos de la columna educacion
set_ciudad= set(df['Ciudad'].tolist()) #obtener los valores unicos de la columna ciudad#
# print(set_genero,set_edu,set_ciudad) #imprimir valores unicos


#Paso 1, tratar valores negativos.

df["Edad"] = df["Edad"].apply(lambda x: 0 if x < 0 else x) #reemplazar valores negativos por 0
df["Ingresos"] = df["Ingresos"].apply(lambda x: 0 if x < 0 else x) #reemplazar valores negativos por 0
df["Hijos"] = df["Hijos"].apply(lambda x: 0 if x < 0 else x) #reemplazar valores negativos por 0

#paso 2, imputar valores faltantes

for column in ['Edad', 'Ingresos', 'Hijos']:
    median_value = df[column].median() #calcular la mediana de la columna
    df[column].fillna(median_value, inplace=True) #rellenar los valores faltantes con la mediana
    
for column in["Género","Ciudad"]:
    mode_value = df[column].mode()[0] #calcular la moda de la columna
    df[column].fillna(mode_value, inplace=True) #rellenar los valores faltantes con la moda

#paso 3, mapeo de valores categoricos

education_mapping = {
    "Bachelors": "Universitario",
    "matre":"Maestria",
    "pHd":"Doctorado",
    "no educatio":"Sin educacion"
}
df["Nivel_Educación"].replace(education_mapping, inplace=True) #reemplazar valores de la columna educacion
df["Nivel_Educación"].fillna("Sin educacion", inplace=True) #rellenar valores faltantes de la columna educacion

#Casteo de tipos de datos
df["Edad"] = df["Edad"].astype(int) #cambiar el tipo de dato de la columna edad a entero
df["Hijos"] = df["Hijos"].astype(int) #cambiar el tipo de dato de la columna hijos a entero
df["Ingresos"] = df["Ingresos"].astype(float) #cambiar el tipo de dato de la columna ingresos a flotante
df["Altura"] = df["Altura"].astype(float) #cambiar el tipo de dato de la columna altura a flotante

print(df) #imprimir dataset
print(df.describe()) #imprimir estadisticas del dataset
print(df.info()) #imprimir informacion del dataset
