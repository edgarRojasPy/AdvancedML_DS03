import numpy as np
import pandas as pd
from datetime import datetime
import re

def cargar_datos(ruta_archivo):
    # Carga los datos del archivo CSV utilizando Numpy
    datos =  np.genfromtxt(ruta_archivo, delimiter=',', skip_header=1,names=True)
    return datos
def cargar_datos_pd(ruta_archivo):
    # Carga los datos del archivo CSV utilizando Pandas.
    datos = pd.read_csv(ruta_archivo, sep=',',header=0)
    return datos
def ver_resumen_nulos(df):
    qna=df.isnull().sum(axis=0)
    qsna=df.shape[0]-qna
    
    ppna=round(100*(qna/df.shape[0]),2)
    aux= {'datos sin NAs en q': qsna, 'Na en q': qna ,'Na en %': ppna}
    na=pd.DataFrame(data=aux)
    resumen_nulos =na.sort_values(by='Na en %',ascending=False)
    return resumen_nulos

def es_fecha_valida(fecha_str, formato="%Y-%m-%d"):
    try:
        datetime.strptime(fecha_str, formato)
        return True
    except ValueError:
        return False
def obtener_filas_no_numericas(df, columnas):
    # Filtra las filas que tienen valores no numéricos en alguna de las columnas especificadas
    filas_no_numericas = df[~df[columnas].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
    return filas_no_numericas

def limpiar_letras_de_numeros(df,columnas):
    # Iterar sobre las columnas y aplicar la extracción de la parte numérica
    for columna in columnas:
        df[columna] = df[columna].astype(str).str.extract(r'(\d+(\.\d+)?)')[0].astype(float)

def obtener_filas_no_fechas(df, columnas, formato):
    """ by chatgpt
    Verifica si las fechas en las columnas especificadas del DataFrame son válidas según un formato dado.

    Args:
    df (pd.DataFrame): El DataFrame a verificar.
    columnas (list): Lista de nombres de columnas a verificar.
    formato (str): Formato de fecha a validar (ej. '%Y-%m-%d').

    Returns:
    pd.DataFrame: Un DataFrame que contiene solo las filas con fechas no válidas.
    """
    # Almacenar las filas con fechas no válidas
    filas_no_validas = pd.DataFrame()
    for columna in columnas:
        if columna in df.columns:
            # Convertir la columna a datetime con el formato especificado
            fechas_invalidas = pd.to_datetime(df[columna], format=formato, errors='coerce')
            # Filtrar las filas donde las fechas son NaT (no válidas)
            filas_invalidas = df[fechas_invalidas.isna()]
            # Agregar filas no válidas al DataFrame
            filas_no_validas = pd.concat([filas_no_validas, filas_invalidas], ignore_index=True)

    return filas_no_validas
# Función para encontrar valores atípicos POR el método de los cuartiles y el rango intercuartílico (IQR). chatgpt
def identificar_atipicos_IQR(df, columnas):
    atipicos = pd.DataFrame()  # DataFrame para almacenar filas con valores atípicos
    
    for columna in columnas:
        # Calcular Q1, Q3 y IQR
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para valores atípicos
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Filtrar filas que tienen valores atípicos
        filas_atipicas = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        
        # Concatenar filas atípicas al DataFrame de atípicos
        atipicos = pd.concat([atipicos, filas_atipicas])
    
    return atipicos.drop_duplicates()  # Eliminar duplicados

def ver_diccionario(titulo,subtitulos):
    # Imprimir el título
    print(titulo)
    print("-" * len(titulo))
    
    # Imprimir cada subtítulo
    for subtitulo in subtitulos:
        print(subtitulo)
def imprimir_bigotes(serie):
    """
    Calcula e imprime los bigotes de una serie de datos.

    Parameters:
    serie (pd.Series): La serie de datos para la cual se calcularán los bigotes.

    Returns:
    None: Imprime los valores de los bigotes.
    """
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1

    # Calcular los bigotes
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    print(f"Límite inferior (bigote inferior): {limite_inferior}")
    print(f"Límite superior (bigote superior): {limite_superior}")