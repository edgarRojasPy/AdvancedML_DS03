import numpy as np
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt

def cargar_datos(ruta_archivo):
    # Carga los datos del archivo CSV utilizando Numpy
    datos =  np.genfromtxt(ruta_archivo, delimiter=',', skip_header=1,names=True)
    return datos
import pandas as pd

def cargar_datos_pd(ruta_archivo, sep=',', header=0):
    """
    Carga los datos del archivo CSV utilizando Pandas.

    Parameters:
    ruta_archivo (str): Ruta del archivo CSV que se va a cargar.
    sep (str): Separador utilizado en el archivo. Por defecto es ','.
    header (int or None): Índice de la fila que contiene los nombres de las columnas. 
                          Por defecto es 0.

    Returns:
    pd.DataFrame: Un DataFrame con los datos cargados del archivo CSV.

    Raises:
    FileNotFoundError: Si no se encuentra el archivo en la ruta especificada.
    Exception: Si ocurre algún otro error durante la carga.
    """
    try:
        # Carga los datos del archivo CSV utilizando Pandas.
        datos = pd.read_csv(ruta_archivo, sep=sep, header=header)
        print("Dataset cargado con éxito.")
        return datos
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en: {ruta_archivo}")
        raise
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        raise

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
def dimensiones(df):
    # Mostrar las primeras filas
    print("Dimensiones del dataset:")
    print(f"Features: {df.shape[1]}, Ejemplos: {df.shape[0]}")
    print("Primeras 5 filas del dataset:")
    df.head()

def plot_varianza_previo(prm_pca):
    # Graficar varianza explicada
    plt.figure(figsize=(12, 6))

    # 1. Varianza explicada por cada componente
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.bar(range(1, len(prm_pca.explained_variance_ratio_) + 1), prm_pca.explained_variance_ratio_)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')

    # 2. Varianza acumulada
    explained_variance_cumulative = prm_pca.explained_variance_ratio_.cumsum()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance_cumulative) + 1), explained_variance_cumulative, marker='o', linestyle='--')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Variación explicada acumulada')
    plt.title('Variación explicada acumulada')

    # Marcar el umbral del 90% como referencia
    threshold = 0.9
    optimal_components = np.argmax(explained_variance_cumulative >= threshold) + 1
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'90% Umbral')
    plt.axvline(x=optimal_components, color='g', linestyle='--', label=f'{optimal_components} componentes')
    # Agregar cuadrículas al gráfico de varianza acumulada
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Mostrar la varianza explicada acumulada para cada componente
    print("Variación explicada acumulada:")
    for i, variance in enumerate(explained_variance_cumulative, 1):
        print(f"Componente {i}: {variance:.2%}")
# Ejemplo de uso
# Asumiendo que 'prm_pca' es el modelo PCA ya ajustado con los datos
# plot_varianza(prm_pca)

def plot_varianza(prm_pca):
    # Graficar varianza explicada
    plt.figure(figsize=(12, 6))

    # 1. Varianza explicada por cada componente
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(prm_pca.explained_variance_ratio_) + 1), prm_pca.explained_variance_ratio_, color='skyblue')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')
    plt.grid(True)
    # Agregar las etiquetas de porcentaje a cada barra
    for i, v in enumerate(prm_pca.explained_variance_ratio_):
        plt.text(i + 1, v + 0.01, f'{v*100:.2f}%', ha='center')

    # 2. Varianza acumulada
    explained_variance_cumulative = prm_pca.explained_variance_ratio_.cumsum()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance_cumulative) + 1), explained_variance_cumulative, marker='o', linestyle='--', color='orange')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Variación Explicada Acumulada')
    plt.title('Variación Explicada Acumulada')
    plt.axhline(y=0.90, color='r', linestyle='--', label=f'90% Umbral')
    plt.axvline(x=np.argmax(explained_variance_cumulative >= 0.90) + 1, color='g', linestyle='--', label='Número óptimo de componentes')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Mostrar la varianza explicada acumulada para cada componente
    print("Variación explicada acumulada:")
    for i, variance in enumerate(explained_variance_cumulative, 1):
        print(f"Componente {i}: {variance:.2%}")

def plot_varianza1(prm_pca):
    # Graficar varianza explicada
    plt.figure(figsize=(15, 6))

    # 1. Varianza explicada por cada componente
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(1, len(prm_pca.explained_variance_ratio_) + 1), prm_pca.explained_variance_ratio_)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')

    # Agregar los porcentajes acumulados encima de las barras
    cumulative_variance = prm_pca.explained_variance_ratio_.cumsum()
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        # Muestra el porcentaje acumulado en cada barra
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{cumulative_variance[i]*100:.2f}%', 
                 ha='center', va='bottom', fontsize=9)

    # 2. Varianza acumulada
    explained_variance_cumulative = prm_pca.explained_variance_ratio_.cumsum()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance_cumulative) + 1), explained_variance_cumulative, marker='o', linestyle='--')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Acumulada')
    plt.title('Varianza Acumulada')

    # Marcar el umbral del 90% como referencia
    threshold = 0.9
    optimal_components = np.argmax(explained_variance_cumulative >= threshold) + 1
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'90% Umbral')
    plt.axvline(x=optimal_components, color='g', linestyle='--', label=f'{optimal_components} componentes')
    # Agregar cuadrículas al gráfico de varianza acumulada
    plt.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Delta o diferencial de la varianza explicada
    plt.subplot(1, 3, 3)
    delta_variance = np.diff(explained_variance_cumulative, prepend=0)  # Diferencial acumulado
    plt.plot(range(1, len(delta_variance) + 1), delta_variance, color='r', marker='o', linestyle='-')
    plt.xlabel('Componente Principal')
    plt.ylabel('Delta Varianza Acumulada')
    plt.title('Delta Varianza Acumulada')

    # Agregar porcentajes en cada nodo
    for i, v in enumerate(delta_variance):
        plt.text(i + 1, v, f'{v*100:.2f}%', color='r', ha='center', va='bottom', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_varianzax(prm_pca):
    # Graficar varianza explicada
    plt.figure(figsize=(15, 6))  # Aumenté el ancho de la figura

    # 1. Varianza explicada por cada componente
    explained_variance = prm_pca.explained_variance_ratio_
    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')
    """""
    bars = plt.bar(range(1, len(prm_pca.explained_variance_ratio_) + 1), prm_pca.explained_variance_ratio_)
    # Agregar los porcentajes acumulados encima de las barras
    cumulative_variance = prm_pca.explained_variance_ratio_.cumsum()
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        # Muestra el porcentaje acumulado en cada barra
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{cumulative_variance[i]*100:.2f}%', 
                 ha='center', va='bottom', fontsize=9)
    """
    # 2. Varianza acumulada
    explained_variance_cumulative = explained_variance.cumsum()
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(explained_variance_cumulative) + 1), explained_variance_cumulative, marker='o', linestyle='--')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Acumulada')
    plt.title('Varianza Acumulada')

    # Marcar el umbral del 90% como referencia
    threshold = 0.9
    optimal_components = np.argmax(explained_variance_cumulative >= threshold) + 1
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'90% Umbral')
    plt.axvline(x=optimal_components, color='g', linestyle='--', label=f'{optimal_components} componentes')

    # 3. Delta o diferencial de la varianza explicada
    plt.subplot(1, 3, 3)
    delta_variance = np.diff(explained_variance_cumulative, prepend=0)  # Diferencial acumulado
    plt.plot(range(1, len(delta_variance) + 1), delta_variance, color='r', marker='o', linestyle='-')
    plt.xlabel('Componente Principal')
    plt.ylabel('Delta Varianza Acumulada')
    plt.title('Delta Varianza Acumulada')

    # Agregar porcentajes en cada nodo
    for i, v in enumerate(delta_variance):
        plt.text(i + 1, v, f'{v*100:.2f}%', color='r', ha='center', va='bottom', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show()
