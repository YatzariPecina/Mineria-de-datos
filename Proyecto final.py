#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os

from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
nip = spacy.load('es_core_news_sm')


# In[39]:


#Importacion y revision
path_data = "D:/inteligencia de negocios/Mineria de datos/data"
df = pd.read_excel(path_data + "/data.xlsx")

#Eliminar filas con valores nulos
df = df.dropna()
#Quitar la marca temporal
df = df.drop('Marca temporal', axis=1)
print(df)


# In[26]:


print(df.describe())


# In[27]:


#Separacion de datos
df_numerico = df.select_dtypes(include=['number'])
df_texto = df.select_dtypes(include=['object', 'string'])


# In[28]:


print(df_numerico)
print(df_texto)


# In[29]:


#Análisis exploratorio de datos (AED)
print(df_numerico.mean())
print(df_numerico.median())
print(df_numerico.std())


# In[30]:


#Histograma
for column in df_numerico.head():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_numerico[column].dropna(), kde=True)
    plt.title(f"Distribución de {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[31]:


# Normalizar texto eliminando acentos y caracteres especiales
def normalize_text(text):
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
# Procesar cada archivo de preguntas
import re
# Procesar cada registro en el DataFrame
for idx, row in df_texto.iterrows():  # Iterar sobre las filas del DataFrame
    for col in df_texto.columns:  # Iterar sobre las columnas
        # Obtener el texto original de la celda
        linea = str(row[col])  # Asegurarse de que sea string

        # Normalizar y eliminar caracteres especiales
        correccion1 = normalize_text(linea)
        linea_sin_caracteres = re.sub(r'[^\w\s]', '', correccion1)

        # Tokenizar y eliminar stopwords
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]

        # Lematizar con spaCy
        doc = nip(' '.join(tokens_sin_stopwords))
        lemas = [token.lemma_ for token in doc]

        # Reemplazar el texto en la celda con los lemas separados por un espacio
        df_texto.at[idx, col] = ' '.join(lemas)


# In[32]:


# Combinar todo el texto en una sola cadena
texto_completo = " ".join(df_texto.astype(str).values.flatten())

# Crear la nube de palabras
nube_palabras = WordCloud(
    width=800, height=400, 
    background_color='white', 
    stopwords=STOPWORDS, 
    colormap='viridis'
).generate(texto_completo)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(nube_palabras, interpolation='bilinear')
plt.axis('off')
plt.title("Nube de Palabras", fontsize=16)
plt.show()


# In[33]:


#Aplicación de modelos de agrupación

# Estandarizar datos
scaler = StandardScaler()
datos_estandarizados = scaler.fit_transform(df_numerico)

# Método del codo para determinar el número óptimo de clústeres
inercia = []
silhouette_scores = []
rango_k = range(2, 11)

for k in rango_k:
    modelo_kmeans = KMeans(n_clusters=k, random_state=42)
    modelo_kmeans.fit(datos_estandarizados)
    inercia.append(modelo_kmeans.inertia_)
    silhouette_scores.append(silhouette_score(datos_estandarizados, modelo_kmeans.labels_))

# Visualización del método del codo
plt.figure(figsize=(10, 6))
plt.plot(rango_k, inercia, marker='o', linestyle='--', color='b')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Visualización de silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(rango_k, silhouette_scores, marker='o', linestyle='--', color='g')
plt.title('Silhouette Score')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Puntaje Silhouette')
plt.grid(True)
plt.show()

# Aplicar K-means con el número óptimo de clústeres
k_optimo = silhouette_scores.index(max(silhouette_scores)) + 2
modelo_final = KMeans(n_clusters=k_optimo, random_state=42)
df['Cluster'] = modelo_final.fit_predict(datos_estandarizados)

# Visualización de los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_numerico.columns[0], 
    y=df_numerico.columns[1], 
    hue=df['Cluster'], 
    palette='Set1', 
    data=df
)
plt.title(f'Visualización de Clústeres con k={k_optimo}')
plt.show()


# In[40]:


# Limpieza de nombres de columnas
df.columns = df.columns.str.strip()
print(df.columns.tolist())


# In[41]:


# Definir columnas según tu dataset
numeric_columns = [
    '¿Cuántos géneros diferentes sueles leer anualmente? (En caso de seleccionar la opción "Otra..." poner un número por favor)',
    '¿Cuántos libros (físico o digital) has terminado este año?   (En caso de seleccionar la opción "Otra..." poner un número por favor)',
    '¿Cuántas personas en tu familia o círculo cercano leen frecuentemente?    (En caso de seleccionar la opción "Otra..." poner un número por favor)',
    '¿Cuántas horas dedicas a leer diariamente?    (En caso de seleccionar la opción "Otra..." poner un número por favor)',
    '¿Cuántos días a la semana dedicas a la lectura?'
]

label_column = '¿Qué opinas sobre la lectura como pasatiempo?'

# Crear etiquetas de satisfacción
satisfaction_mapping = {
    "Me parece aburrido/a": 0,
    "Es una actividad neutral": 1,
    "Es interesante y disfrutable": 2,
    "Es una de mis actividades favoritas": 3
}

# Mapear etiquetas y manejar valores faltantes
df['Etiqueta'] = df[label_column].map(satisfaction_mapping)

# Eliminar registros con valores faltantes en X o y
df_clean = df.dropna(subset=numeric_columns + ['Etiqueta'])

# Preparar datos para entrenamiento
X = df_clean[numeric_columns]
y = df_clean['Etiqueta']

# Eliminar la columna de etiqueta original si ya no es necesaria
df_clean = df_clean.drop(label_column, axis=1)


# In[42]:


print(X)


# In[43]:


print(y)


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Verificar si hay suficientes muestras después de eliminar valores faltantes
if len(X) > 0 and len(y) > 0:
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar modelo SVM
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test_scaled)

    # Evaluar modelo
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Reporte de Clasificación:")
    print(classification_rep)

    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.show()
else:
    print("No hay suficientes datos después de eliminar valores faltantes.")
