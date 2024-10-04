from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

# Parámetros globales
title = "stories" # Nombre del directorio que contiene los archivos
alpha = 0.3 # Factor de ponderación

# Lista de carpetas que contienen los documentos
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
folders[0] = folders[0][:len(folders[0])-1]

dataset = [] # Lista para almacenar los archivos y sus títulos
c = False

# Recorrer todas las carpetas y extraer información de cada archivo
for i in folders:
    file = open(i+"/index.html", 'r')
    text = file.read().strip()
    file.close()

    # Buscar nombres y títulos de los archivos en el texto
    file_name = re.findall('><A HREF="(.*)">', text)
    file_title = re.findall('<BR><TD> (.*)\n', text)

    # Ajustar los archivos si es la primera iteración
    if c == False:
        file_name = file_name[2:]
        c = True
        
    print(len(file_name), len(file_title))

    # Añadir a la lista de datos cada archivo con su título
    for j in range(len(file_name)):
        dataset.append((str(i) +"/"+ str(file_name[j]), file_title[j]))

len(dataset)
N = len(dataset) # Número total de documentos

# Función para imprimir el contenido de un documento dado su ID
def print_doc(id):
    print(dataset[id])
    file = open(dataset[id][0], 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    print(text)

# Convertir el texto a minúsculas
def convert_lower_case(data):
    return np.char.lower(data)

# Eliminar palabras de parada (stop words) del texto
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

# Eliminar signos de puntuación del texto
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

# Eliminar apóstrofes del texto
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

# Realizar el stemming del texto (reducir las palabras a su raíz)
def stemming(data):
    stemmer= PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

# Convertir números en palabras
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            pass
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

# Función de preprocesamiento completa que aplica todos los pasos anteriores
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) 
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) 
    data = remove_punctuation(data) 
    data = remove_stop_words(data) 
    return data

# Listas para almacenar los textos y títulos procesados
processed_text = []
processed_title = []

# Procesar cada documento y su título
for i in dataset[:N]:
    file = open(i[0], 'r', encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()

    processed_text.append(word_tokenize(str(preprocess(text))))
    processed_title.append(word_tokenize(str(preprocess(i[1]))))

# Crear un diccionario para almacenar la frecuencia de documentos por palabra
DF = {}

# Contar la frecuencia de palabras en los textos y títulos procesados
for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = processed_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])

# Tamaño del vocabulario total
total_vocab_size = len(DF)
total_vocab = [x for x in DF]
print(total_vocab[:20])

# Función para obtener la frecuencia de documentos de una palabra dada
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

# Calcular el TF-IDF para cada palabra en cada documento
tf_idf = {}

for i in range(N):
    tokens = processed_text[i]
    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        tf_idf[i, token] = tf*idf

# Calcular el TF-IDF para los títulos
tf_idf_title = {}

for i in range(N):
    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_title[i])
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        tf_idf_title[i, token] = tf*idf

# Ponderar los valores de TF-IDF
for i in tf_idf:
    tf_idf[i] *= alpha
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

len(tf_idf)

# Calcular el score de coincidencia para un query dado
def matching_score(k, query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    print("Matching Score")
    print("\nQuery:", query)
    print(tokens)
    query_weights = {}
    for key in tf_idf:
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    l = [i[0] for i in query_weights[:10]]
    print(l)

# Calcular la similitud coseno entre dos vectores
def cosine_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Matriz D para almacenar vectores de documentos
D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

# Generar el vector del query
def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))
        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

# Calcular la similitud coseno para un query dado
def cosine_similarity(k, query):
    print("Cosine Similarity")
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    d_cosines = []
    query_vector = gen_vector(tokens)
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
    out = np.array(d_cosines).argsort()[-k:][::-1]
    print(out)

Q = cosine_similarity(10, "Without the drive of Rebeccah's insistence, Kate lost her momentum. She stood next a slatted oak bench, canisters still clutched, surveying")

# Imprimir el contenido del documento de ID 200
print_doc(200)
