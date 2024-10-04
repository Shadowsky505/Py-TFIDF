from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import os, numpy as np, re, math

title, alpha = "stories", 0.3  # Parámetros globales
folders = [x[0] for x in os.walk(os.getcwd() + '/' + title + '/')]
folders[0] = folders[0][:-1]

dataset, c = [], False
for i in folders:
    text = open(i + "/index.html", 'r').read().strip()
    file_name, file_title = re.findall('><A HREF="(.*)">', text), re.findall('<BR><TD> (.*)\n', text)
    if not c: file_name, c = file_name[2:], True
    dataset += [(i + "/" + f, t) for f, t in zip(file_name, file_title)]

N = len(dataset)

def print_doc(id):
    with open(dataset[id][0], 'r', encoding='cp1250') as f: print(dataset[id], f.read().strip())

# Funciones de preprocesamiento
def preprocess(data):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    data = re.sub(r"[^\w\s]", " ", data).lower()
    tokens = [stemmer.stem(num2words(w) if w.isdigit() else w) for w in word_tokenize(data) 
              if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

# Procesar textos y títulos
processed_text = [word_tokenize(preprocess(open(i[0], 'r', encoding="utf8", errors='ignore').read())) for i in dataset]
processed_title = [word_tokenize(preprocess(i[1])) for i in dataset]

# Frecuencia de documentos por palabra
DF = {w: set() for doc in processed_text + processed_title for w in doc}
for i, doc in enumerate(processed_text + processed_title):
    for w in set(doc): DF[w].add(i)
DF = {k: len(v) for k, v in DF.items()}

# Calcular TF-IDF
def calculate_tf_idf(docs):
    tf_idf = {}
    for i, tokens in enumerate(docs):
        words_count = len(tokens)
        counter = Counter(tokens)
        for token in set(tokens):
            tf = counter[token] / words_count
            idf = math.log((N + 1) / (DF.get(token, 0) + 1))
            tf_idf[i, token] = tf * idf
    return tf_idf

tf_idf, tf_idf_title = calculate_tf_idf(processed_text), calculate_tf_idf(processed_title)
tf_idf = {k: v * alpha + tf_idf_title.get(k, 0) for k, v in tf_idf.items()}

# Similitud coseno
def cosine_similarity(query):
    query_tokens = word_tokenize(preprocess(query))
    query_vector = gen_vector(query_tokens)
    d_cosines = [np.dot(query_vector, D[d]) / (np.linalg.norm(query_vector) * np.linalg.norm(D[d])) for d in range(N)]
    best_match = np.argmax(d_cosines)  # Obtener el índice con mayor similitud coseno
    return best_match

# Vector del query
def gen_vector(tokens):
    Q = np.zeros(len(DF))
    for w in tokens:
        tf, idf = tokens.count(w) / len(tokens), math.log((N + 1) / (DF.get(w, 0) + 1))
        Q[total_vocab.index(w)] = tf * idf
    return Q

# Matriz D de vectores
total_vocab = list(DF)
D = np.zeros((N, len(total_vocab)))
for (doc, token), value in tf_idf.items():
    D[doc, total_vocab.index(token)] = value

# Ejecutar similitud coseno y mostrar el archivo con mayor similitud
best_match_index = cosine_similarity("TINDERBOX")
print("Archivo con mayor similitud coseno:", dataset[best_match_index][0])
