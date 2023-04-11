import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


# Cargar preguntas y respuestas desde un archivo CSV
preguntas_y_respuestas = pd.read_csv('preguntas.csv')

# Preprocesar datos de entrada
X = []
stop_words = set(stopwords.words('spanish'))

for pregunta in preguntas_y_respuestas['Pregunta']:
    palabras = word_tokenize(pregunta.lower())
    palabras = [p for p in palabras if p not in stop_words]
    X.append(palabras)

# Preprocesar datos de salida
Y = []
for respuesta in preguntas_y_respuestas['Respuesta']:
    palabras = word_tokenize(respuesta.lower())
    Y.append(palabras)

# Convertir palabras en índices
palabras = set([palabra for pregunta in X for palabra in pregunta] + [palabra for respuesta in Y for palabra in respuesta])
palabra_a_indice = {palabra: indice for indice, palabra in enumerate(palabras)}
indice_a_palabra = {indice: palabra for palabra, indice in palabra_a_indice.items()}

X_indices = [[palabra_a_indice[palabra] for palabra in pregunta] for pregunta in X]
Y_indices = [[palabra_a_indice[palabra] for palabra in respuesta] for respuesta in Y]

# Crear matrices de entrada y salida para entrenamiento
max_len = max([len(pregunta) for pregunta in X_indices])
X_padded = np.zeros((len(X_indices), max_len))
for i, pregunta in enumerate(X_indices):
    X_padded[i, :len(pregunta)] = pregunta

max_len = max([len(respuesta) for respuesta in Y_indices])
Y_padded = np.zeros((len(Y_indices), max_len))
for i, respuesta in enumerate(Y_indices):
    Y_padded[i, :len(respuesta)] = respuesta

# LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences

model = Sequential()
model.add(Embedding(len(palabras), 100, input_length=X_padded.shape[1]))
model.add(LSTM(128))
model.add(Dense(len(palabras), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Ejemplo de uso de la función
def responder_pregunta(pregunta):
    # Preprocesar pregunta
    palabras = word_tokenize(pregunta.lower())
    palabras = [p for p in palabras if p not in stop_words]
    pregunta_indices = [palabra_a_indice[palabra] for palabra in palabras]
    pregunta_indices_padded = pad_sequences([pregunta_indices], maxlen=X_padded.shape[1], padding='post')
    
    # Obtener respuesta del modelo
    respuesta_indices_padded = model.predict(pregunta_indices_padded).argmax(axis=-1)
    respuesta_indices = [indice_a_palabra[indice] for indice in respuesta_indices_padded[0]]
    respuesta = ' '.join(respuesta_indices)
    
    return respuesta


# Cargar datos de CSV en un DataFrame de Pandas
data = pd.read_csv('preguntas.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Mostrar el número de ejemplos de entrenamiento y prueba
print("Ejemplos de entrenamiento: ", len(train_data))
print("Ejemplos de prueba: ", len(test_data))


# Entrenamiento del modelo
model.fit(X_padded, Y_padded, epochs=50, verbose=2)
    
# Guardar modelo entrenado
model.save('modelo_lstm.h5')
