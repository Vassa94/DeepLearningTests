import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import json

# Cargar preguntas y respuestas desde un archivo JSON
with open('preguntas.json', 'r') as f:
    preguntas_y_respuestas = json.load(f)

def responder_pregunta(pregunta):
    palabras = pregunta.lower().split()
    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('spanish'))


    # Eliminar stopwords de la pregunta
    palabras = [p for p in palabras if p not in stop_words]

    for p, r in preguntas_y_respuestas.items():
        if all([palabra in p.lower() for palabra in palabras]):
            return r

    # Si la pregunta no está en la lista, solicitar respuesta al usuario y agregar al diccionario
    print("Lo siento, no sé la respuesta a esa pregunta.")
    respuesta_usuario = input("Por favor ingrese la respuesta: ")
    preguntas_y_respuestas[pregunta] = respuesta_usuario
    
    # Guardar preguntas y respuestas actualizadas en el archivo JSON
    with open('preguntas.json', 'w') as f:
        json.dump(preguntas_y_respuestas, f, indent=4)

    return "La respuesta ha sido guardada en la base de conocimiento."

# Ejemplo de uso de la función
pregunta = input("Ingresa tu pregunta: ")
respuesta = responder_pregunta(pregunta)
print(respuesta)
