import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Descarga los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Función para responder preguntas
def responder_pregunta(pregunta):
    # Tokeniza la pregunta en palabras y oraciones
    palabras = word_tokenize(pregunta)
    oraciones = sent_tokenize(pregunta)

    # Elimina las palabras vacías y las stopwords
    palabras = [palabra for palabra in palabras if palabra.lower() not in stopwords.words('spanish')]

    print('Palabras después de eliminar las stopwords:', palabras)
    
    # Agrega reglas para responder preguntas específicas
    if 'capital' in palabras and 'españa' in palabras:
        return 'La capital de España es Madrid.'
    else:
        return 'Lo siento, no sé la respuesta a esa pregunta.'

# Ejemplo de uso de la función
pregunta = '¿Cuál es la capital de España?'
respuesta = responder_pregunta(pregunta)
print(respuesta)

