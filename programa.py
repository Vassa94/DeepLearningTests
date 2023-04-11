import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pyttsx3

# Descarga los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

# Inicializa el sintetizador de voz
sintetizador = pyttsx3.init()

# Función para responder preguntas
def responder_pregunta(pregunta):
    # Tokeniza la pregunta en palabras y oraciones
    palabras = word_tokenize(pregunta)
    oraciones = sent_tokenize(pregunta)

    # Elimina las palabras vacías y las stopwords
    palabras = [palabra for palabra in palabras if palabra.lower() not in stopwords.words('spanish')]

    # Agrega reglas para responder preguntas específicas
    if 'capital' in palabras and 'españa' in palabras:
        respuesta = 'La capital de España es Madrid.'
    else:
        respuesta = 'Lo siento, no sé la respuesta a esa pregunta.'

    # Sintetiza la respuesta en voz
    sintetizar_voz(respuesta)

    return respuesta

# Función para sintetizar la respuesta en voz
def sintetizar_voz(respuesta):
    sintetizador.say(respuesta)
    sintetizador.runAndWait()

# Ejemplo de uso de la función
pregunta = '¿Cuál es la capital de España?'
respuesta = responder_pregunta(pregunta)
print(respuesta)
