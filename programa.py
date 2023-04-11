import nltk

nltk.download('all')

from nltk.tokenize import word_tokenize

sentence = "Hola, ¿cómo estás?"

tokens = word_tokenize(sentence)

print(tokens)
