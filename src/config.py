import os

# Rutas de archivos
BASE_PATH = "/Users/franky/Desktop/Recipe recommender/datasets"
DATA_PATH = os.path.join(BASE_PATH, "recetas_ready.csv")
TENDENCIAS_JSON = os.path.join(BASE_PATH, "tendencias.json")
RAG_JSON = os.path.join(BASE_PATH, "recetas_para_rag.json")

# Configuración MLflow
EXPERIMENT_NAME = "Recipe_Innovation_RAG"

# Listas de limpieza y NLP
LISTA_RUIDO = [
    'salt', 'water', 'olive oil', 'oil', 'black pepper', 
    'garlic', 'onion', 'butter', 'sugar', 'flour', 'all-purpose flour'
]
BASICOS = ['salt', 'water', 'olive oil']
EXTRA_STOPS = {'step', 'minutes', 'seconds', 'approx', 'recipe', 'using', 'prepared', 'place', 'put'}

# Parámetros del Modelo
MODEL_NAME_OLLAMA = "llama3"
VECTOR_SIZE = 100