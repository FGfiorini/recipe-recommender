import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

def tokenize_directions(texto, stop_words):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    return [w for w in texto.split() if w not in stop_words and len(w) > 2]

def train_and_recommend(df, ingredientes_estrella, stop_words):
    # Filtro de tendencia
    def tiene_tendencia(lista):
        receta_norm = [str(i).strip().lower() for i in lista]
        return any(ing in receta_norm for ing in ingredientes_estrella)

    df_filtrado = df[df['clean_ingredients'].apply(tiene_tendencia)].copy()
    
    # Word2Vec
    df_filtrado['tokens_directions'] = df_filtrado['directions'].apply(lambda x: tokenize_directions(x, stop_words))
    model = Word2Vec(sentences=df_filtrado['tokens_directions'].tolist(), vector_size=100, window=5, min_count=1, sg=1)
    
    def get_recipe_vector(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100)

    df_filtrado['recipe_vector'] = df_filtrado['tokens_directions'].apply(get_recipe_vector)
    
    # Buscador de ganador por score
    df_filtrado['score_tendencia'] = df_filtrado['clean_ingredients'].apply(
        lambda x: sum(1 for ing in ingredientes_estrella if ing in [str(i).lower() for i in x])
    )
    
    id_ganador = df_filtrado['score_tendencia'].idxmax()
    vector_base = df_filtrado.loc[id_ganador, 'recipe_vector'].reshape(1, -1)
    
    # Similitud
    todos_vectores = np.stack(df_filtrado['recipe_vector'].values)
    sims = cosine_similarity(vector_base, todos_vectores).flatten()
    
    indices_finales = sims.argsort()[-11:-1][::-1]
    return df_filtrado.iloc[indices_finales], id_ganador, model


# =================================================================
# SIMULATION: UPLOAD GENERATED RECIPE TO GCP BUCKET
# =================================================================
# import os
# from google.cloud import storage

# def upload_recipe_to_gcp(recipe_text, recipe_name):
#     # 1. Configuración de credenciales y cliente
#     # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account.json"
#     # client = storage.Client()
#     # bucket = client.get_bucket('my-chef-ai-recipes')

#     # 2. Formatear el nombre del archivo (slug)
#     # filename = f"{recipe_name.lower().replace(' ', '_')}.txt"
#     # blob = bucket.blob(f"generated_recipes/{filename}")

#     # 3. Subir el contenido
#     # blob.upload_from_string(recipe_text)
#     # print(f"☁️ Recipe successfully saved to GCP: {filename}")
#     pass

# # Ejecución de la simulación
# # recipe_content = response['message']['content']
# # upload_recipe_to_gcp(recipe_content, "Symphony of Tropics")