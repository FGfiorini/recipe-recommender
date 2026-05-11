from fastapi import FastAPI
import pandas as pd
from src import config, rag_engine, model_training
import uvicorn

app = FastAPI(title="Recipe Innovation API")

# Cargamos los datos necesarios al iniciar
df_ready = pd.read_csv(config.DATA_PATH)
# Aquí podrías cargar un modelo Word2Vec ya entrenado en lugar de re-entrenar

@app.get("/generate_innovation")
async def get_recipe():
    # 1. Ejecutamos la lógica de recomendación (simplificada para la API)
    # En un caso real, leerías los ingredientes estrella desde MLflow o un JSON
    # Para este snippet, asumimos que usamos los datos cargados
    
    # Supongamos que ya tienes los índices de las mejores recetas
    # df_rec = ... (lógica de model_training)
    
    # 2. Generar con RAG
    # recipe = rag_engine.generate_innovative_recipe(df_rec, config.MODEL_NAME_OLLAMA)
    
    return {"recipe": "Aquí aparecerá la receta generada por el RAG"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)