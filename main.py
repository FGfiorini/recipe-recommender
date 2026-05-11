import mlflow
import pandas as pd
from src import config, preprocess, analytics, model_training, rag_engine, utils
from nltk.corpus import stopwords

def main():
    utils.setup_nltk()
    utils.check_ollama()
    
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # 1. Carga y Limpieza
        df_raw = pd.read_csv(config.DATA_PATH)
        df = preprocess.clean_data(df_raw)
        mlflow.log_param("total_recipes", len(df))

        # 2. Análisis de Tendencias
        tendencias = analytics.get_trend_ingredients(df, config.LISTA_RUIDO, config.BASICOS)
        mlflow.log_dict({"tendencias": tendencias}, "tendencias.json")

        # 3. Recomendador (Word2Vec)
        stop_words = set(stopwords.words('english'))
        stop_words.update(config.EXTRA_STOPS)
        
        df_rec, id_win, w2v_model = model_training.train_and_recommend(df, tendencias, stop_words)
        mlflow.log_metric("max_trend_score", df_rec['score_tendencia'].max())

        # 4. Generación RAG
        final_recipe = rag_engine.generate_innovative_recipe(df_rec, config.MODEL_NAME_OLLAMA)
        
        # Guardar resultado
        with open("receta_final.md", "w") as f:
            f.write(final_recipe)
        mlflow.log_artifact("receta_final.md")
        
        print("✨ Proceso completado. Receta generada y registrada en MLflow.")

if __name__ == "__main__":
    main()