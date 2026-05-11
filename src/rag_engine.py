import ollama

def generate_innovative_recipe(df_rag, model_name):
    contexto = ""
    for _, row in df_rag.iterrows():
        contexto += f"\n--- REFERENCIA: {row['recipe_name']} ---\n"
        contexto += f"INGREDIENTES: {row['ingredients']}\n"
        contexto += f"INSTRUCCIONES: {row['directions']}\n"

    prompt_sistema = """
    You are a creative High-End Cuisine Chef. Design an innovative recipe based on the references.
    FORMAT: Recipe Name, Metadata, Ingredients (bulleted), Instructions.
    Output must be in English.
    """
    
    response = ollama.chat(model=model_name, messages=[
        {'role': 'system', 'content': prompt_sistema},
        {'role': 'user', 'content': f"Technical info:\n{contexto}"}
    ])
    return response['message']['content']