import pandas as pd

def get_trend_ingredients(df, ruido, basicos):
    # Sets de ingredientes
    dulces_ings = set([ing for lista in df[df['w_sugar'] == 1]['clean_ingredients'] for ing in lista])
    saladas_ings = set([ing for lista in df[df['w_sugar'] == 0]['clean_ingredients'] for ing in lista])
    interseccion = dulces_ings.intersection(saladas_ings)

    # Filtrar recetas saladas innovadoras
    recetas_tendencia = df[
        (df['w_sugar'] == 0) & 
        (df['clean_ingredients'].apply(lambda x: any(ing in interseccion for ing in x)))
    ]

    # Conteo filtrando ruido
    conteo = pd.Series([
        ing for lista in recetas_tendencia['clean_ingredients'] 
        for ing in lista if ing in interseccion and ing not in ruido
    ]).value_counts()
    
    return conteo.head(10).index.tolist()