import pandas as pd
import re
from ingredient_parser import parse_ingredient


"""
======== Simulación obtencion de los ultimos 2 meses de recetas ========
import pandas as pd
from datetime import timedelta

# --- SIMULACIÓN DE FILTRADO POR FECHA (Últimos 2 meses) ---

# 1. Aseguramos que la columna 'date' sea de tipo datetime
df['date'] = pd.to_datetime(df['date'])

# 2. Obtenemos la fecha más reciente en el dataset (Simulación de 'hoy' en los datos)
latest_date = df['date'].max()

# 3. Calculamos la fecha de corte (hace 60 días)
cutoff_date = latest_date - pd.Timedelta(days=60)

# 4. Filtramos el dataframe original
# Seleccionamos recetas con rating > 4 Y que estén dentro del rango de los últimos 2 meses
df_recent = df[(df['rating'] > 4) & (df['date'] >= cutoff_date)].copy()

print(f"✅ Dataframe filtrado por fecha.")
print(f"📅 Rango: {cutoff_date.date()} a {latest_date.date()}")
print(f"📊 Recetas disponibles en este periodo: {len(df_recent)}")

# --- CONTINUACIÓN DEL PROCESO RAG ---

# Ahora usamos 'df_recent' para buscar tus recetas recomendadas
df_rag = df_recent[df_recent['recipe_name'].isin(nombres_seleccionados)]

# Si por el filtro de fecha alguna de las 10 recetas se pierde, 
# el proceso seguirá con las que queden disponibles en ese rango."""


def clean_data(df):
    # Parsing de ingredientes
    df['ingredients_list'] = df['ingredients'].apply(lambda x: [i.strip() for i in x.split(',')])
    
    def extract_ingredient_names(ingredients_list):
        parsed = [parse_ingredient(x) for x in ingredients_list]
        return [p.name[0].text for p in parsed if p.name]

    df['clean_ingredients'] = df['ingredients_list'].apply(extract_ingredient_names)
    
    # Manejo de fechas y azúcar
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['w_sugar'] = df['clean_ingredients'].apply(
        lambda lista: 1 if any('sugar' in ing.lower() for ing in lista) else 0
    )
    
    # Limpieza de tiempos
    columnas_tiempo = ['prep_time', 'cook_time', 'total_time']
    for col in columnas_tiempo:
        df[col] = df[col].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
    
    # Renombrar y filtrar calidad
    df = df.rename(columns={'prep_time': 'prep_time_min', 'cook_time': 'cook_time_min', 'total_time': 'total_time_min'})
    df = df.drop(columns=['url','cuisine_path','img_src','timing','yield'], errors='ignore')
    df = df[df['rating'] > 4].copy()
    
    return df.reset_index(drop=True)