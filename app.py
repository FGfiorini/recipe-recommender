import gradio as gr
import requests

def fetch_innovation():
    try:
        # Llamada al endpoint de FastAPI
        response = requests.get("http://localhost:8000/generate_innovation")
        if response.status_code == 200:
            return response.json()["recipe"]
        return "Error en la conexión con el servidor."
    except Exception as e:
        return f"Error: {str(e)}"

# Interfaz de Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍳 AI Culinary Innovator")
    gr.Markdown("Presiona el botón para generar una receta basada en ingredientes tendencia.")
    
    with gr.Row():
        btn = gr.Button("Generar Nueva Creación", variant="primary")
    
    output = gr.Markdown(label="Receta Sugerida")
    
    btn.click(fn=fetch_innovation, outputs=output)

if __name__ == "__main__":
    demo.launch(server_port=7860)