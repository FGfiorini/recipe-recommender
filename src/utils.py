import nltk
import ollama
import subprocess
import time
import os

def setup_nltk():
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt')
    nltk.download('stopwords')

def check_ollama():
    try:
        ollama.list()
    except Exception:
        if os.name == 'nt':
            subprocess.Popen(["ollama", "serve"], creationflags=8, stdout=-1, stderr=-1)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=-1, stderr=-1)
        time.sleep(5)