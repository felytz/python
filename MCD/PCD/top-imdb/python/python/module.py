#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import urllib.request as urllib2
#import tarfile
#import os.path
import requests
#from os import path
from bs4 import BeautifulSoup
#import time

def cpm(path): # content per line of a text file
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read() 
        lines = content.splitlines()  # List: content per line
        return lines

def title(url):
    # Definir encabezados para la solicitud
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Obtener el contenido de la página web
    webpage = requests.get(url, headers=headers)
    
    # Revisar si la solicitud fue exitosa
    if webpage.status_code == 200:
        soup = BeautifulSoup(webpage.content, "html.parser")
        # Extraer el título de la película
        title_tag = soup.find("h1")
        movie_title = title_tag.get_text(strip=True) if title_tag else "Título no encontrado"
    return movie_title