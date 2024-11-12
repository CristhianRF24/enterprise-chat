import requests
import os
from dotenv import load_dotenv

load_dotenv()

GRAPHDB_URL = os.getenv("GRAPHDB_URL")
REPOSITORY_NAME = os.getenv("REPO_NAME")

def load_ttl_to_graphdb(file_path):
    url = f"{GRAPHDB_URL}/repositories/{REPOSITORY_NAME}/statements"
    headers = {"Content-Type": "application/x-turtle"}

    try:
        with open(file_path, 'rb') as ttl_file:
            response = requests.post(url, data=ttl_file, headers=headers)
            if response.status_code == 204:
                print("Archivo TTL cargado exitosamente a GraphDB")
            else:
                print(f"Error al cargar TTL: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error al intentar cargar el archivo: {e}")
