
from fastapi import HTTPException
import openai
from requests import Session
from app.api.v1.endpoints.files import get_database_schema
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

def generate_sql_query(user_query: str, db: Session) -> str:
    schema = get_database_schema(db)
    
    system_message = f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return the SQL query inside a JSON structure with the key "sql_query".
    
    <example>
    {{
        "sql_query": "SELECT m.nombre AS nombre_mascota, m.tipo, p.nombre AS nombre_dueno FROM mascota m JOIN persona p ON m.persona_id = p.id", 
        
        "original_query": "Muestra los nombres de las mascotas y sus dueños."
      }}
    </example>
    
    <schema>
    {schema}
    </schema>
    """

    
    # Entrada del modelo con la pregunta del usuario
    input_text = f"System: {system_message}\nUser: {user_query}"

    # Tokenizar el texto de entrada
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generar la salida utilizando el modelo Mistral
    outputs = model.generate(**inputs, max_length=300)

    # Decodificar la salida en texto
    response_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        # Intentar convertir la respuesta a JSON para extraer la consulta SQL
        response_json = json.loads(response_content)
        sql_query = response_json.get("sql_query") 
        return sql_query
    except json.JSONDecodeError:
        # Manejar errores en el formato de la respuesta
        raise HTTPException(status_code=500, detail="Error parsing the AI response.")