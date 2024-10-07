from fastapi import HTTPException
from requests import Session
from app.api.v1.endpoints.files import get_database_schema
import json
import re
import requests


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-Small-Instruct-2409/v1/chat/completions"
headers = {"Authorization": "Bearer your_token_here"}


def generate_sql_query(user_query: str, db: Session) -> str:
    schema = get_database_schema(db)

    # Mensaje del sistema con el esquema y ejemplo
    system_message = f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return the SQL query inside a JSON structure with the key "sql_query".

    <example>
    {{
        "sql_query": "SELECT m.nombre AS nombre_mascota, m.tipo, p.nombre AS nombre_dueno
        FROM mascota m
        JOIN persona p ON m.persona_id = p.id", 
        "original_query": "Muestra los duenos de los perros"
      }}
    </example>

    <schema>
    {schema}
    </schema>
    """
    

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    try:
        response = requests.post(API_URL, headers=headers, json={"messages": messages})
        response.raise_for_status() 

        response_content = response.json()
        print("Response from API:", response_content)

        sql_query_response = response_content['choices'][0]['message']['content']
        sql_query_json = re.sub(r'```json\n|\n```', '', sql_query_response).strip()
        sql_query_dict = json.loads(sql_query_json)

      # Get the SQL query
        sql_query = sql_query_dict.get("sql_query")
        if not sql_query:
            raise HTTPException(status_code=500, detail="No SQL query found in the response.")

        return sql_query 

    except requests.exceptions.HTTPError as e:
      
        detail = response.json().get("detail", "Error sin detallar.")
        raise HTTPException(status_code=500, detail=f"HTTP error occurred: {detail}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

