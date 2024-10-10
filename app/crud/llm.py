from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_db_schema
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

def generate_sql_query(user_query: str, db: Session, model: str) -> str:
    schema = get_db_schema(db)

    system_message = create_system_message(schema)

    if model == "openai":
        return _call_openai(system_message, user_query)
    elif model == "mistral":
        return _call_mistral(system_message, user_query)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")


def create_system_message(schema: str) -> str:
  
    return f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return the SQL query inside a JSON structure with the key "sql_query".
    <example>
    {{
        "sql_query": "SELECT * FROM files",
        "original_query": "Show me all the files"
    }}
    </example>
    <schema>
    {schema}
    </schema>
    """


def _call_openai(system_message: str, user_query: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
        )
        response_content = response.choices[0].message["content"]
        return extract_sql_query(response_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")


def _call_mistral(system_message: str, user_query: str) -> str:
    try:
        API_URL = os.getenv("API_URL")
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]

        response = requests.post(API_URL, headers=headers, json={"messages": messages})
        response.raise_for_status()

        response_content = response.json()
        sql_query_response = response_content['choices'][0]['message']['content']
        
        return extract_sql_query(sql_query_response)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Mistral API: {str(e)}")


def extract_sql_query(response_content: str) -> str:
    try:
        match = re.search(r'{.*}', response_content, re.DOTALL)
        
        if not match:
            raise HTTPException(status_code=500, detail="No JSON found in model response.")
        
        sql_query_json = match.group(0)
        # Convert the JSON block into a Python object
        response_json = json.loads(sql_query_json)
        sql_query = response_json.get("sql_query")
        
        if not sql_query:
            raise HTTPException(status_code=500, detail="No SQL query found in the response.")
        
        return sql_query
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the model response.")
