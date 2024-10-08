from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_database_schema
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

def generate_sql_query(user_query: str, db: Session, model: str ) -> str:
    schema = get_database_schema(db)

    system_message = f"""
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
       
    # print used model
    print(f"Using model: {model}")
    if model == "openai":
        return _call_openai(system_message, user_query)
    elif model == "mistral":
        return _call_mistral(system_message, user_query)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")


def _call_openai(system_message: str, user_query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    )

    response_content = response.choices[0].message["content"]
    
    try:
        response_json = json.loads(response_content)
        sql_query = response_json.get("sql_query")
        return sql_query
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the OpenAI response.")


def _call_mistral(system_message: str, user_query: str) -> str:
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

 
    sql_query_json = re.sub(r'```json\n|\n```', '', sql_query_response).strip()
    
    try:
        response_json = json.loads(sql_query_json)
        sql_query = response_json.get("sql_query")
        return sql_query
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the Mistral response.")
