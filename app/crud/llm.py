from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_database_schema
from dotenv import load_dotenv
import json
import re
import os

from app.rdf_generator import generate_ttl

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
        return _call_openai(system_message, user_query, "sql")
    elif model == "mistral":
        return _call_mistral(system_message, user_query, "sql")
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")

def generate_sparql_query(user_query: str, db: Session, model: str ) -> str:
    generate_ttl()
    with open("output.ttl", 'r') as f:
        rdf_content = f.read()
    
    system_message = f"""
    Given the following RDF schema, write a SPARQL query that retrieves the requested information.
    Return only the SPARQL query inside a JSON structure with the key "sparql_query".
    <example>
    {{
        "sparql_query": "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }}",
        "original_query": "Get all triples from the dataset"
    }}
    </example>
    <rdf_schema>
    {rdf_content}
    </rdf_schema>
    """
       
    print(f"Using model: {model} sparql")
    if model == "openai":
        return _call_openai(system_message, user_query, "sparql")
    elif model == "mistral":
        return _call_mistral(system_message, user_query, "sparql")
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")


def _call_openai(system_message: str, user_query: str, query_type: str) -> str:
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
        if query_type == "sql":
            return response_json.get("sql_query")
        elif query_type == "sparql":
            return response_json.get("sparql_query")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the OpenAI response.")


def _call_mistral(system_message: str, user_query: str, query_type: str) -> str:
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
    
    query_json = re.sub(r'```json\n|\n```', '', sql_query_response).strip()
    
    try:
        response_json = json.loads(query_json)
        if query_type == "sql":
            return response_json.get("sql_query")
        elif query_type == "sparql":
            return response_json.get("sparql_query")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the Mistral response.")
