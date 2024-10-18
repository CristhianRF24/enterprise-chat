from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_db_schema
from dotenv import load_dotenv
import json
import re
import os
from app.rdf_generator import generate_schema, generate_ttl

load_dotenv()

def generate_sql_query(user_query: str, db: Session, model: str) -> str:
    schema = get_db_schema(db)

    system_message = create_system_message(schema)

    if model == "openai":
        return _call_openai(system_message, user_query, "sql")
    elif model == "mistral":
        return _call_mistral(system_message, user_query, "sql")
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")

def create_system_message(schema: str) -> str:
  
    return f"""
    Given the following schema, write a SQL query that retrieves the requested information.
    Return ONLY and STRICTLY the SQL query inside a JSON structure with the key "sql_query".
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

def generate_sparql_query(user_query: str, db: Session, model: str ) -> str:
    rdf_content = generate_schema()
    print(rdf_content)
    
    system_message = f"""
    Given the following RDF schema, write a SPARQL query that retrieves the requested information.
    Return ONLY and STRICTLY the JSON structure with the key "sparql_query" as a output. AVOID any solution description. temperature 0 
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
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
        )
        response_content = response.choices[0].message["content"]
        response_json = json.loads(response_content)
        if query_type == "sql":
            return extract_sql_query(response_content)
        elif query_type == "sparql":
            return response_json.get("sparql_query")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the OpenAI response.")


def _call_mistral(system_message: str, user_query: str, query_type: str) -> str:
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
    
        print("MIRA",sql_query_response)

        query_json = re.sub(r'```json\n|\n```', '', sql_query_response).strip()
        response_json = json.loads(query_json)
        print(query_json)
        
        if query_type == "sql":
            return extract_sql_query(sql_query_response)
        elif query_type == "sparql":
            return response_json.get("sparql_query")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the Mistral response.")

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
