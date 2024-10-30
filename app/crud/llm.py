from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_db_schema
from dotenv import load_dotenv
from app.rdf_generator import generate_schema, generate_ttl
from sqlparse.sql import IdentifierList, Identifier
import json
import re
import os
import sqlparse


load_dotenv()

"""
This function looks up table and column names in the user's query
and extracts the relevant subschema for those tables.
"""
def extract_relevant_schema(user_query: str, schema: dict) -> str:
    # Get the contents of the schema
    schema_content = schema.get('schema', {})
    print("Schema Content:", schema_content)  

   # Analyze the SQL query
    parsed_query = sqlparse.parse(user_query)[0]
    tokens = [token for token in parsed_query.tokens if not token.is_whitespace]
    print("Parsed Tokens:", tokens) 

    relevant_tables = set()

    # Find relevant tables in the schema   
    for table_name in schema_content.keys():
        for token in tokens:
            # Compares the table name, ignoring case
            if str(token).lower() == table_name.lower():
                relevant_tables.add(table_name)
            # Also checks if the token is a substring of the table name
            elif table_name.lower() in str(token).lower():
                relevant_tables.add(table_name)

    print("Relevant Tables Found:", relevant_tables) 
    # Create a subschema with only the relevant tables and columns
    sub_schema = {table: schema_content[table] for table in relevant_tables if table in schema_content}
    
    return json.dumps(sub_schema, indent=4)





def generate_sql_query(user_query: str, db: Session, model: str) -> str:
    
    full_schema = get_db_schema(db)
    print("Full Database Schema:", full_schema)
    print("Schema keys:", full_schema.keys())
    # Extract the relevant subschema based on the user query
    relevant_schema = extract_relevant_schema(user_query, full_schema)
    print("Relevant Database Schema:", relevant_schema)
    
    system_message = create_system_message(relevant_schema)

    if model == "openai":
        return _call_openai(system_message, user_query, "sql")
    elif model == "mistral":
        return _call_mistral(system_message, user_query, "sql")
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")


def create_system_message(schema: str) -> str:
    return f"""
    Given the following schema, write ONLY the SQL query that retrieves the requested information.
    1. If the user query is in Spanish, first translate it to English.
    2. Based on the translated or original query, generate the SQL query that retrieves the requested information.
    3. Validate the generated SQL to ensure it is safe and syntactically correct before returning it.
    
    Return the SQL query in this JSON format:
    {{
        "sql_query": "SELECT * FROM city;",
        "original_query": "Show me all the city"
    }}
    You must STRICTLY follow this format and return ONLY the JSON. Do not provide explanations or additional information.
    <schema>
    {schema}
    </schema>
    """


def generate_sparql_query(user_query: str, db: Session, model: str ) -> str:
    rdf_content = generate_schema()
    print(rdf_content)
    
    system_message = f"""
    Given the following RDF schema, write a SPARQL query that retrieves the requested information.
    Return ONLY and STRICTLY the JSON structure with the key "sparql_query" as output, avoiding any solution description.

    {{
        "sparql_query": "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }}",
        "original_query": "Get all triples from the dataset"
    }}
    </example>
    <rdf_schema>
    {rdf_content}
    </rdf_schema>
    """
       
    print(f"Using model: {model} for SPARQL")
    if model == "openai":
        return _call_openai(system_message, user_query, "sparql")
    elif model == "mistral":
        return _call_mistral(system_message, user_query, "sparql")
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")
    

# Function to generate a readable response
def generate_human_readable_response(sql_results: list, user_query: str, model: str) -> str:
    
    results_text = "\n".join([", ".join([f"{key}: {value}" for key, value in row.items()]) for row in sql_results])
    
    system_message = f"""
    Convert the following SQL results into a user-friendly summary:
    {results_text}
    provide ONLY AND STRICTLY a clear and concise description in spanish.
    """

    

    if model == "openai":
        return _call_openai_for_response(system_message)
    elif model == "mistral":
        return _call_mistral_for_response(system_message)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")

# Support function to generate readable response with OpenAI
def _call_openai_for_response(system_message: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_message}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating human-readable response.")

#Support function to generate readable response with Mistral
def _call_mistral_for_response(system_message: str) -> str:
    try:
        API_URL = os.getenv("API_URL")
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
        
        response = requests.post(API_URL, headers=headers, json={"messages": [{"role": "user", "content": system_message}]})
        response.raise_for_status()
        response_content = response.json()
        return response_content['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating human-readable response.")
    
    
    
    

def _call_openai(system_message: str, user_query: str, query_type: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
        )

        total_tokens = response['usage']['total_tokens']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        
        print(f"Tokens del prompt (incluye schema): {prompt_tokens}")
        print(f"Tokens del completado: {completion_tokens}")
        print(f"Tokens totales: {total_tokens}")

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
    
        print("MIRA", sql_query_response)

        query_json = re.sub(r'```json\n|\n```', '', sql_query_response).strip()
        response_json = json.loads(query_json)
        
        print("query_json:", query_json)
        print("response_json:", response_json)
        
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
        response_json = json.loads(sql_query_json)
        sql_query = response_json.get("sql_query")
        
        if not sql_query:
            raise HTTPException(status_code=500, detail="No SQL query found in the response.")
        
        return sql_query
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the model response.")
