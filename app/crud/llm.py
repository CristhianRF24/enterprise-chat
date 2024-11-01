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
import nltk
import sqlparse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


load_dotenv()

nltk.download('wordnet')

def extract_relevant_schema(user_query: str, schema: dict) -> str:
    # Inicializa el lematizador y las stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))  # Asegúrate de tener nltk.corpus.stopwords descargado

    # Obtener el contenido del esquema
    schema_content = schema.get('schema', {})

    # Analizar la consulta del usuario
    parsed_query = sqlparse.parse(user_query)[0]
    tokens = [str(token).lower() for token in parsed_query.tokens if not token.is_whitespace]
    tokens = [word for token in tokens for word in token.split()]  # Divide tokens en palabras individuales
    print("Parsed Tokens:", tokens)

    # Eliminar stop words de los tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]
    print("Filtered Tokens:", filtered_tokens)

    relevant_tables = set()

    # Lemmatiza los tokens filtrados
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    print("Lemmatized Tokens:", lemmatized_tokens)

    # Buscar tablas relevantes en el esquema
    for table_name in schema_content.keys():
        for token in lemmatized_tokens:
            # Comparar nombres de tablas y tokens
            if re.match(re.escape(table_name.lower()), token):
                relevant_tables.add(table_name)
            if re.sub(r's$', '', token) == re.sub(r's$', '', table_name.lower()):
                relevant_tables.add(table_name)
            elif re.sub(r'$', '', table_name.lower()) in token:
                relevant_tables.add(table_name)

    print("Relevant Tables Found:", relevant_tables)

    # Crear un diccionario para filtrar las tablas y añadir tablas relacionadas
    final_tables = {}
    for table_name in relevant_tables:
        if table_name not in final_tables:
            final_tables[table_name] = schema_content[table_name]

        # Verificar si hay llaves foráneas en la tabla actual
        for column in schema_content[table_name].get('columns', []):
            column_name = column.get('COLUMN_NAME', '').lower()
            # Si la columna indica una relación, intenta agregar la tabla relacionada
            if column_name.endswith('_id'):
                related_table_name = column_name.rsplit('_', 1)[0]
                if related_table_name in schema_content:
                    final_tables[related_table_name] = schema_content[related_table_name]

    print("Final Relevant Tables:", final_tables.keys())

    # Crear el subesquema con las tablas relevantes
    sub_schema = {table: schema_content[table] for table in final_tables.keys()}

    return json.dumps(sub_schema, indent=4)

def translate_query(user_query: str, model: str) -> str:
    # Crea un mensaje del sistema para la traducción
    system_message = f"""
    Translate the following query from Spanish to English:
    "{user_query}"
    Return ONLY the translated text without any additional information.
    """

    if model == "openai":
        return _call_openai_for_translation(system_message)
    elif model == "mistral":
        return _call_mistral_for_translation(system_message)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")

# Función de soporte para la traducción con OpenAI
def _call_openai_for_translation(system_message: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_message}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating translation response.")

# Función de soporte para la traducción con Mistral
def _call_mistral_for_translation(system_message: str) -> str:
    try:
        API_URL = os.getenv("API_URL")
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}

        response = requests.post(API_URL, headers=headers, json={"messages": [{"role": "user", "content": system_message}]})
        print("Response status code:", response.status_code)
        print("Response content:", response.text)
        response.raise_for_status()
        response_content = response.json()
        return response_content['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating translation response: {str(e)}")




def generate_sql_query(user_query: str, db: Session, model: str) -> str:
    
    full_schema = get_db_schema(db)
 
    translate_query_response = translate_query(user_query, model)
    relevant_schema = extract_relevant_schema(translate_query_response, full_schema)
    print("Relevant Database Schema:", relevant_schema)
    
    system_message = create_system_message(relevant_schema)

    if model == "openai":
        return _call_openai(system_message, translate_query_response, "sql")
    elif model == "mistral":
        return _call_mistral(system_message, translate_query_response, "sql")
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
        total_tokens = response['usage']['total_tokens']
        print(f"Tokens totales: {total_tokens}")
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
        print("MIRA", response_json)
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