from fastapi import HTTPException
import openai
import requests
from requests import Session
from app.api.v1.endpoints.files import get_db_schema
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.rdf_generator import generate_ttl
import json
import torch
import re
import os

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

headers = {
    "Authorization": f"Bearer {huggingface_token}"
}


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b", use_auth_token=huggingface_token,  cache_dir="E:/huggingface/models")
    model = AutoModelForCausalLM.from_pretrained(
        "chatdb/natural-sql-7b",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=huggingface_token
    )
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise HTTPException(status_code=500, detail=f"Error al cargar el modelo de Hugging Face: {str(e)}")


def generate_sql_query(user_query: str, db: Session, model: str) -> str:
    
    schema = get_db_schema(db)
    system_message = create_system_message(schema)

    if model == "openai":
        return _call_openai(system_message, user_query, "sql")
    elif model == "natural-sql":
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
    generate_ttl()
    with open("output.ttl", 'r') as f:
        rdf_content = f.read()
    
    system_message = f"""
    Given the following RDF schema, write a SPARQL query that retrieves the requested information.
    Return ONLY and STRICTLY  the JSON structure with the key "sparql_query" as a output avoid any solution description. temperature 0 
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
    elif model == "natural-sql":
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
       
        full_prompt = f"{system_message}\nUser: {user_query}"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=1024, temperature=0.7, do_sample=True)
        response_content = tokenizer.decode(outputs[0], skip_special_tokens=True)    
           
        query_json = re.sub(r'```json\n|\n```', '', response_content).strip()
        response_json = json.loads(query_json)

        if query_type == "sql":
            return extract_sql_query(response_content)
        elif query_type == "sparql":
            return response_json.get("sparql_query")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing the model response.")


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
