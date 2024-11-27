import json
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from requests import Session
from sentence_transformers import SentenceTransformer, util
from app.api.v1.endpoints import files
from app.crud.llm import generate_human_readable_response, generate_sparql_query, generate_sql_query, query_huggingface_api_with_roles
from app.db.db import execute_sql_query, get_db, is_sql_query_safe
from rdflib import Graph

app = FastAPI()

class SQLQueryRequest(BaseModel):
    user_query: str
    model: str
class SQLQueryResponse(BaseModel): 
    results: str
    
class QueryRequest(BaseModel):
    user_query: str
    model: str

class QueryResponse(BaseModel):
    query: str 
    results: dict 

class UserQueryRequest(BaseModel):
    query: str
    model: str 


VECTOR_STORE_FOLDER = './vector_stores'
VECTOR_STORE_JSON = f'{VECTOR_STORE_FOLDER}/vector_store.json'

app.include_router(files.router, prefix="/files", tags=["files"])

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_store(file_path):
    with open(file_path, "r") as f:
        vector_store = json.load(f)
    return vector_store

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/generate_sql/", response_model=SQLQueryResponse) 
async def generate_sql(request: SQLQueryRequest, db: Session = Depends(get_db)):
    sql_query = generate_sql_query(request.user_query, db, model=request.model)
    print('sql_query', sql_query)

    if not is_sql_query_safe(sql_query):
        raise HTTPException(status_code=400, detail="Generated SQL query is unsafe.")
    results = execute_sql_query(sql_query, db)
    
    print('results', results)
    
    human_readable_response = generate_human_readable_response(results, request.user_query, request.model)
    
   
    return SQLQueryResponse(results=human_readable_response)
    
def load_graph():
    knowledge_graph = Graph()
    knowledge_graph.parse("output.ttl", format="turtle")  
    return knowledge_graph

def filter_relevant_chunks(query, chunks, top_k=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = similarities.topk(k=top_k)
    return [chunks[i] for i in top_results.indices]

@app.post("/generate_sparql/", response_model=QueryResponse)
async def generate_sparql(request: QueryRequest, db: Session = Depends(get_db)):
    sparql_query = generate_sparql_query(request.user_query, db, model=request.model)
    print('sparql_query', sparql_query)
    
    knowledge_graph = load_graph() 
    results = knowledge_graph.query(sparql_query)
    results_dict = {}
    for row in results:
        for var in results.vars:
            results_dict[str(var)] = str(row[var])  

    return QueryResponse(query=sparql_query, results=results_dict)

@app.post('/ask/')
async def ask_question(request: UserQueryRequest, db: Session = Depends(get_db)):
    if not Path(f"{VECTOR_STORE_FOLDER}/vector_store.json").is_file():
        raise HTTPException(status_code=404, detail="Índice vectorial no encontrado. Por favor, sube un archivo primero.")
    
    vector_store = load_vector_store(VECTOR_STORE_JSON)
    
    try:
        relevant_chunks = filter_relevant_chunks(request.query, vector_store, top_k=3)
        
        system_message = """
        You are an artificial intelligence assistant specialized in summarizing and analyzing text. 
        Provide clear, accurate, and concise answers in Spanish based on the provided text.
        Limit your response to ONLY answering the question based on the context.
        """
        combined_chunks = "\n".join([chunk['text'] for chunk in relevant_chunks])
        user_message = f"Context: {combined_chunks}\n\nQuestion: {request.query}"
        
        response = query_huggingface_api_with_roles(system_message, user_message)
        
        choices = response.get("choices", [])
        if choices:
            model_response = choices[0].get("message", {}).get("content", "")
        else:
            model_response = "Could not get a response from the model."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar la consulta: {str(e)}")

    return {"query": request.query, "response": model_response}
