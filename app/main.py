from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from requests import Session
from app.api.v1.endpoints import files
from app.crud.agent import get_limited_schema, humanize_response, process_query_with_sql_agent
from app.crud.llm import generate_human_readable_response, generate_sparql_query, generate_sql_query
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
    
class QueryRequest(BaseModel):
    question: str

app.include_router(files.router, prefix="/files", tags=["files"])

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/generate_sql/", response_model=SQLQueryResponse) 
async def generate_sql(request: SQLQueryRequest, db: Session = Depends(get_db)):
    sql_query = await generate_sql_query(request.user_query, db, model=request.model)
    print('sql_query', sql_query)

    if not is_sql_query_safe(sql_query):
        raise HTTPException(status_code=400, detail="Generated SQL query is unsafe.")
    results = execute_sql_query(sql_query, db)
    
    print('results', results)
    
    human_readable_response = await generate_human_readable_response(results, request.user_query, request.model)
    
   
    return SQLQueryResponse(results=human_readable_response)
    
def load_graph():
    knowledge_graph = Graph()
    knowledge_graph.parse("output.ttl", format="turtle")  
    return knowledge_graph

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



# Endpoint para procesar consultas SQL
@app.post("/queryAgent")
def query(request: QueryRequest):
    question = request.question
    try:
        # Procesar la consulta con el agente
        sql_response = process_query_with_sql_agent(question)
        # Humanizar la respuesta
        human_response = humanize_response("Consulta generada automáticamente", sql_response)
        return {"question": question, "response": human_response}
    except Exception as e:
        print(f"Error al procesar la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar la consulta.")

