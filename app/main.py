from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from requests import Session
from app.api.v1.endpoints import files
from app.crud.llm import generate_sparql_query, generate_sql_query
from app.db.db import execute_sql_query, get_db, is_sql_query_safe
from rdflib import Graph

app = FastAPI()

class SQLQueryRequest(BaseModel):
    user_query: str
    model: str
class SQLQueryResponse(BaseModel): 
    results: list
    
class QueryRequest(BaseModel):
    user_query: str
    model: str

class QueryResponse(BaseModel):
    query: str 
    results: dict 

app.include_router(files.router, prefix="/files", tags=["files"])

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
  
    return SQLQueryResponse(results=results)
    
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