from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from requests import Session
from app.api.v1.endpoints import files
from app.crud.llm import generate_sql_query
from app.db.db import get_db, is_sql_query_safe

app = FastAPI()

class SQLQueryRequest(BaseModel):
    user_query: str

# Si SQLQueryResponse no está en files.py, puedes definirla aquí
class SQLQueryResponse(BaseModel):
    sql_query: str

app.include_router(files.router, prefix="/files", tags=["files"])

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/generate_sql/", response_model=SQLQueryResponse)  # Cambiado a SQLQueryResponse aquí
async def generate_sql(request: SQLQueryRequest, db: Session = Depends(get_db)):
    sql_query = generate_sql_query(request.user_query, db)
    
    if not is_sql_query_safe(sql_query):
        raise HTTPException(status_code=400, detail="Generated SQL query is unsafe.")

    return SQLQueryResponse(sql_query=sql_query)
