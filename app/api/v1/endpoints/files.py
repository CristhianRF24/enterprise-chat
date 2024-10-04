from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from app.crud.file_crud import create_file, file_exists
from app.db.db import SessionLocal, get_database_schema, get_db
from pathlib import Path
from pydantic import BaseModel
from app.pdf_processing_pipeline import PDFProcessingPipeline 
from app.crud.vector_store_crud import create_or_update_vector_store, get_vector_store 
from dotenv import load_dotenv

import json

load_dotenv()
router = APIRouter()

UPLOAD_FOLDER = './uploaded_files'
VECTOR_STORE_FOLDER = './vector_stores'
VECTOR_STORE_JSON = f'{VECTOR_STORE_FOLDER}/vector_store.json'

class QueryRequest(BaseModel):
    query: str
    k: int


@router.post('/uploadfile/')
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    
    # Create the folder if it does not exist
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_STORE_FOLDER).mkdir(parents=True, exist_ok=True)

    #save the file to the file systema
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"

    existing_file = file_exists(db, filename=file.filename, filepath=file_location)
    if existing_file:
        raise HTTPException(status_code=400, detail="El archivo ya existe en tu BD")

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
        
    # Save the file information to the database
    db_file = create_file(db, filename=file.filename, filetype=file.content_type, filepath=file_location)
    
    pipeline = PDFProcessingPipeline(file_location)
    new_vector_store = pipeline.run_pipeline()

    if Path(VECTOR_STORE_JSON).is_file():
        with open(VECTOR_STORE_JSON, 'r') as f:
            combined_store = json.load(f).get("vector_store", [])
    else:
        combined_store = []

    combined_store.extend(new_vector_store)
    
    with open(VECTOR_STORE_JSON, 'w') as f:
        json.dump({"vector_store": combined_store}, f)

    create_or_update_vector_store(db, VECTOR_STORE_JSON)

    return {"filename": file.filename, "file_id": db_file.id}
 
 

@router.post("/run_sql/")
def run_sql(query: str):
    try:
        # Clean the query to avoid problems and remove whitespace
        query = query.strip().rstrip(';')
        session = SessionLocal()
        result = session.execute(text(query))

        # get  results
        rows = result.fetchall()

        if rows:
            data = []
            for row in rows:
                # Convert each row into a dictionary using the column names
                data.append(dict(zip(result.keys(), row)))

            return {"results": data}
        else:
            return {"error": "The query returned an empty result"}

    except Exception as e:
        return {"error": str(e)}

    finally:
        session.close()


@router.get("/db/schema/")
def get_db_schema(db: Session = Depends(get_db)):
    schema_dict = json.loads(get_database_schema())
    return {"schema": schema_dict}

@router.get('/vector_store/')
async def get_vector_store_endpoint(db: Session = Depends(get_db)):
    vector_store_entry = get_vector_store(db)
    if not vector_store_entry:
        raise HTTPException(status_code=404, detail="No vector store found.")
    
    vector_store_path = vector_store_entry.filepath
    try:
        with open(vector_store_path, 'r') as json_file:
            vector_store_data = json.load(json_file) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading vector store: {str(e)}")
    
    return {"vector_store": vector_store_data}

@router.post("/search/")
async def search(query_request: QueryRequest, db: Session = Depends(get_db)):
    vector_store = get_vector_store(db)
    if not vector_store:
        raise HTTPException(status_code=404, detail="No vector store found.")

    vector_store_path = vector_store.filepath
    try:
        with open(vector_store_path, "r") as json_file:
            vector_store_data = json.load(json_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading vector store: {str(e)}")

    if "vector_store" not in vector_store_data:
        raise HTTPException(status_code=500, detail="Invalid vector store structure.")

    vectors = vector_store_data["vector_store"]

    query = query_request.query.lower()
    k = query_request.k

    results = []
    for document in vectors:
        if query in json.dumps(document).lower(): 
            results.append(document)

    results = results[:k]

    if not results:
        return {"results": "No matches found for your query"}

    return {"results": results}


