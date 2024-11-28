import re
import time
import unicodedata
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from langchain_text_splitters import CharacterTextSplitter
import pdfplumber
import rdflib
import spacy
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from translate import Translator
from app.crud.file_crud import create_file, file_exists
from app.db.db import SessionLocal, get_database_schema, get_db
from pathlib import Path
from pydantic import BaseModel
from app.pdf_processing_pipeline import PDFProcessingPipeline 
from app.crud.vector_store_crud import create_or_update_vector_store, get_vector_store 
from dotenv import load_dotenv
from app.graphdb_integration import load_ttl_to_graphdb
import json
from rdflib import Graph
from langchain.embeddings import OpenAIEmbeddings
from app.rdf_generator import generate_ttl
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langdetect import detect
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
import os
import requests
from app.rdf_generator import generate_ttl

load_dotenv()
router = APIRouter()
nlp = spacy.load("es_core_news_sm")
openai_api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = './uploaded_files'
VECTOR_STORE_FOLDER = './vector_stores'
VECTOR_STORE_JSON = f'{VECTOR_STORE_FOLDER}/vector_store.json'

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    query: str
    k: int

class UserQueryRequest(BaseModel):
    query: str
    model: str 

def clean_text(raw_text):
    text = re.sub(r'\x00', '', text)
    text = re.sub(r'\n+', ' ', text) 
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9\s.,;:¿?¡!]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

def split_text_into_chunks(text, chunk_size=500):
    sentences = re.split(r'(?<=\.)\s+', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk + sentence) > chunk_size:
            chunks.append(chunk)
            chunk = sentence
        else:
            chunk += " " + sentence
    if chunk:
        chunks.append(chunk)
    return chunks

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return lemmatized

def normalize_text(text: str) -> str:
    text = re.sub(r'\x00', '', text)  
    text = re.sub(r'\n+', ' ', text)  
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9\s.,;:¿?¡!]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    
    return text.lower()

def process_pdf(file_path: str, chunk_size: int = 500) -> list:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or "" 

    text = normalize_text(text)
    print("Texto limpio:", text) 
    chunks = split_text_into_chunks(text, chunk_size=chunk_size)
    
    return chunks

def create_vector_index(texts):
    embeddings = embedding_model.encode(texts, convert_to_tensor=False).tolist()
    vector_store = [{"text": text, "embedding": embedding} for text, embedding in zip(texts, embeddings)]
    return vector_store

def save_vector_store(vector_store, file_path):
    with open(file_path, "w") as f:
        json.dump(vector_store, f)

def load_vector_store(file_path):
    with open(file_path, "r") as f:
        vector_store = json.load(f)
    return vector_store

def combine_vector_stores(existing_store, new_store):
    combined_store = existing_store + new_store
    return combined_store

def create_tools(vector_store):
    retriever = vector_store.as_retriever()
    tool = Tool(
        name="Document Retriever",
        func=lambda query: retriever.invoke(query),  
        description="Usado para recuperar documentos relevantes para la consulta"
    )
    return [tool]

def create_agent(tools):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

@router.post('/uploadfile/')
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_STORE_FOLDER).mkdir(parents=True, exist_ok=True)
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"

    existing_file = file_exists(db, filename=file.filename, filepath=file_location)
    if existing_file:
        raise HTTPException(status_code=400, detail="El archivo ya existe en tu BD")

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
        
    db_file = create_file(db, filename=file.filename, filetype="pdf", filepath=file_location)
    
    texts = process_pdf(file_location)
    new_vector_store = create_vector_index(texts)

    if Path(VECTOR_STORE_JSON).is_file():
        existing_vector_store = load_vector_store(VECTOR_STORE_JSON)
        combined_vector_store = combine_vector_stores(existing_vector_store, new_vector_store)
    else:
        combined_vector_store = new_vector_store

    save_vector_store(combined_vector_store, VECTOR_STORE_JSON)

    create_or_update_vector_store(db, VECTOR_STORE_JSON)

    return {"filename": file.filename, "file_id": db_file.id}
 

@router.get('/check-pdf-loaded/')
async def check_pdf_loaded():
    pdf_files = [f for f in Path(UPLOAD_FOLDER).iterdir() if f.suffix.lower() == '.pdf']

    if pdf_files:
        return {"pdf_loaded": True}
    else:
        return {"pdf_loaded": False}

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

@router.post("/generate-and-load-ttl/")
def generate_and_load_ttl_endpoint():
    try:
        output_path = generate_ttl()

        load_ttl_to_graphdb(output_path)

        return {"message": "Archivo TTL generado y cargado en GraphDB con éxito", "file_path": output_path}
    except Exception as e:
        return {"error": str(e)}

def load_graph():
    g = Graph()
    g.parse("output.ttl", format="turtle")  
    return g

@router.post("/sparql/")
def sparql_query(query):
    g = load_graph()
    results = g.query(query)
    return results