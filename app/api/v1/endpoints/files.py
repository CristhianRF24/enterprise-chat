import re
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import openai
import rdflib
import spacy
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from app.crud.file_crud import create_file, file_exists
from app.db.db import SessionLocal, get_database_schema, get_db
from pathlib import Path
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from app.pdf_processing_pipeline import PDFProcessingPipeline 
from app.crud.vector_store_crud import create_or_update_vector_store, get_vector_store 
from dotenv import load_dotenv
from app.graphdb_integration import load_ttl_to_graphdb
import json
from rdflib import Graph
from langchain.vectorstores import FAISS
from app.rdf_generator import generate_ttl
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import os

load_dotenv()
router = APIRouter()
nlp = spacy.load("es_core_news_sm")
openai_api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = './uploaded_files'
VECTOR_STORE_FOLDER = './vector_stores'
VECTOR_STORE_JSON = f'{VECTOR_STORE_FOLDER}/vector_store.json'

total_input_tokens = 0
total_output_tokens = 0

class QueryRequest(BaseModel):
    query: str
    k: int

# Modelos de entrada y salida
class UserQueryRequest(BaseModel):
    query: str
    model: str  # Puedes agregar más parámetros si es necesario para el agente.

class UserQueryResponse(BaseModel):
    query: str
    response: str
    total_input_tokens: int
    total_output_tokens: int

# Para contar tokens, puedes añadir la siguiente función
def count_tokens_in_input_and_output(query: str, response: str):
    global total_input_tokens, total_output_tokens

    # Contar tokens de entrada (para el 'query')
    try:
        input_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # o cualquier modelo que estés utilizando
            messages=[{"role": "user", "content": query}]
        )
        input_tokens = input_response['usage']['total_tokens']
    except Exception as e:
        print(f"Error al contar los tokens de entrada: {e}")
        input_tokens = 0

    # Contar tokens de salida (para la respuesta generada)
    try:
        output_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # o cualquier modelo que estés utilizando
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
        )
        output_tokens = output_response['usage']['total_tokens'] - input_tokens
    except Exception as e:
        print(f"Error al contar los tokens de salida: {e}")
        output_tokens = 0

    # Acumular los tokens en las variables globales
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    return input_tokens, output_tokens

def clean_text(raw_text):
    text = re.sub(r"metadata=\{.*?\}", "", raw_text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"(?<![\.\?!])\n", " ", text) 
    text = re.sub(r"Ignoring.*?object.*?\n", "", text)
    text = re.sub(r"Special Issue.*?\n", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text

# Función para lematizar texto
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return lemmatized

# Procesar y dividir PDFs en fragmentos
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    processed_texts = []
    for doc in documents:
        cleaned_text = clean_text(doc.page_content)
        lemmatized_text = lemmatize_text(cleaned_text)
        doc.page_content = lemmatized_text
        processed_texts.append(doc)
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(processed_texts)

def create_vector_index(texts, existing_index=None):
    embeddings = OpenAIEmbeddings()
    if existing_index:
        existing_index.add_documents(texts)
        return existing_index
    return FAISS.from_documents(texts, embeddings)

# Crear herramientas para el agente
def create_tools(vector_store):
    # Crear una herramienta de búsqueda basada en el vector store
    retriever = vector_store.as_retriever()
    tool = Tool(
        name="Document Retriever",
        func=lambda query: retriever.invoke(query),  # Cambiar a invoke
        description="Usado para recuperar documentos relevantes para la consulta"
    )
    return [tool]

# Inicializar el agente
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
    db_file = create_file(db, filename=file.filename, filetype="pdf", filepath=file_location)
    
    texts = process_pdf(file_location)

    new_vector_store = create_vector_index(texts)

    if Path(VECTOR_STORE_JSON).is_file():
        existing_index = FAISS.load_local(VECTOR_STORE_FOLDER, OpenAIEmbeddings())
        existing_index.add_documents(new_vector_store.docstore._dict.values())
    else:
        existing_index = new_vector_store

    existing_index.save_local(VECTOR_STORE_FOLDER)

    create_or_update_vector_store(db, VECTOR_STORE_JSON)

    return {"filename": file.filename, "file_id": db_file.id}
 
@router.post('/ask/')
async def ask_question(request: UserQueryRequest, db: Session = Depends(get_db)):
    """
    Endpoint para recibir una pregunta del usuario y responder utilizando el agente.
    """
    # Verificar que el archivo de vector_store existe
    if not Path(f"{VECTOR_STORE_FOLDER}/index.faiss").is_file() or not Path(f"{VECTOR_STORE_FOLDER}/index.pkl").is_file():
        raise HTTPException(status_code=404, detail="Índice vectorial no encontrado. Por favor, sube un archivo primero.")
    
    # Cargar vector_store desde el archivo JSON
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(VECTOR_STORE_FOLDER, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el vector store: {str(e)}")

    # Crear herramientas y el agente
    tools = create_tools(vector_store)
    agent = create_agent(tools)

    # Ejecutar el agente con la consulta del usuario
    try:
        response = agent.invoke(request.query)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar el agente: {str(e)}")

    return {"query": request.query, "response": response}

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