import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from glob import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import spacy
from langchain.agents import initialize_agent, Tool, AgentType

load_dotenv()

# Configuración de OpenAI API desde el archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

nlp = spacy.load("es_core_news_sm")

# Función para limpiar texto
def clean_text(raw_text):
    text = re.sub(r"metadata=\{.*?\}", "", raw_text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"(?<![\.\?!])\n", " ", text) 
    text = re.sub(r"Ignoring.*?object.*?\n", "", text)
    text = re.sub(r"Special Issue.*?\n", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return lemmatized

# Función para cargar y procesar varios PDFs con limpieza
def load_and_process_pdfs(folder_path):
    pdf_files = glob(f"{folder_path}/*.pdf")  # Buscar todos los PDFs en la carpeta
    all_texts = []

    for pdf_file in pdf_files:
        print(f"Cargando {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        # Limpieza del texto de cada documento
        for doc in documents:
            cleaned_text = clean_text(doc.page_content)
            lemmatized_text = lemmatize_text(cleaned_text)
            doc.page_content = lemmatized_text         
        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)
    
    return all_texts

# Crear índice vectorial
def create_vector_index(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

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

# Flujo principal
def main():
    # Ruta a la carpeta con PDFs
    pdf_folder = "C:/Users/BELEN/Monsters university/10mo semestre/Practica pre profesional/Zekiri/enterprise-chat/uploaded_files"
    print("Cargando y procesando los PDFs...")
    texts = load_and_process_pdfs(pdf_folder)
    print("Documentos procesados con éxito.")

    # Crear índice vectorial
    print("Creando índice vectorial...")
    vector_store = create_vector_index(texts)
    print("Índice vectorial creado.")

    # Crear herramientas para el agente
    tools = create_tools(vector_store)

    # Crear el agente
    agent = create_agent(tools)

    # Interactuar con el usuario
    print("\n¡Listo! Puedes empezar a hacer preguntas.")
    while True:
        query = input("\nPregunta: ")
        if query.lower() in ["salir", "exit", "quit"]:
            print("¡Adiós!")
            break
        response = agent.invoke(query)
        print(f"Respuesta: {response}")

if __name__ == "__main__":
    main()
