import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()  # Carga las variables de entorno desde el archivo .env


# Carga la URL de la base de datos desde el archivo .env
URL_DATABASE = os.getenv('DATABASE_URL')

engine = create_engine(URL_DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def is_sql_query_safe(sql_query):
    prohibited_phrases = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
        "EXEC", "--", ";", "/*", "*/", "@@", "@", "CREATE", "SHUTDOWN",
        "GRANT", "REVOKE"
    ]
    for phrase in prohibited_phrases:
        if phrase.lower() in sql_query.lower():
            return False
    return True