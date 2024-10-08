import os
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
from typing import List, Dict


load_dotenv() 

db_config = {
    "host": os.getenv('DB_HOST'), 
    "user": os.getenv('DB_USER'),       
    "password":  os.getenv('DB_PASS'),      
    "database":  os.getenv('DB_NAME'), 
    "port": os.getenv('DB_PORT')        
}

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
        


def execute_sql_query(query: str, db: Session) -> List[Dict]:
    try:
       
        query = query.strip().rstrip(';')
      
        if not is_sql_query_safe(query):
            raise Exception("SQL query is unsafe")

        result = db.execute(text(query))
        rows = result.fetchall()

        if rows:
            data = []
            for row in rows:
                data.append(dict(zip(result.keys(), row)))
            return data
        else:
            return []

    except Exception as e:
        print(f"Error: {e}")
        return []
    
    
def is_sql_query_safe(sql_query):
    prohibited_phrases = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
        "EXEC", "--", "/*", "*/", "@@", "@", "CREATE", "SHUTDOWN",
        "GRANT", "REVOKE"
    ]
    for phrase in prohibited_phrases:
        if phrase.lower() in sql_query.lower():
            return False
    return True