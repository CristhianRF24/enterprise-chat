from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from app.crud.file_crud import create_file, file_exists
from app.db.db import SessionLocal, get_db
from pathlib import Path



router = APIRouter()

UPLOAD_FOLDER = './uploaded_files'


@router.post('/uploadfile/')
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    
    # Create the folder if it does not exist
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    
    #save the file to the file systema
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"

    existing_file = file_exists(db, filename=file.filename, filepath=file_location)
    if existing_file:
        raise HTTPException(status_code=400, detail="El archivo ya existe en tu BD")

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
        
    # Save the file information to the database
    db_file = create_file(db, filename=file.filename, filetype=file.content_type, filepath=file_location)
    
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
def get_database_schema(db: Session = Depends(get_db)):
    schema = {}
    try:
        
        connection = db.connection()
        inspector = inspect(connection)  
        tables = inspector.get_table_names()

        for table in tables:
            schema[table] = {
                "columns": [],
                "foreign_keys": [],
                "primary_keys": []
            }
            #  add columns
            for column in inspector.get_columns(table):
                schema[table]["columns"].append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": column["default"]
                })
            # add foreign keys
            schema[table]["foreign_keys"] = inspector.get_foreign_keys(table)
            # add primary keys
            primary_key_info = inspector.get_pk_constraint(table)
            schema[table]["primary_keys"] = primary_key_info["constrained_columns"] if "constrained_columns" in primary_key_info else []

        return {"schema": schema}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting schema: {str(e)}")





