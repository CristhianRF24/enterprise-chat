from sqlalchemy.orm import Session
from app.models.vector_store import VectorStore

def create_vector_store(db: Session, filepath: str):
    db_store = VectorStore(filepath=filepath)
    db.add(db_store)
    db.commit()
    db.refresh(db_store)
    return db_store

def get_vector_store(db: Session):
    return db.query(VectorStore).first()

def create_or_update_vector_store(db: Session, filepath: str):
    existing_store = get_vector_store(db)

    if existing_store:
        existing_store.filepath = filepath
        db.commit()
        return existing_store
    else:
        return create_vector_store(db, filepath)