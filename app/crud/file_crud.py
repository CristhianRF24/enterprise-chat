
from sqlalchemy.orm import Session
from app.models.file import File

def create_file(db: Session, filename: str, filetype: str, filepath: str):
    db_file = File(filename=filename, filetype=filetype, filepath=filepath)
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file    


def file_exists(db: Session, filename: str, filepath: str):
    return db.query(File).filter((File.filename == filename) | (File.filepath == filepath)).first()

