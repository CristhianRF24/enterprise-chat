from sqlalchemy import Column, Integer, String
from app.db.db import Base

class File(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filetype = Column(String(50), nullable=False)
    filepath = Column(String(255), nullable=False)