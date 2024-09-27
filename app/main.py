from fastapi import FastAPI
from app.api.v1.endpoints import files

app = FastAPI()

app.include_router(files.router, prefix="/files", tags=["files"])
