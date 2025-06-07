# main.py
# This file is part of the FastAPI application that serves as the entry point for the API.
# It includes the necessary imports, sets up the FastAPI app, and includes routers for different functionalities.

from fastapi import FastAPI
from app.api.upload import router as upload_router
from routes.chat import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Add this middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (in production, restrict this!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api", tags=["Upload and Search"])

# Register the /chat endpoint
app.include_router(chat_router, tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
