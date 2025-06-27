from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import base
from routes import data
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

async def startup_span():

     settings = get_settings()

     llm_provider_factory = LLMProviderFactory(settings)
     vectordb_provider_factory = VectorDBProviderFactory(settings)

    # generation client
     app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
     app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
     app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
     app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,embedding_size=settings.EMBEDDING_MODEL_SIZE)

     app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
     app.vectordb_client.connect()

async def shutdown_span():
    app.vectordb_client.disconnect()

app.router.on_startup.append(startup_span)
app.router.on_shutdown.append(shutdown_span)


app.include_router(base.base_router)
app.include_router(data.data_router)