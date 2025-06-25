from fastapi import FastAPI, APIRouter, Depends, UploadFile, status,Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController,ProcessController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest
from uuid import uuid4
from stores.llm.templates.prompt_template import PromptTemplate

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    

    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": file_id,
            }
        )

@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, process_request: ProcessRequest, request: Request,
):

    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size

    process_controller = ProcessController(project_id=project_id)

    file_content = process_controller.get_file_content(file_id=file_id)

    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size
    )

    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )
    
    embedding_client = request.app.embedding_client
    vectordb_client = request.app.vectordb_client

        
    texts = []
    vectors = []
    metadata = []
    record_ids = []

    collection_name = f"{project_id}_{file_id}"

    print(collection_name)
    # Create collection if not exist or reset if needed
    vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=embedding_client.embedding_size,
        do_reset=True
    )
    for idx, chunk in enumerate(file_chunks):
        if not hasattr(chunk, "text") or not chunk.text.strip():
            continue
        clean_text = chunk.text.strip()
        vector = embedding_client.embed_text(clean_text)
        if vector is None:
            continue  # skip failed embeddings

        texts.append(clean_text)

        vectors.append(vector)

        metadata.append({
            "file_id": file_id,
            "chunk_index": idx,
            "project_id": project_id,
        })
        record_ids.append(str(uuid4()))

    if not texts:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.PROCESSING_FAILED.value}
        )

    success = vectordb_client.insert_many(
        collection_name=collection_name,
        texts=texts,
        vectors=vectors,
        metadata=metadata,
        record_ids=record_ids
    )

    if not success:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.PROCESSING_FAILED.value}
        )

    return {
        "signal": ResponseSignal.PROCESSING_SUCCESS.value,
        "chunks_stored": len(texts),
        "collection": collection_name
    }


@data_router.post("/query/{project_id}")
async def query_endpoint(
    request: Request,
    project_id: str
):
    # Get the raw request body (question and top_k)
    body = await request.json()
    question = body.get("question")
    top_k = body.get("top_k", 5)  # Default to 5 if top_k is not provided
    file_id = body.get("file_id")  # Assuming file_id is part of the request

    metadata_filter = {}
    metadata_filter["file_id"] = file_id


    if not question:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "Missing question"}
        )

    # Embed the user question
    embedding_client = request.app.embedding_client
    question_vector = embedding_client.embed_text(question)

    vectordb_client = request.app.vectordb_client
    collection_name = f"{project_id}_{file_id}"


    if question_vector is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "Question embedding failed"}
        )

    # Search for top-k relevant chunks from ChromaDB
    
    search_results = vectordb_client.search_by_vector(
        collection_name=collection_name,
        vector=question_vector,
        limit=top_k,
        metadata_filter=metadata_filter
    
        )
    


    if not search_results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "No relevant chunks found"}
        )

    # Prepare context (top-k chunks) for LLM
    context = "\n".join([doc for doc in search_results['documents'][0]])

    # Generate AI response using the context and question
    generation_client = request.app.generation_client

    prompt_template = PromptTemplate()
    prompt = prompt_template.create_question_prompt(question, context)


    answer = generation_client.generate_text(prompt)

    if not answer:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "Answer generation failed"}
        )

    # Prepare response with the answer and sources
   
    sources = [
        {
         "text": doc,  # result is a dictionary
         "file_id": metadata['file_id'],
         "chunk_index": metadata['chunk_index']

        }
        for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0])
    ]

    return JSONResponse(
        content={
            "answer": answer,
            "sources": sources
        }
    )
