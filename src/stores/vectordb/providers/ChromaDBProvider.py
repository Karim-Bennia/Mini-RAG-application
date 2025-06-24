from chromadb import PersistentClient
from chromadb.config import Settings
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from typing import List, Optional, Dict, Any
import logging
import os

class ChromaDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str = "cosine"):
        self.db_path = db_path
        self.client: Optional[PersistentClient] = None
        self.distance_method = distance_method

        if self.distance_method not in [e.value for e in DistanceMethodEnums]:
            raise ValueError(f"Unsupported distance method: {self.distance_method}")
        
        self.logger = logging.getLogger(__name__)
    

    def connect(self):
        self.client = PersistentClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        try:
            _ = self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def list_all_collections(self) -> List:
        return self.client.list_collections()

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        return self.client.get_collection(collection_name=collection_name)
         

    def delete_collection(self, collection_name: str):
      if self.is_collection_existed(collection_name):  
         self.client.delete_collection(collection_name=collection_name)

    def create_collection(self, collection_name: str, embedding_size: int , do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name)
        if not self.is_collection_existed(collection_name):
            self.client.create_collection(name=collection_name, metadata={
                        "hnsw:space": self.distance_method,
                    })

    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None
    ):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            return False

        try:
            col = self.client.get_collection(collection_name=collection_name)
            col.add(documents=[text], embeddings=[vector], metadatas=[metadata], ids=[record_id])
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert record: {e}")
            return False

    def insert_many(
        self,
        collection_name: str,
        texts: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        record_ids: Optional[List[str]] = None,
        batch_size: int = 50
    ):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            return False

        try:
            col = self.client.get_collection(collection_name=collection_name)

            if metadata is None:
                metadata = [{} for _ in texts]

            if record_ids is None:
                record_ids = [str(i) for i in range(len(texts))]

            for i in range(0, len(texts), batch_size):
                col.add(
                    documents=texts[i:i+batch_size],
                    embeddings=vectors[i:i+batch_size],
                    metadatas=metadata[i:i+batch_size],
                    ids=record_ids[i:i+batch_size]
                )
            return True

        except Exception as e:
            self.logger.error(f"Failed to insert batch: {e}")
            return False

    def search_by_vector(
        self,
        collection_name: str,
        vector: List[float],
        limit: int
    ) -> List[Dict[str, Any]]:
        try:
            col = self.client.get_collection(collection_name=collection_name)
            results = col.query(query_embeddings=[vector], n_results=limit)
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
