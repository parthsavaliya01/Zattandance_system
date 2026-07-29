# app/db/qdrant_service.py
# establish qdrant connection, create database and upload embeddings
import logging
import threading
from typing import Any, Dict, List
from  qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    _client: QdrantClient = None
    _lock = threading.Lock()

    def connect(self) -> QdrantClient:
        """
        Return a shared Qdrant client (one per process).
        Recreate only if missing or dead.
        """
        with self._lock:
            if self._client is None:
                logger.info("[QdrantService] Creating new Qdrant client")
                self._client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=5.0,
                )
            return self._client

    def reconnect(self) -> QdrantClient:
        """
        Force reconnect (used after transport failure).
        """
        with self._lock:
            logger.warning("[QdrantService] Reconnecting Qdrant client")
            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5.0,
            )
            return self._client

    def create_collection(self, qdrant_client, collection_name: str, vector_size: int):
        try:
            existing = qdrant_client.get_collections().collections
            if any (c.name == collection_name for c in existing):
                logger.info(f"[QdrantService] Collection '{collection_name}' already exists")
                return  
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            logger.info(f"[QdrantService] Created collection '{collection_name}' (size={vector_size})")
        except Exception as e:
            logger.exception(f"[QdrantService] Failed to create collection '{collection_name}': {e}")
            raise


    def upsert_points(self, collection_name: str, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str] = None):
        try:
            client=self.connect()
            points = []
            for i, vec in enumerate(vectors):
                points.append(PointStruct(id=ids[i], vector=vec, payload=metadatas[i]))
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"[QdrantService] Upserted {len(points)} points into '{collection_name}'")
        except Exception as e:
            logger.exception(f"[QdrantService] Failed upsert for collection '{collection_name}': {e}")
            raise