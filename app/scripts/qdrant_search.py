# app/scripts/qdrant_search.py
import logging
import traceback
from qdrant_client.http.exceptions import ResponseHandlingException
from app.db.qdrant_service import QdrantService
from app.core.config import settings
import numpy as np

logger = logging.getLogger(__name__)
qdrant_service = QdrantService()

class QdrantSearchEngine:
    def __init__(self):
        try:
            self.client = qdrant_service.connect()
        except Exception as e:
            logger.error(f"[QDRANT] Failed to connect: {e}")
            logger.error(traceback.format_exc())
            self.client = None

    def search(self, embedding, limit=5):
        try:
            if embedding is None:
                logger.warning("[QDRANT] Received None embedding for search")
                return []
            
            if self.client is None:
                logger.error("[QDRANT] No client available. Search aborted.")
                return []
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten().tolist()

            if not isinstance(embedding, list) or len(embedding) == 0:
                logger.error("[QDRANT] Invalid embedding format")
                return []
            
            result = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=embedding,
                limit=limit
            )

            matches = []
            for r in result.points:
                matches.append({
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                })

            return matches
        
        except ResponseHandlingException:
            logger.error("[QDRANT] Connection dropped, reconnecting")
            logger.error(traceback.format_exc())

            self.client = qdrant_service.reconnect()
            return []
        
        except Exception as e:
            logger.error(f"[QDRANT] Search failed: {e}")
            logger.error(traceback.format_exc())
            return []
