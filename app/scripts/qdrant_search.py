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
            if self.client:
                for col in [settings.QDRANT_COLLECTION_CENTROIDS, settings.QDRANT_COLLECTION_EMBEDDINGS]:
                    try:
                        exists = self.client.collection_exists(col)
                    except Exception:
                        exists = any(c.name == col for c in self.client.get_collections().collections)
                    if not exists:
                        logger.info(f"[Qdrant] Collection '{col}' does not exist. Creating it.")
                        qdrant_service.create_collection(self.client, col, vector_size=512)
        except Exception as e:
            logger.error(f"[QDRANT] Failed to connect: {e}")
            logger.error(traceback.format_exc())
            self.client = None

    def search(self, embedding, limit=None):
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
            
            top_k = settings.TOP_K if hasattr(settings, "TOP_K") else 3
            centroid_threshold = settings.CENTROID_THRESHOLD if hasattr(settings, "CENTROID_THRESHOLD") else 0.35
            verification_threshold = settings.VERIFICATION_THRESHOLD if hasattr(settings, "VERIFICATION_THRESHOLD") else 0.38

            # Step 1: Search ONLY doctor_centroids
            result = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                query=embedding,
                limit=top_k
            )

            for r in result.points:
                centroid_score = r.score
                if centroid_score < centroid_threshold:
                    continue
                
                payload = r.payload or {}
                doctor_id = payload.get("doctor_id")
                doctor_name = payload.get("doctor_name", "Unknown")
                
                if not doctor_id:
                    continue

                # Step 2: Retrieve all embeddings of that doctor from doctor_embeddings
                from qdrant_client.http import models
                emb_results, _ = self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_EMBEDDINGS,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doctor_id",
                                match=models.MatchValue(value=doctor_id)
                            )
                        ]
                    ),
                    with_vectors=True,
                    limit=100
                )
                
                similarities = []
                for emb_r in emb_results:
                    if emb_r.vector is not None:
                        sim = np.dot(np.array(embedding), np.array(emb_r.vector))
                        similarities.append(sim)
                
                if not similarities:
                    continue
                
                avg_sim = sum(similarities) / len(similarities)
                count_exceeded = sum(1 for s in similarities if s >= verification_threshold)
                
                # Step 3: Verification rules
                if avg_sim >= verification_threshold or count_exceeded >= 3:
                    # Verified! Return this doctor
                    return [{
                        "id": r.id,
                        "score": avg_sim,
                        "payload": {
                            "doctor_id": doctor_id,
                            "doctor_name": doctor_name,
                            "profile_path": payload.get("profile_path", ""),
                            "centroid_score": centroid_score,
                            "avg_similarity": avg_sim,
                            "count_exceeded": count_exceeded
                        }
                    }]

            return []
        
        except ResponseHandlingException:
            logger.error("[QDRANT] Connection dropped, reconnecting")
            logger.error(traceback.format_exc())

            self.client = qdrant_service.reconnect()
            return []
        
        except Exception as e:
            logger.error(f"[QDRANT] Search failed: {e}")
            logger.error(traceback.format_exc())
            return []

    def search_candidate(self, embedding):
        try:
            if embedding is None or self.client is None:
                return "Unknown", 0.0
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten().tolist()

            # Search ONLY doctor_centroids for the single best candidate
            result = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                query=embedding,
                limit=1
            )
            if not result.points:
                return "Unknown", 0.0

            r = result.points[0]
            centroid_score = r.score
            payload = r.payload or {}
            doctor_id = payload.get("doctor_id")
            doctor_name = payload.get("doctor_name", "Unknown")

            if not doctor_id:
                return "Unknown", centroid_score

            # Retrieve all embeddings of that doctor to compute average similarity
            from qdrant_client.http import models
            emb_results, _ = self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION_EMBEDDINGS,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doctor_id",
                            match=models.MatchValue(value=doctor_id)
                        )
                    ]
                ),
                with_vectors=True,
                limit=100
            )

            similarities = []
            for emb_r in emb_results:
                if emb_r.vector is not None:
                    sim = np.dot(np.array(embedding), np.array(emb_r.vector))
                    similarities.append(sim)

            if not similarities:
                return doctor_name, centroid_score

            avg_sim = sum(similarities) / len(similarities)
            return doctor_name, avg_sim
        except Exception as e:
            logger.error(f"[QDRANT] search_candidate failed: {e}")
            return "Unknown", 0.0
