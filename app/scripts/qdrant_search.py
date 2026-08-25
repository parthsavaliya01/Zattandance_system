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

    def search(self, embedding, limit=5, expected_doctor_id: str = None):
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
            
            top_k = limit if limit is not None else (settings.TOP_K if hasattr(settings, "TOP_K") else 5)
            verification_threshold_global = settings.VERIFICATION_THRESHOLD if hasattr(settings, "VERIFICATION_THRESHOLD") else 0.36

            # Step 1: Search ONLY doctor_centroids
            result = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                query=embedding,
                limit=top_k
            )

            # Log Stage 1 candidate ranks and scores
            if settings.TELEMETRY_LOGGING_ENABLED:
                logger.info(f"[STAGE 1 CENTROID] Query returned {len(result.points)} candidates (top_k={top_k}):")
                for rank, r in enumerate(result.points, 1):
                    payload = r.payload or {}
                    logger.info(f"  Rank {rank}: doctor_id={payload.get('doctor_id')}, name={payload.get('doctor_name')}, centroid_score={r.score:.4f}")

            # Telemetry for expected doctor if not in top_k
            if expected_doctor_id and not any((r.payload or {}).get("doctor_id") == expected_doctor_id for r in result.points):
                from qdrant_client.http import models
                specific_res, _ = self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doctor_id",
                                match=models.MatchValue(value=expected_doctor_id)
                            )
                        ]
                    ),
                    limit=1
                )
                if specific_res and specific_res[0].vector is not None:
                    # Specific centroid score is cosine similarity = dot product of normalized vectors
                    specific_score = float(np.dot(np.array(embedding), np.array(specific_res[0].vector)))
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        logger.info(f"[STAGE 1 CENTROID] Expected doctor {expected_doctor_id} was NOT in top_k. Centroid score: {specific_score:.4f}")
                else:
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        logger.info(f"[STAGE 1 CENTROID] Expected doctor {expected_doctor_id} has no centroid registered.")

            verified_candidates = []

            for r in result.points:
                centroid_score = r.score
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
                        similarities.append(float(sim))
                
                if not similarities:
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        logger.info(f"[STAGE 2 VERIFICATION] Doctor {doctor_id} ({doctor_name}): No enrollment embeddings found.")
                    continue
                
                # Retrieve adaptive threshold if enabled, else use global threshold
                if settings.ADAPTIVE_THRESHOLD_ENABLED:
                    adaptive_t = payload.get("adaptive_threshold")
                    if adaptive_t is not None:
                        verification_threshold = adaptive_t
                        is_adaptive = "Enabled"
                        has_calibration = True
                        spread_val = payload.get("intra_doctor_spread")
                    else:
                        verification_threshold = verification_threshold_global
                        is_adaptive = "Enabled (Fallback to Global)"
                        has_calibration = False
                        spread_val = None
                else:
                    verification_threshold = verification_threshold_global
                    is_adaptive = "Disabled"
                    has_calibration = False
                    spread_val = None

                avg_sim = sum(similarities) / len(similarities)
                count_exceeded = sum(1 for s in similarities if s >= verification_threshold)
                
                # Verification rules
                passed = (avg_sim >= verification_threshold or count_exceeded >= 3)
                
                # Log Stage 2 results
                if settings.TELEMETRY_LOGGING_ENABLED or passed:
                    calibration_status = "Calibration Exists" if has_calibration else "No Calibration (Fallback to Global)"
                    spread_str = f", Spread: {spread_val:.4f}" if spread_val is not None else ""
                    logger.info(
                        f"[STAGE 2 VERIFICATION] Doctor: {doctor_name} ({doctor_id}). "
                        f"5 similarities: {[round(s, 4) for s in similarities]}. "
                        f"Average similarity: {avg_sim:.4f}. "
                        f"Count >= threshold: {count_exceeded}. "
                        f"Verification threshold: {verification_threshold:.4f} (Adaptive: {is_adaptive}, Status: {calibration_status}{spread_str}). "
                        f"Passed: {passed}."
                    )

                if passed and not verified_candidates:
                    # Keep the first verified candidate to return
                    verified_candidates.append({
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
                    })

            # Stage 2 logging for expected doctor if not shortlisted in Stage 1
            if expected_doctor_id and not any((r.payload or {}).get("doctor_id") == expected_doctor_id for r in result.points):
                from qdrant_client.http import models
                emb_results_exp, _ = self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION_EMBEDDINGS,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doctor_id",
                                match=models.MatchValue(value=expected_doctor_id)
                            )
                        ]
                    ),
                    with_vectors=True,
                    limit=100
                )
                similarities_exp = []
                for emb_r in emb_results_exp:
                    if emb_r.vector is not None:
                        sim = np.dot(np.array(embedding), np.array(emb_r.vector))
                        similarities_exp.append(float(sim))
                
                if similarities_exp:
                    specific_res_exp, _ = self.client.scroll(
                        collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doctor_id",
                                    match=models.MatchValue(value=expected_doctor_id)
                                )
                            ]
                        ),
                        limit=1
                    )
                    payload_exp = specific_res_exp[0].payload or {} if specific_res_exp else {}
                    if settings.ADAPTIVE_THRESHOLD_ENABLED:
                        adaptive_t_exp = payload_exp.get("adaptive_threshold")
                        if adaptive_t_exp is not None:
                            verification_threshold_exp = adaptive_t_exp
                            is_adaptive_exp = "Enabled"
                            has_calibration_exp = True
                            spread_val_exp = payload_exp.get("intra_doctor_spread")
                        else:
                            verification_threshold_exp = verification_threshold_global
                            is_adaptive_exp = "Enabled (Fallback to Global)"
                            has_calibration_exp = False
                            spread_val_exp = None
                    else:
                        verification_threshold_exp = verification_threshold_global
                        is_adaptive_exp = "Disabled"
                        has_calibration_exp = False
                        spread_val_exp = None
                    
                    avg_sim_exp = sum(similarities_exp) / len(similarities_exp)
                    count_exceeded_exp = sum(1 for s in similarities_exp if s >= verification_threshold_exp)
                    passed_exp = (avg_sim_exp >= verification_threshold_exp or count_exceeded_exp >= 3)
                    
                    if settings.TELEMETRY_LOGGING_ENABLED or passed_exp:
                        calibration_status_exp = "Calibration Exists" if has_calibration_exp else "No Calibration (Fallback to Global)"
                        spread_str_exp = f", Spread: {spread_val_exp:.4f}" if spread_val_exp is not None else ""
                        logger.info(
                            f"[STAGE 2 VERIFICATION] Expected Doctor (NOT IN SHORTLIST): {expected_doctor_id}. "
                            f"5 similarities: {[round(s, 4) for s in similarities_exp]}. "
                            f"Average similarity: {avg_sim_exp:.4f}. "
                            f"Count >= threshold: {count_exceeded_exp}. "
                            f"Verification threshold: {verification_threshold_exp:.4f} (Adaptive: {is_adaptive_exp}, Status: {calibration_status_exp}{spread_str_exp}). "
                            f"Passed: {passed_exp}."
                        )

            return verified_candidates
        
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
