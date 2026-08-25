# app/services/store_embedding.py.py
from hashlib import md5
import logging
import os
import time
import cv2

# import torch
from app.db.qdrant_service import QdrantService
from app.scripts.embeddings import extract_embedding_from_image
from app.core.config import settings
import uuid
from app.utils.face_augmentations import generate_face_variants

logger = logging.getLogger(__name__)

qdrant_service = QdrantService()
collection_name = settings.QDRANT_COLLECTION_NAME

from app.services.gpu_worker import GPU_TASK_QUEUE, GPU_RESULT_MAP
import uuid, time, numpy as np


def _get_front_filename_keywords():
    raw = settings.ENROLL_FRONT_FILENAME_KEYWORDS or ""
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _is_front_profile_image(file_name: str) -> bool:
    if not settings.ENROLL_FRONT_ONLY:
        return True
    name = file_name.lower()
    keywords = _get_front_filename_keywords()
    return any(keyword in name for keyword in keywords)


def _front_image_priority(file_name: str) -> int:
    name = file_name.lower()
    if "front_face" in name:
        return 0
    if "front_profile" in name:
        return 1
    if "front" in name:
        return 2
    return 99

def get_embedding_from_face(face_rgb, timeout=3.0):
    task_id = uuid.uuid4().hex
    GPU_TASK_QUEUE.put((task_id, {
        "type": "face",
        "data": face_rgb
    }))

    start = time.time()
    while task_id not in GPU_RESULT_MAP:
        if time.time() - start > timeout:
            return None
        time.sleep(0.002)

    faces = GPU_RESULT_MAP.pop(task_id)
    return faces[0]["embedding"] if faces else None


from app.db.mysql_db import db_connection

def get_doctor_details_from_db(doctor_id: str):
    conn = db_connection()
    if conn is None:
        return {}
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT full_name, department_name FROM tbl_doctors WHERE id = %s", (doctor_id,))
        row = cur.fetchone()
        if row:
            return {
                "doctor_name": row.get("full_name"),
                "department": row.get("department_name")
            }
    except Exception as e:
        logger.error(f"Error fetching doctor details from DB: {e}")
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()
    return {}

def compute_calibration_metrics(embeddings_list):
    """
    Compute intra-doctor spread (mean pairwise cosine similarity) and adaptive threshold.
    """
    spread = 1.0
    if len(embeddings_list) > 1:
        similarities = []
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                v1 = np.array(embeddings_list[i])
                v2 = np.array(embeddings_list[j])
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
                sim = np.dot(v1_norm, v2_norm)
                similarities.append(float(sim))
        spread = sum(similarities) / len(similarities)

    # Compute adaptive threshold
    alpha = settings.ADAPTIVE_THRESHOLD_ALPHA if hasattr(settings, "ADAPTIVE_THRESHOLD_ALPHA") else 0.1
    min_t = settings.MIN_ADAPTIVE_THRESHOLD if hasattr(settings, "MIN_ADAPTIVE_THRESHOLD") else 0.32
    max_t = settings.MAX_ADAPTIVE_THRESHOLD if hasattr(settings, "MAX_ADAPTIVE_THRESHOLD") else 0.40
    global_t = settings.VERIFICATION_THRESHOLD if hasattr(settings, "VERIFICATION_THRESHOLD") else 0.36
    
    adaptive_threshold = global_t - alpha * (1.0 - spread)
    adaptive_threshold = max(min_t, min(adaptive_threshold, max_t))
    
    return float(spread), float(adaptive_threshold)


def store_single_image(face_path: str,
                       doctor_id: str,
                       doctor_name: str,
                       ):

    try:
        if not os.path.exists(face_path):
            logger.error(f"[store_single_image] face_path does not exist: {face_path}")
            return None
        
        face_rgb, facial_area = extract_embedding_from_image(face_path)

        if face_rgb is None:
            logger.error("[store_single_image] Failed to extract face crop")
            return None
        
        qdrant_client = qdrant_service.connect()
        
        # Ensure collections exist
        for col in [settings.QDRANT_COLLECTION_CENTROIDS, settings.QDRANT_COLLECTION_EMBEDDINGS]:
            try:
                exists = qdrant_client.collection_exists(col)
            except Exception:
                exists = any(c.name == col for c in qdrant_client.get_collections().collections)
            if not exists:
                qdrant_service.create_collection(qdrant_client, col, vector_size=512)

        emb = get_embedding_from_face(face_rgb)
        if emb is None:
            logger.error(f"[store_single_image] Failed to extract embedding from {face_path}")
            return None

        # Fetch doctor details from DB for payload
        details = get_doctor_details_from_db(doctor_id)
        real_doctor_name = details.get("doctor_name") or doctor_name
        department = details.get("department") or "General"

        # Determine image_number by scrolling existing points
        from qdrant_client.http import models
        existing_points, _ = qdrant_client.scroll(
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

        # Check if this profile_path already exists to overwrite it
        target_point = None
        for p in existing_points:
            if p.payload and p.payload.get("profile_path") == os.path.abspath(face_path):
                target_point = p
                break

        if target_point:
            image_number = target_point.payload.get("image_number", 1)
        else:
            existing_nums = {p.payload.get("image_number") for p in existing_points if p.payload}
            image_number = 1
            for i in range(1, 6):
                if i not in existing_nums:
                    image_number = i
                    break
            else:
                image_number = len(existing_points) + 1

        raw_id = f"{doctor_id}_img_{image_number}"
        point_id = int(md5(raw_id.encode()).hexdigest()[:16], 16)
        
        metadata = {
            "doctor_id": doctor_id,
            "image_number": image_number,
            "doctor_name": real_doctor_name,
            "profile_path": os.path.abspath(face_path),
            "facial_area": facial_area,
            "department": department
        }

        qdrant_service.upsert_points(
            settings.QDRANT_COLLECTION_EMBEDDINGS,
            vectors=[emb.tolist()],
            metadatas=[metadata],
            ids=[point_id]
        )
        logger.info(f"[store_single_image] Successfully stored embedding for {face_path} at slot {image_number} for doctor {doctor_id}")

        # Fetch all embeddings again to compute/update centroid
        all_points, _ = qdrant_client.scroll(
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

        embeddings_list = [p.vector for p in all_points if p.vector is not None]
        if embeddings_list:
            embs_arr = np.array(embeddings_list)
            centroid = np.mean(embs_arr, axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-6)

            spread, adaptive_threshold = compute_calibration_metrics(embeddings_list)
            
            warning_limit = settings.INTRA_DOCTOR_SPREAD_WARNING_THRESHOLD if hasattr(settings, "INTRA_DOCTOR_SPREAD_WARNING_THRESHOLD") else 0.50
            if spread < warning_limit:
                logger.warning(
                    f"[INCONSISTENT ENROLLMENT] Doctor {doctor_id} has high embedding spread! "
                    f"Mean pairwise similarity is {spread:.4f}, below warning threshold {warning_limit}"
                )

            centroid_raw_id = f"{doctor_id}_centroid"
            centroid_point_id = int(md5(centroid_raw_id.encode()).hexdigest()[:16], 16)

            centroid_payload = {
                "doctor_id": doctor_id,
                "doctor_name": real_doctor_name,
                "department": department,
                "profile_path": os.path.abspath(face_path),
                "intra_doctor_spread": spread,
                "adaptive_threshold": adaptive_threshold
            }

            qdrant_service.upsert_points(
                settings.QDRANT_COLLECTION_CENTROIDS,
                vectors=[centroid_norm.tolist()],
                metadatas=[centroid_payload],
                ids=[centroid_point_id]
            )
            logger.info(f"[store_single_image] Successfully updated centroid for doctor {doctor_id} with spread {spread:.4f} and adaptive threshold {adaptive_threshold:.4f}")

        return 1
    except Exception as e:
        logger.exception(f"❌ [store_single_image] Error storing {face_path}: {e}")
        return None

def store_multiple_images_parallel(root_dir: str,
                                   collection_name: str = "doctors",
                                   batch_size: int = 200
                                   ):
    try:
        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            logger.error(f"[store_multiple_images_parallel] root_dir does not exist or is not a directory: {root_dir}")
            return None
        
        logger.info(f"🔍 Scanning directory: {root_dir}")

        img_exts = (".jpg", ".jpeg", ".png")
        doctors = os.listdir(root_dir)
        start_time = time.time()
        uploaded_count = 0

        qdrant_client = qdrant_service.connect()
        # Ensure collections exist
        for col in [settings.QDRANT_COLLECTION_CENTROIDS, settings.QDRANT_COLLECTION_EMBEDDINGS]:
            try:
                exists = qdrant_client.collection_exists(col)
            except Exception:
                exists = any(c.name == col for c in qdrant_client.get_collections().collections)
            if not exists:
                qdrant_service.create_collection(qdrant_client, col, vector_size=512)

        for doctor in doctors:
            doctor_path = os.path.join(root_dir, doctor)
            if not os.path.isdir(doctor_path):
                continue

            image_files = [f for f in os.listdir(doctor_path) if f.lower().endswith(img_exts)]
            if not image_files:
                continue

            selected_files = image_files
            if settings.ENROLL_FRONT_ONLY:
                front_files = [f for f in image_files if _is_front_profile_image(f)]
                if front_files:
                    selected_files = [sorted(front_files, key=_front_image_priority)[0]]

            doctor_id = doctor
            details = get_doctor_details_from_db(doctor_id)
            doctor_name = details.get("doctor_name") or doctor_id
            department = details.get("department") or "General"

            # Clean start for this doctor to be idempotent
            from qdrant_client.http import models
            qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION_EMBEDDINGS,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doctor_id",
                                match=models.MatchValue(value=doctor_id)
                            )
                        ]
                    )
                )
            )
            qdrant_client.delete(
                collection_name=settings.QDRANT_COLLECTION_CENTROIDS,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doctor_id",
                                match=models.MatchValue(value=doctor_id)
                            )
                        ]
                    )
                )
            )

            embeddings = []
            payloads = []
            ids = []

            for idx, file in enumerate(selected_files):
                face_path = os.path.join(doctor_path, file)
                try:
                    face_rgb, facial_area = extract_embedding_from_image(face_path)
                    if face_rgb is None:
                        continue
                    
                    emb = get_embedding_from_face(face_rgb)
                    if emb is None:
                        continue

                    image_number = idx + 1
                    raw_id = f"{doctor_id}_img_{image_number}"
                    point_id = int(md5(raw_id.encode()).hexdigest()[:16], 16)

                    embeddings.append(emb.tolist())
                    ids.append(point_id)
                    payloads.append({
                        "doctor_id": doctor_id,
                        "image_number": image_number,
                        "doctor_name": doctor_name,
                        "profile_path": os.path.abspath(face_path),
                        "facial_area": facial_area,
                        "department": department
                    })
                except Exception as e:
                    logger.error(f"Error processing {face_path} in parallel: {e}")
                    continue

            if embeddings:
                # Upsert to doctor_embeddings
                qdrant_service.upsert_points(
                    settings.QDRANT_COLLECTION_EMBEDDINGS,
                    vectors=embeddings,
                    metadatas=payloads,
                    ids=ids
                )
                uploaded_count += len(embeddings)

                # Compute centroid
                embs_arr = np.array(embeddings)
                centroid = np.mean(embs_arr, axis=0)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-6)

                spread, adaptive_threshold = compute_calibration_metrics(embeddings)

                warning_limit = settings.INTRA_DOCTOR_SPREAD_WARNING_THRESHOLD if hasattr(settings, "INTRA_DOCTOR_SPREAD_WARNING_THRESHOLD") else 0.50
                if spread < warning_limit:
                    logger.warning(
                        f"[INCONSISTENT ENROLLMENT] Doctor {doctor_id} has high embedding spread! "
                        f"Mean pairwise similarity is {spread:.4f}, below warning threshold {warning_limit}"
                    )

                centroid_raw_id = f"{doctor_id}_centroid"
                centroid_point_id = int(md5(centroid_raw_id.encode()).hexdigest()[:16], 16)

                centroid_payload = {
                    "doctor_id": doctor_id,
                    "doctor_name": doctor_name,
                    "department": department,
                    "profile_path": payloads[0]["profile_path"],
                    "intra_doctor_spread": spread,
                    "adaptive_threshold": adaptive_threshold
                }

                qdrant_service.upsert_points(
                    settings.QDRANT_COLLECTION_CENTROIDS,
                    vectors=[centroid_norm.tolist()],
                    metadatas=[centroid_payload],
                    ids=[centroid_point_id]
                )
                logger.info(f"Enrolled doctor {doctor_id} with {len(embeddings)} original embeddings and centroid. Spread: {spread:.4f}, Threshold: {adaptive_threshold:.4f}")

        logger.info(f"✅ Completed bulk enrollment in {time.time() - start_time:.2f}s")
        return uploaded_count
    except Exception as e:
        logger.exception(f"❌ [store_multiple_images_parallel] Fatal error: {e}")
        return None
