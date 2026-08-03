# app/scripts/migrate_embeddings.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import logging
import time
from hashlib import md5
import numpy as np

# Ensure path is correct for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.config import settings
from app.db.qdrant_service import QdrantService
from app.db.mysql_db import db_connection
from app.scripts.embeddings import extract_embedding_from_image
from app.services.store_embedding import get_embedding_from_face
from app.services.gpu_worker import GPUWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

def get_doctor_details(doctor_id):
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
        logger.error(f"Error fetching doctor {doctor_id} from DB: {e}")
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()
    return {}

def run_migration():
    logger.info("Initializing GPUWorker...")
    gpu_worker = GPUWorker()
    gpu_worker.start()
    logger.info("GPUWorker started. Preparing migration...")

    qdrant_service = QdrantService()
    client = qdrant_service.connect()
    
    # 1. Delete old collection (doctors_adaface)
    old_collection = settings.QDRANT_COLLECTION_NAME
    try:
        exists = client.collection_exists(old_collection)
    except Exception:
        exists = any(c.name == old_collection for c in client.get_collections().collections)
    if exists:
        logger.info(f"Deleting old collection: {old_collection}")
        client.delete_collection(old_collection)

    # Ensure new collections exist
    for col_name in [settings.QDRANT_COLLECTION_CENTROIDS, settings.QDRANT_COLLECTION_EMBEDDINGS]:
        try:
            exists = client.collection_exists(col_name)
        except Exception:
            exists = any(c.name == col_name for c in client.get_collections().collections)
        if not exists:
            qdrant_service.create_collection(client, col_name, vector_size=512)

    # 2. Read each doctor's original five images
    doctors_root = "/home/admin1/Zydus/Backend/uploads/doctors"
    if not os.path.exists(doctors_root):
        logger.error(f"Doctors upload folder not found at {doctors_root}")
        return

    doctors = [d for d in os.listdir(doctors_root) if os.path.isdir(os.path.join(doctors_root, d))]
    logger.info(f"Found {len(doctors)} doctor directories to migrate.")

    img_exts = (".jpg", ".jpeg", ".png")

    for doctor_id in doctors:
        doctor_dir = os.path.join(doctors_root, doctor_id)
        image_files = [f for f in os.listdir(doctor_dir) if f.lower().endswith(img_exts)]
        if not image_files:
            logger.warning(f"No images found for doctor {doctor_id}")
            continue

        logger.info(f"Processing doctor {doctor_id} ({len(image_files)} images)...")
        details = get_doctor_details(doctor_id)
        doctor_name = details.get("doctor_name") or doctor_id
        department = details.get("department") or "General"

        # Delete existing points for this doctor in new collections (for idempotency)
        from qdrant_client.http import models
        client.delete(
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
        client.delete(
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

        for idx, file in enumerate(image_files):
            face_path = os.path.join(doctor_dir, file)
            face_rgb, facial_area = extract_embedding_from_image(face_path)
            if face_rgb is None:
                logger.warning(f"Failed face crop extraction for {face_path}")
                continue

            emb = get_embedding_from_face(face_rgb)
            if emb is None:
                logger.warning(f"Failed embedding generation for {face_path}")
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

        if not embeddings:
            logger.warning(f"No embeddings generated for doctor {doctor_id}")
            continue

        # Upload original embeddings
        qdrant_service.upsert_points(
            settings.QDRANT_COLLECTION_EMBEDDINGS,
            vectors=embeddings,
            metadatas=payloads,
            ids=ids
        )

        # Compute centroid
        embs_arr = np.array(embeddings)
        centroid = np.mean(embs_arr, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-6)

        centroid_raw_id = f"{doctor_id}_centroid"
        centroid_point_id = int(md5(centroid_raw_id.encode()).hexdigest()[:16], 16)

        centroid_payload = {
            "doctor_id": doctor_id,
            "doctor_name": doctor_name,
            "department": department,
            "profile_path": payloads[0]["profile_path"]
        }

        # Upload centroid
        qdrant_service.upsert_points(
            settings.QDRANT_COLLECTION_CENTROIDS,
            vectors=[centroid_norm.tolist()],
            metadatas=[centroid_payload],
            ids=[centroid_point_id]
        )
        logger.info(f"Finished migrating doctor {doctor_id}. Centroid and {len(embeddings)} embeddings uploaded.")

    logger.info("Migration completed successfully.")

if __name__ == "__main__":
    run_migration()
