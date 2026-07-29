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


def store_single_image(face_path: str,
                       doctor_id: str,
                       doctor_name: str,
                       ):

    try:
        if not os.path.exists(face_path):
            logger.error(f"[store_single_image] face_path does not exist: {face_path}")
            return None
        
        output_dir = "test_output/variants"
        os.makedirs(output_dir, exist_ok=True)
        
        face_rgb, facial_area = extract_embedding_from_image(face_path)
        
        # if emb is None or getattr(emb, "size", 0) == 0:
        #     logger.error(f"[store_single_image] Failed to extract embedding from {face_path}")
        #     return None

        if face_rgb is None:
            logger.error("[store_single_image] Failed to extract face crop")
            return None
        
        face_variants = generate_face_variants(face_rgb)

        stored = 0
        qdrant_client = qdrant_service.connect()

        try:
            collection_exists = qdrant_client.collection_exists(collection_name)
        except Exception:
            collection_exists = any(c.name == collection_name for c in qdrant_client.get_collections().collections)

        for idx, variant in enumerate(face_variants):
            try:
                variant_bgr = cv2.cvtColor(variant, cv2.COLOR_RGB2BGR)
                variant_filename = f"{doctor_name}_var{idx}.jpg"
                variant_save_path = os.path.join(output_dir, variant_filename)
                cv2.imwrite(variant_save_path, variant_bgr)
                variant = cv2.cvtColor(variant_bgr, cv2.COLOR_BGR2RGB)

                
                emb = get_embedding_from_face(variant)
                if not collection_exists:
                    logger.info(
                        f"[store_single_image] Creating collection '{collection_name}' "
                        f"(vector size={emb.shape[0]})"
                    )
                    qdrant_service.create_collection(qdrant_client, collection_name, vector_size=emb.shape[0])
                    collection_exists=True

                raw_id = f"{doctor_id}_{uuid.uuid4()}"
                point_id = int(md5(raw_id.encode()).hexdigest()[:16], 16)
                metadata = {
                    "doctor_id": doctor_id,
                    "doctor_name": doctor_name,
                    "profile_path": os.path.abspath(face_path),
                    "facial_area": facial_area
                }

                qdrant_service.upsert_points(collection_name, vectors=[emb.tolist()], metadatas=[metadata], ids=[point_id])
                stored += 1
                logger.info(
                    f"[store_single_image] Successfully stored embedding for {face_path} for variant {idx} "
                    f"for doctor_id {doctor_id}")
            except Exception as e:
                logger.warning(f"[store_single_image] Failed variant {idx}: {e}")
        return stored
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

        # Collect files
        img_exts = (".jpg", ".jpeg", ".png")
        face_items = []
        results = []
        doctors = os.listdir(root_dir)
        start_time = time.time()

        for doctor in doctors:
            doctor_path = os.path.join(root_dir, doctor)
            if not os.path.isdir(doctor_path):
                logger.error(f"⚠️ Skipping {doctor_path} - not a directory")
                continue

            image_files = [f for f in os.listdir(doctor_path) if f.lower().endswith(img_exts)]
            if not image_files:
                continue

            selected_files = image_files
            if settings.ENROLL_FRONT_ONLY:
                front_files = [f for f in image_files if _is_front_profile_image(f)]
                if not front_files:
                    logger.warning(
                        f"[store_multiple_images_parallel] No front profile image found for doctor folder {doctor_path}; skipping"
                    )
                    continue
                selected_files = [sorted(front_files, key=_front_image_priority)[0]]

            for file in selected_files:
                face_path = os.path.join(doctor_path, file)
                try:
                    logger.info(f"[store_multiple_images_parallel] Extracting embedding from {face_path}....")
                    emb, facial_area = extract_embedding_from_image(face_path)
                    if emb is None or getattr(emb, "size", 0) == 0:
                        logger.warning(f"[store_multiple_images_parallel] ❌ Failed to extract embedding from {face_path}")
                        continue
                    result = {
                        "doctor_id": doctor,
                        "doctor_name": doctor,
                        "face_path": face_path,
                        "embedding": emb.tolist(),
                        "facial_area": facial_area
                    }
                    results.append(result)
                    logger.info(f"[store_multiple_images_parallel] ✅ Extracted embedding from {face_path}")
                except Exception as e:
                    logger.exception(f"❌ [store_multiple_images_parallel] Error processing {face_path}: {e}")
                    continue

        logger.info(f"✅ Extracted {len(results)} embeddings in {time.time() - start_time:.2f}s ")
        logger.info(f"📸 Found {len(results)} images for {len(doctors)}")

        if not results:
            logger.error("[store_multiple_images_parallel] ❌ No embeddings extracted")
            return 0
    

        # Connect Qdrant
        qdrant_client = qdrant_service.connect()
        logger.info(f"[store_multiple_images_parallel] ✅ Connected to Qdrant")

        # Create collection based on the first embedding
        vector_size = len(results[0]["embedding"])
        try:
            collection_exists = qdrant_client.collection_exists(collection_name)
        except Exception:
            collection_exists = any(c.name == collection_name for c in qdrant_client.get_collections().collections)

        if not collection_exists:
            logger.info(f"[store_multiple_images_parallel] Collection '{collection_name}' does not exist. Creating it with vector size {vector_size}...")
            qdrant_service.create_collection(qdrant_client, collection_name, vector_size)
            logger.info(f"[store_multiple_images_parallel] ✅ Created collection '{collection_name}' with vector size {vector_size}")

        logger.info(f"[store_multiple_images_parallel] ⬆️ Uploading embeddings to Qdrant...")

        start_time = time.time()


        # Batch upload
        batch_vectors = []
        batch_ids = []
        batch_meta = []
        uploaded = 0

        for item in results:
            batch_vectors.append(item["embedding"])
            raw_id = f"{item['doctor_id']}_{uuid.uuid4()}"
            batch_ids.append(int(md5(raw_id.encode()).hexdigest()[:16], 16))
            batch_meta.append({
                "doctor_id": item["doctor_id"],
                "doctor_name": item["doctor_name"],
                "profile_path": os.path.abspath(item["face_path"]),
                "facial_area": item["facial_area"]
            })

            if len(batch_vectors) >= batch_size:
                try:
                    qdrant_service.upsert_points(qdrant_client, collection_name, batch_vectors, batch_meta, batch_ids)
                    uploaded += len(batch_vectors)
                    logger.info(f"[store_multiple_images_parallel] Uploaded batch of {len(batch_vectors)}")
                except Exception as e:
                    logger.exception(f"[store_multiple_images_parallel] Failed to upload a batch: {e}")
                finally:
                    batch_vectors, batch_meta, batch_ids = [], [], []

        # final upload
        if batch_vectors:
            try:
                uploaded += len(batch_vectors)
                qdrant_service.upsert_points(qdrant_client, collection_name, batch_vectors, batch_meta, batch_ids)
            except Exception as e:
                logger.exception(f"[store_multiple_images_parallel] Failed to upload final batch: {e}")

        logger.info(f"✅ Completed — uploaded {uploaded} embeddings to Qdrant collection '{collection_name}' in {time.time() - start_time:.2f}s")
        return uploaded
    except Exception as e:
        logger.exception(f"❌ [store_multiple_images_parallel] Fatal error: {e}")
        return None
