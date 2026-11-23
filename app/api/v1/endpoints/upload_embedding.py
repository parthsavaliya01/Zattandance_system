# app/api/v1/endpoints/upload_embedding.py
import logging
import os
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas.image_request import (
    BulkImageRequest,
    DoctorEmbeddingReplaceRequest,
    SingleImageRequest,
)
from app.services.store_embedding import (
    delete_doctor_embeddings,
    replace_doctor_embeddings,
    store_multiple_images_parallel,
    store_single_image,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/store_embedding", tags=["store_embedding"])

DOCTOR_IMAGES_ROOT = Path(
    os.getenv(
        "DOCTOR_IMAGES_ROOT",
        "/home/admin1/Zydus/Backend/uploads/doctors",
    )
)


def _is_front_profile_path(path: str) -> bool:
    file_name = os.path.basename(path).lower()
    keywords = [
        x.strip().lower()
        for x in (settings.ENROLL_FRONT_FILENAME_KEYWORDS or "").split(",")
        if x.strip()
    ]
    return any(keyword in file_name for keyword in keywords)


def _validate_doctor_image_paths(doctor_id: str, image_paths: list[str]) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", doctor_id):
        raise HTTPException(status_code=400, detail="Invalid doctor ID")

    doctor_directory = (DOCTOR_IMAGES_ROOT / doctor_id).resolve()
    validated_paths = []

    for image_path in image_paths:
        try:
            resolved_path = Path(image_path).resolve(strict=True)
            resolved_path.relative_to(doctor_directory)
        except (OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Enrollment images must be existing files in that doctor's upload folder",
            )

        if not resolved_path.is_file() or resolved_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            raise HTTPException(
                status_code=400,
                detail="Enrollment image must be a JPG, JPEG, or PNG file",
            )
        validated_paths.append(str(resolved_path))

    return validated_paths


@router.post("/replace_doctor_embeddings")
async def replace_doctor_embedding_data(request: DoctorEmbeddingReplaceRequest):
    try:
        image_paths = _validate_doctor_image_paths(
            request.doctor_id,
            request.image_paths,
        )
        if not image_paths:
            raise HTTPException(status_code=400, detail="At least one doctor image is required")

        result = replace_doctor_embeddings(
            doctor_id=request.doctor_id,
            doctor_name=request.doctor_name,
            image_paths=image_paths,
        )
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[replace_doctor_embedding_data] Failed for doctor %s",
            request.doctor_id,
        )
        raise HTTPException(status_code=500, detail="Doctor embedding replacement failed") from e


@router.delete("/doctors/{doctor_id}/embeddings")
async def remove_doctor_embedding_data(doctor_id: str):
    try:
        result = delete_doctor_embeddings(doctor_id)
        return {"status": "success", **result}
    except Exception as e:
        logger.exception(
            "[remove_doctor_embedding_data] Failed for doctor %s",
            doctor_id,
        )
        raise HTTPException(status_code=500, detail="Doctor embedding deletion failed") from e


@router.post("/generate_single_embedding")
async def upload_single_embedding(request: SingleImageRequest):
    """Generate + upload embedding for one image using its file path."""
    try:
        if not os.path.exists(request.profile_picture_path):
            raise HTTPException(status_code=400, detail="face_path does not exist")

        if settings.ENROLL_FRONT_ONLY and not _is_front_profile_path(request.profile_picture_path):
            logger.info(
                f"[upload_single_embedding] Skipping non-front profile image: {request.profile_picture_path}"
            )
            return {
                "status": "skipped",
                "message": "Only front profile photo is enrolled",
                "face_path": request.profile_picture_path,
                "doctor_id": request.doctor_id,
            }

        stored = store_single_image(
            face_path=request.profile_picture_path,
            doctor_id=request.doctor_id,
            doctor_name=request.doctor_name
        )

        if stored is None:
            logger.error(f"[upload_single_embedding] Embedding extraction/storage failed for {request.profile_picture_path}")
            raise HTTPException(status_code=500, detail="Embedding extraction or storage failed")

        return {
            "status": "success",
            "message": "Embedding stored in Qdrant",
            "total profiles": stored,
            "face_path": request.profile_picture_path,
            "doctor_id": request.doctor_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[upload_single_embedding] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.post("/generate_multiple_embeddings")
async def upload_bulk_embeddings(request: BulkImageRequest):
    """Scan doctor folders and upload embeddings for all images found."""
    try:
        if not os.path.exists(request.root_dir):
            raise HTTPException(status_code=400, detail="root_dir does not exist")

        embeddings_length = store_multiple_images_parallel(
            root_dir=request.root_dir,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        if embeddings_length is None:
            raise HTTPException(
                status_code=500,
                detail="Fatal error occurred while generating embeddings"
            )
        
        if embeddings_length == 0:
            raise HTTPException(
                status_code=500,
                detail="No embeddings generated — check logs for DeepFace or CUDA errors"
            )
        return {
            "status": "success",
            "message": f"Bulk embedding upload completed for {embeddings_length} images",
            "root_dir": request.root_dir,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[upload_bulk_embeddings] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
