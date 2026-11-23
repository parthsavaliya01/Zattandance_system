
from pydantic import BaseModel
from typing import List


class SingleImageRequest(BaseModel):
    profile_picture_path: str
    doctor_id: str
    doctor_name: str


class BulkImageRequest(BaseModel):
    root_dir: str


class DoctorEmbeddingReplaceRequest(BaseModel):
    doctor_id: str
    doctor_name: str
    image_paths: List[str]