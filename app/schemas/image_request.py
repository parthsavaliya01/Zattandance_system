
from pydantic import BaseModel


class SingleImageRequest(BaseModel):
    profile_picture_path: str
    doctor_id: str
    doctor_name: str


class BulkImageRequest(BaseModel):
    root_dir: str