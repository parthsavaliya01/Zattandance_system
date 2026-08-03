import os
import time
from typing import Optional
from fastapi import APIRouter, Query
from app.db.qdrant_service import QdrantService
from app.core.config import settings

router = APIRouter(prefix="/image_logs", tags=["logs"])

# Cache for doctor_id -> doctor_name mapping
doctor_name_cache = {}
last_cache_time = 0
CACHE_TTL = 300  # 5 minutes

def get_doctor_mapping():
    global doctor_name_cache, last_cache_time
    if time.time() - last_cache_time < CACHE_TTL and doctor_name_cache:
        return doctor_name_cache
    
    try:
        q = QdrantService()
        client = q.connect()
        res, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION_CENTROIDS, 
            with_payload=True, 
            limit=10000
        )
        mapping = {}
        for p in res:
            if p.payload:
                doc_id = str(p.payload.get("doctor_id"))
                name = p.payload.get("doctor_name", "Unknown")
                mapping[doc_id] = name
        
        doctor_name_cache = mapping
        last_cache_time = time.time()
    except Exception as e:
        print(f"[Image Logs API] Failed to fetch doctor mapping: {e}")
        
    return doctor_name_cache

@router.get("/")
def get_image_logs(
    start_date: Optional[str] = Query(None, description="Format: YYYYMMDD"),
    end_date: Optional[str] = Query(None, description="Format: YYYYMMDD"),
    doctor_name: Optional[str] = Query(None, description="Case insensitive partial match"),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None)
):
    recognized_dir = "outputs/recognized"
    frames_dir = "outputs/frames"
    
    mapping = get_doctor_mapping()
    logs = []
    
    if not os.path.exists(recognized_dir):
        return []
        
    files = os.listdir(recognized_dir)
    # Sort files by newest first (based on filename timestamp)
    files.sort(reverse=True)
    
    for filename in files:
        if not filename.endswith(".jpg"):
            continue
            
        # Example filename: 11112_1234555_0.401_20260527_173619.jpg
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) < 5:
            continue
            
        camera_id = parts[0]
        doctor_id = parts[1]
        score_str = parts[2]
        date_str = parts[3]
        time_str = parts[4]
        
        try:
            score = float(score_str)
        except ValueError:
            continue
            
        doc_name = mapping.get(doctor_id, "Unknown")
        
        # Apply filters
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue
        if doctor_name and doctor_name.lower() not in doc_name.lower():
            continue
            
        # Format the display date
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        
        # The frame image format: {camera_id}_{doctor_id}_{YYYYMMDD}_{HHMMSS}.jpg
        frame_filename = f"{camera_id}_{doctor_id}_{date_str}_{time_str}.jpg"
        frame_url = f"/outputs/frames/{frame_filename}"
        
        # Fallback to detected_face if frame doesn't exist (though it should)
        if not os.path.exists(os.path.join(frames_dir, frame_filename)):
            frame_url = f"/outputs/detected_face/{frame_filename}"
            
        logs.append({
            "camera_id": camera_id,
            "doctor_id": doctor_id,
            "doctor_name": doc_name,
            "score": score,
            "date_formatted": formatted_date,
            "face_url": f"/outputs/recognized/{filename}",
            "frame_url": frame_url
        })
        
    return logs
