from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os

router = APIRouter()

MISSED_PROOFS_DIR = "outputs/missed_proofs"

@router.get("/missed_proofs")
def get_missed_proofs():
    """Returns a list of all missed attendance proof images."""
    if not os.path.exists(MISSED_PROOFS_DIR):
        return JSONResponse(content={"proofs": []})

    proofs = []
    
    # Iterate through date directories
    for date_dir in sorted(os.listdir(MISSED_PROOFS_DIR), reverse=True):
        date_path = os.path.join(MISSED_PROOFS_DIR, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        # Iterate through categories (blurry, invalid_angle, low_score)
        for category in os.listdir(date_path):
            category_path = os.path.join(date_path, category)
            if not os.path.isdir(category_path):
                continue
                
            for filename in os.listdir(category_path):
                if not filename.endswith(".jpg"):
                    continue
                    
                # Parse filename to get details if possible
                # e.g. 20260625_115755_SCORE-0.250_Unknown_cam111.jpg
                score = "N/A"
                name = "N/A"
                
                if category == "low_score":
                    parts = filename.split("_")
                    for p in parts:
                        if p.startswith("SCORE-"):
                            score = p.replace("SCORE-", "")
                    
                    # Name is usually between SCORE and cam111
                    try:
                        score_idx = next(i for i, p in enumerate(parts) if p.startswith("SCORE-"))
                        cam_idx = next(i for i, p in enumerate(parts) if p.startswith("cam"))
                        if cam_idx > score_idx + 1:
                            name = "_".join(parts[score_idx+1:cam_idx])
                    except Exception:
                        pass
                        
                elif category == "blurry":
                    parts = filename.split("_")
                    for p in parts:
                        if p.startswith("var-"):
                            score = p.replace("var-", "")

                url = f"/outputs/missed_proofs/{date_dir}/{category}/{filename}"
                
                proofs.append({
                    "date": date_dir,
                    "category": category,
                    "filename": filename,
                    "url": url,
                    "score": score,
                    "name": name.replace("_", " ") if name != "N/A" else name,
                    "timestamp": parts[1] if "parts" in locals() and len(parts) > 1 else ""
                })
                
    return JSONResponse(content={"proofs": proofs})
