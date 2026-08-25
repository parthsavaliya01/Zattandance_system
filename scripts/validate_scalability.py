# scripts/validate_scalability.py
import sys
import os
import time
import uuid
import random
import logging
from hashlib import md5
import numpy as np

# Ensure app is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.db.qdrant_service import QdrantService
from app.scripts.qdrant_search import QdrantSearchEngine
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_scalability")

def generate_random_unit_vector(dim=512):
    vec = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)

def generate_variations(centroid, num_vars=5, noise_std=0.2):
    variations = []
    for _ in range(num_vars):
        noise = np.random.normal(0, noise_std, len(centroid))
        var = centroid + noise
        var_norm = var / (np.linalg.norm(var) + 1e-9)
        variations.append(var_norm)
    return variations

def main():
    logger.info("Initializing Scalability & Accuracy Validation Test (20 -> 220 Doctors)...")
    
    q_service = QdrantService()
    client = q_service.connect()
    
    if not client:
        logger.error("Failed to connect to Qdrant. Make sure Qdrant is running and accessible.")
        return
        
    centroids_col = settings.QDRANT_COLLECTION_CENTROIDS
    embeddings_col = settings.QDRANT_COLLECTION_EMBEDDINGS
    
    logger.info(f"Target Centroids Collection: {centroids_col}")
    logger.info(f"Target Embeddings Collection: {embeddings_col}")
    
    # 1. Inspect existing database state
    try:
        c_info = client.get_collection(centroids_col)
        e_info = client.get_collection(embeddings_col)
        logger.info(f"Existing Centroids Count: {c_info.points_count}")
        logger.info(f"Existing Embeddings Count: {e_info.points_count}")
    except Exception as e:
        logger.error(f"Error fetching collection info: {e}")
        return

    # Retrieve existing real doctors
    logger.info("Retrieving existing doctors from Qdrant...")
    scroll_res, _ = client.scroll(
        collection_name=centroids_col,
        limit=1000,
        with_vectors=True,
        with_payload=True
    )
    
    real_doctors = []
    for p in scroll_res:
        payload = p.payload or {}
        doc_id = payload.get("doctor_id")
        if doc_id and not doc_id.startswith("synthetic_doc_"):
            real_doctors.append({
                "id": p.id,
                "doctor_id": doc_id,
                "doctor_name": payload.get("doctor_name", "Unknown"),
                "vector": p.vector,
                "payload": payload
            })
            
    num_real = len(real_doctors)
    logger.info(f"Found {num_real} real enrolled doctors in Qdrant.")
    
    # Fetch their embeddings
    real_embeddings = {}
    for doc in real_doctors:
        doc_id = doc["doctor_id"]
        emb_scroll, _ = client.scroll(
            collection_name=embeddings_col,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doctor_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=100,
            with_vectors=True
        )
        real_embeddings[doc_id] = [p.vector for p in emb_scroll if p.vector is not None]
        logger.info(f"  Doctor '{doc['doctor_name']}' ({doc_id}) has {len(real_embeddings[doc_id])} enrollment embeddings.")

    # 2. Form Baseline Cohort (exactly 20 doctors)
    # If we have less than 20, we supplement with temporary baseline synthetic doctors
    baseline_doctors = list(real_doctors)
    baseline_supplemental_ids = []
    
    num_baseline_needed = 20
    if len(baseline_doctors) < num_baseline_needed:
        supplement_count = num_baseline_needed - len(baseline_doctors)
        logger.info(f"Supplementing baseline with {supplement_count} synthetic doctors to reach {num_baseline_needed}...")
        
        for i in range(supplement_count):
            doc_id = f"synthetic_doc_base_{i}"
            doc_name = f"Synthetic Base Doc {i}"
            centroid = generate_random_unit_vector()
            variations = generate_variations(centroid, num_vars=5, noise_std=0.18)
            
            # Save in list
            baseline_doctors.append({
                "doctor_id": doc_id,
                "doctor_name": doc_name,
                "vector": centroid.tolist()
            })
            baseline_supplemental_ids.append(doc_id)
            
            # Upsert centroid
            centroid_raw_id = f"{doc_id}_centroid"
            c_pid = int(md5(centroid_raw_id.encode()).hexdigest()[:16], 16)
            client.upsert(
                collection_name=centroids_col,
                points=[
                    models.PointStruct(
                        id=c_pid,
                        vector=centroid.tolist(),
                        payload={
                            "doctor_id": doc_id,
                            "doctor_name": doc_name,
                            "department": "Synthetic-Base",
                            "intra_doctor_spread": 0.85,
                            "adaptive_threshold": settings.VERIFICATION_THRESHOLD
                        }
                    )
                ]
            )
            
            # Upsert embeddings
            emb_points = []
            real_embeddings[doc_id] = []
            for idx, var in enumerate(variations):
                raw_id = f"{doc_id}_img_{idx+1}"
                pid = int(md5(raw_id.encode()).hexdigest()[:16], 16)
                emb_points.append(
                    models.PointStruct(
                        id=pid,
                        vector=var.tolist(),
                        payload={
                            "doctor_id": doc_id,
                            "doctor_name": doc_name,
                            "image_number": idx + 1,
                            "department": "Synthetic-Base"
                        }
                    )
                )
                real_embeddings[doc_id].append(var.tolist())
                
            client.upsert(collection_name=embeddings_col, points=emb_points)

    logger.info(f"Baseline cohort established with {len(baseline_doctors)} doctors.")

    # Instantiate search engine
    search_engine = QdrantSearchEngine()
    
    # Generate 100 random vectors for open-set FAR testing (unknown/non-enrolled faces)
    logger.info("Generating 100 random unit vectors for open-set FAR testing...")
    unknown_embeddings = [generate_random_unit_vector().tolist() for _ in range(100)]
    
    # ── PHASE 1: EVALUATE BASELINE (20 DOCTORS) ──────────────────────────────
    logger.info("=========================================")
    logger.info("PHASE 1: RUNNING BASELINE (20 DOCTORS) EVALUATION")
    logger.info("=========================================")
    
    p1_results = evaluate_gallery(baseline_doctors, real_embeddings, unknown_embeddings, search_engine, client, centroids_col)
    
    # ── PHASE 2: SCALE UP (ADD 200 SYNTHETIC DOCTORS) ────────────────────────
    logger.info("=========================================")
    logger.info("PHASE 2: ENROLLING 200 ADDITIONAL DOCTORS (TOTAL 220)")
    logger.info("=========================================")
    
    added_doc_ids = []
    num_scale_up = 200
    
    logger.info(f"Generating and uploading {num_scale_up} synthetic doctors...")
    
    # Bulk prep to speed up upserts
    scale_centroids = []
    scale_embeddings = []
    
    for i in range(num_scale_up):
        doc_id = f"synthetic_doc_scale_{i}"
        doc_name = f"Synthetic Scale Doc {i}"
        added_doc_ids.append(doc_id)
        
        centroid = generate_random_unit_vector()
        variations = generate_variations(centroid, num_vars=5, noise_std=0.18)
        
        # Centroid Point
        centroid_raw_id = f"{doc_id}_centroid"
        c_pid = int(md5(centroid_raw_id.encode()).hexdigest()[:16], 16)
        scale_centroids.append(
            models.PointStruct(
                id=c_pid,
                vector=centroid.tolist(),
                payload={
                    "doctor_id": doc_id,
                    "doctor_name": doc_name,
                    "department": "Synthetic-Scale",
                    "intra_doctor_spread": 0.85,
                    "adaptive_threshold": settings.VERIFICATION_THRESHOLD
                }
            )
        )
        
        # Embedding Points
        for idx, var in enumerate(variations):
            raw_id = f"{doc_id}_img_{idx+1}"
            pid = int(md5(raw_id.encode()).hexdigest()[:16], 16)
            scale_embeddings.append(
                models.PointStruct(
                    id=pid,
                    vector=var.tolist(),
                    payload={
                        "doctor_id": doc_id,
                        "doctor_name": doc_name,
                        "image_number": idx + 1,
                        "department": "Synthetic-Scale"
                    }
                )
            )
            
    # Upsert in chunks
    chunk_size = 50
    logger.info(f"Upserting centroids in chunks of {chunk_size}...")
    for idx in range(0, len(scale_centroids), chunk_size):
        client.upsert(collection_name=centroids_col, points=scale_centroids[idx:idx+chunk_size])
        
    logger.info(f"Upserting embeddings in chunks of {chunk_size * 5}...")
    for idx in range(0, len(scale_embeddings), chunk_size * 5):
        client.upsert(collection_name=embeddings_col, points=scale_embeddings[idx:idx+chunk_size*5])
        
    logger.info("Scale-up enrollment complete. Wait 2 seconds for indexes to update...")
    time.sleep(2.0)
    
    # ── PHASE 3: EVALUATE EXPANDED GALLERY (220 DOCTORS) ──────────────────────
    logger.info("=========================================")
    logger.info("PHASE 3: RUNNING EXPANDED (220 DOCTORS) EVALUATION")
    logger.info("=========================================")
    
    p2_results = evaluate_gallery(baseline_doctors, real_embeddings, unknown_embeddings, search_engine, client, centroids_col)

    # ── PHASE 4: CLEAN UP SYNTHETIC DATA ─────────────────────────────────────
    logger.info("=========================================")
    logger.info("PHASE 4: CLEANING UP SYNTHETIC DATA")
    logger.info("=========================================")
    
    cleanup_synthetic_data(client, centroids_col, embeddings_col, baseline_supplemental_ids + added_doc_ids)
    
    # ── PHASE 5: PRINT COMPREHENSIVE REPORT ──────────────────────────────────
    print_report(p1_results, p2_results, len(baseline_doctors), len(baseline_doctors) + num_scale_up)

def evaluate_gallery(doctors, doctor_embeddings, unknown_embeddings, search_engine, client, centroids_col):
    """
    Run closed-set and open-set evaluations against the current Qdrant database.
    """
    closed_set_total = 0
    closed_set_correct = 0
    closed_set_not_in_shortlist = 0
    closed_set_failed_verification = 0
    closed_set_wrong_id = 0
    
    centroid_ranks = []
    search_latencies = []
    
    # Closed-set evaluation: Query each enrolled doctor's embeddings
    for doc in doctors:
        doc_id = doc["doctor_id"]
        embeddings = doctor_embeddings.get(doc_id, [])
        if not embeddings:
            continue
            
        for emb in embeddings:
            closed_set_total += 1
            
            # Time the query
            start_t = time.perf_counter()
            results = search_engine.search(emb)
            elapsed = (time.perf_counter() - start_t) * 1000.0
            search_latencies.append(elapsed)
            
            # Check Stage 1 Centroid Shortlist Rank
            top_k = settings.TOP_K if hasattr(settings, "TOP_K") else 3
            centroid_res = client.query_points(
                collection_name=centroids_col,
                query=emb,
                limit=30
            )
            
            rank = -1
            for r_idx, r in enumerate(centroid_res.points):
                if (r.payload or {}).get("doctor_id") == doc_id:
                    rank = r_idx + 1
                    break
            
            centroid_ranks.append(rank)
            
            # Check recognition results
            if not results:
                # Recognition failed
                if rank == -1 or rank > top_k:
                    closed_set_not_in_shortlist += 1
                else:
                    closed_set_failed_verification += 1
            else:
                matched_id = results[0]["payload"].get("doctor_id")
                if matched_id == doc_id:
                    closed_set_correct += 1
                else:
                    closed_set_wrong_id += 1
                    
    # Open-set evaluation: Query 100 unknown random face vectors
    open_set_total = len(unknown_embeddings)
    open_set_false_accepts = 0
    highest_false_score = 0.0
    
    for emb in unknown_embeddings:
        results = search_engine.search(emb)
        if results:
            open_set_false_accepts += 1
            score = results[0]["score"]
            if score > highest_false_score:
                highest_false_score = score
                
    # Compile statistics
    avg_latency = sum(search_latencies) / len(search_latencies)
    accuracy = (closed_set_correct / closed_set_total) * 100.0 if closed_set_total > 0 else 0
    far = (open_set_false_accepts / open_set_total) * 100.0
    
    # Centroid shortlist recall
    top_k_val = settings.TOP_K if hasattr(settings, "TOP_K") else 3
    in_top_3_count = sum(1 for r in centroid_ranks if 1 <= r <= 3)
    in_top_5_count = sum(1 for r in centroid_ranks if 1 <= r <= 5)
    in_top_10_count = sum(1 for r in centroid_ranks if 1 <= r <= 10)
    not_in_list_count = sum(1 for r in centroid_ranks if r == -1)
    
    recall_top_k = sum(1 for r in centroid_ranks if 1 <= r <= top_k_val) / len(centroid_ranks) * 100.0
    
    return {
        "closed_set_total": closed_set_total,
        "closed_set_correct": closed_set_correct,
        "closed_set_not_in_shortlist": closed_set_not_in_shortlist,
        "closed_set_failed_verification": closed_set_failed_verification,
        "closed_set_wrong_id": closed_set_wrong_id,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "centroid_ranks": centroid_ranks,
        "in_top_3": in_top_3_count,
        "in_top_5": in_top_5_count,
        "in_top_10": in_top_10_count,
        "not_in_list": not_in_list_count,
        "recall_top_k": recall_top_k,
        "open_set_total": open_set_total,
        "open_set_false_accepts": open_set_false_accepts,
        "far": far,
        "highest_false_score": highest_false_score
    }

def cleanup_synthetic_data(client, centroids_col, embeddings_col, doc_ids):
    """
    Remove all temporary synthetic points from Qdrant.
    """
    logger.info(f"Removing {len(doc_ids)} synthetic doctor profiles from Qdrant...")
    for idx in range(0, len(doc_ids), 50):
        chunk = doc_ids[idx:idx+50]
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="doctor_id",
                    match=models.MatchAny(any=chunk)
                )
            ]
        )
        client.delete(collection_name=centroids_col, points_selector=models.FilterSelector(filter=filt))
        client.delete(collection_name=embeddings_col, points_selector=models.FilterSelector(filter=filt))
    logger.info("Cleanup complete.")

def print_report(p1, p2, n1, n2):
    """
    Format and output evaluation comparison results.
    """
    print("\n" + "="*60)
    print("      FACE ATTENDANCE SYSTEM SCALABILITY EVALUATION REPORT      ")
    print("="*60)
    print(f"Gallery Size Baseline:  {n1} doctors")
    print(f"Gallery Size Scaled:    {n2} doctors (Added 200 Synthetic Profiles)")
    print("-"*60)
    
    print(f"1. CLOSED-SET RECOGNITION ACCURACY:")
    print(f"   - Baseline ({n1} Docs) Accuracy: {p1['accuracy']:.2f}% ({p1['closed_set_correct']}/{p1['closed_set_total']})")
    print(f"   - Scaled   ({n2} Docs) Accuracy: {p2['accuracy']:.2f}% ({p2['closed_set_correct']}/{p2['closed_set_total']})")
    print(f"   - Accuracy Delta:            {p2['accuracy'] - p1['accuracy']:.2f}%")
    print(f"   - Wrong Identity Matches:    Baseline: {p1['closed_set_wrong_id']} | Scaled: {p2['closed_set_wrong_id']}")
    print(f"   - Failed Verifications:      Baseline: {p1['closed_set_failed_verification']} | Scaled: {p2['closed_set_failed_verification']}")
    print(f"   - Shortlist Misses:          Baseline: {p1['closed_set_not_in_shortlist']} | Scaled: {p2['closed_set_not_in_shortlist']}")
    print("-"*60)
    
    top_k_val = settings.TOP_K if hasattr(settings, "TOP_K") else 3
    print(f"2. STAGE 1 CENTROID SHORTLIST RECALL (TOP K = {top_k_val}):")
    print(f"   - Baseline ({n1} Docs) Recall:   {p1['recall_top_k']:.2f}%")
    print(f"   - Scaled   ({n2} Docs) Recall:   {p2['recall_top_k']:.2f}%")
    print(f"   - Centroid Rank Distribution (Baseline):")
    print(f"       Rank 1-3: {p1['in_top_3']} | Rank 4-5: {p1['in_top_5'] - p1['in_top_3']} | Rank 6-10: {p1['in_top_10'] - p1['in_top_5']} | Miss (>10): {p1['not_in_list']}")
    print(f"   - Centroid Rank Distribution (Scaled):")
    print(f"       Rank 1-3: {p2['in_top_3']} | Rank 4-5: {p2['in_top_5'] - p2['in_top_3']} | Rank 6-10: {p2['in_top_10'] - p2['in_top_5']} | Miss (>10): {p2['not_in_list']}")
    print("-"*60)
    
    print(f"3. OPEN-SET FALSE ACCEPTANCE RATE (FAR) (Threshold = {settings.VERIFICATION_THRESHOLD}):")
    print(f"   - Baseline ({n1} Docs) FAR:      {p1['far']:.2f}% ({p1['open_set_false_accepts']}/{p1['open_set_total']})")
    print(f"   - Scaled   ({n2} Docs) FAR:      {p2['far']:.2f}% ({p2['open_set_false_accepts']}/{p2['open_set_total']})")
    print(f"   - Max Cosine Score of False: Baseline: {p1['highest_false_score']:.4f} | Scaled: {p2['highest_false_score']:.4f}")
    print("-"*60)
    
    print(f"4. LATENCY & PERFORMANCE:")
    print(f"   - Baseline ({n1} Docs) Latency:  {p1['avg_latency_ms']:.2f} ms")
    print(f"   - Scaled   ({n2} Docs) Latency:  {p2['avg_latency_ms']:.2f} ms")
    print(f"   - Latency Increase:          {p2['avg_latency_ms'] - p1['avg_latency_ms']:.2f} ms")
    print("="*60)
    print("RECOMMENDATION DECISION GUIDELINE:")
    
    # Rule engine for recommendations
    if p2['accuracy'] < 95.0 or (p2['recall_top_k'] < p1['recall_top_k'] - 2.0):
        print("🚨 CRITICAL: Shortlist truncation detected! True centroids are being pushed out of Top K.")
        print(f"   Action: INCREASE TOP_K in .env (e.g. set TOP_K=10). Run this script again to verify.")
    elif p2['far'] > 0.0:
        print("⚠️ WARNING: Open-set False Acceptance detected!")
        print(f"   Action: Increase VERIFICATION_THRESHOLD slightly (e.g., to 0.40) or enable ADAPTIVE_THRESHOLD_ENABLED.")
    else:
        print("✅ SUCCESS: System scaled to 220 doctors with negligible accuracy impact.")
        print("   Action: No architectural changes required. The current system is suitable for production.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
