# AI Attendance System — Architecture & Operations Record

Living technical reference for the face-recognition attendance service. Update it after every significant change.
Last updated: 2026-11-25.

---

## 1. System overview

Two RTSP cameras (entry, exit) feed a single FastAPI process that detects, tracks, and recognises doctors. A recognised doctor produces an event on RabbitMQ; a separate consumer writes the attendance row to MySQL. The admin panel (Node backend in `~/Zydus/Backend`) reads MySQL and manages doctors, cameras, and enrollment.

```
MySQL tbl_cameras ──(every 10 s)──► FaceRecognition.sync_cameras ──► one StreamWorker per camera
  StreamWorker (4 threads)
    capture   : cv2 RTSP (FFMPEG) ──► frame_queue (max 5, drops oldest)
    yolo      : best.pt head detector + ByteTrack (conf 0.4) + blur/size gates ──► good_face_queue (max 6)
    inference : extract_face_embedding ──► GPU_TASK_QUEUE ──► GPUWorker thread
                  SCRFD (InsightFace buffalo_sc) on full frame → keep face overlapping YOLO box
                  → 5-point affine align to 112×112 → AdaFace IR-101 ONNX → 512-d L2-normalised vector
              ──► QdrantSearchEngine.search ──► per-track 2-of-3 ──► 300 s cooldown
              ──► save crops (outputs/) + publish to RabbitMQ "face_recognition_events"
    heartbeat
attendance_consumer ──► entry/exit debounce (120 s) ──► MySQL tbl_attendance
Admin backend ──► reads tbl_attendance (joins doctor by id OR employee_id)
```

## 2. Components

| Component | File / function | Input → Output |
|---|---|---|
| Entrypoint | `main.py` `lifespan` | Starts GPUWorker, loads YOLO, starts camera manager thread |
| Camera config | `app/db/camera_db.py` `fetch_active_cameras`, `normalize_camera_stream_url` | `tbl_cameras` rows (`is_active=1, is_deleted=0`, area entry/exit) → `{camera_id, url, area}` |
| Camera manager | `app/services/face_recognition_manager.py` `sync_cameras`, `monitor` | Starts/stops/restarts StreamWorker threads when DB config changes |
| Stream worker | `app/services/stream_worker.py` | Frames → attendance events |
| GPU worker | `app/services/gpu_worker.py` `GPUWorker._process_frame`, `_embed`, `get_insight_app` | Full frame + YOLO box → `[{embedding, facial_area}]` |
| YOLO loader | `app/models/load_yolo_model.py` | `app/models/best.pt` (one instance per camera) |
| Matching | `app/scripts/qdrant_search.py` `QdrantSearchEngine.search` | 512-d vector → `[top1]` or `[]` |
| Enrollment | `app/api/v1/endpoints/upload_embedding.py`, `app/services/store_embedding.py`, `app/scripts/embeddings.py` | Doctor images → Qdrant points |
| Publisher | `app/scripts/rabbitmq_publisher.py`, `publisher_singleton.py` | Event dict → queue |
| Consumer | `app/services/attendance_consumer.py` → `app/scripts/rabbitmq_consumer.py` | Event → `tbl_attendance` |
| Attendance DB | `app/db/attendance_handler.py` `AttendanceService` | Insert / close sessions |
| Health | `app/services/health_service.py`, `app/api/v1/endpoints/health.py` | Camera threads, Qdrant, RabbitMQ, **GPU** |

## 3. Models and embedding format

| Model | File | Runtime | Notes |
|---|---|---|---|
| Head detector + tracker | `app/models/best.pt` (MD5 prefix `cb35afc0562d`) | ultralytics 8.4.7 / torch, ByteTrack | `track(persist=True, conf=0.4, iou=0.5)` |
| Face detector + landmarks | InsightFace `buffalo_sc` (SCRFD-500M) | onnxruntime-gpu 1.23.2 | `det_size=640`, min det score 0.45 |
| Face embedding | `app/models/adaface_ir101.onnx` + `.onnx.data` | onnxruntime-gpu 1.23.2 | BGR, mean 0.5 / std 0.5, output L2-normalised 512-d |

Model files are git-ignored. They are also baked into the Docker image by `COPY`, but the running container uses the bind-mounted host copies (`./:/app`).

## 4. Face registration (enrollment)

1. The admin panel saves doctor photos in `~/Zydus/Backend/uploads/doctors/<id>/` (fields `profile_picture`, `front_face`, `left_face`, `right_face`, `down_face`).
2. On create/edit the backend calls `POST /api/v1/store_embedding/replace_doctor_embeddings` with `doctor_id = tbl_doctors.id` (numeric), then deletes the legacy `employee_id` points.
3. On delete it calls `DELETE /api/v1/store_embedding/doctors/{id}/embeddings` for both `id` and `employee_id`, then sets `tbl_doctors.is_deleted = 1`.
4. Qdrant collections (cosine, 512-d):
   - `doctor_embeddings`: one point per image; payload `doctor_id, doctor_name, department, profile_path, facial_area, image_number`.
   - `doctor_centroids`: one point per doctor; payload `doctor_id, doctor_name, department, adaptive_threshold, intra_doctor_spread`.
5. ID scheme: doctors enrolled before 2026-11-23 use `employee_id` (e.g. `NEETA Mam`); doctors enrolled through the new flow use the numeric `id` (e.g. `153`). The admin attendance join handles both.

## 5. Matching and acceptance rules

Settings in `app/core/config.py`, overridable in `.env`:

| Rule | Value |
|---|---|
| Stage 1 | `doctor_centroids` top-K (`TOP_K=5`) |
| Stage 2 | Cosine against all embeddings of each candidate; `passed = avg ≥ VERIFICATION_THRESHOLD (0.36) AND ≥2 embeddings ≥ 0.36` |
| Candidate used | Only centroid rank 1; if it fails, the frame is rejected |
| Ambiguity | Reject if top-1 score < 0.400 and (top-1 − top-2) < `SCORE_MARGIN_THRESHOLD` (0.010) |
| Temporal | Per ByteTrack track, last `FRAME_BUFFER_SIZE=3` verified frames; fire when the same doctor appears `MIN_MATCHES=2` times |
| Special rule | "Nancy Amin" requires score ≥ 0.42 |
| Cooldown | 300 s per doctor per camera worker |
| Consumer debounce | Entry: ignore within 120 s of last `in_time`; Exit: ignore within 120 s of last `out_time`, or within 10 s of entry |
| Adaptive thresholds | Stored in centroids but `ADAPTIVE_THRESHOLD_ENABLED=False` |

Silent rejections: low score, ambiguous match, invalid face area, and no-face frames are not logged unless `TELEMETRY_LOGGING_ENABLED=True`. Missed-proof images are saved only for `unconfirmed` (verified once, but temporal failed).

## 6. Database (MySQL `zydus_attendance`, on the host at `172.17.0.1:3306`)

| Table | Used for | Key columns |
|---|---|---|
| `tbl_cameras` | Camera config (soft-delete; a new row per edit) | `camera_id, camera_stream_url, monitoring_area, is_active, is_deleted` |
| `tbl_doctors` | Doctor master | `id, employee_id, full_name, *_face, is_active, is_deleted` |
| `tbl_attendance` | Sessions | `id, doctor_id (id or employee_id), attendance_date, in_time, out_time, entry_camera_id, exit_camera_id, in_image_path, out_image_path, total_hours` |

## 7. API (prefix `/api/v1`, port 8000)

- `POST /store_embedding/replace_doctor_embeddings`, `DELETE /store_embedding/doctors/{doctor_id}/embeddings`, `POST /store_embedding/generate_single_embedding`, `POST /store_embedding/generate_multiple_embeddings`
- `POST /face-recognition/start`, `GET /face-recognition/status`
- `POST /cameras/sync`, `GET /cameras/status`, `POST /start/{camera_id}`, `POST /stop/{camera_id}`
- `GET /health/` (includes `gpu`), `GET /health/liveness`, `GET /health/readiness`
- `GET /image_logs/`, `GET /missed_proofs`
- Static: `/outputs`, `/recordings`

### Web pages (static, served by the AI service from `outputs/`)

| Page | URL | Data | Shows |
|---|---|---|---|
| Dashboard | `http://<server>:8000/outputs/dashboard.html` | `GET /api/v1/image_logs/` (+ `/missed_proofs` for the Captured Faces card) | Confirmed attendance events (each `ATTENDANCE FIRED` saves `outputs/recognized/` crop + `outputs/frames/` full frame) |
| Entry Missed Proofs | `/outputs/entry_missed_proofs.html` | `GET /api/v1/missed_proofs` | Unconfirmed proofs from entry cameras |
| Exit Missed Proofs | `/outputs/exit_missed_proofs.html` | `GET /api/v1/missed_proofs` | Unconfirmed proofs from exit cameras |

- Entry vs exit: `CAMERA_TYPES = {'111': 'entry', '11112': 'exit'}` at the top of each page's script (same rule as `setup_missed_proof_logger`). Update it if camera IDs change.
- Summary cards: Total Entry / Total Exit = confirmed events on entry / exit cameras; Total Confirmed Attendance = Entry + Exit; Total Captured Faces = confirmed faces + missed-proof faces in the same filters. These are AI recognition events, not MySQL `tbl_attendance` rows (the consumer drops duplicates within 120 s).
- Filters: date range, name and min score go to the API; ID, entry/exit and camera are filtered in the browser. Missed-proof pages filter entirely in the browser.
- Limitations: full frames exist only from July 2026 onward (older records show a placeholder); missed proofs store only a padded face crop (no full frame) and no doctor ID; names come from Qdrant, so deleted doctors show "Unknown".
- Missed proofs are unconfirmed by definition: when a track confirms, its proofs are deleted (`MISSED PROOF CLEANUP` in `stream_worker.py`).
- History: built by an IDE agent Jun–Aug 2026 and lost (`outputs/` was git-ignored); restored 2026-11-25 from the 2026-07-28 zip (dashboard, Jun 23 version), editor Local History (missed proofs, Jul 31), and the exact nav bar from agent transcripts. `.gitignore` now tracks `outputs/*.html`. **Any cleanup of `outputs/` must keep `*.html`.**

## 8. Docker services (`docker-compose.yml`, project `attendance_system_clean`)

| Service | Container | Notes |
|---|---|---|
| `qdrant` | `qdrant` | Bind mount `./qdrant_data` (root-owned) |
| `rabbitmq` | `rabbitmq` | Named volume `rabbitmq_data`; healthcheck |
| `api` | `face_recognition_api` | `runtime: nvidia`; bind mount `./:/app` (host code is the live code); uvicorn without `--reload`, so code changes need a container restart |
| `rabbitmq_consumer` | `attendance_consumer` | Same image; `python -m app.services.attendance_consumer` |

Image base: `nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`, Python 3.11. `torch` is not pinned in `requirements.txt` (it comes in through ultralytics).

Host GPU: **NVIDIA GeForce RTX 5070 (PCI 10de:2f04). This GPU requires the NVIDIA *open* kernel module** (`nvidia-driver-580-open`). The proprietary `nvidia-driver-580` loads but exposes no devices.

## 9. Logging

All log timestamps are in **IST** (`settings.TIMEZONE`, via `_LocalTimeFormatter` in `app/core/logging_config.py`, added 2026-11-25). Before that change the logs were in container UTC (5½ h behind MySQL and the admin panel). To find attendance quickly, search camera logs for `ATTENDANCE FIRED`.

- `logs/faceRecognition.log`: app-wide (startup, `[GPU-CHECK]`, Qdrant, enrollment)
- `logs/camera_<id>.log`: per-camera decisions (`[FACE_ATTENDANCE]`, `ATTENDANCE FIRED`, `COOLDOWN`)
- `logs/missed_proof_<area>.log`, `logs/rabbitmq_consumer.log`
- `docker logs face_recognition_api`: also contains onnxruntime/torch warnings (e.g. CPU fallback). History persists across container restarts.

---

## 10. Incident record — November 2026 attendance drop

**Symptom:** registered doctors walking past were often not recognised; attendance was not captured.

**Baseline (`tbl_attendance` rows/day):** 41–48 on weekdays in late October 2026 → 1–14/day from 2026-11-05.

**Confirmed root cause (primary):** the GPU became unusable on 2026-11-05.
- An unattended-upgrade moved the NVIDIA userspace to 580.178.04 while the 580.173.02 open kernel module stayed loaded (mismatch warnings from 2026-10-10).
- 2026-11-05 09:59: a container restart failed with an NVML driver/library mismatch.
- 10:56: `apt-get purge ^nvidia-.*` removed `nvidia-driver-580-open`. The reinstall (`nvidia-driver-525`) pulled in the proprietary `nvidia-driver-580`.
- After the 11:34 reboot, the kernel logged `GPU 10de:2f04 requires use of the NVIDIA open kernel modules`, and `nvidia-smi` reported "No devices were found".
- onnxruntime fell back to `CPUExecutionProvider` and torch reported `CUDA unknown error`. YOLO, SCRFD and AdaFace all ran on CPU (0.24–1.44 s per frame). ByteTrack fragmented tracks, so the 2-of-3 confirmation rarely completed.

**Other findings:**
- The entry camera (111) was replaced around 2026-11-03 (new URL `/video/live?channel=1&subtype=0`, 1920×1080 instead of 3200×1800). It had 0 events from Nov 3 to 22 and reconnected after commit `48e3be0` (RTSP password URL-normalisation).
- The 2026-11-23 `git restore` / `rm -rf` was **not** the cause. Recognition code at HEAD matches the October runtime (`e9b2870`); the only difference is the missed-proof save toggles. Libraries and the AdaFace model are unchanged. Stored embeddings are compatible with the restored model.
- 7 doctors were deleted on purpose through the admin panel (2026-11-23/25).
- 32 of 63 active doctors have no embeddings. This is pre-existing: they were never enrolled in the current collections.
- Enrollment quality: many doctors' 5 stored vectors are identical (1 effective photo); two doctors (`SN65464`, `PS6495`) have only 2 embeddings.
- `~/attendance_remaining_backup.tar.gz` is truncated (unexpected EOF) and contains no code. Docker image `0130ff4641d5` (built 2026-11-05 13:15) holds a `COPY . .` snapshot of the code as of that time.
- Minor: the RabbitMQ health check raises `'RabbitMQPublisher' object has no attribute 'channel'`; a Qdrant reconnect drops that frame; timed-out GPU tasks leave orphan entries in `GPU_RESULT_MAP`.

**Fix applied (2026-11-25):**
1. Host: `sudo apt-get install -y nvidia-driver-580-open` + reboot. Verified: `nvidia-smi` shows the RTX 5070; the kernel log shows "Open Kernel Module 580.178.04"; the container log shows `Applied providers: ['CUDAExecutionProvider', ...]` with no fallback; the GPU process uses 1886 MiB.
2. Code (guard against silent recurrence): `gpu_worker.py` and `load_yolo_model.py` log `[GPU-CHECK]` (an ERROR if not on GPU); `/api/v1/health/` adds a `gpu` field (`ok` / `cpu_fallback`). No change to matching behaviour.

**Live test (2026-11-25 12:37 IST, GPU active):** a doctor enrolled with embeddings (ID `153`) was recognised on the exit camera (score 0.435) and the entry camera (score 0.405). Both events were saved (`tbl_attendance` 116056 exit, 116057 entry). Per-frame inference is now 0.02–0.06 s, down from 0.24–1.44 s on CPU. Telemetry showed every other face seen in that window scoring ≤ 0.25 against all enrolled doctors (well below 0.36), so all were correctly rejected with no false accepts. `TELEMETRY_LOGGING_ENABLED=True` was used temporarily (12:28–12:50 IST) and then removed: it buried the attendance lines under thousands of `MULTI-FRAME MISS` lines and wrote about 50 MB/hour. Turn it on again only to debug a specific missed doctor.

**Acceptance target (approved by the owner):** over 3 consecutive weekdays, each still-enrolled doctor's daily attendance rate reaches at least 80% of their 2026-10-01 to 2026-11-02 weekday rate. No unknown person or wrong doctor is accepted in a manual review of `outputs/recognized/`.

**Remaining work:**
- [ ] Restart `api` to load the `[GPU-CHECK]` guard; confirm `/api/v1/health/` reports `gpu.status = ok`.
- [ ] 3-weekday live verification against the baseline.
- [ ] Validate the new entry camera with real doctors (so far only the test user has fired on it); adjust the camera's angle and position if needed.
- [ ] Hold NVIDIA packages from unattended-upgrades to prevent a repeat.
- [ ] Optional: compare the Docker image `0130ff4641d5` snapshot with HEAD to confirm no other uncommitted work was lost.
- [ ] Optional: enroll the 32 doctors without embeddings; re-enroll weak (duplicate-vector) enrollments.
