# app/api/v1/endpoints/health.py
from fastapi import APIRouter
from app.services.health_service import HealthService

router = APIRouter(prefix="/health", tags=["Health"])
health_service = HealthService()

@router.get("/")
def system_health():
    return health_service.system_health()

@router.get("/liveness")
def liveness():
    return {"status": "alive"}

@router.get("/readiness")
def readiness():
    return health_service.readiness()
