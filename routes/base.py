from fastapi import APIRouter
from .pneumonia_route import router as pneumonia_cls_route

router = APIRouter()
router.include_router(pneumonia_cls_route, prefix = "/pneumonia_classification")