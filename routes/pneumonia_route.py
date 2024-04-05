import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import File, UploadFile
from fastapi import APIRouter
from schemas.pneumonia_schema import PneumoniaResponse
from config.pneumonia_cfg import ModelConfig
from models.pneumonia_predictor import Predictor

router = APIRouter()
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device = ModelConfig.DEVICE
)

@router.post("/predict")
async def predict(file_upload: UploadFile = File(...)):
    response = await predictor.predict(file_upload.file)
    
    return PneumoniaResponse(**response)