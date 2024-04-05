import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from middleware import LogMiddleware, setup_cors
from routes.base import router

app = FastAPI()
setup_cors(app)
app.include_router(router)