import torch
import uvicorn
from fastapi import FastAPI

from src.router.router import router


app = FastAPI(
    title="dermatology-medgemma-service"
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7021
    )