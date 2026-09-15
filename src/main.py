"""FastAPI Application Entrypoint."""

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Hardware Pareto Optimizer API",
    description="Multi-objective constrained Bayesian optimization (qEHVI) with parametric CAD export",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
