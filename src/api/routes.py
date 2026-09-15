"""FastAPI application endpoints."""
from fastapi import APIRouter
from src.schemas.optimization import OptimizationRequest, OptimizationResponse
from src.schemas.cad import CADParametricSpec, CADExportResponse
from src.optimizer.qehvi_optimizer import QEHVIHardwareOptimizer
from src.cad.exporter import CADExporter

router = APIRouter()
optimizer = QEHVIHardwareOptimizer()


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "hardware-pareto-optimizer"}


@router.post("/optimize", response_model=OptimizationResponse)
def optimize_design(request: OptimizationRequest):
    return optimizer.optimize(request)


@router.post("/export-cad", response_model=CADExportResponse)
def export_cad(spec: CADParametricSpec):
    return CADExporter.export(spec)
