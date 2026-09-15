"""Unit and Integration Tests for Hardware Pareto Optimizer & API."""

import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas.cad import CADParametricSpec
from src.cad.exporter import CADExporter
from src.schemas.optimization import OptimizationRequest
from src.optimizer.qehvi_optimizer import QEHVIHardwareOptimizer


client = TestClient(app)


def test_cad_exporter_geometry() -> None:
    spec = CADParametricSpec(
        part_name="test_heatsink",
        length_mm=100.0,
        width_mm=50.0,
        base_thickness_mm=5.0,
        fin_height_mm=25.0,
        fin_count=10,
        fin_thickness_mm=1.0,
        material="Aluminum_6061-T6",
        material_density_g_cm3=2.7,
    )
    result = CADExporter.export(spec)

    assert result.part_name == "test_heatsink"
    assert result.computed_volume_cm3 > 0.0
    assert result.computed_mass_grams > 0.0
    assert result.total_surface_area_cm2 > 0.0
    assert "cube" in result.openscad_script
    assert len(result.feature_tree["features"]) == 2


def test_qehvi_optimization_engine() -> None:
    optimizer = QEHVIHardwareOptimizer()

    # 6 historical evaluations of a 2D parameter space, 2 objectives to maximize
    observed_x = [
        [10.0, 5.0],
        [20.0, 10.0],
        [30.0, 15.0],
        [40.0, 20.0],
        [50.0, 25.0],
        [60.0, 30.0],
    ]
    # Synthetic objectives: [ -thermal_res, -mass ]
    observed_obj = [
        [-0.8, -150.0],
        [-0.6, -200.0],
        [-0.5, -280.0],
        [-0.4, -360.0],
        [-0.35, -450.0],
        [-0.3, -560.0],
    ]
    bounds = [[10.0, 80.0], [5.0, 40.0]]
    ref_point = [-1.0, -600.0]

    request = OptimizationRequest(
        observed_x=observed_x,
        observed_obj=observed_obj,
        param_bounds=bounds,
        ref_point=ref_point,
        num_candidates=1,
    )

    response = optimizer.optimize(request)

    assert len(response.pareto_frontier) > 0
    assert len(response.candidates) == 1
    assert response.hypervolume > 0.0
    candidate_params = response.candidates[0].parameters
    assert bounds[0][0] <= candidate_params[0] <= bounds[0][1]
    assert bounds[1][0] <= candidate_params[1] <= bounds[1][1]


def test_api_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_cad_export() -> None:
    payload = {
        "part_name": "chassis_cooler",
        "length_mm": 80.0,
        "width_mm": 40.0,
        "base_thickness_mm": 4.0,
        "fin_height_mm": 20.0,
        "fin_count": 8,
        "fin_thickness_mm": 1.2,
        "material": "Copper_C11000",
        "material_density_g_cm3": 8.96,
    }
    response = client.post("/api/v1/export-cad", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["part_name"] == "chassis_cooler"
    assert data["computed_mass_grams"] > 0
    assert "openscad_script" in data
