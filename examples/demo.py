"""Interactive Multi-Objective Constrained Hardware Optimization Demo."""

from src.schemas.cad import CADParametricSpec
from src.cad.exporter import CADExporter
from src.schemas.optimization import OptimizationRequest
from src.optimizer.qehvi_optimizer import QEHVIHardwareOptimizer


def main() -> None:
    print("=== Hardware Pareto Optimizer & CAD Exporter Demo ===")

    print("\n1. Running Parametric CAD Synthesis for Aluminum Heatsink...")
    cad_spec = CADParametricSpec(
        part_name="chassis_heatsink_v1",
        length_mm=120.0,
        width_mm=60.0,
        base_thickness_mm=6.0,
        fin_height_mm=35.0,
        fin_count=12,
        fin_thickness_mm=1.5,
        material="Aluminum_6061-T6",
        material_density_g_cm3=2.70,
    )
    cad_result = CADExporter.export(cad_spec)
    print(f"   Computed Volume:       {cad_result.computed_volume_cm3} cm³")
    print(f"   Computed Mass:         {cad_result.computed_mass_grams} grams")
    print(f"   Total Surface Area:    {cad_result.total_surface_area_cm2} cm²")
    print(f"   Features in Tree:      {len(cad_result.feature_tree['features'])} (Baseplate Extrude + Fin Pattern)")

    print("\n2. Executing BoTorch qEHVI Multi-Objective Optimization Step...")
    optimizer = QEHVIHardwareOptimizer()
    observed_x = [
        [20.0, 10.0],
        [40.0, 15.0],
        [60.0, 20.0],
        [80.0, 25.0],
    ]
    observed_obj = [
        [-0.95, -120.0],
        [-0.70, -210.0],
        [-0.50, -340.0],
        [-0.35, -510.0],
    ]
    bounds = [[10.0, 100.0], [5.0, 30.0]]
    ref_point = [-1.5, -600.0]

    request = OptimizationRequest(
        observed_x=observed_x,
        observed_obj=observed_obj,
        param_bounds=bounds,
        ref_point=ref_point,
        num_candidates=1,
    )

    opt_res = optimizer.optimize(request)
    print(f"   Identified Pareto Frontier Size: {len(opt_res.pareto_frontier)} non-dominated design points")
    print(f"   Current Hypervolume:            {opt_res.hypervolume:.4f}")
    candidate = opt_res.candidates[0]
    print(f"   Next Suggested Candidate:        {candidate.parameters}")
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()
