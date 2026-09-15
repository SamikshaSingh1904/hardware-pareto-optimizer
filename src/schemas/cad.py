"""Parametric CAD Pydantic Schemas."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class CADParametricSpec(BaseModel):
    """Parametric specification defining a physical heat sink / chassis component."""

    part_name: str = Field(default="heatsink_chassis", description="Component identifier")
    length_mm: float = Field(..., ge=10.0, le=500.0, description="Overall component length in mm")
    width_mm: float = Field(..., ge=10.0, le=300.0, description="Overall component width in mm")
    base_thickness_mm: float = Field(..., ge=1.0, le=50.0, description="Baseplate thickness in mm")
    fin_height_mm: float = Field(..., ge=5.0, le=150.0, description="Cooling fin height in mm")
    fin_count: int = Field(..., ge=2, le=100, description="Number of parallel cooling fins")
    fin_thickness_mm: float = Field(..., ge=0.5, le=10.0, description="Fin thickness in mm")
    material: str = Field(default="Aluminum_6061-T6", description="Hardware material alloy")
    material_density_g_cm3: float = Field(default=2.70, gt=0.0, description="Alloy density in g/cm³")


class CADExportResponse(BaseModel):
    """Response containing computed physical attributes and parametric CAD tree."""

    part_name: str
    material: str
    computed_volume_cm3: float
    computed_mass_grams: float
    total_surface_area_cm2: float
    feature_tree: Dict[str, Any]
    openscad_script: str
