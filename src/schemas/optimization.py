"""Pydantic schemas for Multi-Objective Constrained Hardware Optimization."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParetoPoint(BaseModel):
    parameters: List[float]
    objectives: List[float]
    constraints: Optional[List[float]] = None
    is_feasible: bool = True


class CandidatePoint(BaseModel):
    parameters: List[float]
    predicted_objectives: Optional[List[float]] = None
    acquisition_value: float


class OptimizationRequest(BaseModel):
    """Payload for submitting observed experiments and receiving new Pareto-optimal candidates."""

    observed_x: List[List[float]] = Field(..., description="Historical design vectors (N, D)")
    observed_obj: List[List[float]] = Field(..., description="Multi-objective values (N, M) [to MAXIMIZE]")
    observed_con: Optional[List[List[float]]] = Field(
        default=None, description="Inequality constraints (N, C) where con(x) <= 0 is feasible"
    )
    param_bounds: List[List[float]] = Field(..., description="Lower and upper bounds for each parameter (D, 2)")
    ref_point: List[float] = Field(..., description="Reference point for hypervolume calculation (M,)")
    num_candidates: int = Field(default=1, ge=1, le=8, description="Number of parallel candidates (q) to generate")


class OptimizationResponse(BaseModel):
    """Response containing Pareto-optimal design points and next suggested design candidate."""

    pareto_frontier: List[ParetoPoint]
    candidates: List[CandidatePoint]
    hypervolume: float
    num_evaluated: int
