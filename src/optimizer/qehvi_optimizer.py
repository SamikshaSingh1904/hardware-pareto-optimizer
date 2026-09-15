"""Multi-Objective Constrained Optimizer using BoTorch qEHVI."""

from typing import Tuple, List, Optional
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.schemas.optimization import (
    OptimizationRequest,
    OptimizationResponse,
    ParetoPoint,
    CandidatePoint,
)


class QEHVIHardwareOptimizer:
    """Multi-Objective Constrained Hardware Optimizer using qEHVI."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.dtype = dtype

    def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        device = torch.device("cpu")

        train_x = torch.tensor(request.observed_x, dtype=self.dtype, device=device)
        train_obj = torch.tensor(request.observed_obj, dtype=self.dtype, device=device)
        ref_point = torch.tensor(request.ref_point, dtype=self.dtype, device=device)
        bounds = torch.tensor(request.param_bounds, dtype=self.dtype, device=device).T  # (2, D)

        has_constraints = request.observed_con is not None and len(request.observed_con) > 0
        if has_constraints:
            train_con = torch.tensor(request.observed_con, dtype=self.dtype, device=device)
            # Feasibility: con(x) <= 0
            is_feasible = (train_con <= 0).all(dim=-1)
        else:
            train_con = None
            is_feasible = torch.ones(train_x.size(0), dtype=torch.bool, device=device)

        # 1. Compute Pareto frontier among feasible points
        if is_feasible.any():
            feasible_obj = train_obj[is_feasible]
            non_dom = is_non_dominated(feasible_obj)
            pareto_x = train_x[is_feasible][non_dom]
            pareto_obj = feasible_obj[non_dom]

            # Compute current hypervolume
            hv_calculator = Hypervolume(ref_point=ref_point)
            current_hv = float(hv_calculator.compute(pareto_obj))
        else:
            pareto_x = torch.empty((0, train_x.size(1)), dtype=self.dtype)
            pareto_obj = torch.empty((0, train_obj.size(1)), dtype=self.dtype)
            current_hv = 0.0

        pareto_points = []
        for i in range(pareto_x.size(0)):
            pareto_points.append(
                ParetoPoint(
                    parameters=[float(v) for v in pareto_x[i]],
                    objectives=[float(v) for v in pareto_obj[i]],
                    is_feasible=True,
                )
            )

        # 2. Fit Multi-Objective Gaussian Process Models
        models = []
        num_objectives = train_obj.size(-1)
        for m in range(num_objectives):
            gp_m = SingleTaskGP(train_x, train_obj[:, m : m + 1])
            mll = ExactMarginalLogLikelihood(gp_m.likelihood, gp_m)
            fit_gpytorch_mll(mll)
            models.append(gp_m)

        if has_constraints:
            num_constraints = train_con.size(-1)
            for c in range(num_constraints):
                gp_c = SingleTaskGP(train_x, train_con[:, c : c + 1])
                mll = ExactMarginalLogLikelihood(gp_c.likelihood, gp_c)
                fit_gpytorch_mll(mll)
                models.append(gp_c)

        model_list = ModelListGP(*models)

        # 3. Setup Partitioning & qEHVI Acquisition Function
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pareto_obj if pareto_obj.size(0) > 0 else ref_point.unsqueeze(0),
        )

        acq_func = qExpectedHypervolumeImprovement(
            model=model_list,
            ref_point=ref_point,
            partitioning=partitioning,
        )

        # 4. Optimize Acquisition Function
        candidates, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=request.num_candidates,
            num_restarts=5,
            raw_samples=30,
        )

        candidate_list = []
        for cand in candidates:
            candidate_list.append(
                CandidatePoint(
                    parameters=[float(v) for v in cand],
                    acquisition_value=float(acq_value),
                )
            )

        return OptimizationResponse(
            pareto_frontier=pareto_points,
            candidates=candidate_list,
            hypervolume=current_hv,
            num_evaluated=train_x.size(0),
        )
