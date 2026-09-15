# Hardware Pareto Optimizer

A production-grade Bayesian Multi-Objective Constrained Optimization engine using BoTorch ($q$-Expected Hypervolume Improvement / $q$EHVI) coupled with a parametric CAD synthesis and JSON feature-tree exporter for mechanical/hardware design spaces.

## Architecture Overview

```
[ Parametric Hardware Domain (Thermal, Mass, Stress) ]
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  BoTorch Constrained qEHVI Optimizer                   │
│  - Multi-Task / ModelList Gaussian Processes           │
│  - Non-dominated Hypervolume Partitioning               │
│  - Simultaneous parallel multi-point acquisitions      │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI Microservice Engine                           │
│  - RESTful endpoints: /optimize & /export-cad          │
│  - Pydantic v2 Contract Validation                     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[ Parametric CAD JSON Feature Tree + OpenSCAD Models ]
```

## Mathematical Optimization Formulation

We formulate the hardware design task as a multi-objective constrained problem:

$$\max_{\mathbf{x} \in \mathcal{X}} \quad \mathbf{f}(\mathbf{x}) = \left[ -R_{\text{th}}(\mathbf{x}),\, -M(\mathbf{x}) \right]^\top \quad \text{s.t.} \quad c_j(\mathbf{x}) \le 0 \;\; \forall j \in \{1, \dots, C\}$$

Under candidate set $X = \{\mathbf{x}_1, \dots, \mathbf{x}_q\}$, the $q$EHVI acquisition function computes:

$$\alpha_{q\text{EHVI}}(X) = \mathbb{E}\left[ \text{HV}\left( \mathcal{P} \cup \mathbf{f}(X) \right) - \text{HV}(\mathcal{P}) \right] \prod_{i=1}^q \mathbb{P}\left(\mathbf{c}(\mathbf{x}_i) \le \mathbf{0}\right)$$

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run test suite
pytest tests/ -v

# Launch the FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
