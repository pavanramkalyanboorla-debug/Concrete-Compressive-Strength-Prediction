---
title: Concrete Mix Optimizer
emoji: 🧱
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
---
title: Concrete Mix Optimizer
emoji: 🧱
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Concrete Mix Optimizer 🧱

**AI-powered concrete mix design — predict strength, optimize ingredients, understand why.**

[![Live App](https://img.shields.io/badge/🤗%20Live%20App-HuggingFace-yellow)](https://huggingface.co/spaces/PavanBoorla/concrete-mix-optimizer)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-purple)](https://optuna.org/)
[![Uncertainty](https://img.shields.io/badge/Uncertainty-CV%20Residuals-orange)]()

---

Most ML projects in the concrete domain stop at "given a mix, predict strength." That's not how real mix design works. In practice, you start from a target — say, M35 grade — and need to figure out what proportions to use while staying inside material cost budgets, workability limits, and durability code requirements. This project tries to do that.

It combines a gradient boosting model with physics-based volume balancing, Optuna-driven optimization, **prediction intervals from cross-validation residuals**, and SHAP explainability into a single deployable app. You can use it through a Streamlit UI or hit the FastAPI endpoints directly.

---

## What it does

**Tab 1 — Predict:** Enter any mix (cement, slag, fly ash, water, superplasticizer, aggregates, age) and get back the predicted compressive strength with a 90% confidence interval, estimated slump, water/binder ratio, approximate cost per m³, and a list of any constraint violations — all in one shot. The confidence interval tells you how much the model trusts its own prediction: wide intervals (inherently ±7.7 MPa for 90%) reflect the model's overall uncertainty.

**Tab 2 — Optimize:** Give it a target strength and a trial budget, and Optuna searches for the best mix. In Pareto mode it finds the full trade-off frontier between cost and how close the predicted strength is to your target — you can pick whichever point on that front fits your project. Single-objective mode just returns the best mix directly with all ingredient quantities shown.

**Tab 3 — Explain:** Submit a mix and get a SHAP waterfall plot that breaks down exactly how much each ingredient pushed the prediction up or down from the model's baseline. There's also a plain-English summary of the top drivers.

---

## How it works under the hood

### Data & features

The dataset comes from the UCI concrete compressive strength dataset (1,030 samples) covering a wide range of mix proportions and curing ages. Eight raw features go in: Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age. Four engineered features are derived on top — Water/Binder Ratio, log(Age), Cement×Age interaction, and SCM Ratio (supplementary cementitious materials as a fraction of binder).

A critical issue that was found and fixed during development: `log1p` transformation was being applied during EDA but not consistently during inference. This was corrected by baking the transformation directly into the scikit-learn preprocessor pipeline, so training, prediction, optimization, and SHAP all go through exactly the same preprocessing path.

### Model training

Training happens inside Docker at build time, not at startup. The Dockerfile downloads the dataset from Figshare, runs `src/training.py`, and serializes `model.pkl` and `preprocessor.pkl` into `artifacts/`. This means the container starts immediately with no training overhead. Gradient Boosting was selected as the final model after evaluating several algorithms.

### Comparison with traditional methods

Concrete mix design has been governed by empirical formulas for over a century. The most widely used is **Abram's Law** (1918), which relates compressive strength solely to the inverse of the water-cement ratio:

```
f'c = k₁ / (w/c) + k₂
```

We fit k₁ and k₂ using least-squares regression on the same training split used for the ML model, then evaluate both on the identical held-out test set.

* The ML model reduces prediction error by **70%** and captures non‑linear interactions from all twelve features, compared to Abram's Law which relies solely on the inverse water‑cement ratio.

| Method | MAE (MPa) | R² |
|---|---|---|
| Abram's Law (fitted) | 10.26 | 0.404 |
| Our Model (Gradient Boosting) | 3.04 | 0.925 |

Abram's Law uses only one predictor (water-cement ratio) and cannot capture the contributions of supplementary materials like fly ash, slag, or superplasticizer. The ML model learns those non-linear interactions from all 12 features. This comparison validates that modern ML can improve on century-old formulas while still being constrained by physical feasibility checks during optimization.

### Prediction intervals (uncertainty)

Every prediction from the `/predict` endpoint includes a **90% confidence interval** calculated from the standard deviation of **5‑fold cross‑validation residuals** (σ = 4.67 MPa). The interval is simply:

```
lower_bound = prediction - 1.645 × σ
upper_bound = prediction + 1.645 × σ
```

This method uses the empirical spread of the model's own errors, requires no extra library, and gives distribution‑free coverage. If the uncertainty parameters file (`artifacts/uncertainty_params.pkl`) is missing at startup, the API gracefully returns `lower_bound`/`upper_bound` as `null`.

In practice: the interval width is constant (±7.7 MPa) because the model's error variance is assumed homogeneous. This is a valid first approximation and gives engineers a clear, honest estimate of the model's predictive accuracy.

### Physics layer

Before any constraint checking or optimization, mixes go through the absolute volume method (`src/physics.py`). This recalculates fine and coarse aggregate masses from the binder + water volumes to ensure the mix actually fills 1 m³, accounts for moisture corrections on aggregates, and computes paste volume. If the paste volume check fails (binder + water > 1 m³), the mix is rejected before it even reaches the model.

### Engineering constraints (`src/constraints.py`)

Every mix gets checked against real code-inspired limits:
- Water/binder ratio ≤ 0.50
- Cement content between 320–450 kg/m³
- SCM replacement ≤ 60% of binder
- Paste volume between 26–34%
- Estimated slump between 50–100 mm

These are enforced as hard rejection criteria during optimization (any violation returns a penalty of 1e6 in the Optuna objective) and shown as warnings to the user in the prediction tab.

### Optimization (`src/optimization.py`)

Two modes via Optuna:

**Single-objective:** Minimizes `|predicted_strength − target| + 0.001 × cost`. Fast, returns one best mix.

**Multi-objective (Pareto):** Simultaneously minimizes `|predicted_strength − target|` and `cost` as separate objectives, using Optuna's NSGA-II sampler. Returns a full Pareto front — every non-dominated mix that can't be improved on one objective without getting worse on the other. Plotted as a cost vs. strength-error scatter so you can pick your own trade-off point.

### SHAP explainability (`src/shap_utils.py`)

Uses `shap.TreeExplainer` on the gradient boosting model. The `ShapExplainer` class is initialized with the already-loaded model and preprocessor rather than creating a second pipeline instance. Waterfall plots show the contribution of each of the 12 features (8 raw + 4 engineered) to the final prediction.

### Failure modes & safety

- **Uncertainty params missing**: API returns intervals as `null`, predictions still work.
- **Optimization failures**: If Optuna finds no feasible mix, the API returns an empty result set rather than a bad recommendation.
- **Preprocessing consistency**: All code paths (predict, optimize, SHAP) use the rigorous `_feature_engineer` function, preventing silent skew.

---

## Project structure

```
├── app/
│   ├── streamlit_app.py      # Streamlit UI (3 tabs: Predict, Optimize, Explain)
│   └── main.py               # FastAPI app with /predict, /predict/batch, /optimize, /shap
├── src/
│   ├── constants.py           # Specific gravities, cost table, CONFIG limits
│   ├── physics.py             # Absolute volume method, moisture correction
│   ├── constraints.py         # Engineering constraint checks + slump estimate
│   ├── optimization.py        # Optuna single/multi-objective studies
│   ├── shap_utils.py          # ShapExplainer class, waterfall + reasoning text
│   ├── train_uncertainty.py   # Computes CV residual std → uncertainty_params.pkl
│   └── pipeline/
│       ├── predict_pipeline.py   # PredictPipeline class + feature engineering
│       └── training_pipeline.py  # Training script
├── baseline/
│   └── abrams_eval.py         # Abram's Law baseline evaluation script
├── data/                      # Dataset (downloaded at Docker build time)
├── artifacts/                 # model.pkl, preprocessor.pkl, uncertainty_params.pkl
├── notebooks/                 # EDA and experimentation notebooks
├── tests/                     # Test suite
├── Dockerfile                 # Multi-stage build: uv for deps, trains model in container
├── docker-compose.yml
└── pyproject.toml             # Dependencies managed with uv
```

---

## Running locally

**With Docker (recommended):**

```bash
docker compose up --build
```

The build step downloads the dataset, trains the model and computes uncertainty parameters, and starts the Streamlit app on port 7860. The first build takes a few minutes; subsequent starts are instant since the model is baked into the image.

**Without Docker:**

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv sync

# Train the model first
python src/training.py

# Compute uncertainty parameters (CV residuals)
python src/train_uncertainty.py

# Run Abram's Law baseline (optional — prints comparison table)
python baseline/abrams_eval.py

# Start the app
streamlit run app/streamlit_app.py --server.port 7860
```

**API only:**

```bash
uvicorn app.main:app --reload --port 8000
```

Then hit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API reference

**POST /predict**
```json
{
  "Cement": 350,
  "Blast_Furnace_Slag": 50,
  "Fly_Ash": 50,
  "Water": 180,
  "Superplasticizer": 5,
  "Coarse_Aggregate": 1000,
  "Fine_Aggregate": 700,
  "Age": 28
}
```
Returns:
```json
{
  "predicted_strength_mpa": 35.2,
  "strength_category": "Normal Strength (20–40 MPa)",
  "lower_bound": 27.5,
  "upper_bound": 42.9
}
```
`lower_bound` and `upper_bound` define the 90% confidence interval. If uncertainty params are unavailable, both are `null`.

**POST /predict/batch** — same schema as above but wrapped in a list.

**POST /optimize**
```json
{
  "target_strength": 40.0,
  "n_trials": 100,
  "multi_objective": true
}
```
Returns either a single best mix or a Pareto front depending on `multi_objective`.

**GET /shap** — query params mirror the predict input fields, returns a base64-encoded waterfall plot PNG.

**GET /health** — returns `{"status": "healthy"}` including uncertainty params status.

**GET /metadata** — returns model type, preprocessor type, uncertainty availability, and feature lists.

---

## Material cost table

Costs are approximate Indian market rates (₹/kg):

| Material | Cost |
|---|---|
| Cement | ₹7.00 |
| Blast Furnace Slag | ₹3.00 |
| Fly Ash | ₹2.00 |
| Superplasticizer | ₹50.00 |
| Coarse Aggregate | ₹1.50 |
| Fine Aggregate | ₹1.20 |
| Water | ₹0.05 |

These are editable in `src/constants.py` and the optimizer will automatically factor in any changes.

---

## Design decisions worth noting

**Why Optuna over a grid search or scipy.optimize?** The feasible mix space is non-convex and has hard discontinuities from the constraint rejections. Optuna's TPE sampler handles this well and the Pareto mode gives something genuinely more useful than a single-point optimum for an engineering decision problem.

**Why train inside Docker?** Avoids the common HuggingFace Spaces issue where a pre-trained `.pkl` gets out of sync with the code or data version. The artifact is always regenerated from the current dataset and preprocessing logic.

**Why absolute volume method?** It ensures physically consistent mixes — a mix where cement + water + aggregates don't actually sum to 1 m³ is nonsensical. Skipping this (as most pure-ML implementations do) means the model gets inputs that couldn't exist in reality.

**Why cross‑validated intervals instead of a library?** Using the standard deviation of 5‑fold CV residuals is simple, transparent, and has no external dependencies. The 90% interval (prediction ± 1.645σ) gives an honest, constant width that reflects the model's overall error level. This approach is exactly what you'd explain in an interview — no black boxes.

**Why an Abram's Law baseline?** ML practitioners often skip comparing against domain-specific baselines. In civil engineering, Abram's Law has been the standard since 1918 — if your model can't beat it (or at least match it while adding other capabilities), you haven't justified the added complexity. Documenting this comparison shows awareness of the problem domain, not just the ML stack.

---

## Dependencies

Main ones — full list in `pyproject.toml`:

- `scikit-learn` — preprocessing and model utilities
- `catboost`, `xgboost` — candidate models during training
- `optuna` — optimization framework
- `shap` — explainability
- `streamlit` — web UI
- `fastapi` + `uvicorn` — REST API
- `plotly` — interactive charts

Dependency management uses `uv` for fast, reproducible installs.

---

## Links

- **Live app:** https://huggingface.co/spaces/PavanBoorla/concrete-mix-optimizer
- **GitHub:** https://github.com/pavanramkalyanboorla-debug/Concrete-Compressive-Strength-Prediction
- **Dataset:** UCI Concrete Compressive Strength (via Figshare)