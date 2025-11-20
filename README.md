# EcoSim

EcoSim is a deterministic economic simulation that models households, firms, and the government across weekly ticks. The first 52 ticks form a warm-up period where government baseline firms stabilize the economy before private entrepreneurship and adaptive pricing kick in. After that, households can open firms, compete on price/quality, and react to wages, wellbeing, and transfers.

## Getting Started

1. **Create a virtual environment** (optional)
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Populate the sample database (optional for analysis dashboards)**
   ```bash
   python backend/generate_sample_data.py
   ```
   This script creates `backend/sample_data/ecosim_sample.db` so notebooks and dashboards can run without waiting on a large simulation.
4. **Run a simulation**
   ```bash
   python backend/demo_skill_experience.py
   ```
   or use `backend/generate_sample_data.py` to run the engine and export structured data.

## Repository Layout

| Path | Description |
| --- | --- |
| `backend/` | Core simulation engine, agents, and FastAPI surface area. |
| `frontend/` | UI experiments and dashboards. |
| `docs/` | All simulation design specs and technical memos. |
| `data/`, `ecosim_chartjs/` | Supporting resources for visualization experiments. |

## Documentation Map

All Markdown documentation now lives under `docs/`:

- `docs/DATA_SPECIFICATION.md` – schema for exported simulation data.
- `docs/DATA_SPECIFICATION_old.md` – legacy schema kept for reference.
- `docs/DYNAMIC_FEATURES.md` – description of the new dynamic agents and markets.
- `docs/IMPLEMENTATION_SUMMARY.md` – changelog-style overview of major engine upgrades.
- `docs/REDESIGN_FEATURES.md` – design doc for just-in-time production, pricing, etc.
- `docs/SKILL_EXPERIENCE_SYSTEM.md` – details on worker experience and skill mechanics.

Use `docs/README.md` (below) for a curated guide if you only need the highlights.

## Support / Questions

Open an issue or leave TODOs inside the relevant doc. The `docs` folder provides the authoritative specs, so updates should be reflected there before changing code.
