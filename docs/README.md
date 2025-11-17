# Documentation Guide

All EcoSim reference material is grouped here so you can jump directly to the spec you need without digging through backend files.

## Quick Links

| Document | Purpose |
| --- | --- |
| [DATA_SPECIFICATION.md](DATA_SPECIFICATION.md) | Canonical tables and fields emitted by the simulation. |
| [DYNAMIC_FEATURES.md](DYNAMIC_FEATURES.md) | Explanation of the redesigned agent behaviors and market mechanics. |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Narrative of major engine milestones and architectural decisions. |
| [REDESIGN_FEATURES.md](REDESIGN_FEATURES.md) | Motivation and details for the pricing, production, and wellbeing overhaul. |
| [SKILL_EXPERIENCE_SYSTEM.md](SKILL_EXPERIENCE_SYSTEM.md) | Deep dive on worker skill growth and morale modifiers. |
| [DATA_SPECIFICATION_old.md](DATA_SPECIFICATION_old.md) | Archived schema retained for backward compatibility checks. |

## How to Use This Folder

1. **Start with `DYNAMIC_FEATURES.md`** if you need a conceptual overview of the economy redesign.
2. **Consult `DATA_SPECIFICATION.md`** when building dashboards or pipelines using simulation output.
3. **Reference `IMPLEMENTATION_SUMMARY.md`** before modifying the engine so you know why decisions were made.
4. **Check `SKILL_EXPERIENCE_SYSTEM.md`** when tuning labor, morale, or wellbeing parameters.

If you add new docs, keep them in this folder and link them here so contributors always have a single index.
