# EcoSim Testing Guide

## Available Tests

### ✅ Household Agent Test Suite
**File**: `test_household_agent.py`
**Status**: Unit tests - all passing

**What it tests**:
- Household creation and initialization
- Labor supply planning (job search with unemployment benefits)
- Consumption planning (purchase decisions with market prices)
- Employment status changes and wage tracking
- Skill development through work experience (passive growth)
- Goods inventory consumption (10% depletion per tick)
- Wellbeing system (happiness, morale, health tracking)
- Income and spending (wages, taxes, transfers, purchases)
- **20-tick simulation** showing realistic household behavior

**How to run**:
```bash
cd backend
python test_household_agent.py
```

**Expected output**: 8 unit tests + 1 simulation, all passing ✅

---

### ✅ Firm Behavior Integration Test
**File**: `test_firm_behavior.py`
**Status**: 52-tick simulation - working

**What it tests**:
- Production and inventory management over time
- Hiring and layoff decisions
- Wage adjustments based on labor market
- Price adjustments based on supply/demand
- Cash flow and profitability tracking
- Quality levels over time
- **Tracks 3 firms every 10 ticks for 52 ticks (1 year)**

**How to run**:
```bash
cd backend
python test_firm_behavior.py
```

**Expected output**: Detailed tables showing firm state every 10 ticks ✅

---

### ✅ Government Behavior Integration Test
**File**: `test_government_behavior.py`
**Status**: 52-tick simulation - working

**What it tests**:
- Tax collection (wage and profit taxes)
- Unemployment benefit distribution
- Transfer budget management
- Policy adjustments based on economic conditions
- Fiscal balance tracking
- Impact on unemployment and happiness
- **Tracks government every 10 ticks for 52 ticks (1 year)**

**How to run**:
```bash
cd backend
python test_government_behavior.py
```

**Expected output**: Government state tables every 10 ticks + final analysis ✅

---

## Test Coverage

| Agent Type | Test File | Type | Status |
|------------|-----------|------|--------|
| **HouseholdAgent** | `test_household_agent.py` | Unit tests | ✅ Complete |
| **FirmAgent** | `test_firm_behavior.py` | Integration (52 ticks) | ✅ Complete |
| **GovernmentAgent** | `test_government_behavior.py` | Integration (52 ticks) | ✅ Complete |
| **Full Economy** | `run_large_simulation.py` | Large-scale (500+ ticks) | ✅ Available |

---

## Running Large-Scale Simulations

For testing the complete system with all agents interacting:

```bash
cd backend
python run_large_simulation.py
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for configuration and analysis details.

---

## Test Development Guidelines

When creating new tests, follow the pattern in `test_household_agent.py`:

1. Clear docstring explaining what's tested
2. `print_section()` helper for formatted output
3. Individual test functions with descriptive names
4. Detailed print statements showing progress
5. Clear assertions with helpful error messages
6. Quick simulation (20-30 ticks) showing real behavior
7. Summary section listing all verified behaviors

