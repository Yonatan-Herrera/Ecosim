# Configuration System Usage Guide

## 📚 Overview

All simulation parameters are now centralized in [backend/config.py](backend/config.py). This replaces **400+ scattered "magic numbers"** with a structured, hierarchical configuration system.

---

## 🎯 Quick Start

### **Import the Config**
```python
from config import CONFIG

# Access any parameter
warmup_period = CONFIG.time.warmup_ticks  # 52
food_elasticity = CONFIG.households.food_elasticity  # 0.5
pid_gain = CONFIG.firms.pid_control_scaling  # 0.05
```

### **Modify Parameters at Runtime**
```python
# Change a value before running simulation
CONFIG.firms.target_inventory_weeks = 4.0  # Was 2.0
CONFIG.households.food_elasticity = 0.8  # Was 0.5

# Then run the simulation
economy = create_large_economy()
economy.step()  # Uses new config values
```

---

## 📊 Config Structure

```
SimulationConfig (CONFIG)
├── time                 # Time-related constants
├── households           # Household behavioral parameters (90+)
├── firms                # Firm behavioral parameters (80+)
├── government           # Government policy parameters (40+)
├── labor_market         # Labor market matching (15+)
└── market               # Market mechanics (15+)
```

---

## 🔧 Sub-Configurations

### **1. TimeConfig** (`CONFIG.time`)
**Purpose**: Universal time constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ticks_per_year` | 52 | One tick = one week |
| `warmup_ticks` | 52 | Warm-up period (1 year) |

**Example**:
```python
# Speed up simulation (monthly ticks instead of weekly)
CONFIG.time.ticks_per_year = 12
CONFIG.time.warmup_ticks = 12  # 1 year warmup
```

---

### **2. HouseholdBehaviorConfig** (`CONFIG.households`)
**Purpose**: All household decision-making parameters

#### **Key Categories**:

**Price Elasticity** (NEW)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `food_elasticity` | 0.5 | Inelastic (necessity) |
| `services_elasticity` | 0.8 | Somewhat elastic |
| `housing_elasticity` | 1.5 | Elastic (luxury) |
| `max_food_budget_share` | 0.6 | Max budget for food (substitution) |

**Example**:
```python
# Make households more responsive to food prices
CONFIG.households.food_elasticity = 1.0

# Allow households to spend more on food when needed
CONFIG.households.max_food_budget_share = 0.75
```

**Unemployment Response** (NEW)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_spend_fraction` | 0.3 | Spending when scared (30%) |
| `confidence_multiplier` | 0.5 | Up to 80% when confident |

**Example**:
```python
# Make households more cautious during recession
CONFIG.households.min_spend_fraction = 0.2  # Save 80%

# Or more aggressive during boom
CONFIG.households.min_spend_fraction = 0.4  # Save only 60%
```

**Wellbeing Dynamics**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `happiness_decay_rate` | 0.01 | Decay per tick |
| `morale_decay_rate` | 0.02 | Faster than happiness |
| `health_decay_rate` | 0.005 | Slowest decay |

---

### **3. FirmBehaviorConfig** (`CONFIG.firms`)
**Purpose**: Production, pricing, and hiring behavior

#### **Key Categories**:

**PID Pricing** (NEW - Prompt 3)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_inventory_weeks` | 2.0 | Target weeks of supply |
| `pid_pressure_decay` | 0.7 | Integral decay |
| `pid_integral_gain` | 0.1 | Integral coefficient |
| `pid_control_scaling` | 0.05 | Overall gain |
| `pid_adjustment_min` | 0.80 | Max -20% price cut |
| `pid_adjustment_max` | 1.20 | Max +20% price hike |

**Example**:
```python
# More aggressive price adjustments
CONFIG.firms.pid_control_scaling = 0.10  # Double the gain

# Target higher inventory buffer
CONFIG.firms.target_inventory_weeks = 4.0  # 4 weeks instead of 2
```

**Diminishing Returns** (NEW - Prompt 5)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `production_scaling_exponent` | 0.9 | Non-linear production |
| `min_base_productivity` | 1.0 | Safety floor |

**Example**:
```python
# Stronger diminishing returns (harder to scale)
CONFIG.firms.production_scaling_exponent = 0.85

# Weaker diminishing returns (monopolies easier)
CONFIG.firms.production_scaling_exponent = 0.95
```

**Firm Personalities**
| Personality | Investment | Risk | Price Adj | Wage Adj | R&D | Hires |
|-------------|-----------|------|-----------|----------|-----|-------|
| Aggressive | 15% | 0.9 | 10% | 15% | 8% | 3 |
| Moderate | 5% | 0.5 | 5% | 10% | 5% | 2 |
| Conservative | 2% | 0.2 | 2% | 5% | 2% | 1 |

**Example**:
```python
# Make aggressive firms even more aggressive
CONFIG.firms.aggressive_price_adjustment = 0.15  # Was 0.10
CONFIG.firms.aggressive_max_hires = 5  # Was 3
```

---

### **4. GovernmentPolicyConfig** (`CONFIG.government`)
**Purpose**: Fiscal policy and automatic stabilizers

**Tax Rates**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_wage_tax_rate` | 0.15 | 15% income tax |
| `default_profit_tax_rate` | 0.20 | 20% corporate tax |
| `default_unemployment_benefit` | 30.0 | Per-tick payment |

**Example**:
```python
# Implement progressive taxation
CONFIG.government.default_wage_tax_rate = 0.25  # Higher taxes
CONFIG.government.default_unemployment_benefit = 50.0  # Higher safety net
```

**Automatic Stabilizers**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `high_unemployment_threshold` | 0.15 | Trigger expansion (15%) |
| `low_unemployment_threshold` | 0.03 | Trigger contraction (3%) |
| `large_deficit_threshold` | -10000.0 | Trigger austerity |
| `large_surplus_threshold` | 50000.0 | Trigger stimulus |

**Example**:
```python
# More aggressive counter-cyclical policy
CONFIG.government.high_unemployment_threshold = 0.10  # Trigger earlier
CONFIG.government.high_unemployment_benefit_increase = 1.15  # +15% not +5%
```

---

### **5. LaborMarketConfig** (`CONFIG.labor_market`)
**Purpose**: Wage determination and labor matching

**Skill & Experience Premiums**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_skill_premium` | 0.5 | 50% wage premium for high skills |
| `max_experience_premium` | 0.5 | 50% wage premium for experience |
| `experience_premium_per_year` | 0.05 | 5% per year |

**Example**:
```python
# Reward experience more heavily
CONFIG.labor_market.experience_premium_per_year = 0.10  # 10% per year
CONFIG.labor_market.max_experience_premium = 1.0  # Up to 100% premium
```

---

### **6. MarketMechanicsConfig** (`CONFIG.market`)
**Purpose**: Price ceiling, firm entry/exit

**Price Ceiling**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_ceiling` | 50.0 | Max price before tax |
| `price_ceiling_tax_rate` | 0.25 | 25% tax on excess |

**Example**:
```python
# Implement strict price controls
CONFIG.market.price_ceiling = 30.0  # Lower ceiling
CONFIG.market.price_ceiling_tax_rate = 0.50  # 50% punitive tax
```

**Firm Dynamics**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bankruptcy_threshold` | -1000.0 | Firms exit below this |
| `max_private_competitors` | 5 | Limit firm creation |
| `new_firm_demand_threshold` | 1000.0 | Min cash to enter |

---

## 🎮 Common Use Cases

### **Use Case 1: Test Keynesian Economics**
```python
# High government spending, progressive taxes
CONFIG.government.default_wage_tax_rate = 0.30
CONFIG.government.default_profit_tax_rate = 0.35
CONFIG.government.default_unemployment_benefit = 60.0
CONFIG.government.infrastructure_investment_budget = 5000.0

# Aggressive automatic stabilizers
CONFIG.government.high_unemployment_benefit_increase = 1.20
CONFIG.government.deficit_tax_increase = 1.01  # Gentle tax increases
```

### **Use Case 2: Test Austrian Economics**
```python
# Low taxes, minimal intervention
CONFIG.government.default_wage_tax_rate = 0.05
CONFIG.government.default_profit_tax_rate = 0.10
CONFIG.government.default_unemployment_benefit = 10.0

# No automatic stabilizers
CONFIG.government.high_unemployment_benefit_increase = 1.0  # No change
CONFIG.government.deficit_tax_increase = 1.0  # No change
```

### **Use Case 3: Test Inflation Shock**
```python
# Make households more sensitive to prices
CONFIG.households.food_elasticity = 1.2  # Very elastic
CONFIG.households.services_elasticity = 1.5  # Very elastic

# Allow firms to adjust prices quickly
CONFIG.firms.pid_control_scaling = 0.15  # Aggressive PID
CONFIG.firms.pid_adjustment_max = 1.50  # Allow +50% price hikes
```

### **Use Case 4: Test Market Concentration**
```python
# Weaken diminishing returns (allow monopolies)
CONFIG.firms.production_scaling_exponent = 0.98  # Almost linear

# Limit firm entry
CONFIG.market.max_private_competitors = 1
CONFIG.market.new_firm_demand_threshold = 100000.0  # Very high barrier
```

---

## 🔬 A/B Testing Example

```python
# Baseline scenario
baseline_config = CONFIG  # Default values

# Test scenario: Higher elasticity
test_config = SimulationConfig()
test_config.households.food_elasticity = 1.0
test_config.households.services_elasticity = 1.5

# Run both
baseline_economy = create_large_economy(baseline_config)
test_economy = create_large_economy(test_config)

for tick in range(500):
    baseline_economy.step()
    test_economy.step()

# Compare outcomes
print(f"Baseline unemployment: {baseline_economy.unemployment_rate}")
print(f"Test unemployment: {test_economy.unemployment_rate}")
```

---

## 📈 Parameter Sensitivity Analysis

**High Impact Parameters** (change these first):
1. `CONFIG.firms.production_scaling_exponent` → Market structure
2. `CONFIG.households.food_elasticity` → Household survival
3. `CONFIG.firms.pid_control_scaling` → Price stability
4. `CONFIG.government.default_unemployment_benefit` → Safety net
5. `CONFIG.labor_market.max_skill_premium` → Wage inequality

**Low Impact Parameters** (fine-tuning):
1. `CONFIG.households.inventory_depletion_threshold` → 0.001 vs 0.002
2. `CONFIG.firms.pid_safety_epsilon` → 1e-3 vs 1e-4
3. `CONFIG.households.happiness_decay_rate` → 0.01 vs 0.015

---

## ⚠️ Important Notes

1. **Thread Safety**: CONFIG is a global singleton. Modifying it during simulation affects all agents.
2. **Validation**: Some parameters are validated in `__post_init__()`. Invalid values raise `ValueError`.
3. **Units**: Time is in ticks. Money is in abstract currency units.
4. **Derived Values**: Some parameters depend on `ticks_per_year` (e.g., experience premiums).

---

## 🚀 Next Steps

1. **Update agents.py**: Replace hardcoded values with `CONFIG.*` (in progress)
2. **Add validation**: More comprehensive bounds checking
3. **Create presets**: Pre-configured economic models (Keynesian, Austrian, MMT)
4. **Web UI**: Real-time tuning interface
5. **ML optimization**: Auto-tune CONFIG for desired outcomes

---

**Questions?** See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for implementation details.
