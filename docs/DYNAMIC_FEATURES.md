# Dynamic Economy Features

## Overview

The EcoSim simulation now includes several dynamic features that make the economy more realistic and responsive to changing conditions:

1. **Skill Development System**
2. **Goods Consumption & Inventory Depletion**
3. **Firm Bankruptcy & Exit**
4. **New Firm Creation & Market Entry**
5. **Dynamic Government Policy**

---

## 1. Skill Development System

### Passive Skill Growth

Workers automatically improve their skills through work experience with diminishing returns:

```python
# In HouseholdAgent.apply_labor_outcome()
if self.is_employed and employer_category is not None:
    # Passive skill growth (diminishing returns)
    skill_improvement = self.skill_growth_rate * (1.0 - self.skills_level)
    self.skills_level = min(1.0, self.skills_level + skill_improvement)
```

**Parameters:**
- `skill_growth_rate`: 0.001 (0.1% improvement per tick at low skills)
- Diminishing returns: Growth slows as skills approach 1.0

**Example:**
- Worker at skill 0.3: Gains ~0.0007 per tick (0.07% improvement)
- Worker at skill 0.8: Gains ~0.0002 per tick (0.02% improvement)
- After 1000 ticks employed, skill 0.3 → ~0.55

### Active Education Investment

Households can invest cash to accelerate skill development:

```python
# Call this method to invest in education
household.invest_in_education(investment_amount=1000.0)
```

**Mechanics:**
- Cost: $1000 per 0.1 skill points at low skills
- Diminishing returns: Harder to improve at higher skill levels
- Formula: `skill_gain = investment * 0.0001 * (1.0 - current_skill)`

**Example:**
- $1000 invested at skill 0.2: Gains 0.08 skill points
- $1000 invested at skill 0.8: Gains 0.02 skill points

---

## 2. Goods Consumption & Inventory Depletion

### Problem Solved

Previously, households accumulated infinite goods without consuming them. Now goods are consumed each tick.

### Implementation

```python
# In HouseholdAgent.consume_goods()
consumption_rate = 0.1  # Consume 10% of inventory per tick

for good in goods_inventory:
    consumed = inventory[good] * consumption_rate
    inventory[good] -= consumed
```

**Parameters:**
- `consumption_rate`: 0.1 (10% per tick)
- Half-life: ~7 ticks (goods deplete by 50% in 7 ticks)

**Economic Impact:**
- Households must continuously purchase goods
- Creates sustained demand for firm output
- Prevents infinite inventory accumulation

---

## 3. Firm Bankruptcy & Exit

### Bankruptcy Threshold

Firms with cash below a threshold are removed from the economy:

```python
bankruptcy_threshold = -1000.0  # Firms below this exit
```

### Exit Process

When a firm goes bankrupt:
1. All employees are laid off
2. Employees' `employer_id` set to `None`
3. Employees' `wage` set to 0.0
4. Firm removed from economy
5. Tracking dictionaries cleaned up

**Example Scenario:**
- Firm starts with $5000 cash
- Each tick, pays 3 workers $50 each = $150/tick
- No sales for 40 ticks: $5000 - (40 * $150) = -$1000
- Firm goes bankrupt and exits

---

## 4. New Firm Creation & Market Entry

### Creation Conditions

New firms are created when:
1. Total firms < minimum threshold (default: 3)
2. Total household cash > $1000 (demand exists)

### Creation Process

```python
new_firm = FirmAgent(
    firm_id=next_available_id,
    good_name=f"{category}Product{id}",
    cash_balance=5000.0,  # Starting capital
    inventory_units=0.0,
    good_category=chosen_category,  # Food, Housing, or Services
    quality_level=5.0,  # Medium quality
    wage_offer=50.0,
    price=10.0,
    expected_sales_units=50.0,
    production_capacity_units=100.0
)
```

### Category Selection

New firms enter the least-represented category:
- Count firms in each category (Food, Housing, Services)
- Choose category with fewest existing firms
- Promotes market diversity

**Example:**
- Current: 2 Food firms, 1 Housing firm, 0 Service firms
- New firm will be created in Services category

---

## 5. Dynamic Government Policy

### Policy Adjustment Triggers

Government adjusts policies every tick based on:
- **Unemployment rate**: Affects benefits and transfers
- **Fiscal balance**: Affects tax rates
- **Economic activity**: Affects transfer budget

### Unemployment-Based Adjustments

```python
if unemployment_rate > 0.15:  # High unemployment (>15%)
    unemployment_benefit_level *= 1.05  # Increase benefits by 5%
    transfer_budget *= 1.1  # Increase transfers by 10%

elif unemployment_rate < 0.03:  # Low unemployment (<3%)
    unemployment_benefit_level *= 0.98  # Reduce benefits by 2%
```

### Fiscal Balance Adjustments

```python
if government.cash_balance < -10000:  # Large deficit
    wage_tax_rate = min(0.30, wage_tax_rate * 1.02)  # Increase taxes
    profit_tax_rate = min(0.35, profit_tax_rate * 1.02)

elif government.cash_balance > 50000:  # Large surplus
    wage_tax_rate = max(0.05, wage_tax_rate * 0.98)  # Reduce taxes
    profit_tax_rate = max(0.10, profit_tax_rate * 0.98)
```

### Counter-Cyclical Policy

```python
if unemployment_rate > 0.10 and cash_balance > 0:
    # Recession: Increase transfers
    transfer_budget = min(30000, transfer_budget * 1.05)

elif unemployment_rate < 0.05 and cash_balance > 20000:
    # Boom: Reduce transfers
    transfer_budget = max(5000, transfer_budget * 0.98)
```

### Policy Bounds

All policies have min/max limits:
- Wage tax rate: 5% - 30%
- Profit tax rate: 10% - 35%
- Unemployment benefits: $20 - $50 per tick
- Transfer budget: $5,000 - $30,000 per tick

---

## Simulation Flow (Updated)

The economy now executes 15 phases per tick:

1. Firms plan production, labor, prices, wages
2. Households plan labor supply and consumption
3. Labor market matching (with skill/experience premiums)
4. Apply labor outcomes (with skill growth)
5. Firms apply production (with experience-based productivity)
6. Goods market clearing
7. Government plans taxes
8. Government plans transfers
9. Apply sales, profits, taxes to firms
10. Apply income, taxes, transfers, purchases to households
11. **Households consume goods from inventory** ← NEW
12. **Handle firm bankruptcies and exits** ← NEW
13. **Create new firms if needed** ← NEW
14. **Government adjusts policies** ← NEW
15. Update statistics

---

## Configuration Parameters

All parameters can be tuned:

| Feature | Parameter | Default Value | Description |
|---------|-----------|---------------|-------------|
| **Skill Growth** | `skill_growth_rate` | 0.001 | Passive skill improvement per tick |
| **Education** | `education_cost_per_skill_point` | 1000.0 | Cost to gain 0.1 skill points |
| **Consumption** | `consumption_rate` | 0.1 | Fraction of inventory consumed per tick |
| **Bankruptcy** | `bankruptcy_threshold` | -1000.0 | Cash level triggering firm exit |
| **Firm Entry** | `min_firms` | 3 | Minimum number of firms to maintain |
| **Firm Entry** | `starting_capital` | 5000.0 | Initial cash for new firms |
| **Gov Policy** | `high_unemployment` | 0.15 | Threshold for increasing benefits |
| **Gov Policy** | `low_unemployment` | 0.03 | Threshold for reducing benefits |
| **Gov Policy** | `deficit_threshold` | -10000 | Threshold for raising taxes |
| **Gov Policy** | `surplus_threshold` | 50000 | Threshold for lowering taxes |

---

## Economic Implications

### Skill System
- Workers naturally improve over time
- High-skilled workers command premium wages
- Education investment provides accelerated advancement
- Creates realistic wage inequality based on experience

### Consumption System
- Eliminates infinite inventory accumulation
- Creates continuous demand for goods
- Households must earn to maintain consumption
- Realistic goods flow through economy

### Firm Dynamics
- Poorly-managed firms fail and exit
- Market maintains minimum competition
- New entrants keep markets competitive
- Promotes economic renewal

### Government Adaptation
- Counter-cyclical fiscal policy
- Automatic stabilizers during recessions
- Tax adjustments based on fiscal health
- Responds to unemployment dynamically

---

## Future Enhancements

Potential additions to make the system even more dynamic:

1. **Household Birth/Death**: Population dynamics
2. **Firm Investment in Capacity**: Expand when profitable
3. **Technology Shocks**: Productivity improvements
4. **Credit Markets**: Firms can borrow to avoid bankruptcy
5. **Monetary Policy**: Interest rates, money supply
6. **International Trade**: Import/export dynamics
7. **Education Institutions**: Formal training centers
8. **Labor Unions**: Collective bargaining

---

## Testing

Run the comprehensive test:

```bash
python test_dynamic_economy.py
```

This tests:
- ✓ Skill growth over 100 ticks
- ✓ Goods consumption and inventory depletion
- ✓ Firm bankruptcy detection
- ✓ New firm creation
- ✓ Government policy adaptation

---

## Summary

The EcoSim simulation is now a **fully dynamic** economic system where:

- **Workers grow and learn** through experience and education
- **Goods flow realistically** with consumption and depletion
- **Firms compete, fail, and enter** creating market dynamics
- **Government adapts policies** responding to economic conditions

This creates a rich, self-evolving economy suitable for studying complex phenomena like recessions, inequality, and policy impacts.
