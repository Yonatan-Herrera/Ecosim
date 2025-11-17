# Economic Redesign Features

## Overview

The EcoSim simulation has been fundamentally redesigned to create a more dynamic, realistic economy with:

1. **Just-in-Time Production** - Firms no longer stockpile inventory
2. **Demand-Based Pricing** - Prices respond to what people are willing to buy
3. **Firm Personality System** - Aggressive vs conservative firm strategies
4. **Government Investment** - Infrastructure, technology, and social programs
5. **Happiness/Morale/Health System** - Worker wellbeing affects performance

---

## 1. Just-in-Time Production

### Problem Solved

Previously, firms accumulated large inventories and stopped hiring workers, creating unemployment. The economy would stagnate as firms focused on inventory management rather than employment.

### New System

Firms now produce to **replace what was sold**, not to build inventory:

```python
# OLD: Target inventory approach
target_inventory = target_inventory_multiplier * expected_sales
desired_production = max(target_inventory - current_inventory, 0)

# NEW: Just-in-time approach
desired_production = max(expected_sales, minimum_production_floor)
```

### Impact

- **Continuous employment demand**: Firms must keep hiring to meet ongoing sales
- **No inventory accumulation**: Production matches consumption
- **More realistic**: Resembles modern supply chain management
- **Better job creation**: Economy sustains employment levels

### Configuration

```python
# In FirmAgent.plan_production_and_labor()
small_positive_floor = 10.0  # Minimum production to maintain employment
```

---

## 2. Demand-Based Pricing

### Problem Solved

Previously, prices were based on cost-plus markup, not demand. This didn't reflect what people were willing to pay or market conditions.

### New System

Prices now adjust aggressively based on **sell-through rate** (fraction of inventory sold):

| Sell-Through Rate | Demand Level | Price Change |
|-------------------|--------------|--------------|
| > 90% | Very High | +15% (3x adjustment rate) |
| 70-90% | Good | +7.5% (1.5x adjustment rate) |
| 50-70% | Moderate | +2.5% (0.5x adjustment rate) |
| 30-50% | Weak | -5% (1x adjustment rate) |
| < 30% | Very Weak | -10% (2x adjustment rate) |

```python
# Example with price_adjustment_rate = 0.05
if sell_through_rate > 0.9:
    price_next = price * 1.15  # +15%
elif sell_through_rate < 0.3:
    price_next = price * 0.90  # -10%
```

### Impact

- **Market-driven prices**: Reflect actual demand, not just costs
- **Faster equilibrium**: Prices adjust quickly to supply/demand
- **Competitive dynamics**: Firms compete on price sensitivity to demand
- **No artificial floors**: Prices can drop if no one is buying

### Government Baseline Firm

The government can operate a baseline firm (field: `baseline_firm_id`) that sets reference pricing for private firms to benchmark against.

---

## 3. Firm Personality System

### Problem Solved

All firms behaved identically. Real economies have diverse firm strategies.

### New System

Firms are assigned one of three personalities (deterministically based on firm_id):

#### Aggressive Firms (33%)
- **Investment**: 15% of profits reinvested
- **Risk tolerance**: 0.9 (very high)
- **Price adjustment**: 10% per tick (rapid)
- **Wage bidding**: 15% adjustment (aggressive)
- **R&D spending**: 8% of revenue

**Strategy**: High risk, high reward - grab market share quickly

#### Conservative Firms (33%)
- **Investment**: 2% of profits reinvested
- **Risk tolerance**: 0.2 (very low)
- **Price adjustment**: 2% per tick (gradual)
- **Wage bidding**: 5% adjustment (cautious)
- **R&D spending**: 2% of revenue

**Strategy**: Play it safe - stable, predictable growth

#### Moderate Firms (33%)
- **Investment**: 5% of profits reinvested
- **Risk tolerance**: 0.5 (balanced)
- **Price adjustment**: 5% per tick (standard)
- **Wage bidding**: 10% adjustment (balanced)
- **R&D spending**: 5% of revenue

**Strategy**: Balanced approach - middle ground

### Implementation

```python
# Firm creation
personality_index = new_firm_id % 3
personality = ["aggressive", "conservative", "moderate"][personality_index]
new_firm.set_personality(personality)
```

### Impact

- **Market diversity**: Different firms behave differently
- **Competitive dynamics**: Aggressive firms disrupt, conservative firms stabilize
- **Realistic**: Mirrors real-world firm heterogeneity
- **Strategic depth**: Users can observe different business strategies

---

## 4. Government Investment System

### Problem Solved

Government could only tax and transfer. Real governments invest in infrastructure, technology, and social programs.

### New System

Government has three investment capabilities:

#### A. Infrastructure Investment

**Budget**: $1,000 per tick (configurable)

**Effect**: Boosts economy-wide productivity

```python
# Each $1000 invested → +0.5% productivity multiplier
productivity_gain = (investment / 1000.0) * 0.005
infrastructure_productivity_multiplier += productivity_gain
```

**Impact**:
- All workers produce more
- Higher GDP
- Compounds over time

#### B. Technology Investment

**Budget**: $500 per tick (configurable)

**Effect**: Improves product quality across economy

```python
# Each $500 invested → +0.5% quality multiplier
quality_gain = (investment / 500.0) * 0.005
technology_quality_multiplier += quality_gain
```

**Impact**:
- All firms' products improve
- Higher consumer satisfaction
- Better competitiveness

#### C. Social Investment

**Budget**: $750 per tick (configurable)

**Effect**: Boosts happiness/health (healthcare, amenities, culture)

```python
# Each $750 invested → +0.5% happiness multiplier
happiness_gain = (investment / 750.0) * 0.005
social_happiness_multiplier += happiness_gain
```

**Impact**:
- Workers happier and healthier
- Better performance
- Higher quality of life

### Investment Logic

Government only invests when it has surplus:

```python
if government.cash_balance > 10000.0:  # Keep reserve
    government.make_investments()
```

### Example Timeline

| Tick | Gov Cash | Infrastructure Invested | Productivity Multiplier |
|------|----------|------------------------|------------------------|
| 0 | $10,000 | $0 | 1.00x |
| 1 | $12,000 | $1,000 | 1.005x (+0.5%) |
| 10 | $25,000 | $1,000 | 1.050x (+5.0%) |
| 100 | $50,000 | $1,000 | 1.500x (+50%) |

---

## 5. Happiness/Morale/Health System

### Problem Solved

Workers were identical regardless of their circumstances. Real workers' performance depends on wellbeing.

### New System

Each household has three wellbeing metrics:

#### Happiness (0-1 scale)

**Affected by**:
- Employment status (employed: +2%, unemployed: -3%)
- Consumption (goods > 10: +1%, goods < 2: -2%)
- Government social programs (multiplier bonus)
- Natural decay (-1% per tick)

**Impact**: Affects engagement and consumption

#### Morale (0-1 scale)

**Affected by**:
- Wage satisfaction (wage ≥ expected: +3%, underpaid: -5%)
- Employment status (unemployed: -5%)
- Natural decay (-2% per tick, faster than happiness)

**Impact**: Affects day-to-day work performance (50% weight)

#### Health (0-1 scale)

**Affected by**:
- Consumption (goods > 15: +1%, goods < 5: -2%)
- Government healthcare (multiplier bonus)
- Natural decay (-0.5% per tick, slowest)

**Impact**: Affects capacity and stamina (30% weight)

### Performance Multiplier

Workers' overall performance is calculated as:

```python
wellbeing_score = (morale * 0.5) + (health * 0.3) + (happiness * 0.2)
performance_multiplier = 0.5 + (wellbeing_score * 1.0)
```

**Range**: 0.5x (terrible wellbeing) to 1.5x (perfect wellbeing)

### Example Scenarios

| Scenario | Happiness | Morale | Health | Performance |
|----------|-----------|--------|--------|-------------|
| Thriving worker (employed, well-paid, healthy) | 0.9 | 0.9 | 0.95 | 1.43x |
| Struggling worker (employed, underpaid) | 0.6 | 0.5 | 0.7 | 1.05x |
| Unemployed worker (no income, poor health) | 0.3 | 0.2 | 0.5 | 0.78x |
| Destitute worker (long unemployed) | 0.1 | 0.1 | 0.3 | 0.61x |

### Integration with Production

```python
# In _calculate_experience_adjusted_production()
wellbeing_multiplier = household.get_performance_multiplier()
productivity_multiplier *= wellbeing_multiplier
```

A worker with 0.6x wellbeing performance produces 40% less than a worker with 1.5x performance!

---

## Updated Simulation Flow

The economy now executes these phases per tick:

1. Firms plan production, labor, prices, wages
2. Households plan labor supply and consumption
3. Labor market matching (with skill/experience premiums)
4. Apply labor outcomes (with skill growth)
5. **Firms apply production (with wellbeing and infrastructure multipliers)** ← UPDATED
6. Goods market clearing
7. Government plans taxes
8. Government plans transfers
9. Apply sales, profits, taxes to firms
10. Apply income, taxes, transfers, purchases to households
11. Households consume goods from inventory
11.5. **Government makes investments (infrastructure, technology, social)** ← NEW
11.75. **Households update wellbeing (happiness, morale, health)** ← NEW
12. Handle firm bankruptcies and exits
13. **Create new firms with random personalities** ← UPDATED
14. Government adjusts policies
15. Update statistics

---

## Configuration Parameters

### Just-in-Time Production

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `small_positive_floor` | `FirmAgent.plan_production_and_labor()` | 10.0 | Minimum production units per tick |

### Demand-Based Pricing

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `price_adjustment_rate` | `FirmAgent` | 0.05 | Base price adjustment rate |
| Aggressive multiplier | `plan_pricing()` | 3.0 | Price change for sell-through > 90% |
| Weak multiplier | `plan_pricing()` | 2.0 | Price change for sell-through < 30% |

### Firm Personalities

| Personality | Investment | Risk | Price Adj | Wage Adj | R&D |
|-------------|------------|------|-----------|----------|-----|
| Aggressive | 15% | 0.9 | 10% | 15% | 8% |
| Moderate | 5% | 0.5 | 5% | 10% | 5% |
| Conservative | 2% | 0.2 | 2% | 5% | 2% |

### Government Investment

| Investment Type | Budget | Effect per $1000 | Multiplier Field |
|-----------------|--------|------------------|------------------|
| Infrastructure | $1,000 | +0.5% productivity | `infrastructure_productivity_multiplier` |
| Technology | $500 | +0.5% quality | `technology_quality_multiplier` |
| Social | $750 | +0.5% happiness | `social_happiness_multiplier` |

### Wellbeing System

| Metric | Decay Rate | Weight in Performance | Range |
|--------|------------|----------------------|-------|
| Happiness | 1% | 20% | 0-1 |
| Morale | 2% | 50% | 0-1 |
| Health | 0.5% | 30% | 0-1 |

**Performance range**: 0.5x (all metrics = 0) to 1.5x (all metrics = 1.0)

---

## Economic Implications

### Just-in-Time Production

**Pros**:
- Continuous job creation
- No idle capital in inventory
- Responsive to demand changes

**Cons**:
- More sensitive to demand shocks
- Less buffer for downturns

### Demand-Based Pricing

**Pros**:
- Prices reflect market conditions
- Faster price discovery
- Better resource allocation

**Cons**:
- Higher price volatility
- Can create feedback loops

### Firm Personalities

**Pros**:
- Market diversity
- Realistic competition
- Different risk/reward profiles

**Cons**:
- Aggressive firms may destabilize markets
- Conservative firms may stagnate

### Government Investment

**Pros**:
- Counter-cyclical stabilization
- Long-term growth
- Improved living standards

**Cons**:
- Requires fiscal discipline
- Effects are gradual
- Can crowd out private investment if over-used

### Wellbeing System

**Pros**:
- Realistic labor dynamics
- Feedback between economy and workers
- Quality of life matters

**Cons**:
- Can amplify downturns (unhappy → unproductive → more unhappy)
- Requires balanced economy to maintain wellbeing

---

## Testing the New System

Run the existing test with updated expectations:

```bash
python test_dynamic_economy.py
```

Expected outcomes:
- Firms continuously hire to meet demand
- Prices fluctuate based on sales
- Mix of aggressive, moderate, and conservative firms
- Government invests surplus funds
- Worker wellbeing varies based on employment and wages

---

## Future Enhancements

Potential additions to make the system even more realistic:

1. **Credit markets**: Firms can borrow to avoid bankruptcy
2. **Firm investment in capacity**: Expand production when profitable
3. **Healthcare system**: Explicit healthcare goods and services
4. **Education institutions**: Schools as economic actors
5. **Labor unions**: Collective bargaining
6. **International trade**: Import/export dynamics
7. **Business cycles**: Endogenous booms and recessions
8. **Innovation shocks**: Technology breakthroughs

---

## Summary

The redesigned EcoSim creates a **demand-driven, dynamic economy** where:

- **Firms produce to meet demand**, not build inventory
- **Prices reflect willingness to pay**, not just costs
- **Firms have diverse strategies**, creating realistic competition
- **Government invests in growth**, not just redistribution
- **Workers' wellbeing matters**, affecting productivity

This creates a rich, evolving economy suitable for studying complex phenomena like:
- Recessions and recoveries
- Inequality dynamics
- Policy effectiveness
- Firm competition
- Quality of life trade-offs
