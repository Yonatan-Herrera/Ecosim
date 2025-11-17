# Skill-Based and Experience-Based Wage & Productivity System

## Overview

The EcoSim simulation now implements a sophisticated skill and experience-based system where:
1. **Workers with higher skills earn higher wages**
2. **Workers accumulate industry-specific experience over time**
3. **Experience increases both wages and productivity**

This creates realistic labor market dynamics where skilled, experienced workers are more valuable to firms.

---

## Implementation Details

### 1. Skill-Based Wages

Workers are paid based on their `skills_level` (0.0 to 1.0):

```python
skill_premium = skills_level * 0.5  # Up to 50% premium for max skill
```

**Example:**
- Base wage offer: $50/tick
- Worker with `skills_level = 1.0` earns: $50 × 1.5 = **$75/tick**
- Worker with `skills_level = 0.5` earns: $50 × 1.25 = **$62.50/tick**
- Worker with `skills_level = 0.0` earns: $50 × 1.0 = **$50/tick**

### 2. Experience Accumulation

Each tick a household is employed in a category, they gain 1 tick of experience in that category:

```python
# In HouseholdAgent
category_experience: Dict[str, int]  # category -> ticks worked

# Accumulated in apply_labor_outcome()
if self.is_employed and employer_category is not None:
    if employer_category not in self.category_experience:
        self.category_experience[employer_category] = 0
    self.category_experience[employer_category] += 1
```

**Categories:**
- `"Food"` - Food production industry
- `"Housing"` - Housing/construction industry
- `"Services"` - Service industry
- Any custom category defined by firms

### 3. Experience-Based Wage Premium

Experience provides an additional wage premium:

```python
# Assume 52 ticks per year
experience_years = experience_ticks / 52.0
experience_premium = min(experience_years * 0.05, 0.5)  # 5% per year, capped at 50%
```

**Example Timeline:**
- **Year 0 (0 ticks):** 0% experience premium
- **Year 1 (52 ticks):** 5% experience premium
- **Year 5 (260 ticks):** 25% experience premium
- **Year 10+ (520+ ticks):** 50% experience premium (capped)

### 4. Total Wage Calculation

The final wage combines base wage, skill premium, and experience premium:

```python
actual_wage = wage_offer * (1.0 + skill_premium + experience_premium)
```

**Example: Experienced Skilled Worker**
- Firm wage offer: $50/tick
- Worker `skills_level`: 0.8
- Worker experience: 5 years (260 ticks) in the same category
- Skill premium: 0.8 × 0.5 = 0.4 (40%)
- Experience premium: 5 × 0.05 = 0.25 (25%)
- **Final wage: $50 × (1.0 + 0.4 + 0.25) = $50 × 1.65 = $82.50/tick**

### 5. Experience-Based Productivity

Experienced workers produce more output per tick:

```python
# Base multiplier
productivity_multiplier = 1.0

# Add skill bonus (max 25% for skills_level = 1.0)
skill_bonus = household.skills_level * 0.25

# Add experience bonus (5% per year, capped at 50%)
experience_years = experience_ticks / 52.0
experience_bonus = min(experience_years * 0.05, 0.5)

productivity_multiplier += skill_bonus + experience_bonus
```

**Impact on Production:**
- Firms with experienced workforces produce more units per tick
- Higher productivity means lower unit costs (same wage bill, more output)
- This creates incentive for firms to retain experienced workers

**Example:**
- Base productivity: 10 units/worker/tick
- Experienced worker (5 years, skills 0.5):
  - Skill bonus: 0.5 × 0.25 = 0.125 (12.5%)
  - Experience bonus: 5 × 0.05 = 0.25 (25%)
  - Total multiplier: 1.375
  - **Actual productivity: 10 × 1.375 = 13.75 units/worker/tick**

---

## Economic Implications

### For Workers
- **Incentive to stay in one industry:** Switching categories resets experience
- **Value of skills:** High-skilled workers command premium wages immediately
- **Career progression:** Wages naturally increase over time with experience
- **Industry specialization:** Deep experience in one category is valuable

### For Firms
- **Retention matters:** Experienced workers are more productive
- **Training costs:** New hires are less productive initially
- **Wage pressure:** Must pay premiums to retain experienced workers
- **Productivity gains:** Long-term employees improve profitability

### Market Dynamics
- **Natural wage inequality:** Emerges from skill and experience differences
- **Sector-specific human capital:** Workers become specialized
- **Hiring competition:** Firms compete for experienced workers in their category
- **Economic mobility:** Low-skill workers can build experience over time

---

## Data Export

The experience system adds new fields to household data:

```python
{
    "household_id": 42,
    "skills_level": 0.75,
    "wage": 82.50,
    "category_experience": {
        "Food": 260,      # 5 years in food industry
        "Housing": 52     # 1 year in housing (previous job)
    }
}
```

### Useful Analytics

**Track experience distribution:**
```sql
SELECT good_category,
       AVG(category_experience) as avg_experience_ticks,
       AVG(category_experience) / 52.0 as avg_experience_years
FROM (
    SELECT h.household_id, f.good_category,
           JSON_EXTRACT(h.category_experience, CONCAT('$.', f.good_category)) as category_experience
    FROM households h
    JOIN firms f ON h.employer_id = f.firm_id
    WHERE h.is_employed = 1
)
GROUP BY good_category;
```

**Analyze wage premiums:**
```sql
SELECT
    h.household_id,
    h.skills_level,
    h.category_experience,
    h.wage,
    f.wage_offer as base_wage,
    (h.wage / f.wage_offer - 1.0) as total_premium
FROM households h
JOIN firms f ON h.employer_id = f.firm_id
WHERE h.is_employed = 1
ORDER BY total_premium DESC;
```

---

## Configuration Parameters

All parameters can be tuned in the code:

| Parameter | Current Value | Description |
|-----------|---------------|-------------|
| `skill_premium_rate` | 0.5 | Max wage premium for skill=1.0 (50%) |
| `experience_premium_rate` | 0.05 | Wage premium per year (5%) |
| `experience_premium_cap` | 0.5 | Max experience premium (50%) |
| `ticks_per_year` | 52 | Conversion factor for years |
| `productivity_skill_bonus_rate` | 0.25 | Max productivity bonus for skill=1.0 (25%) |
| `productivity_experience_bonus_rate` | 0.05 | Productivity bonus per year (5%) |
| `productivity_experience_cap` | 0.5 | Max productivity bonus (50%) |

To modify these, search for the relevant calculations in:
- [economy.py](/Users/aymanislam/Ecosim/backend/economy.py) - `_match_labor()` and `_calculate_experience_adjusted_production()`
- [agents.py](/Users/aymanislam/Ecosim/backend/agents.py) - `HouseholdAgent.apply_labor_outcome()`

---

## Testing

Run the test suite to verify the system:

```bash
python test_skill_experience_system.py
```

This tests:
1. ✓ Skill-based wage differentiation
2. ✓ Experience accumulation tracking
3. ✓ Experience-based productivity bonuses

---

## Future Enhancements

Potential improvements to the system:

1. **Education/Training System:**
   - Households can invest in skills training
   - Costs cash but increases `skills_level`
   - Firms might subsidize training

2. **Experience Decay:**
   - Unused experience slowly decays
   - Encourages continuous employment

3. **Cross-Category Experience Transfer:**
   - Some experience transfers between related categories
   - Example: "Food" and "Services" might have 25% transfer

4. **Firm-Specific Experience:**
   - Track time at specific firm vs. category
   - Firm-specific bonuses for loyalty

5. **Skill Levels by Category:**
   - Different skills for different industries
   - More realistic specialization

---

## Summary

The skill and experience system creates realistic labor market dynamics where:
- Workers are rewarded for their skills and experience
- Firms benefit from retaining experienced workers
- Natural wage inequality emerges from individual differences
- Career progression happens organically through experience accumulation

This system provides a foundation for analyzing income inequality, labor market dynamics, and the value of human capital in the simulated economy.
