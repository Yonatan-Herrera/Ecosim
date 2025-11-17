# EcoSim Data Specification for Data Team

## Overview
This document specifies the data outputs from the EcoSim economic simulation engine. The simulation generates deterministic, tick-based time-series data across multiple agent types (Households, Firms, Government).

## Target Use Cases
- **Visualization**: Time-series charts, distributions, network graphs
- **Database Storage**: SQLite, PostgreSQL, or time-series databases
- **ML Modeling**: Prediction models for economic indicators, agent behavior forecasting
- **User Dashboards**: Real-time or historical views of economic metrics

---

## Data Schema

### 1. Simulation Metadata

```python
{
    "simulation_id": str,           # Unique identifier for this simulation run
    "start_time": datetime,         # Wall-clock start time
    "current_tick": int,            # Current simulation tick number
    "config": {
        "num_households": int,
        "num_firms": int,
        "initial_conditions": dict,
    }
}
```

---

### 2. Household Agent Data (Per Tick)

**Table: `households`**

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `tick` | int | Simulation tick number | Time-series indexing |
| `household_id` | int | Unique household identifier | Agent tracking |
| `skills_level` | float | Skills (0-1) | Income inequality analysis |
| `age` | int | Household age | Demographic analysis |
| `cash_balance` | float | Current cash holdings | Wealth distribution |
| `employer_id` | int/null | Current employer (null if unemployed) | Employment network |
| `wage` | float | Current wage rate | Income distribution |
| `is_employed` | bool | Employment status | Unemployment rate |
| `goods_inventory` | JSON | `{good_name: quantity}` | Consumption patterns |
| `consumption_budget_share` | float | Propensity to consume (0-1) | Savings rate analysis |
| `category_weights` | JSON | `{category: weight}` | Preference analysis |
| `quality_preference_weight` | float | Quality elasticity | Consumer behavior |
| `price_sensitivity` | float | Price elasticity | Demand modeling |
| `expected_wage` | float | Wage expectation | Expectation dynamics |
| `reservation_wage` | float | Minimum acceptable wage | Labor market analysis |

**Example Row:**
```json
{
    "tick": 150,
    "household_id": 42,
    "skills_level": 0.75,
    "age": 35,
    "cash_balance": 1250.50,
    "employer_id": 7,
    "wage": 55.00,
    "is_employed": true,
    "goods_inventory": {"Food": 10.5, "Housing": 1.0, "Services": 3.2},
    "category_weights": {"Food": 0.35, "Housing": 0.40, "Services": 0.25},
    "quality_preference_weight": 1.2,
    "price_sensitivity": 0.8
}
```

---

### 3. Firm Agent Data (Per Tick)

**Table: `firms`**

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `tick` | int | Simulation tick number | Time-series indexing |
| `firm_id` | int | Unique firm identifier | Agent tracking |
| `good_name` | str | Product name | Product market analysis |
| `good_category` | str | Category (Food/Housing/Services) | Sector analysis |
| `quality_level` | float | Product quality (0-10) | Competition analysis |
| `cash_balance` | float | Current cash holdings | Firm solvency |
| `inventory_units` | float | Stock on hand | Supply analysis |
| `employees` | JSON | `[household_ids]` | Employment network |
| `employee_count` | int | Number of workers | Firm size distribution |
| `wage_offer` | float | Current wage offer | Labor market dynamics |
| `price` | float | Product price | Price dynamics |
| `unit_cost` | float | Production cost per unit | Profitability analysis |
| `markup` | float | Price markup over cost | Pricing strategy |
| `expected_sales_units` | float | Sales expectation | Forecast accuracy |
| `production_capacity_units` | float | Max production | Capacity utilization |
| `rd_spending_rate` | float | R&D as % of revenue | Innovation investment |
| `quality_level` | float | Product quality (0-10) | Quality competition |
| `accumulated_rd_investment` | float | Lifetime R&D spending | Innovation tracking |

**Example Row:**
```json
{
    "tick": 150,
    "firm_id": 7,
    "good_name": "BasicFood",
    "good_category": "Food",
    "quality_level": 6.5,
    "cash_balance": 5000.00,
    "inventory_units": 150.0,
    "employees": [42, 13, 91],
    "employee_count": 3,
    "wage_offer": 55.00,
    "price": 12.50,
    "unit_cost": 10.00,
    "markup": 0.25,
    "expected_sales_units": 100.0,
    "production_capacity_units": 200.0,
    "rd_spending_rate": 0.05,
    "accumulated_rd_investment": 2500.00
}
```

---

### 4. Government Data (Per Tick)

**Table: `government`**

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `tick` | int | Simulation tick number | Time-series indexing |
| `cash_balance` | float | Government reserves | Fiscal health |
| `wage_tax_rate` | float | Tax rate on wages (0-1) | Policy analysis |
| `profit_tax_rate` | float | Tax rate on profits (0-1) | Policy analysis |
| `unemployment_benefit_level` | float | Per-tick payment to unemployed | Safety net analysis |
| `min_cash_threshold` | float | Minimum household cash target | Poverty threshold |
| `transfer_budget` | float | Max transfers per tick | Fiscal capacity |
| `total_wage_taxes_collected` | float | Tax revenue from wages | Revenue analysis |
| `total_profit_taxes_collected` | float | Tax revenue from profits | Revenue analysis |
| `total_transfers_paid` | float | Total transfers distributed | Spending analysis |
| `fiscal_balance` | float | Revenue - Spending | Deficit/surplus |

**Example Row:**
```json
{
    "tick": 150,
    "cash_balance": 50000.00,
    "wage_tax_rate": 0.15,
    "profit_tax_rate": 0.20,
    "unemployment_benefit_level": 30.00,
    "total_wage_taxes_collected": 1500.00,
    "total_profit_taxes_collected": 800.00,
    "total_transfers_paid": 600.00,
    "fiscal_balance": 1700.00
}
```

---

### 5. Aggregate Metrics (Per Tick)

**Table: `aggregate_metrics`**

| Metric | Type | Description | Visualization |
|--------|------|-------------|---------------|
| `tick` | int | Simulation tick | X-axis |
| `total_gdp` | float | Sum of all production value | Line chart |
| `unemployment_rate` | float | % of households unemployed | Line chart |
| `mean_wage` | float | Average wage across employed | Line chart |
| `median_wage` | float | Median wage | Distribution analysis |
| `gini_coefficient` | float | Wealth inequality (0-1) | Inequality tracking |
| `mean_household_cash` | float | Average household wealth | Wealth trends |
| `median_household_cash` | float | Median household wealth | Wealth trends |
| `total_firm_inventory` | float | Sum of all inventory | Supply analysis |
| `mean_price_level` | float | Average price across goods | Inflation tracking |
| `inflation_rate` | float | % change in price level | Macroeconomic indicator |
| `num_active_firms` | int | Firms with positive cash | Firm survival |
| `num_bankrupt_firms` | int | Firms with negative cash | Bankruptcy tracking |
| `government_deficit` | float | Negative fiscal balance | Fiscal health |

**Example Row:**
```json
{
    "tick": 150,
    "total_gdp": 10000.00,
    "unemployment_rate": 0.08,
    "mean_wage": 52.50,
    "median_wage": 50.00,
    "gini_coefficient": 0.35,
    "mean_household_cash": 1500.00,
    "median_household_cash": 1200.00,
    "total_firm_inventory": 500.0,
    "mean_price_level": 11.25,
    "inflation_rate": 0.02,
    "num_active_firms": 10,
    "num_bankrupt_firms": 0,
    "government_deficit": -500.00
}
```

---

### 6. Transaction Data (Per Tick)

**Table: `transactions`**

Records all economic transactions for network analysis and flow tracking.

| Field | Type | Description |
|-------|------|-------------|
| `tick` | int | When transaction occurred |
| `transaction_id` | str | Unique identifier |
| `transaction_type` | str | "purchase", "wage_payment", "tax", "transfer" |
| `from_agent_id` | int/str | Payer ID |
| `from_agent_type` | str | "household", "firm", "government" |
| `to_agent_id` | int/str | Recipient ID |
| `to_agent_type` | str | Agent type |
| `amount` | float | Transaction value |
| `good_name` | str/null | If purchase, what good |
| `quantity` | float/null | If purchase, quantity |

**Example Rows:**
```json
[
    {
        "tick": 150,
        "transaction_id": "150_purchase_42_7",
        "transaction_type": "purchase",
        "from_agent_id": 42,
        "from_agent_type": "household",
        "to_agent_id": 7,
        "to_agent_type": "firm",
        "amount": 125.00,
        "good_name": "BasicFood",
        "quantity": 10.0
    },
    {
        "tick": 150,
        "transaction_id": "150_wage_7_42",
        "transaction_type": "wage_payment",
        "from_agent_id": 7,
        "from_agent_type": "firm",
        "to_agent_id": 42,
        "to_agent_type": "household",
        "amount": 55.00,
        "good_name": null,
        "quantity": null
    }
]
```

---

## Data Export Methods

### Method 1: SQLite Database (Recommended for 10K-100K agents)

```python
import sqlite3
from typing import List

def export_to_sqlite(economy: Economy, tick: int, db_path: str):
    """Export current tick data to SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables (if not exist)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS households (
            tick INTEGER,
            household_id INTEGER,
            skills_level REAL,
            age INTEGER,
            cash_balance REAL,
            employer_id INTEGER,
            wage REAL,
            is_employed BOOLEAN,
            goods_inventory TEXT,
            PRIMARY KEY (tick, household_id)
        )
    """)

    # Bulk insert household data
    household_rows = [
        (tick, h.household_id, h.skills_level, h.age, h.cash_balance,
         h.employer_id, h.wage, h.is_employed, json.dumps(h.goods_inventory))
        for h in economy.households
    ]

    cursor.executemany(
        "INSERT INTO households VALUES (?,?,?,?,?,?,?,?,?)",
        household_rows
    )

    conn.commit()
    conn.close()
```

### Method 2: CSV Export (for single-tick snapshots)

```python
import csv

def export_households_to_csv(households: List[HouseholdAgent], tick: int, filepath: str):
    """Export household data to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'tick', 'household_id', 'skills_level', 'age', 'cash_balance',
            'employer_id', 'wage', 'is_employed'
        ])
        writer.writeheader()

        for h in households:
            writer.writerow({
                'tick': tick,
                'household_id': h.household_id,
                'skills_level': h.skills_level,
                'age': h.age,
                'cash_balance': h.cash_balance,
                'employer_id': h.employer_id,
                'wage': h.wage,
                'is_employed': h.is_employed,
            })
```

### Method 3: Parquet (for large-scale analytics)

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def export_to_parquet(economy: Economy, tick: int, output_dir: str):
    """Export data to Parquet format for efficient analytics."""

    # Convert households to DataFrame
    household_df = pd.DataFrame([h.to_dict() for h in economy.households])
    household_df['tick'] = tick

    # Write to partitioned Parquet
    table = pa.Table.from_pandas(household_df)
    pq.write_to_dataset(
        table,
        root_path=f'{output_dir}/households',
        partition_cols=['tick']
    )
```

---

## Visualization Recommendations

### 1. Time-Series Charts (Line Graphs)
- **Unemployment Rate Over Time**: Track labor market health
- **Mean Wage Over Time**: Monitor wage dynamics
- **Government Cash Balance**: Fiscal health
- **Inflation Rate**: Price stability
- **Gini Coefficient**: Inequality trends

### 2. Distributions (Histograms/Box Plots)
- **Household Wealth Distribution**: Show inequality at a point in time
- **Firm Size Distribution**: Number of employees per firm
- **Wage Distribution**: Income inequality
- **Quality Level Distribution**: Product competition landscape

### 3. Network Graphs
- **Employment Network**: Households → Firms connections
- **Transaction Flow**: Money flow between agents
- **Market Share**: Firm competition within categories

### 4. Heatmaps
- **Category Spending Patterns**: Household preferences across categories
- **Firm Performance Matrix**: Quality vs. Price positioning
- **Correlation Matrix**: Relationships between economic indicators

### 5. Dashboards
- **Real-Time Metrics**: Current tick's key indicators
- **Historical Trends**: Multi-tick time-series
- **Agent Details**: Drill-down into specific households/firms

---

## ML Modeling Opportunities

### 1. Prediction Models
- **Unemployment Forecasting**: Predict unemployment rate N ticks ahead
- **Firm Bankruptcy Prediction**: Binary classification based on firm financials
- **Inflation Prediction**: Forecast price level changes

### 2. Clustering
- **Household Segmentation**: Cluster by wealth, consumption patterns
- **Firm Strategies**: Identify pricing/quality strategy clusters

### 3. Causal Analysis
- **Policy Impact**: Effect of tax rate changes on unemployment
- **Quality Investment ROI**: Relationship between R&D spending and market share

### 4. Anomaly Detection
- **Recession Detection**: Identify early warning signs
- **Market Failures**: Detect when markets stop clearing efficiently

---

## Example SQL Queries

```sql
-- Unemployment rate over time
SELECT tick,
       100.0 * SUM(CASE WHEN is_employed = 0 THEN 1 ELSE 0 END) / COUNT(*) as unemployment_rate
FROM households
GROUP BY tick
ORDER BY tick;

-- Wealth inequality (top 10% vs bottom 50%)
WITH wealth_percentiles AS (
    SELECT tick, household_id, cash_balance,
           NTILE(10) OVER (PARTITION BY tick ORDER BY cash_balance) as wealth_decile
    FROM households
)
SELECT tick,
       AVG(CASE WHEN wealth_decile = 10 THEN cash_balance END) as top_10_pct_wealth,
       AVG(CASE WHEN wealth_decile <= 5 THEN cash_balance END) as bottom_50_pct_wealth
FROM wealth_percentiles
GROUP BY tick;

-- Firm performance by category
SELECT tick, good_category,
       COUNT(*) as num_firms,
       AVG(quality_level) as avg_quality,
       AVG(price) as avg_price,
       SUM(inventory_units) as total_inventory
FROM firms
GROUP BY tick, good_category;
```

---

## Performance Considerations

### Storage Estimates (per 1000 ticks)
- **10 Households, 5 Firms**: ~100 KB
- **1,000 Households, 100 Firms**: ~10 MB
- **10,000 Households, 1,000 Firms**: ~100 MB
- **100,000 Households, 10,000 Firms**: ~1 GB

### Recommendations
- **< 10K agents**: SQLite is sufficient
- **10K-100K agents**: PostgreSQL with indexing
- **> 100K agents**: Time-series database (InfluxDB, TimescaleDB)
- **Analytics workloads**: Export to Parquet, use DuckDB or Apache Spark

---

## Data Quality & Validation

### Invariants to Check
1. **Cash Conservation**: Sum of all agent cash should remain constant (modulo government deficit/surplus)
2. **Employment Consistency**: If household has employer_id, that firm must list them in employees
3. **Inventory Non-Negative**: All inventory values >= 0
4. **Price Floor**: All prices >= firm.min_price

### Data Validation Script
```python
def validate_tick_data(economy: Economy, tick: int) -> List[str]:
    """Return list of validation errors."""
    errors = []

    # Check cash conservation
    total_cash = sum(h.cash_balance for h in economy.households)
    total_cash += sum(f.cash_balance for f in economy.firms)
    total_cash += economy.government.cash_balance

    if abs(total_cash - expected_total) > 1e-6:
        errors.append(f"Tick {tick}: Cash conservation violated")

    # Check employment consistency
    for household in economy.households:
        if household.is_employed:
            employer = next((f for f in economy.firms if f.firm_id == household.employer_id), None)
            if employer is None or household.household_id not in employer.employees:
                errors.append(f"Tick {tick}: Household {household.household_id} employment inconsistent")

    return errors
```

---

## Contact & Support

For questions about data formats or access:
- **Simulation Engine Team**: Maintains agent logic and economy.py
- **Data Team**: Responsible for storage, visualization, and ML modeling

**Data Access API** (to be implemented):
```python
# Example usage
from ecosim import DataExporter

exporter = DataExporter(economy)
exporter.export_tick(tick=150, format='sqlite', path='./data/sim.db')
exporter.export_aggregate_metrics(ticks=range(0, 1000), format='csv')
```
