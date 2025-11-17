# EcoSim Data Specification for Data Team

## Overview

This document specifies the data outputs from the redesigned EcoSim economic simulation engine. The simulation generates deterministic, tick-based time-series data across multiple agent types (Households, Firms, Government) with new dynamic features including wellbeing systems, firm personalities, and government investment.

## Performance Requirements

### Scale Targets

The data infrastructure should support:
- **Ideal scale**: **100,000 agents** (100K households + firms)
- **Stretch goal**: **1,000,000 agents** (1M total)
- **Minimum ticks**: 1,000 ticks for meaningful patterns
- **Recommended**: 10,000 ticks for long-term dynamics

### Storage Implications

| Scale | Households | Firms | Ticks | Estimated Storage | Recommended DB |
|-------|------------|-------|-------|-------------------|----------------|
| Small | 1,000 | 100 | 1,000 | ~50 MB | SQLite |
| Medium | 10,000 | 1,000 | 1,000 | ~500 MB | PostgreSQL |
| Large | 100,000 | 10,000 | 1,000 | ~5 GB | PostgreSQL + Indexing |
| **Ideal** | **100,000** | **10,000** | **10,000** | **~50 GB** | **TimescaleDB / InfluxDB** |
| Stretch | 1,000,000 | 100,000 | 1,000 | ~50 GB | Distributed DB (Cassandra) |

---

## Sample Data

**Pre-generated sample database available**: `sample_data/ecosim_sample.db`

- **50 households** with varying skills, ages, wealth
- **7 firms** across 3 categories (Food, Housing, Services)
- **200 ticks** of simulation data
- **All new features** included (wellbeing, personalities, government investment)

Use this database to:
- ✅ Develop visualization prototypes
- ✅ Test ML models
- ✅ Validate database schemas
- ✅ Build dashboards without running simulations

---

## New Features (Since Last Update)

### 1. Wellbeing System
Households now have happiness, morale, and health metrics that affect performance.

### 2. Firm Personalities
Firms have distinct strategies: aggressive, moderate, or conservative.

### 3. Government Investment
Government invests in infrastructure (productivity), technology (quality), and social programs (happiness).

### 4. Just-in-Time Production
Firms no longer stockpile inventory - they produce to replace what was sold.

### 5. Demand-Based Pricing
Prices adjust based on sell-through rates, not cost-plus markup.

See [REDESIGN_FEATURES.md](REDESIGN_FEATURES.md) for complete details on all new features.

---

## Data Schema (Updated)

### Household Table (households)

| Field | Type | Description | New? | Use Case |
|-------|------|-------------|------|----------|
| `tick` | int | Simulation tick number | | Time-series indexing |
| `household_id` | int | Unique household identifier | | Agent tracking |
| `skills_level` | float | Skills (0-1) | | Income inequality analysis |
| `age` | int | Household age | | Demographic analysis |
| `cash_balance` | float | Current cash holdings | | Wealth distribution |
| `employer_id` | int/null | Current employer | | Employment network |
| `wage` | float | Current wage rate | | Income distribution |
| `is_employed` | bool | Employment status | | Unemployment rate |
| **`happiness`** | **float** | **Happiness (0-1)** | **✅** | **Wellbeing tracking** |
| **`morale`** | **float** | **Morale (0-1)** | **✅** | **Performance analysis** |
| **`health`** | **float** | **Health (0-1)** | **✅** | **Productivity modeling** |
| **`performance_multiplier`** | **float** | **Performance (0.5-1.5)** | **✅** | **Output calculation** |
| **`food_experience`** | **int** | **Ticks worked in Food** | **✅** | **Career tracking** |
| **`housing_experience`** | **int** | **Ticks worked in Housing** | **✅** | **Career tracking** |
| **`services_experience`** | **int** | **Ticks worked in Services** | **✅** | **Career tracking** |

### Firm Table (firms)

| Field | Type | Description | New? | Use Case |
|-------|------|-------------|------|----------|
| `tick` | int | Simulation tick number | | Time-series indexing |
| `firm_id` | int | Unique firm identifier | | Agent tracking |
| `good_name` | str | Product name | | Product analysis |
| `good_category` | str | Food/Housing/Services | | Sector analysis |
| `quality_level` | float | Quality (0-10) | | Competition |
| `cash_balance` | float | Cash holdings | | Solvency |
| `inventory_units` | float | Stock on hand | | Supply |
| `employee_count` | int | Number of workers | | Firm size |
| `wage_offer` | float | Wage offer | | Labor market |
| `price` | float | Product price | | Price dynamics |
| **`personality`** | **str** | **"aggressive"/"moderate"/"conservative"** | **✅** | **Strategy** |
| **`investment_propensity`** | **float** | **% of profits invested** | **✅** | **Investment behavior** |
| **`risk_tolerance`** | **float** | **Risk (0-1)** | **✅** | **Clustering** |

### Government Table (government)

| Field | Type | Description | New? | Use Case |
|-------|------|-------------|------|----------|
| `tick` | int | Simulation tick | | Time-series |
| `cash_balance` | float | Reserves | | Fiscal health |
| `wage_tax_rate` | float | Tax rate (0-1) | | Policy |
| `profit_tax_rate` | float | Tax rate (0-1) | | Policy |
| **`infrastructure_productivity_multiplier`** | **float** | **Productivity boost** | **✅** | **Growth** |
| **`technology_quality_multiplier`** | **float** | **Quality boost** | **✅** | **Innovation** |
| **`social_happiness_multiplier`** | **float** | **Happiness boost** | **✅** | **Wellbeing policy** |

### Aggregate Metrics Table (aggregate_metrics)

| Metric | Type | Description | New? |
|--------|------|-------------|------|
| `tick` | int | Simulation tick | |
| `unemployment_rate` | float | % unemployed | |
| `mean_wage` | float | Average wage | |
| `mean_household_cash` | float | Average wealth | |
| **`mean_happiness`** | **float** | **Average happiness** | **✅** |
| **`mean_morale`** | **float** | **Average morale** | **✅** |
| **`mean_health`** | **float** | **Average health** | **✅** |
| **`mean_performance`** | **float** | **Average performance** | **✅** |

See [generate_sample_data.py](generate_sample_data.py) for complete schema implementation.

---

## Visualization Guide

### 🔹 Simple Visualizations (Start Here)

#### 1. Basic Time-Series

```python
# Unemployment Over Time
plt.plot(df['tick'], df['unemployment_rate'] * 100)
plt.xlabel('Tick')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time')
```

```python
# Wellbeing Metrics
plt.plot(df['tick'], df['mean_happiness'], label='Happiness')
plt.plot(df['tick'], df['mean_morale'], label='Morale')
plt.plot(df['tick'], df['mean_health'], label='Health')
plt.legend()
```

**Tools**: Matplotlib, Seaborn, Plotly

#### 2. Distributions

```python
# Wealth Distribution at Tick 1000
tick_1000 = households[households['tick'] == 1000]
plt.hist(tick_1000['cash_balance'], bins=50)
plt.xlabel('Cash Balance ($)')
plt.ylabel('Count')
plt.title('Wealth Distribution')
```

**Tools**: Matplotlib, Seaborn

---

### 🔸 Intermediate Visualizations

#### 3. Multi-Panel Dashboards

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Unemployment', 'Mean Wage', 'Gov Cash', 'Happiness')
)

fig.add_trace(go.Scatter(x=df['tick'], y=df['unemployment_rate']), row=1, col=1)
fig.add_trace(go.Scatter(x=df['tick'], y=df['mean_wage']), row=1, col=2)
fig.add_trace(go.Scatter(x=df['tick'], y=df['government_cash']), row=2, col=1)
fig.add_trace(go.Scatter(x=df['tick'], y=df['mean_happiness']), row=2, col=2)

fig.show()
```

**Tools**: Plotly Dash, Streamlit

#### 4. Firm Strategy Comparison

```python
# Compare aggressive vs conservative firms
aggressive = firms[firms['personality'] == 'aggressive'].groupby('tick')['price'].mean()
conservative = firms[firms['personality'] == 'conservative'].groupby('tick')['price'].mean()

plt.plot(aggressive.index, aggressive.values, label='Aggressive', color='red')
plt.plot(conservative.index, conservative.values, label='Conservative', color='green')
plt.legend()
```

#### 5. Scatter Plots

```python
# Wage vs Performance
tick_500 = households[households['tick'] == 500]
plt.scatter(tick_500['wage'], tick_500['performance_multiplier'], alpha=0.5)
plt.xlabel('Wage ($)')
plt.ylabel('Performance Multiplier')
```

---

### 🔹 Advanced Visualizations

#### 6. Correlation Heatmaps

```python
import seaborn as sns

corr_data = aggregate_metrics[['unemployment_rate', 'mean_wage',
                                'mean_happiness', 'government_cash']].corr()

sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
plt.title('Economic Indicators Correlation')
```

#### 7. Network Graphs (Employment)

```python
import networkx as nx

G = nx.Graph()
tick_100 = households[households['tick'] == 100]

for _, h in tick_100.iterrows():
    if h['is_employed']:
        G.add_edge(f"H{h['household_id']}", f"F{h['employer_id']}")

nx.draw(G, with_labels=True, node_size=300)
```

**Tools**: NetworkX, PyVis

#### 8. Animated Charts

```python
import plotly.express as px

fig = px.histogram(households, x='cash_balance', animation_frame='tick',
                   range_x=[0, 10000], nbins=50)
fig.show()
```

#### 9. 3D Scatter (Wellbeing)

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tick_1000 = households[households['tick'] == 1000]
ax.scatter(tick_1000['happiness'], tick_1000['morale'], tick_1000['health'],
           c=tick_1000['performance_multiplier'], cmap='viridis')
```

#### 10. Sankey Diagrams (Money Flow)

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(label = ["Firms", "Households", "Government"]),
    link = dict(
      source = [0, 1, 1],
      target = [1, 0, 2],
      value = [wage_payments, purchases, taxes]
  ))])
fig.show()
```

---

## Team Tasks & Assignments

### Task 1: Database Schema Design ⚙️
**Assignee**: Database Engineer
**Time**: 2-3 days

**Deliverables**:
- PostgreSQL/TimescaleDB schema for 100K agents
- Indexes on `(tick, agent_id)`, `tick`, `employer_id`
- Partitioning by tick ranges (1000-tick partitions)
- Views for current state, aggregates

```sql
CREATE TABLE households_partitioned (
    tick INTEGER NOT NULL,
    household_id INTEGER NOT NULL,
    ...
    PRIMARY KEY (tick, household_id)
) PARTITION BY RANGE (tick);
```

---

### Task 2: Data Export Module 💾
**Assignee**: Backend Engineer
**Time**: 3-4 days

**Deliverables**:
- `DataExporter` class in `data_exporter.py`
- Methods: `export_tick_snapshot()`, `export_batch()`, `export_aggregate_metrics()`
- Support: SQLite, PostgreSQL, CSV, Parquet, JSON
- Optimize for 100K agents (batch inserts)

```python
exporter = DataExporter(economy)
exporter.export_tick_snapshot(tick=1000, format='postgresql')
exporter.export_batch(start_tick=0, end_tick=10000, format='parquet')
```

---

### Task 3: Real-Time Dashboard 📊
**Assignee**: Frontend/Visualization Engineer
**Time**: 5-7 days

**Deliverables**:
- Interactive dashboard (Plotly Dash or Streamlit)
- Visualizations: Unemployment, wage, wealth dist, wellbeing, gov cash, firm survival
- Filters: Tick range, category, personality
- Current tick stats cards
- Auto-refresh for live simulation

---

### Task 4: ML Modeling Pipeline 🤖
**Assignee**: Data Scientist
**Time**: 7-10 days

**Deliverables**:
1. **Prediction models**:
   - Unemployment forecasting (LSTM/ARIMA)
   - Firm bankruptcy prediction (Random Forest/XGBoost)
   - Wage prediction (regression)
2. **Clustering**:
   - Household segmentation (K-means)
   - Firm strategy clusters
3. **Jupyter notebooks** with examples

Use `sample_data/ecosim_sample.db` for prototyping.

---

### Task 5: Data Quality Monitoring ✅
**Assignee**: QA/Data Engineer
**Time**: 3-4 days

**Deliverables**:
- Validation checks: cash conservation, employment consistency, value bounds
- `validate_simulation_data(db_path)` function
- Data quality reports per tick
- Alerts for violations

```python
def validate_simulation_data(db_path, tick):
    errors = []
    # Check cash conservation
    # Check employment consistency
    # Check non-negative values
    return errors
```

---

### Task 6: Performance Benchmarking ⚡
**Assignee**: Performance Engineer
**Time**: 4-5 days

**Deliverables**:
- Benchmark at 1K, 10K, 100K, 1M agents
- Measure: ticks/second, memory, DB write speed
- Profile bottlenecks (`cProfile`, `py-spy`)
- Optimize labor market matching, goods clearing, DB inserts
- Hardware requirements doc

---

## SQL Query Examples

### Unemployment Rate
```sql
SELECT tick,
       100.0 * SUM(CASE WHEN is_employed = 0 THEN 1 ELSE 0 END) / COUNT(*) AS unemployment_rate
FROM households
GROUP BY tick
ORDER BY tick;
```

### Wellbeing by Employment
```sql
SELECT is_employed,
       AVG(happiness) AS avg_happiness,
       AVG(morale) AS avg_morale,
       AVG(health) AS avg_health
FROM households
WHERE tick = 1000
GROUP BY is_employed;
```

### Firm Performance by Personality
```sql
SELECT personality,
       COUNT(*) AS num_firms,
       AVG(cash_balance) AS avg_cash,
       AVG(employee_count) AS avg_employees
FROM firms
WHERE tick = 1000
GROUP BY personality;
```

### Government Investment Impact
```sql
SELECT tick,
       infrastructure_productivity_multiplier,
       technology_quality_multiplier,
       social_happiness_multiplier
FROM government
ORDER BY tick;
```

---

## Using Sample Data

**Load in Python**:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('sample_data/ecosim_sample.db')
households = pd.read_sql_query("SELECT * FROM households", conn)
firms = pd.read_sql_query("SELECT * FROM firms", conn)
metrics = pd.read_sql_query("SELECT * FROM aggregate_metrics", conn)
conn.close()
```

**Load in R**:
```r
library(DBI)
library(RSQLite)

conn <- dbConnect(RSQLite::SQLite(), "sample_data/ecosim_sample.db")
households <- dbReadTable(conn, "households")
firms <- dbReadTable(conn, "firms")
dbDisconnect(conn)
```

---

## Contact & Support

For questions:
- **Simulation Engine Team**: @engine-team (maintains agents, economy)
- **Data Team Lead**: @data-lead (coordinates viz, ML, database)

**Office Hours**: Tuesdays 2-3 PM

---

## Reference Documents

- [REDESIGN_FEATURES.md](REDESIGN_FEATURES.md) - Complete feature documentation
- [generate_sample_data.py](generate_sample_data.py) - Sample data generation script
- [DYNAMIC_FEATURES.md](DYNAMIC_FEATURES.md) - Original dynamic features spec

---

## Quick Start Checklist

For new data team members:

- [ ] Read [REDESIGN_FEATURES.md](REDESIGN_FEATURES.md)
- [ ] Download `sample_data/ecosim_sample.db`
- [ ] Load sample data in Python/R
- [ ] Create first visualization (unemployment over time)
- [ ] Review assigned task from list above
- [ ] Ask questions in #data-team channel
