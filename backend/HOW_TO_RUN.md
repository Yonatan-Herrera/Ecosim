# How to Run the EcoSim Large-Scale Simulation

## Quick Start

Just run the script - it's pre-configured for 10,000 agents and 500 ticks:

```bash
cd /Users/aymanislam/intern/Ecosim/backend
python run_large_simulation.py
```

## What It Does

The script will:
1. Create 10,000 households
2. Create 33 firms (3 baseline government firms + 30 private firms)
3. Run for 500 simulation ticks
4. Export data to SQLite database every 10 ticks
5. Generate summary JSON file

## Output Files

After completion, you'll find in `sample_data/`:
- `ecosim_10k_balanced.db` - Full SQLite database with all tick data
- `simulation_10k_balanced_summary.json` - Summary statistics

## Current Configuration

```python
NUM_HOUSEHOLDS = 10000          # Number of household agents
NUM_FIRMS_PER_CATEGORY = 10     # Private firms per category (Food, Housing, Services)
NUM_TICKS = 500                 # Simulation length

EXPORT_EVERY_N_TICKS = 10       # Database export frequency
```

## Expected Runtime

- **10,000 agents, 500 ticks**: ~3-5 minutes
- Average: ~300-400ms per tick

## Key Features Implemented

### 1. Price Ceiling Tax
- 25% tax on revenue when prices exceed $50
- Helps government fiscal health
- Discourages excessive pricing

### 2. Balanced Competition
- **Baseline (government) firms**: Low quality (Q3.0), safety net
- **Private firms**: High quality (Q5.0-7.7), competitive advantages
- Private firms designed to outcompete government over time

### 3. Quality Cap
- Maximum quality improvement: 5% (multiplier capped at 1.05)
- Prevents unrealistic quality perfection

## Monitoring Progress

The script prints progress every 10 ticks:
```
Tick | Time(s) | Firms | Unemploy |   Happiness | Avg Wage | Gov Cash
```

## Analyzing Results

Query the database with SQLite:

```bash
sqlite3 sample_data/ecosim_10k_balanced.db

# Example queries:
SELECT * FROM aggregate_metrics WHERE tick = 499;
SELECT good_name, cash_balance, employee_count FROM firms WHERE tick = 499 ORDER BY cash_balance DESC;
```

## Troubleshooting

**Database already exists error:**
The script automatically removes the old database before each run.

**Out of memory:**
Reduce `NUM_HOUSEHOLDS` or `NUM_TICKS` in the script.

**Slow performance:**
- Reduce `NUM_FIRMS_PER_CATEGORY`
- Increase `EXPORT_EVERY_N_TICKS` to export less frequently
