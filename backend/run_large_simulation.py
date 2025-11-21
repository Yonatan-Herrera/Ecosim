"""
Run large-scale EcoSim simulation with 10,000 agents.

This script creates a larger economy and runs it for a specified number of ticks.
Progress is printed every 10 ticks to monitor performance and economic indicators.
"""

import json
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np

from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def create_large_economy(num_households: int = 10000, num_firms_per_category: int = 10):
    """
    Create a large economy with specified number of agents.

    Args:
        num_households: Number of household agents to create
        num_firms_per_category: Number of firms per essential category

    Returns:
        Economy instance
    """
    print(f"Creating economy with {num_households} households...")

    essential_categories = ["Food", "Housing", "Services"]

    # Create government with scaled parameters
    gov = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        unemployment_benefit_level=40.0,
        transfer_budget=num_households * 200.0,  # Scale with population
        cash_balance=num_households * 3000.0     # Scale with population
    )

    baseline_prices = {
        "Food": 5.0,
        "Housing": 15.0,
        "Services": 7.0,
    }

    # Create firms
    firms: List[FirmAgent] = []
    next_firm_id = 1

    # Create baseline firms (government-controlled "safety net")
    # These should be LOWER quality than private firms, and eventually die out
    print(f"Creating {len(essential_categories)} baseline firms...")
    for category in essential_categories:
        baseline_firm = FirmAgent(
            firm_id=next_firm_id,
            good_name=f"Baseline{category}",
            cash_balance=2_000_000.0,  # Reduced from 10M - still comfortable but not infinite
            inventory_units=20_000.0,   # Reduced from 100k - modest starting inventory
            good_category=category,
            quality_level=3.0,          # LOW quality (on 0-10 scale) - government basic goods
            wage_offer=25.0,
            price=baseline_prices.get(category, 5.0),
            expected_sales_units=num_households * 0.1,
            production_capacity_units=100_000.0,  # Reduced from 200k
            units_per_worker=40.0,
            productivity_per_worker=12.0,  # Lower productivity than private firms
            personality="conservative",
            is_baseline=True,
            baseline_production_quota=num_households * 0.15  # Reduced quota
        )
        baseline_firm.set_personality("conservative")
        gov.register_baseline_firm(category, baseline_firm.firm_id)
        firms.append(baseline_firm)
        next_firm_id += 1

    # Create competitive private firms (HIGHER quality than government)
    print(f"Creating {num_firms_per_category * len(essential_categories)} competitive firms...")
    personalities = ["aggressive", "moderate", "conservative"]

    for category in essential_categories:
        for i in range(num_firms_per_category):
            personality = personalities[i % len(personalities)]
            competitive_firm = FirmAgent(
                firm_id=next_firm_id,
                good_name=f"{category}Co{i+1}",
                cash_balance=800_000.0,  # Increased from 500k - competitive with baseline
                inventory_units=8_000.0,  # Increased starting inventory
                good_category=category,
                quality_level=5.0 + (i * 0.3),  # HIGHER quality: 5.0-7.7 (vs baseline 3.0)
                wage_offer=25.0 + (i * 3.0),    # Higher wages to attract talent
                price=baseline_prices.get(category, 5.0) * (0.95 + i * 0.03),  # Competitive pricing
                expected_sales_units=num_households * 0.03,
                production_capacity_units=60_000.0,  # Better capacity
                units_per_worker=40.0,
                productivity_per_worker=15.0 + (i * 0.8),  # HIGHER productivity
                personality=personality,
                is_baseline=False
            )
            competitive_firm.set_personality(personality)
            firms.append(competitive_firm)
            next_firm_id += 1

    # Create households with distributed characteristics
    print(f"Creating {num_households} households...")
    households = []

    for i in range(num_households):
        # Distribute skills across population (0.2 to 0.95)
        skill_level = min(0.95, 0.2 + (i / num_households) * 0.75)

        # Distribute ages (22 to 62)
        age = 22 + (i % 40)

        # Distribute starting cash (500 to 2000)
        base_cash = 500.0 + (i % 100) * 15.0

        households.append(
            HouseholdAgent(
                household_id=i,
                skills_level=skill_level,
                age=age,
                cash_balance=base_cash
            )
        )

        # Progress indicator for large populations
        if (i + 1) % 1000 == 0:
            print(f"  Created {i + 1}/{num_households} households...")

    print(f"✓ Economy created successfully!")
    print(f"  Total agents: {len(households) + len(firms) + 1}")
    print(f"    - Households: {len(households)}")
    print(f"    - Firms: {len(firms)}")
    print(f"    - Government: 1")
    print()

    return Economy(households=households, firms=firms, government=gov)


def init_database(db_path: str):
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Households table
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
            happiness REAL,
            morale REAL,
            health REAL,
            performance_multiplier REAL,
            food_experience INTEGER,
            housing_experience INTEGER,
            services_experience INTEGER,
            PRIMARY KEY (tick, household_id)
        )
    """)

    # Firms table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS firms (
            tick INTEGER,
            firm_id INTEGER,
            good_name TEXT,
            good_category TEXT,
            quality_level REAL,
            cash_balance REAL,
            inventory_units REAL,
            employee_count INTEGER,
            employees TEXT,
            wage_offer REAL,
            price REAL,
            unit_cost REAL,
            markup REAL,
            expected_sales_units REAL,
            production_capacity_units REAL,
            personality TEXT,
            investment_propensity REAL,
            risk_tolerance REAL,
            PRIMARY KEY (tick, firm_id)
        )
    """)

    # Government table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS government (
            tick INTEGER PRIMARY KEY,
            cash_balance REAL,
            wage_tax_rate REAL,
            profit_tax_rate REAL,
            unemployment_benefit_level REAL,
            transfer_budget REAL,
            infrastructure_productivity_multiplier REAL,
            technology_quality_multiplier REAL,
            social_happiness_multiplier REAL
        )
    """)

    # Aggregate metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aggregate_metrics (
            tick INTEGER PRIMARY KEY,
            total_households INTEGER,
            total_firms INTEGER,
            unemployment_rate REAL,
            mean_wage REAL,
            median_wage REAL,
            mean_household_cash REAL,
            median_household_cash REAL,
            mean_happiness REAL,
            mean_morale REAL,
            mean_health REAL,
            mean_performance REAL,
            total_firm_cash REAL,
            mean_price REAL,
            government_cash REAL
        )
    """)

    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_households_tick ON households(tick)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_firms_tick ON firms(tick)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_households_employed ON households(is_employed)")

    conn.commit()
    conn.close()


def compute_household_stats(households: List[HouseholdAgent]) -> Dict[str, float]:
    """Vectorized snapshot of household metrics for reuse."""
    if not households:
        return {
            "unemployment_rate": 0.0,
            "mean_wage": 0.0,
            "median_wage": 0.0,
            "mean_cash": 0.0,
            "median_cash": 0.0,
            "mean_happiness": 0.0,
            "mean_morale": 0.0,
            "mean_health": 0.0,
            "mean_performance": 0.0
        }

    cash = np.array([h.cash_balance for h in households], dtype=float)
    happiness = np.array([h.happiness for h in households], dtype=float)
    morale = np.array([h.morale for h in households], dtype=float)
    health = np.array([h.health for h in households], dtype=float)
    performance = np.array([h.get_performance_multiplier() for h in households], dtype=float)
    employment = np.array([1.0 if h.is_employed else 0.0 for h in households], dtype=float)
    employed_wages = np.array([h.wage for h in households if h.is_employed], dtype=float)

    unemployment_rate = 1.0 - (employment.mean() if employment.size else 0.0)
    mean_wage = float(employed_wages.mean()) if employed_wages.size else 0.0
    median_wage = float(np.median(employed_wages)) if employed_wages.size else 0.0

    return {
        "unemployment_rate": unemployment_rate,
        "mean_wage": mean_wage,
        "median_wage": median_wage,
        "mean_cash": float(cash.mean()),
        "median_cash": float(np.median(cash)),
        "mean_happiness": float(happiness.mean()),
        "mean_morale": float(morale.mean()),
        "mean_health": float(health.mean()),
        "mean_performance": float(performance.mean())
    }


def export_tick_data(
    economy: Economy,
    tick: int,
    conn: sqlite3.Connection,
    household_stats: Dict[str, float] | None = None
):
    """Export current tick data using an open database connection."""
    cursor = conn.cursor()

    # Export households
    household_rows = []
    for h in economy.households:
        household_rows.append((
            tick,
            h.household_id,
            h.skills_level,
            h.age,
            h.cash_balance,
            h.employer_id,
            h.wage,
            h.is_employed,
            json.dumps(h.goods_inventory),
            h.happiness,
            h.morale,
            h.health,
            h.get_performance_multiplier(),
            h.category_experience.get("Food", 0),
            h.category_experience.get("Housing", 0),
            h.category_experience.get("Services", 0)
        ))

    cursor.executemany(
        "INSERT INTO households VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        household_rows
    )

    # Export firms
    firm_rows = []
    for f in economy.firms:
        firm_rows.append((
            tick,
            f.firm_id,
            f.good_name,
            f.good_category,
            f.quality_level,
            f.cash_balance,
            f.inventory_units,
            len(f.employees),
            json.dumps(f.employees),
            f.wage_offer,
            f.price,
            f.unit_cost,
            f.markup,
            f.expected_sales_units,
            f.production_capacity_units,
            f.personality,
            f.investment_propensity,
            f.risk_tolerance
        ))

    cursor.executemany(
        "INSERT INTO firms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        firm_rows
    )

    # Export government
    gov = economy.government
    cursor.execute(
        "INSERT INTO government VALUES (?,?,?,?,?,?,?,?,?)",
        (tick, gov.cash_balance, gov.wage_tax_rate, gov.profit_tax_rate,
         gov.unemployment_benefit_level, gov.transfer_budget,
         gov.infrastructure_productivity_multiplier,
         gov.technology_quality_multiplier,
         gov.social_happiness_multiplier)
    )

    # Calculate and export aggregate metrics
    stats = household_stats or compute_household_stats(economy.households)
    total_firm_cash = sum(f.cash_balance for f in economy.firms)
    mean_price = sum(f.price for f in economy.firms) / len(economy.firms) if economy.firms else 0.0

    cursor.execute(
        "INSERT INTO aggregate_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            tick,
            len(economy.households),
            len(economy.firms),
            stats["unemployment_rate"],
            stats["mean_wage"],
            stats["median_wage"],
            stats["mean_cash"],
            stats["median_cash"],
            stats["mean_happiness"],
            stats["mean_morale"],
            stats["mean_health"],
            stats["mean_performance"],
            total_firm_cash,
            mean_price,
            gov.cash_balance
        )
    )

    conn.commit()


def main():
    """Run large-scale simulation."""
    print("=" * 80)
    print("ECOSIM LARGE-SCALE SIMULATION (10,000 AGENTS)")
    print("=" * 80)
    print()

    # Configuration
    NUM_HOUSEHOLDS = 10000
    NUM_FIRMS_PER_CATEGORY = 10
    NUM_TICKS = 500  # Run for 500 ticks to see dynamics
    EXPORT_EVERY_N_TICKS = 10  # Export to DB every 10 ticks

    # Create output directory
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)

    # Create economy
    start_time = time.time()
    economy = create_large_economy(NUM_HOUSEHOLDS, NUM_FIRMS_PER_CATEGORY)
    creation_time = time.time() - start_time
    print(f"Economy creation time: {creation_time:.2f} seconds")
    print()

    # Initialize database (remove existing file if present)
    db_path = output_dir / "ecosim_10k_balanced.db"
    if db_path.exists():
        db_path.unlink()  # Delete existing database
        print(f"Removed existing database: {db_path}")
    print(f"Initializing database: {db_path}")
    init_database(str(db_path))
    print()

    # Prepare persistent DB connection
    db_conn = sqlite3.connect(str(db_path))

    # Run simulation
    print(f"Running simulation for {NUM_TICKS} ticks...")
    print(f"(Exporting to database every {EXPORT_EVERY_N_TICKS} ticks)")
    print()
    print("=" * 120)
    print(" TICK | Time(s) |  GDP/Tick | Firms | Unemp% | Avg Wage |   Happiness | Firm Wealth | HH Wealth | Gov Cash")
    print("=" * 120)

    tick_time_history: deque[float] = deque(maxlen=10)
    tick_time_sum = 0.0

    for tick in range(NUM_TICKS):
        tick_start = time.time()

        # Step economy
        economy.step()

        tick_time = time.time() - tick_start
        tick_time_history.append(tick_time)
        tick_time_sum += tick_time
        household_stats = compute_household_stats(economy.households)

        # Export data periodically
        if tick % EXPORT_EVERY_N_TICKS == 0 or tick == NUM_TICKS - 1:
            export_tick_data(economy, tick, db_conn, household_stats)

        # Print progress every 10 ticks
        if tick % 10 == 0 or tick == NUM_TICKS - 1:
            avg_tick_time = sum(tick_time_history) / len(tick_time_history)
            metrics = economy.get_economic_metrics()

            print(f" {tick:4d} | {avg_tick_time:7.3f} | ${metrics['gdp_this_tick']:9,.0f} | "
                  f"{metrics['total_firms']:5d} | {metrics['unemployment_rate']:5.1%} | "
                  f"${metrics['mean_wage']:8.2f} | {metrics['mean_happiness']:11.3f} | "
                  f"${metrics['total_firm_cash']:11,.0f} | ${metrics['total_household_cash']:9,.0f} | "
                  f"${metrics['government_cash']:9,.0f}")

    print()
    total_time = time.time() - start_time
    avg_tick_time = tick_time_sum / NUM_TICKS

    db_conn.close()

    print(f"✓ Simulation complete!")
    print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Average tick time: {avg_tick_time:.3f} seconds")
    print(f"  Ticks per second: {1/avg_tick_time:.2f}")
    print(f"  Database saved to: {db_path}")
    print()

    # Generate summary statistics
    print("Generating summary statistics...")
    conn = sqlite3.connect(str(db_path))

    # Query final state
    final_tick = NUM_TICKS - 1
    final_metrics = conn.execute(
        "SELECT * FROM aggregate_metrics WHERE tick = ?", (final_tick,)
    ).fetchone()

    # Get time series data for key metrics
    metrics_over_time = conn.execute(
        "SELECT tick, unemployment_rate, mean_wage, mean_happiness FROM aggregate_metrics ORDER BY tick"
    ).fetchall()

    summary = {
        "simulation_info": {
            "num_ticks": NUM_TICKS,
            "num_households": NUM_HOUSEHOLDS,
            "num_firms_initial": len(economy.firms),
            "num_firms_final": final_metrics[2] if final_metrics else 0,
            "total_simulation_time_seconds": total_time,
            "average_tick_time_seconds": avg_tick_time
        },
        "final_state": {
            "tick": final_tick,
            "unemployment_rate": final_metrics[3] if final_metrics else 0,
            "mean_wage": final_metrics[4] if final_metrics else 0,
            "median_wage": final_metrics[5] if final_metrics else 0,
            "mean_household_cash": final_metrics[6] if final_metrics else 0,
            "median_household_cash": final_metrics[7] if final_metrics else 0,
            "mean_happiness": final_metrics[8] if final_metrics else 0,
            "mean_morale": final_metrics[9] if final_metrics else 0,
            "mean_health": final_metrics[10] if final_metrics else 0,
            "government_cash": final_metrics[14] if final_metrics else 0,
            "total_firm_cash": final_metrics[12] if final_metrics else 0
        },
        "database_stats": {
            "households_rows": conn.execute("SELECT COUNT(*) FROM households").fetchone()[0],
            "firms_rows": conn.execute("SELECT COUNT(*) FROM firms").fetchone()[0],
            "government_rows": conn.execute("SELECT COUNT(*) FROM government").fetchone()[0],
        },
        "time_series_sample": {
            "ticks": [row[0] for row in metrics_over_time[::10]],  # Every 10th tick
            "unemployment_rate": [row[1] for row in metrics_over_time[::10]],
            "mean_wage": [row[2] for row in metrics_over_time[::10]],
            "mean_happiness": [row[3] for row in metrics_over_time[::10]]
        }
    }

    conn.close()

    # Save summary (remove existing file to avoid stale content)
    summary_path = output_dir / "simulation_10k_balanced_summary.json"
    if summary_path.exists():
        summary_path.unlink()
        print(f"Removed existing summary: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}")
    print()

    # Get final comprehensive metrics
    final_metrics = economy.get_economic_metrics()

    print("=" * 120)
    print("COMPREHENSIVE ECONOMIC DASHBOARD - FINAL STATE")
    print("=" * 120)
    print()

    # GDP & Economic Output
    print("📊 ECONOMIC OUTPUT")
    print("-" * 120)
    total_gdp = sum([row[1] for row in metrics_over_time])
    avg_gdp = total_gdp / len(metrics_over_time) if metrics_over_time else 0
    print(f"  Total GDP (all ticks):        ${total_gdp:15,.0f}")
    print(f"  Average GDP per tick:         ${avg_gdp:15,.0f}")
    print(f"  Final tick GDP:               ${final_metrics['gdp_this_tick']:15,.0f}")
    print(f"  Total economy wealth:         ${final_metrics['total_economy_cash']:15,.0f}")
    print()

    # Labor Market
    print("👥 LABOR MARKET")
    print("-" * 120)
    print(f"  Total households:             {final_metrics['total_households']:15,}")
    print(f"  Employed:                     {final_metrics['employed_count']:15,} ({100 - final_metrics['unemployment_rate']*100:.1f}%)")
    print(f"  Unemployed:                   {final_metrics['unemployed_count']:15,} ({final_metrics['unemployment_rate']*100:.1f}%)")
    print(f"  Mean wage:                    ${final_metrics['mean_wage']:15,.2f}")
    print(f"  Median wage:                  ${final_metrics['median_wage']:15,.2f}")
    print(f"  Wage range:                   ${final_metrics['min_wage']:,.2f} - ${final_metrics['max_wage']:,.2f}")
    print()

    # Household Wellbeing
    print("😊 HOUSEHOLD WELLBEING")
    print("-" * 120)
    print(f"  Mean happiness:               {final_metrics['mean_happiness']:15.3f} (0-1 scale)")
    print(f"  Mean morale:                  {final_metrics['mean_morale']:15.3f} (0-1 scale)")
    print(f"  Mean health:                  {final_metrics['mean_health']:15.3f} (0-1 scale)")
    print(f"  Mean skills:                  {final_metrics['mean_skills']:15.3f} (0-1 scale)")
    print()

    # Household Finances
    print("💰 HOUSEHOLD FINANCES")
    print("-" * 120)
    print(f"  Total household wealth:       ${final_metrics['total_household_cash']:15,.0f}")
    print(f"  Mean household cash:          ${final_metrics['mean_household_cash']:15,.2f}")
    print(f"  Median household cash:        ${final_metrics['median_household_cash']:15,.2f}")
    print()

    # Firm Sector
    print("🏢 FIRM SECTOR")
    print("-" * 120)
    print(f"  Total firms operating:        {final_metrics['total_firms']:15,}")
    print(f"  Total firm wealth:            ${final_metrics['total_firm_cash']:15,.0f}")
    print(f"  Mean firm cash:               ${final_metrics['mean_firm_cash']:15,.2f}")
    print(f"  Median firm cash:             ${final_metrics['median_firm_cash']:15,.2f}")
    print(f"  Total inventory:              {final_metrics['total_firm_inventory']:15,.0f} units")
    print(f"  Total employees:              {final_metrics['total_employees']:15,}")
    print(f"  Mean price:                   ${final_metrics['mean_price']:15,.2f}")
    print(f"  Mean quality:                 {final_metrics['mean_quality']:15.2f} (0-10 scale)")
    print()

    # Government
    print("🏛️  GOVERNMENT FINANCES & POLICY")
    print("-" * 120)
    print(f"  Government cash balance:      ${final_metrics['government_cash']:15,.0f}")
    print(f"  Wage tax rate:                {final_metrics['wage_tax_rate']*100:15.1f}%")
    print(f"  Profit tax rate:              {final_metrics['profit_tax_rate']*100:15.1f}%")
    print(f"  Unemployment benefit:         ${final_metrics['unemployment_benefit']:15,.2f}")
    print(f"  Transfer budget:              ${final_metrics['transfer_budget']:15,.0f}")
    print()
    print(f"  Infrastructure multiplier:    {final_metrics['infrastructure_productivity']:15.3f}x")
    print(f"  Technology multiplier:        {final_metrics['technology_quality']:15.3f}x")
    print(f"  Social happiness multiplier:  {final_metrics['social_happiness']:15.3f}x")
    print()

    # Performance
    print("⚡ SIMULATION PERFORMANCE")
    print("-" * 120)
    print(f"  Total agents:                 {NUM_HOUSEHOLDS + len(economy.firms) + 1:15,}")
    print(f"  Total ticks:                  {NUM_TICKS:15,}")
    print(f"  Total time:                   {total_time:15.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Average tick time:            {avg_tick_time*1000:15.1f} ms")
    print(f"  Ticks per second:             {1/avg_tick_time:15.2f}")
    print()

    print("📁 FILES GENERATED")
    print("-" * 120)
    print(f"  Database:  {db_path}")
    print(f"  Summary:   {summary_path}")
    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
