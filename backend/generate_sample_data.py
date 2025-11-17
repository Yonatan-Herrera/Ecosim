"""
Generate sample database for data team to work with.

Runs a 200-tick simulation and exports all data to multiple formats:
- SQLite database
- CSV files (per-tick snapshots)
- JSON summary statistics
"""

import json
import sqlite3
import csv
from pathlib import Path
from typing import List
from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def create_sample_economy():
    """Create a small economy for sample data generation."""
    essential_categories = ["Food", "Housing", "Services"]

    # Create government and register baseline firms
    gov = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        unemployment_benefit_level=40.0,
        transfer_budget=8000.0,
        cash_balance=150000.0
    )

    baseline_prices = {
        "Food": 5.0,
        "Housing": 15.0,
        "Services": 7.0,
    }

    firms: List[FirmAgent] = []
    next_firm_id = 1
    for category in essential_categories:
        baseline_firm = FirmAgent(
            firm_id=next_firm_id,
            good_name=f"Baseline{category}",
            cash_balance=1_000_000.0,
            inventory_units=10_000.0,
            good_category=category,
            quality_level=1.0,
            wage_offer=25.0,
            price=baseline_prices.get(category, 5.0),
            expected_sales_units=500.0,
            production_capacity_units=20_000.0,
            units_per_worker=40.0,
            productivity_per_worker=15.0,
            personality="conservative",
            is_baseline=True,
            baseline_production_quota=2_000.0
        )
        baseline_firm.set_personality("conservative")
        gov.register_baseline_firm(category, baseline_firm.firm_id)
        firms.append(baseline_firm)
        next_firm_id += 1

    # Create households with modest starting cash so warmup can circulate demand
    households = []
    for i in range(50):
        base_cash = 500.0 + (i % 10) * 150.0
        households.append(
            HouseholdAgent(
                household_id=i,
                skills_level=min(0.95, 0.2 + (i * 0.01)),
                age=22 + (i % 40),
                cash_balance=base_cash
            )
        )

    return Economy(households=households, firms=firms, government=gov)


def init_database(db_path: str):
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Households table with new wellbeing fields
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

    # Firms table with new personality fields
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

    # Government table with investment fields
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

    conn.commit()
    conn.close()


def export_tick_data(economy: Economy, tick: int, db_path: str):
    """Export current tick data to database."""
    conn = sqlite3.connect(db_path)
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
    unemployed = sum(1 for h in economy.households if not h.is_employed)
    employed_wages = [h.wage for h in economy.households if h.is_employed]

    cursor.execute(
        "INSERT INTO aggregate_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            tick,
            len(economy.households),
            len(economy.firms),
            unemployed / len(economy.households),
            sum(employed_wages) / len(employed_wages) if employed_wages else 0.0,
            sorted(employed_wages)[len(employed_wages)//2] if employed_wages else 0.0,
            sum(h.cash_balance for h in economy.households) / len(economy.households),
            sorted([h.cash_balance for h in economy.households])[len(economy.households)//2],
            sum(h.happiness for h in economy.households) / len(economy.households),
            sum(h.morale for h in economy.households) / len(economy.households),
            sum(h.health for h in economy.households) / len(economy.households),
            sum(h.get_performance_multiplier() for h in economy.households) / len(economy.households),
            sum(f.cash_balance for f in economy.firms),
            sum(f.price for f in economy.firms) / len(economy.firms),
            gov.cash_balance
        )
    )

    conn.commit()
    conn.close()


def main():
    """Generate sample data for data team."""
    print("=" * 70)
    print("SAMPLE DATA GENERATION FOR DATA TEAM")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)

    # Create economy
    print("Creating sample economy...")
    print("  - 50 households")
    print("  - 7 firms (mixed personalities)")
    print("  - 1 government")
    print()

    economy = create_sample_economy()

    # Initialize database
    db_path = output_dir / "ecosim_sample.db"
    print(f"Initializing database: {db_path}")
    init_database(str(db_path))
    print()

    # Run simulation for 200 ticks
    num_ticks = 200
    print(f"Running simulation for {num_ticks} ticks...")
    print()
    print("Tick | Firms | Unemploy | Avg Happiness | Avg Wage | Gov Cash")
    print("-" * 70)

    for tick in range(num_ticks):
        # Step economy
        economy.step()

        # Export data
        export_tick_data(economy, tick, str(db_path))

        # Print progress
        if tick % 20 == 0 or tick == num_ticks - 1:
            unemployed = sum(1 for h in economy.households if not h.is_employed)
            unemployment_rate = unemployed / len(economy.households)
            avg_happiness = sum(h.happiness for h in economy.households) / len(economy.households)
            employed_wages = [h.wage for h in economy.households if h.is_employed]
            avg_wage = sum(employed_wages) / len(employed_wages) if employed_wages else 0.0

            print(f"{tick:4d} | {len(economy.firms):5d} | {unemployment_rate:7.1%} | "
                  f"{avg_happiness:13.3f} | ${avg_wage:7.2f} | ${economy.government.cash_balance:8.0f}")

    print()
    print(f"✓ Generated {num_ticks} ticks of data")
    print(f"✓ Database saved to: {db_path}")
    print()

    # Generate summary statistics
    print("Generating summary statistics...")
    conn = sqlite3.connect(str(db_path))

    # Query final state
    final_tick = num_ticks - 1
    final_metrics = conn.execute(
        "SELECT * FROM aggregate_metrics WHERE tick = ?", (final_tick,)
    ).fetchone()

    summary = {
        "simulation_info": {
            "num_ticks": num_ticks,
            "num_households": 50,
            "num_firms_initial": 7,
            "num_firms_final": final_metrics[2] if final_metrics else 0
        },
        "final_state": {
            "unemployment_rate": final_metrics[3] if final_metrics else 0,
            "mean_wage": final_metrics[4] if final_metrics else 0,
            "mean_happiness": final_metrics[8] if final_metrics else 0,
            "mean_morale": final_metrics[9] if final_metrics else 0,
            "mean_health": final_metrics[10] if final_metrics else 0,
            "government_cash": final_metrics[14] if final_metrics else 0
        },
        "database_stats": {
            "households_rows": conn.execute("SELECT COUNT(*) FROM households").fetchone()[0],
            "firms_rows": conn.execute("SELECT COUNT(*) FROM firms").fetchone()[0],
            "government_rows": conn.execute("SELECT COUNT(*) FROM government").fetchone()[0],
        }
    }

    conn.close()

    # Save summary
    summary_path = output_dir / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}")
    print()
    print("Sample data generation complete!")
    print()
    print("Files generated:")
    print(f"  - {db_path} (SQLite database with all tick data)")
    print(f"  - {summary_path} (Summary statistics)")
    print()
    print("Data team can now use this database for:")
    print("  - Visualization development")
    print("  - ML model prototyping")
    print("  - Database schema validation")


if __name__ == "__main__":
    main()
