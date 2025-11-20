"""
Firm Behavior Integration Test

This test runs a 52-tick (1 year) simulation and tracks firm behavior.
Shows firm state every 10 ticks to demonstrate:
- Production and inventory management
- Hiring and wage decisions
- Price adjustments
- Cash flow and profitability
- Quality improvements

Usage:
    python test_firm_behavior.py
"""

import sys
from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_firm_table(firms, tick):
    """Print a formatted table of firm states."""
    print(f"\n--- Tick {tick} ---")
    print(f"{'Firm':<15} {'Cash':<12} {'Inventory':<10} {'Employees':<10} {'Wage':<8} {'Price':<8} {'Quality':<8}")
    print("-" * 100)
    
    for firm in firms:
        print(f"{firm.good_name:<15} ${firm.cash_balance:<11.2f} {firm.inventory_units:<9.1f} "
              f"{len(firm.employees):<10} ${firm.wage_offer:<7.2f} ${firm.price:<7.2f} {firm.quality_level:<7.2f}")


def test_firm_behavior_52_ticks():
    """Run a 52-tick simulation tracking firm behavior."""
    print_section("FIRM BEHAVIOR TEST - 52 Tick Simulation (1 Year)")
    
    # Create small economy
    print("\nCreating economy:")
    print("  - 100 households")
    print("  - 3 firms (Food, Housing, Services)")
    print("  - 1 government")
    
    # Create households
    households = []
    for i in range(100):
        household = HouseholdAgent(
            household_id=i + 1,
            skills_level=0.3 + (i * 0.005),  # Skills from 0.3 to 0.8
            age=25 + (i % 40),
            cash_balance=1000.0 + (i * 10),
            category_weights={"Food": 0.4, "Housing": 0.3, "Services": 0.3}
        )
        households.append(household)
    
    # Create firms
    firms = [
        FirmAgent(
            firm_id=1,
            good_name="FoodCorp",
            cash_balance=50000.0,
            inventory_units=500.0,
            good_category="Food",
            quality_level=5.0,
            wage_offer=50.0,
            price=1.0,
            expected_sales_units=200.0,
            production_capacity_units=1000.0,
            productivity_per_worker=15.0
        ),
        FirmAgent(
            firm_id=2,
            good_name="HousingCo",
            cash_balance=75000.0,
            inventory_units=300.0,
            good_category="Housing",
            quality_level=6.0,
            wage_offer=60.0,
            price=15.0,
            expected_sales_units=150.0,
            production_capacity_units=800.0,
            productivity_per_worker=12.0
        ),
        FirmAgent(
            firm_id=3,
            good_name="ServicesCorp",
            cash_balance=40000.0,
            inventory_units=400.0,
            good_category="Services",
            quality_level=5.5,
            wage_offer=55.0,
            price=12.0,
            expected_sales_units=180.0,
            production_capacity_units=900.0,
            productivity_per_worker=14.0
        )
    ]
    
    # Create government
    government = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        transfer_budget=2000.0,
        unemployment_benefit_level=40.0
    )
    
    # Create economy
    economy = Economy(
        households=households,
        firms=firms,
        government=government
    )
    
    print("\n✓ Economy created successfully")
    print(f"  Initial total household cash: ${sum(h.cash_balance for h in households):,.2f}")
    print(f"  Initial total firm cash: ${sum(f.cash_balance for f in firms):,.2f}")
    print(f"  Initial government cash: ${government.cash_balance:,.2f}")
    
    # Print initial state
    print_firm_table(economy.firms, tick=0)
    
    # Run simulation for 52 ticks (1 year)
    print_section("Running 52-Tick Simulation (52 ticks = 1 year)")
    
    for tick in range(1, 53):
        economy.step()
        
        # Print firm state every 10 ticks
        if tick % 10 == 0 or tick == 52:
            print_firm_table(economy.firms, tick)
    
    # Final summary
    print_section("SIMULATION COMPLETE - Final Analysis")
    
    print("\nFirm Performance Summary:")
    print(f"{'Firm':<15} {'Initial Cash':<14} {'Final Cash':<12} {'Change':<12} {'Employees':<10} {'Avg Quality':<12}")
    print("-" * 100)
    
    initial_cash = [50000.0, 75000.0, 40000.0]
    for i, firm in enumerate(economy.firms):
        cash_change = firm.cash_balance - initial_cash[i]
        change_pct = (cash_change / initial_cash[i]) * 100
        print(f"{firm.good_name:<15} ${initial_cash[i]:<13,.2f} ${firm.cash_balance:<11,.2f} "
              f"{cash_change:>+11,.2f} ({change_pct:>+6.1f}%) {len(firm.employees):<10} {firm.quality_level:<11.2f}")
    
    print("\nKey Observations:")
    print(f"  • Total firms still operating: {len(economy.firms)}")
    print(f"  • Total employees across all firms: {sum(len(f.employees) for f in economy.firms)}")
    print(f"  • Unemployment rate: {sum(1 for h in economy.households if not h.is_employed) / len(economy.households) * 100:.1f}%")
    print(f"  • Average household cash: ${sum(h.cash_balance for h in economy.households) / len(economy.households):,.2f}")
    print(f"  • Government cash balance: ${economy.government.cash_balance:,.2f}")
    
    # Detailed firm analysis
    print("\nDetailed Firm Metrics:")
    for firm in economy.firms:
        print(f"\n  {firm.good_name}:")
        print(f"    Cash: ${firm.cash_balance:,.2f}")
        print(f"    Inventory: {firm.inventory_units:.1f} units")
        print(f"    Employees: {len(firm.employees)}")
        print(f"    Wage: ${firm.wage_offer:.2f}")
        print(f"    Price: ${firm.price:.2f}")
        print(f"    Quality: {firm.quality_level:.2f}")
        print(f"    Expected sales: {firm.expected_sales_units:.1f} units/tick")
    
    print("\n✅ TEST COMPLETE: Firm behavior tracked over 52 ticks")
    return 0


def main():
    """Run the firm behavior test."""
    try:
        return test_firm_behavior_52_ticks()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
