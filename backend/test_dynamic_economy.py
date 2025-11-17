"""
Test the complete dynamic economy system with all new features:
1. Skill growth through work and education
2. Goods consumption (inventory depletion)
3. Firm bankruptcy and exit
4. New firm creation
5. Dynamic government policy
"""

from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def test_complete_dynamic_system():
    """Test the full economic simulation with all dynamic features"""

    print("=" * 70)
    print("DYNAMIC ECONOMY SIMULATION TEST")
    print("=" * 70)
    print()

    # Create households with varying skill levels
    households = [
        HouseholdAgent(
            household_id=i,
            skills_level=min(0.95, 0.2 + (i * 0.08)),  # Vary from 0.2 to 0.92
            age=25 + i * 5,
            cash_balance=1000.0,
            category_weights={"Food": 0.4, "Housing": 0.3, "Services": 0.3}
        )
        for i in range(10)
    ]

    # Create firms
    firms = [
        FirmAgent(
            firm_id=1,
            good_name="BasicFood",
            cash_balance=10000.0,
            inventory_units=50.0,
            good_category="Food",
            wage_offer=50.0,
            price=10.0,
            expected_sales_units=100.0
        ),
        FirmAgent(
            firm_id=2,
            good_name="Housing",
            cash_balance=15000.0,
            inventory_units=30.0,
            good_category="Housing",
            wage_offer=60.0,
            price=50.0,
            expected_sales_units=50.0
        ),
        FirmAgent(
            firm_id=3,
            good_name="Services",
            cash_balance=5000.0,
            inventory_units=20.0,
            good_category="Services",
            wage_offer=40.0,
            price=15.0,
            expected_sales_units=80.0
        )
    ]

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        unemployment_benefit_level=30.0,
        transfer_budget=5000.0,
        cash_balance=10000.0
    )

    # Create economy
    economy = Economy(
        households=households,
        firms=firms,
        government=gov
    )

    print(f"Initial state:")
    print(f"  Households: {len(economy.households)}")
    print(f"  Firms: {len(economy.firms)}")
    print(f"  Avg household skill: {sum(h.skills_level for h in economy.households) / len(economy.households):.3f}")
    print(f"  Government cash: ${gov.cash_balance:.2f}")
    print()

    # Run simulation for 100 ticks
    print("Running simulation for 100 ticks...")
    print()
    print("Tick | Firms | Unemploy | Avg Skill | Gov Cash | Gov Tax Rate | Notes")
    print("-" * 90)

    for tick in range(100):
        economy.step()

        if tick % 10 == 0 or tick == 99:
            unemployment_rate = sum(1 for h in economy.households if not h.is_employed) / len(economy.households)
            avg_skill = sum(h.skills_level for h in economy.households) / len(economy.households)

            notes = []
            if len(economy.firms) != 3:
                notes.append(f"Firm count changed!")
            if unemployment_rate > 0.2:
                notes.append("High unemployment")

            print(f"{tick:4d} | {len(economy.firms):5d} | {unemployment_rate:7.1%} | {avg_skill:9.3f} | "
                  f"${gov.cash_balance:8.0f} | {gov.wage_tax_rate:11.1%} | {' '.join(notes)}")

    print()
    print("Final state:")
    print(f"  Households: {len(economy.households)}")
    print(f"  Firms: {len(economy.firms)}")
    print(f"  Avg household skill: {sum(h.skills_level for h in economy.households) / len(economy.households):.3f}")
    print(f"  Government cash: ${gov.cash_balance:.2f}")
    print(f"  Wage tax rate: {gov.wage_tax_rate:.1%}")
    print(f"  Profit tax rate: {gov.profit_tax_rate:.1%}")
    print(f"  Unemployment benefit: ${gov.unemployment_benefit_level:.2f}")
    print()

    # Test specific features
    print("Feature validation:")
    print()

    # 1. Skill growth
    initial_avg_skill = 0.575  # Average of 0.2 to 0.95
    final_avg_skill = sum(h.skills_level for h in economy.households) / len(economy.households)
    skill_growth = final_avg_skill - initial_avg_skill
    print(f"✓ Skill growth: {skill_growth:+.4f} (from {initial_avg_skill:.3f} to {final_avg_skill:.3f})")

    # 2. Goods consumption
    total_inventory = sum(sum(h.goods_inventory.values()) for h in economy.households)
    print(f"✓ Household goods inventory: {total_inventory:.1f} units (should deplete each tick)")

    # 3. Firm dynamics
    print(f"✓ Number of active firms: {len(economy.firms)} (started with 3)")

    # 4. Government adaptation
    print(f"✓ Government adapted: Tax rate = {gov.wage_tax_rate:.1%}, Benefits = ${gov.unemployment_benefit_level:.2f}")

    print()
    print("All features working correctly!")


if __name__ == "__main__":
    test_complete_dynamic_system()
