"""
Demonstration of the skill and experience-based wage system

This script shows how wages and productivity evolve as workers
gain experience in their industries.
"""

from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def run_career_simulation():
    """Simulate a worker's career progression over 10 years"""

    print("=" * 60)
    print("CAREER PROGRESSION SIMULATION (10 years)")
    print("=" * 60)
    print()

    # Create a worker who will build experience
    worker = HouseholdAgent(
        household_id=1,
        skills_level=0.6,  # Moderately skilled
        age=25,
        cash_balance=5000.0,
        category_weights={"Food": 1.0}
    )

    # Create a firm in food industry
    food_firm = FirmAgent(
        firm_id=1,
        good_name="ProcessedFood",
        cash_balance=100000.0,
        inventory_units=0.0,
        good_category="Food",
        wage_offer=50.0,
        expected_sales_units=200.0,  # High demand to keep worker employed
        productivity_per_worker=10.0
    )

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        transfer_budget=5000.0
    )

    # Create economy
    economy = Economy(
        households=[worker],
        firms=[food_firm],
        government=gov
    )

    print(f"Initial state:")
    print(f"  Worker skills: {worker.skills_level:.2f}")
    print(f"  Base wage offer: ${food_firm.wage_offer:.2f}/tick")
    print(f"  Expected skill premium: {worker.skills_level * 0.5:.1%}")
    print()

    # Track progression over 10 years (520 ticks)
    years = 10
    ticks_per_year = 52
    total_ticks = years * ticks_per_year

    print("Year | Experience | Wage     | Wage Increase | Productivity Multiplier")
    print("-" * 70)

    for tick in range(total_ticks + 1):
        if tick % ticks_per_year == 0:
            year = tick // ticks_per_year
            experience_ticks = worker.category_experience.get("Food", 0)
            wage = worker.wage if worker.is_employed else 0.0

            # Calculate expected productivity multiplier
            skill_bonus = worker.skills_level * 0.25
            exp_years = experience_ticks / 52.0
            exp_bonus = min(exp_years * 0.05, 0.5)
            prod_multiplier = 1.0 + skill_bonus + exp_bonus

            # Calculate wage increase from year 0
            initial_wage = 50.0 * (1.0 + worker.skills_level * 0.5)
            wage_increase = ((wage - initial_wage) / initial_wage * 100) if worker.is_employed else 0

            print(f"{year:4d} | {experience_ticks:10d} | ${wage:7.2f} | {wage_increase:12.1f}% | {prod_multiplier:22.3f}")

        if tick < total_ticks:
            economy.step()

    print()
    print("Final state:")
    print(f"  Total experience: {worker.category_experience.get('Food', 0)} ticks ({worker.category_experience.get('Food', 0) / 52:.1f} years)")
    print(f"  Final wage: ${worker.wage:.2f}/tick" if worker.is_employed else "  Status: Unemployed")
    print(f"  Total cash accumulated: ${worker.cash_balance:.2f}")
    print()


def compare_workers_with_different_skills():
    """Compare career outcomes for workers with different skill levels"""

    print("=" * 60)
    print("SKILL LEVEL COMPARISON (After 5 years)")
    print("=" * 60)
    print()

    # Create three workers with different skill levels
    workers = [
        HouseholdAgent(
            household_id=1,
            skills_level=0.2,
            age=25,
            cash_balance=5000.0,
            category_weights={"Food": 1.0},
            category_experience={"Food": 260}  # 5 years experience
        ),
        HouseholdAgent(
            household_id=2,
            skills_level=0.6,
            age=25,
            cash_balance=5000.0,
            category_weights={"Housing": 1.0},
            category_experience={"Housing": 260}  # 5 years experience
        ),
        HouseholdAgent(
            household_id=3,
            skills_level=1.0,
            age=25,
            cash_balance=5000.0,
            category_weights={"Services": 1.0},
            category_experience={"Services": 260}  # 5 years experience
        ),
    ]

    # Create three firms
    firms = [
        FirmAgent(
            firm_id=1,
            good_name="Food",
            cash_balance=100000.0,
            inventory_units=0.0,
            good_category="Food",
            wage_offer=50.0
        ),
        FirmAgent(
            firm_id=2,
            good_name="Housing",
            cash_balance=100000.0,
            inventory_units=0.0,
            good_category="Housing",
            wage_offer=50.0
        ),
        FirmAgent(
            firm_id=3,
            good_name="Services",
            cash_balance=100000.0,
            inventory_units=0.0,
            good_category="Services",
            wage_offer=50.0
        ),
    ]

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.15,
        profit_tax_rate=0.20,
        transfer_budget=5000.0
    )

    # Create economy
    economy = Economy(
        households=workers,
        firms=firms,
        government=gov
    )

    # Run one tick to establish wages
    economy.step()

    print("Worker | Skills | Experience | Category | Wage     | Skill Prem | Exp Prem | Total Prem")
    print("-" * 90)

    for i, worker in enumerate(workers, 1):
        category = list(worker.category_weights.keys())[0]
        exp_ticks = worker.category_experience.get(category, 0)
        wage = worker.wage if worker.is_employed else 0.0

        # Calculate premiums
        skill_prem = worker.skills_level * 0.5
        exp_years = exp_ticks / 52.0
        exp_prem = min(exp_years * 0.05, 0.5)
        total_prem = skill_prem + exp_prem

        print(f"{i:6d} | {worker.skills_level:6.2f} | {exp_ticks:10d} | {category:8s} | ${wage:7.2f} | {skill_prem:9.1%} | {exp_prem:7.1%} | {total_prem:9.1%}")

    print()
    print("Key insights:")
    print("  - All workers have same experience (5 years)")
    print("  - Higher skill levels result in higher immediate wages")
    print("  - Experience premium is the same for all (25%)")
    print("  - Total wage difference comes from skill differences")
    print()


if __name__ == "__main__":
    run_career_simulation()
    print()
    compare_workers_with_different_skills()
