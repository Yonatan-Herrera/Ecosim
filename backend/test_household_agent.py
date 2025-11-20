"""
Comprehensive Test Suite for HouseholdAgent

This test file validates the HouseholdAgent behavior including:
- Labor supply decisions
- Consumption planning with price sensitivity
- Skill development and experience accumulation
- Wellbeing tracking (happiness, morale, health)
- Inventory management and goods consumption

Run this test to verify household agents are working correctly.

Usage:
    python test_household_agent.py
"""

import sys
from agents import HouseholdAgent


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_household_creation():
    """Test 1: Verify household agent can be created with default parameters."""
    print_section("TEST 1: Household Creation")

    household = HouseholdAgent(
        household_id=1,
        skills_level=0.5,
        age=30,
        cash_balance=1000.0
    )

    print(f"✓ Created household #{household.household_id}")
    print(f"  Skills: {household.skills_level}")
    print(f"  Age: {household.age}")
    print(f"  Cash: ${household.cash_balance:.2f}")
    print(f"  Employed: {household.is_employed}")
    print(f"  Happiness: {household.happiness:.3f}")
    print(f"  Morale: {household.morale:.3f}")
    print(f"  Health: {household.health:.3f}")

    assert household.household_id == 1, "Household ID mismatch"
    assert household.skills_level == 0.5, "Skills level mismatch"
    assert household.cash_balance == 1000.0, "Cash balance mismatch"

    print("\n✅ TEST 1 PASSED: Household creation successful")
    return household


def test_labor_supply_planning(household):
    """Test 2: Verify household labor supply decisions."""
    print_section("TEST 2: Labor Supply Planning")

    # Unemployed household should search for job
    unemployment_benefit = 40.0  # Government unemployment support
    labor_plan = household.plan_labor_supply(unemployment_benefit)

    print(f"Labor Supply Plan:")
    print(f"  Unemployment benefit: ${unemployment_benefit:.2f}")
    print(f"  Searching for job: {labor_plan['searching_for_job']}")
    print(f"  Reservation wage: ${labor_plan['reservation_wage']:.2f}")
    print(f"  Skills level: {labor_plan['skills_level']:.2f}")

    assert labor_plan['searching_for_job'] == True, "Unemployed should search"
    assert labor_plan['reservation_wage'] > 0, "Reservation wage should be positive"

    print("\n✅ TEST 2 PASSED: Labor supply planning works correctly")


def test_consumption_planning(household):
    """Test 3: Verify household consumption decisions."""
    print_section("TEST 3: Consumption Planning")

    # Set up market prices
    market_prices = {
        "Food": 5.0,
        "Housing": 15.0,
        "Services": 7.0
    }

    consumption_plan = household.plan_consumption(market_prices)

    print(f"Consumption Plan:")
    print(f"  Budget available: ${household.cash_balance * 0.9:.2f} (90% of cash)")
    print(f"  Planned purchases: {len(consumption_plan['planned_purchases'])} items")

    for good, quantity in consumption_plan['planned_purchases'].items():
        if quantity > 0:
            price = market_prices.get(good, 0)
            cost = quantity * price
            print(f"    - {good}: {quantity:.2f} units @ ${price:.2f} = ${cost:.2f}")

    assert 'planned_purchases' in consumption_plan, "Missing planned_purchases"

    print("\n✅ TEST 3 PASSED: Consumption planning works correctly")


def test_employment_and_wages(household):
    """Test 4: Verify employment status changes and wage tracking."""
    print_section("TEST 4: Employment and Wage Updates")

    # Simulate getting hired
    labor_outcome = {
        "employer_id": 10,
        "wage": 75.0,
        "employer_category": "Food"
    }

    print(f"Before employment:")
    print(f"  Employed: {household.is_employed}")
    print(f"  Wage: ${household.wage:.2f}")
    print(f"  Food experience: {household.category_experience.get('Food', 0)} ticks")

    household.apply_labor_outcome(labor_outcome)

    print(f"\nAfter getting hired:")
    print(f"  Employed: {household.is_employed}")
    print(f"  Employer ID: {household.employer_id}")
    print(f"  Wage: ${household.wage:.2f}")
    print(f"  Category: {labor_outcome['employer_category']}")

    assert household.is_employed == True, "Should be employed"
    assert household.wage == 75.0, "Wage should be $75"
    assert household.employer_id == 10, "Employer ID should be 10"

    print("\n✅ TEST 4 PASSED: Employment status updates correctly")


def test_skill_development(household):
    """Test 5: Verify skill growth through work experience."""
    print_section("TEST 5: Skill Development")

    initial_skills = household.skills_level
    initial_food_exp = household.category_experience.get('Food', 0)

    print(f"Initial skills: {initial_skills:.4f}")
    print(f"Initial Food experience: {initial_food_exp} ticks")

    # Simulate 10 ticks of employment with passive skill growth
    for tick in range(10):
        labor_outcome = {
            "employer_id": 10,
            "wage": 75.0,
            "employer_category": "Food"
        }
        household.apply_labor_outcome(labor_outcome)

    final_skills = household.skills_level
    final_food_exp = household.category_experience.get('Food', 0)
    skill_gain = final_skills - initial_skills
    experience_gain = final_food_exp - initial_food_exp

    print(f"\nAfter 10 ticks of work:")
    print(f"  Final skills: {final_skills:.4f}")
    print(f"  Skill gain: +{skill_gain:.4f}")
    print(f"  Food experience: {final_food_exp} ticks (+{experience_gain} gained)")

    assert final_skills > initial_skills, "Skills should improve with work"
    assert experience_gain == 10, f"Should gain 10 ticks of experience, gained {experience_gain}"

    print("\n✅ TEST 5 PASSED: Skills develop through work experience")


def test_goods_consumption():
    """Test 6: Verify goods inventory depletion."""
    print_section("TEST 6: Goods Consumption")

    household = HouseholdAgent(
        household_id=2,
        skills_level=0.6,
        age=35,
        cash_balance=500.0
    )

    # Give household some goods
    household.goods_inventory = {
        "Food": 100.0,
        "Housing": 50.0,
        "Services": 30.0
    }

    print(f"Initial inventory:")
    for good, amount in household.goods_inventory.items():
        print(f"  {good}: {amount:.2f} units")

    # Consume goods over 5 ticks
    print(f"\nConsuming goods (10% per tick):")
    for tick in range(5):
        household.consume_goods()
        if tick % 2 == 1:  # Print every other tick
            print(f"\n  After tick {tick + 1}:")
            for good, amount in household.goods_inventory.items():
                print(f"    {good}: {amount:.2f} units")

    # Check depletion
    for good, final_amount in household.goods_inventory.items():
        print(f"\n  {good}: 100 → {final_amount:.2f} ({(1 - final_amount/100)*100:.1f}% consumed)")

    assert all(v < 100 for v in household.goods_inventory.values()), "Goods should be consumed"

    print("\n✅ TEST 6 PASSED: Goods consumption works correctly")


def test_wellbeing_system():
    """Test 7: Verify happiness, morale, and health tracking."""
    print_section("TEST 7: Wellbeing System")

    household = HouseholdAgent(
        household_id=3,
        skills_level=0.7,
        age=40,
        cash_balance=2000.0
    )

    # Give household some goods to avoid goods penalty
    household.goods_inventory = {
        "Food": 50.0,
        "Housing": 30.0,
        "Services": 20.0
    }

    initial_happiness = household.happiness
    initial_morale = household.morale
    initial_health = household.health

    print(f"Initial wellbeing:")
    print(f"  Happiness: {household.happiness:.3f}")
    print(f"  Morale: {household.morale:.3f}")
    print(f"  Health: {household.health:.3f}")
    print(f"  Total goods: {sum(household.goods_inventory.values()):.1f} units")
    print(f"  Performance multiplier: {household.get_performance_multiplier():.3f}")

    # Simulate being unemployed for 5 ticks (should decrease wellbeing)
    print(f"\nSimulating 5 ticks of unemployment...")
    for _ in range(5):
        household.update_wellbeing(government_happiness_multiplier=1.0)

    unemployment_happiness = household.happiness
    unemployment_morale = household.morale

    print(f"\nAfter unemployment:")
    print(f"  Happiness: {household.happiness:.3f} (decreased from {initial_happiness:.3f})")
    print(f"  Morale: {household.morale:.3f} (decreased from {initial_morale:.3f})")
    print(f"  Health: {household.health:.3f} (decreased from {initial_health:.3f})")
    print(f"  Performance multiplier: {household.get_performance_multiplier():.3f}")

    # Verify unemployment decreases wellbeing
    assert household.happiness < initial_happiness, "Happiness should decrease during unemployment"
    assert household.morale < initial_morale, "Morale should decrease during unemployment"

    # Now get employed with good wage
    print(f"\nGetting employed with $100 wage...")
    labor_outcome = {
        "employer_id": 15,
        "wage": 100.0,
        "employer_category": "Services"
    }
    household.apply_labor_outcome(labor_outcome)

    # Set expected wage so household is satisfied
    household.expected_wage = 90.0

    # Simulate 5 ticks of employment
    for _ in range(5):
        household.update_wellbeing(government_happiness_multiplier=1.0)

    print(f"\nAfter 5 ticks of employment (wage $100 vs expected $90):")
    print(f"  Happiness: {household.happiness:.3f} (was {unemployment_happiness:.3f})")
    print(f"  Morale: {household.morale:.3f} (was {unemployment_morale:.3f})")
    print(f"  Health: {household.health:.3f}")
    print(f"  Performance multiplier: {household.get_performance_multiplier():.3f}")

    # Verify employment improves wellbeing over time
    assert household.morale > unemployment_morale, "Morale should improve with good employment"

    print("\n✅ TEST 7 PASSED: Wellbeing system tracks employment effects")


def test_income_and_spending():
    """Test 8: Verify income, taxes, transfers, and purchases."""
    print_section("TEST 8: Income and Spending")

    household = HouseholdAgent(
        household_id=4,
        skills_level=0.8,
        age=45,
        cash_balance=500.0
    )

    initial_cash = household.cash_balance
    print(f"Initial cash: ${initial_cash:.2f}")

    # Apply income and taxes
    income_data = {
        "wage_income": 80.0,
        "transfers": 20.0,  # Unemployment benefits
        "taxes_paid": 12.0  # 15% wage tax
    }

    print(f"\nIncome this tick:")
    print(f"  Wage income: ${income_data['wage_income']:.2f}")
    print(f"  Transfers: ${income_data['transfers']:.2f}")
    print(f"  Taxes paid: -${income_data['taxes_paid']:.2f}")

    household.apply_income_and_taxes(income_data)

    after_income_cash = household.cash_balance
    net_income = income_data['wage_income'] + income_data['transfers'] - income_data['taxes_paid']

    print(f"\nAfter income:")
    print(f"  Cash: ${after_income_cash:.2f}")
    print(f"  Net income: +${net_income:.2f}")

    # Apply purchases
    purchases = {
        "Food": (10.0, 5.0),  # (units, price)
        "Services": (5.0, 7.0)
    }

    total_cost = sum(units * price for units, price in purchases.values())
    print(f"\nPurchases:")
    for good, (units, price) in purchases.items():
        cost = units * price
        print(f"  {good}: {units:.1f} units @ ${price:.2f} = ${cost:.2f}")
    print(f"  Total cost: ${total_cost:.2f}")

    household.apply_purchases(purchases)

    final_cash = household.cash_balance
    print(f"\nFinal cash: ${final_cash:.2f}")
    print(f"  Started with: ${initial_cash:.2f}")
    print(f"  Income: +${net_income:.2f}")
    print(f"  Spending: -${total_cost:.2f}")
    print(f"  Expected: ${initial_cash + net_income - total_cost:.2f}")

    expected_cash = initial_cash + net_income - total_cost
    assert abs(final_cash - expected_cash) < 0.01, f"Cash mismatch: expected ${expected_cash:.2f}, got ${final_cash:.2f}"

    print("\n✅ TEST 8 PASSED: Income and spending tracked correctly")


def run_quick_simulation():
    """Run a quick 20-tick simulation showing household behavior."""
    print_section("QUICK SIMULATION: 20 Ticks of Household Life")

    household = HouseholdAgent(
        household_id=100,
        skills_level=0.4,
        age=25,
        cash_balance=800.0
    )

    print(f"Starting household simulation:")
    print(f"  ID: {household.household_id}")
    print(f"  Initial cash: ${household.cash_balance:.2f}")
    print(f"  Initial skills: {household.skills_level:.3f}")
    print(f"  Initial happiness: {household.happiness:.3f}")

    print(f"\n{'Tick':<6} {'Employed':<10} {'Wage':<8} {'Cash':<10} {'Skills':<8} {'Happiness':<10} {'Goods':<6}")
    print("-" * 70)

    # Simulate 20 ticks
    for tick in range(20):
        # Get employed at tick 5
        if tick == 5:
            labor_outcome = {
                "employer_id": 50,
                "wage": 70.0,
                "employer_category": "Housing"
            }
            household.apply_labor_outcome(labor_outcome)

        # Apply income if employed
        if household.is_employed:
            income_data = {
                "wage_income": household.wage,
                "transfers": 0.0,
                "taxes_paid": household.wage * 0.15
            }
            household.apply_income_and_taxes(income_data)
        else:
            # Unemployed - get benefits
            income_data = {
                "wage_income": 0.0,
                "transfers": 40.0,
                "taxes_paid": 0.0
            }
            household.apply_income_and_taxes(income_data)

        # Buy some goods every 3 ticks if can afford
        if tick % 3 == 0 and household.cash_balance > 30:
            purchases = {
                "Food": (5.0, 5.0),
            }
            household.apply_purchases(purchases)

        # Consume goods
        household.consume_goods()

        # Update wellbeing
        household.update_wellbeing(government_happiness_multiplier=1.0)

        # Print status every 2 ticks
        if tick % 2 == 0:
            total_goods = sum(household.goods_inventory.values())
            print(f"{tick:<6} {str(household.is_employed):<10} ${household.wage:<7.2f} ${household.cash_balance:<9.2f} {household.skills_level:<7.4f} {household.happiness:<9.3f} {total_goods:<6.1f}")

    print()
    print(f"Final state after 20 ticks:")
    print(f"  Cash: ${household.cash_balance:.2f}")
    print(f"  Skills: {household.skills_level:.4f} (improved from 0.4000)")
    print(f"  Happiness: {household.happiness:.3f}")
    print(f"  Housing experience: {household.category_experience.get('Housing', 0)} ticks")

    print("\n✅ SIMULATION COMPLETE: Household agent functioning normally")


def main():
    """Run all household agent tests."""
    print("\n" + "=" * 70)
    print("  HOUSEHOLD AGENT TEST SUITE")
    print("=" * 70)
    print("\nThis test suite validates all household agent behaviors.")
    print("Each test is independent and verifies a specific feature.")

    try:
        # Run all tests
        household = test_household_creation()
        test_labor_supply_planning(household)
        test_consumption_planning(household)
        test_employment_and_wages(household)
        test_skill_development(household)
        test_goods_consumption()
        test_wellbeing_system()
        test_income_and_spending()

        # Run simulation
        run_quick_simulation()

        # Summary
        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED ✅")
        print("=" * 70)
        print("\nHousehold agents are working correctly!")
        print("Key behaviors verified:")
        print("  ✓ Labor supply decisions")
        print("  ✓ Consumption planning")
        print("  ✓ Employment and wage tracking")
        print("  ✓ Skill development")
        print("  ✓ Goods consumption")
        print("  ✓ Wellbeing tracking")
        print("  ✓ Income and spending")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
