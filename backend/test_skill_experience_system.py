"""
Test skill-based and experience-based wage and productivity system
"""

from agents import HouseholdAgent, FirmAgent, GovernmentAgent
from economy import Economy


def test_skill_based_wages():
    """Test that workers with higher skills get higher wages"""
    # Create two households with different skill levels
    skilled_household = HouseholdAgent(
        household_id=1,
        skills_level=1.0,  # Maximum skill
        age=30,
        cash_balance=1000.0,
        category_weights={"Food": 1.0}
    )

    unskilled_household = HouseholdAgent(
        household_id=2,
        skills_level=0.0,  # Minimum skill
        age=30,
        cash_balance=1000.0,
        category_weights={"Food": 1.0}
    )

    # Create a firm that needs workers
    firm = FirmAgent(
        firm_id=1,
        good_name="BasicFood",
        cash_balance=10000.0,
        inventory_units=0.0,
        good_category="Food",
        wage_offer=50.0
    )

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.1,
        profit_tax_rate=0.1,
        transfer_budget=100.0
    )

    # Create economy
    economy = Economy(
        households=[skilled_household, unskilled_household],
        firms=[firm],
        government=gov
    )

    # Run one step to match labor
    economy.step()

    # Check that both households got jobs (firm should hire both)
    # The skilled worker should have a higher wage than the unskilled worker
    print(f"Skilled worker wage: {skilled_household.wage}")
    print(f"Unskilled worker wage: {unskilled_household.wage}")

    # Skilled worker should earn 50% more (base wage 50 * 1.5 = 75)
    assert skilled_household.wage > unskilled_household.wage
    assert skilled_household.wage == 50.0 * 1.5  # 50% skill premium
    assert unskilled_household.wage == 50.0  # No premium

    print("✓ Skill-based wages working correctly")


def test_experience_accumulation():
    """Test that workers accumulate experience when employed"""
    # Create household
    household = HouseholdAgent(
        household_id=1,
        skills_level=0.5,
        age=30,
        cash_balance=1000.0,
        category_weights={"Food": 1.0}
    )

    # Manually set experience to simulate working for 1 year
    household.category_experience["Food"] = 52

    # Verify experience is tracked
    print(f"Experience in Food category: {household.category_experience.get('Food', 0)} ticks")
    assert household.category_experience.get("Food", 0) == 52

    # Create firm
    firm = FirmAgent(
        firm_id=1,
        good_name="BasicFood",
        cash_balance=10000.0,
        inventory_units=0.0,
        good_category="Food",
        wage_offer=50.0
    )

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.1,
        profit_tax_rate=0.1,
        transfer_budget=100.0
    )

    # Create economy
    economy = Economy(
        households=[household],
        firms=[firm],
        government=gov
    )

    # Run one step to hire the worker
    economy.step()

    # Check that wage includes skill and experience premiums
    # Skill premium: 0.5 * 0.5 = 0.25
    # Experience premium: (52/52) * 0.05 = 0.05
    # Total: 50.0 * (1.0 + 0.25 + 0.05) = 50.0 * 1.30 = 65.0
    expected_wage = 50.0 * 1.30
    print(f"Worker wage: {household.wage}")
    print(f"Expected wage: {expected_wage}")

    if household.is_employed:
        assert abs(household.wage - expected_wage) < 0.01
        # After one more tick, experience should increase
        initial_exp = household.category_experience.get("Food", 0)
        economy.step()
        if household.is_employed:  # Still employed
            new_exp = household.category_experience.get("Food", 0)
            print(f"Experience increased from {initial_exp} to {new_exp}")
            assert new_exp == initial_exp + 1

    print("✓ Experience accumulation working correctly")


def test_productivity_bonus():
    """Test that experienced workers produce more"""
    # Create two firms with workers at different experience levels
    experienced_household = HouseholdAgent(
        household_id=1,
        skills_level=0.5,
        age=30,
        cash_balance=1000.0,
        category_weights={"Food": 1.0},
        category_experience={"Food": 52 * 5}  # 5 years of experience
    )

    novice_household = HouseholdAgent(
        household_id=2,
        skills_level=0.5,
        age=25,
        cash_balance=1000.0,
        category_weights={"Housing": 1.0},
        category_experience={"Housing": 0}  # No experience
    )

    # Create two firms
    food_firm = FirmAgent(
        firm_id=1,
        good_name="BasicFood",
        cash_balance=10000.0,
        inventory_units=0.0,
        good_category="Food",
        wage_offer=50.0,
        productivity_per_worker=10.0
    )

    housing_firm = FirmAgent(
        firm_id=2,
        good_name="Housing",
        cash_balance=10000.0,
        inventory_units=0.0,
        good_category="Housing",
        wage_offer=50.0,
        productivity_per_worker=10.0
    )

    # Create government
    gov = GovernmentAgent(
        wage_tax_rate=0.1,
        profit_tax_rate=0.1,
        transfer_budget=100.0
    )

    # Create economy
    economy = Economy(
        households=[experienced_household, novice_household],
        firms=[food_firm, housing_firm],
        government=gov
    )

    # Run one step
    economy.step()

    # Check that experienced worker produces more than novice
    print(f"Food firm inventory (experienced worker): {food_firm.inventory_units}")
    print(f"Housing firm inventory (novice worker): {housing_firm.inventory_units}")

    # Experienced worker: skill bonus (0.5 * 0.25 = 0.125) + experience bonus (5 years * 0.05 = 0.25)
    # Total multiplier: 1.0 + 0.125 + 0.25 = 1.375
    # Production should be higher than base
    assert food_firm.inventory_units > housing_firm.inventory_units

    print("✓ Experience-based productivity working correctly")


if __name__ == "__main__":
    print("Testing skill-based and experience-based system...\n")

    test_skill_based_wages()
    print()

    test_experience_accumulation()
    print()

    test_productivity_bonus()
    print()

    print("All tests passed!")
