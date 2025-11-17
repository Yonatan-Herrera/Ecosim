"""
Unit tests for GovernmentAgent

Tests cover:
- Tax collection on wages and profits
- Transfer planning for eligible households
- Transfer execution
- Invariant enforcement
"""

import pytest
from agents import GovernmentAgent


class TestGovernmentAgent:
    """Test suite for GovernmentAgent functionality"""

    def test_collect_taxes_basic(self):
        """Tax collection should match rates and increase government cash"""
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=100.0,
            cash=1000.0
        )

        total_wages = 5000.0
        total_profits = 2000.0

        initial_cash = gov.cash
        collected = gov.collect_taxes(total_wages, total_profits)

        # Check tax calculation
        expected_wage_tax = 0.2 * 5000.0  # 1000
        expected_profit_tax = 0.3 * 2000.0  # 600
        expected_total = expected_wage_tax + expected_profit_tax  # 1600

        assert abs(collected - expected_total) < 1e-9

        # Check cash increased correctly
        assert abs(gov.cash - (initial_cash + expected_total)) < 1e-9

    def test_collect_taxes_ignores_negative_inputs(self):
        """Negative wages or profits should be treated as zero"""
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=100.0,
            cash=1000.0
        )

        # Test with negative wages
        collected1 = gov.collect_taxes(total_wages=-1000.0, total_profits=1000.0)
        expected1 = 0.0 + (0.3 * 1000.0)  # Only profit tax
        assert abs(collected1 - expected1) < 1e-9

        # Reset and test with negative profits
        gov.cash = 1000.0
        collected2 = gov.collect_taxes(total_wages=1000.0, total_profits=-500.0)
        expected2 = (0.2 * 1000.0) + 0.0  # Only wage tax
        assert abs(collected2 - expected2) < 1e-9

        # Test with both negative
        gov.cash = 1000.0
        collected3 = gov.collect_taxes(total_wages=-1000.0, total_profits=-500.0)
        assert abs(collected3 - 0.0) < 1e-9

    def test_plan_transfers_no_eligible_households(self):
        """When all households are above threshold, no transfers planned"""
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=1000.0,
            min_household_cash=100.0
        )

        # All households have sufficient cash
        households = [
            {"household_id": "h1", "cash": 150.0},
            {"household_id": "h2", "cash": 200.0},
            {"household_id": "h3", "cash": 100.0}  # Exactly at threshold
        ]

        transfers = gov.plan_transfers(households)

        assert transfers == []

    def test_plan_transfers_respects_budget(self):
        """Total planned transfers should equal transfer_budget"""
        transfer_budget = 1000.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=transfer_budget,
            min_household_cash=100.0
        )

        households = [
            {"household_id": "h1", "cash": 50.0},   # Gap: 50
            {"household_id": "h2", "cash": 80.0},   # Gap: 20
            {"household_id": "h3", "cash": 200.0}   # No gap
        ]

        transfers = gov.plan_transfers(households)

        # Calculate total transfer amount
        total_transfer = sum(t["amount"] for t in transfers)

        # Should equal transfer_budget (within floating point precision)
        assert abs(total_transfer - transfer_budget) < 1e-9

    def test_apply_transfers_decreases_cash(self):
        """Applying transfers should decrease government cash by exact amount"""
        initial_cash = 5000.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=1000.0,
            cash=initial_cash
        )

        transfers = [
            {"household_id": "h1", "amount": 100.0},
            {"household_id": "h2", "amount": 150.0},
            {"household_id": "h3", "amount": 250.0}
        ]

        total_transferred = gov.apply_transfers(transfers)

        # Check return value
        expected_total = 100.0 + 150.0 + 250.0
        assert abs(total_transferred - expected_total) < 1e-9

        # Check cash decreased correctly
        expected_cash = initial_cash - expected_total
        assert abs(gov.cash - expected_cash) < 1e-9

    def test_plan_transfers_proportional_allocation(self):
        """Transfers should be allocated proportionally to household gaps"""
        transfer_budget = 100.0
        min_cash = 100.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=transfer_budget,
            min_household_cash=min_cash
        )

        households = [
            {"household_id": "h1", "cash": 50.0},  # Gap: 50
            {"household_id": "h2", "cash": 50.0}   # Gap: 50
        ]

        transfers = gov.plan_transfers(households)

        # Total gap = 100, each household has gap of 50
        # Each should get 50% of budget = 50
        assert len(transfers) == 2
        for transfer in transfers:
            assert abs(transfer["amount"] - 50.0) < 1e-9

    def test_plan_transfers_unequal_gaps(self):
        """Households with larger gaps should receive proportionally more"""
        transfer_budget = 100.0
        min_cash = 100.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=transfer_budget,
            min_household_cash=min_cash
        )

        households = [
            {"household_id": "h1", "cash": 25.0},  # Gap: 75
            {"household_id": "h2", "cash": 75.0}   # Gap: 25
        ]

        transfers = gov.plan_transfers(households)

        # Total gap = 100
        # h1 should get 75% of budget = 75
        # h2 should get 25% of budget = 25
        transfer_dict = {t["household_id"]: t["amount"] for t in transfers}

        assert abs(transfer_dict["h1"] - 75.0) < 1e-9
        assert abs(transfer_dict["h2"] - 25.0) < 1e-9

    def test_tax_rate_wage_validates_range(self):
        """tax_rate_wage must be in [0, 1]"""
        with pytest.raises(ValueError, match="tax_rate_wage"):
            GovernmentAgent(
                tax_rate_wage=1.5,
                tax_rate_profit=0.3,
                transfer_budget=100.0
            )

        with pytest.raises(ValueError, match="tax_rate_wage"):
            GovernmentAgent(
                tax_rate_wage=-0.1,
                tax_rate_profit=0.3,
                transfer_budget=100.0
            )

    def test_tax_rate_profit_validates_range(self):
        """tax_rate_profit must be in [0, 1]"""
        with pytest.raises(ValueError, match="tax_rate_profit"):
            GovernmentAgent(
                tax_rate_wage=0.2,
                tax_rate_profit=1.5,
                transfer_budget=100.0
            )

        with pytest.raises(ValueError, match="tax_rate_profit"):
            GovernmentAgent(
                tax_rate_wage=0.2,
                tax_rate_profit=-0.1,
                transfer_budget=100.0
            )

    def test_transfer_budget_must_be_non_negative(self):
        """transfer_budget cannot be negative"""
        with pytest.raises(ValueError, match="transfer_budget"):
            GovernmentAgent(
                tax_rate_wage=0.2,
                tax_rate_profit=0.3,
                transfer_budget=-100.0
            )

    def test_min_household_cash_must_be_non_negative(self):
        """min_household_cash cannot be negative"""
        with pytest.raises(ValueError, match="min_household_cash"):
            GovernmentAgent(
                tax_rate_wage=0.2,
                tax_rate_profit=0.3,
                transfer_budget=100.0,
                min_household_cash=-50.0
            )

    def test_government_cash_can_go_negative_deficit(self):
        """Government can run a deficit (negative cash balance)"""
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=1000.0,
            cash=100.0  # Small initial balance
        )

        # Apply large transfers that exceed cash
        transfers = [{"household_id": "h1", "amount": 500.0}]
        gov.apply_transfers(transfers)

        # Cash should go negative (deficit spending allowed)
        assert gov.cash < 0
        assert abs(gov.cash - (100.0 - 500.0)) < 1e-9

    def test_zero_tax_rates_collect_nothing(self):
        """With zero tax rates, no taxes should be collected"""
        gov = GovernmentAgent(
            tax_rate_wage=0.0,
            tax_rate_profit=0.0,
            transfer_budget=100.0,
            cash=1000.0
        )

        collected = gov.collect_taxes(total_wages=5000.0, total_profits=2000.0)

        assert collected == 0.0
        assert gov.cash == 1000.0  # Unchanged

    def test_empty_household_list_returns_no_transfers(self):
        """Empty household list should return empty transfer list"""
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=1000.0
        )

        transfers = gov.plan_transfers([])

        assert transfers == []

    def test_empty_transfer_list_changes_nothing(self):
        """Applying empty transfer list should not change cash"""
        initial_cash = 1000.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=100.0,
            cash=initial_cash
        )

        total = gov.apply_transfers([])

        assert total == 0.0
        assert gov.cash == initial_cash

    def test_single_eligible_household_receives_full_budget(self):
        """If only one household is eligible, it gets the entire budget"""
        transfer_budget = 500.0
        gov = GovernmentAgent(
            tax_rate_wage=0.2,
            tax_rate_profit=0.3,
            transfer_budget=transfer_budget,
            min_household_cash=100.0
        )

        households = [
            {"household_id": "h1", "cash": 50.0},
            {"household_id": "h2", "cash": 200.0}
        ]

        transfers = gov.plan_transfers(households)

        assert len(transfers) == 1
        assert transfers[0]["household_id"] == "h1"
        assert abs(transfers[0]["amount"] - transfer_budget) < 1e-9
