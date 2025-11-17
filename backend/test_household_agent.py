"""
Unit tests for HouseholdAgent

Tests cover:
- Demand planning behavior under various conditions
- Price expectation updates
- Income receipt and trade execution
- Invariant enforcement
"""

import pytest
from agents import HouseholdAgent


class TestHouseholdAgent:
    """Test suite for HouseholdAgent functionality"""

    def test_no_cash_no_demand(self):
        """With zero cash, household should plan no demand"""
        household = HouseholdAgent(
            id="h1",
            cash=0.0,
            budget_share=0.8,
            good_weights={"food": 0.5, "clothing": 0.5},
            expected_prices={"food": 10.0, "clothing": 20.0}
        )

        orders = household.plan_demand()

        # Should return empty list or all zero quantities
        assert len(orders) == 0 or all(order["quantity"] == 0 for order in orders)

    def test_prices_double_demand_falls(self):
        """When prices double, total planned quantity should strictly decrease"""
        # Setup household with some cash and preferences
        household = HouseholdAgent(
            id="h1",
            cash=1000.0,
            budget_share=0.8,
            good_weights={"food": 0.6, "clothing": 0.4},
            expected_prices={"food": 10.0, "clothing": 20.0}
        )

        # Plan demand with initial prices
        orders_before = household.plan_demand()
        total_qty_before = sum(order["quantity"] for order in orders_before)

        # Double all expected prices
        household.expected_prices = {"food": 20.0, "clothing": 40.0}

        # Plan demand with doubled prices
        orders_after = household.plan_demand()
        total_qty_after = sum(order["quantity"] for order in orders_after)

        # Total quantity must strictly decrease
        assert total_qty_after < total_qty_before

    def test_expectations_update_toward_observed_price(self):
        """Price expectations should update via exponential smoothing"""
        alpha = 0.3
        household = HouseholdAgent(
            id="h1",
            cash=1000.0,
            expectation_alpha=alpha,
            expected_prices={"food": 100.0}
        )

        old_expected = household.expected_prices["food"]
        observed_price = 80.0

        # Observe new market price
        household.observe_market({"food": observed_price})

        new_expected = household.expected_prices["food"]
        expected_value = alpha * observed_price + (1 - alpha) * old_expected

        # Check exponential smoothing formula
        assert abs(new_expected - expected_value) < 1e-9

    def test_cannot_spend_more_than_cash(self):
        """Total planned spending must never exceed available cash"""
        household = HouseholdAgent(
            id="h1",
            cash=100.0,
            budget_share=1.0,  # Try to spend all cash
            good_weights={"food": 0.5, "clothing": 0.5},
            expected_prices={"food": 10.0, "clothing": 20.0}
        )

        orders = household.plan_demand()

        # Calculate total planned spending
        total_spending = 0.0
        for order in orders:
            good_id = order["good_id"]
            quantity = order["quantity"]
            expected_price = household.expected_prices[good_id]
            total_spending += quantity * expected_price

        # Must not exceed cash
        assert total_spending <= household.cash + 1e-9  # Small tolerance for floating point

    def test_apply_trade_results_updates_cash_and_inventory(self):
        """Executing trades should correctly update cash and inventory"""
        initial_cash = 1000.0
        household = HouseholdAgent(
            id="h1",
            cash=initial_cash,
            inventory={"food": 5.0}
        )

        # Execute some trades
        executed_trades = [
            {"good_id": "food", "quantity": 10.0, "unit_price": 15.0},
            {"good_id": "clothing", "quantity": 3.0, "unit_price": 25.0}
        ]

        household.apply_trade_results(executed_trades)

        # Check cash decreased correctly
        expected_cash = initial_cash - (10.0 * 15.0) - (3.0 * 25.0)
        assert abs(household.cash - expected_cash) < 1e-9

        # Check inventory updated correctly
        assert abs(household.inventory["food"] - 15.0) < 1e-9  # 5 + 10
        assert abs(household.inventory["clothing"] - 3.0) < 1e-9

        # Check non-negativity
        assert household.cash >= 0
        assert all(qty >= 0 for qty in household.inventory.values())

    def test_receive_income_increases_cash(self):
        """Receiving income should increase cash balance"""
        household = HouseholdAgent(id="h1", cash=100.0)

        household.receive_income(50.0)

        assert household.cash == 150.0

    def test_receive_income_rejects_negative(self):
        """Receiving negative income should raise ValueError"""
        household = HouseholdAgent(id="h1", cash=100.0)

        with pytest.raises(ValueError, match="non-negative"):
            household.receive_income(-50.0)

    def test_insufficient_cash_for_trade_raises_error(self):
        """Attempting trade with insufficient cash should raise ValueError"""
        household = HouseholdAgent(id="h1", cash=50.0)

        executed_trades = [
            {"good_id": "food", "quantity": 10.0, "unit_price": 10.0}  # Costs 100
        ]

        with pytest.raises(ValueError, match="Insufficient cash"):
            household.apply_trade_results(executed_trades)

    def test_observe_market_initializes_new_good_price(self):
        """Observing a new good should initialize its expected price"""
        household = HouseholdAgent(id="h1", cash=100.0)

        household.observe_market({"food": 25.0})

        assert "food" in household.expected_prices
        assert household.expected_prices["food"] == 25.0

    def test_budget_share_validates_range(self):
        """budget_share must be in [0, 1]"""
        with pytest.raises(ValueError, match="budget_share"):
            HouseholdAgent(id="h1", cash=100.0, budget_share=1.5)

        with pytest.raises(ValueError, match="budget_share"):
            HouseholdAgent(id="h1", cash=100.0, budget_share=-0.1)

    def test_expectation_alpha_validates_range(self):
        """expectation_alpha must be in (0, 1]"""
        with pytest.raises(ValueError, match="expectation_alpha"):
            HouseholdAgent(id="h1", cash=100.0, expectation_alpha=0.0)

        with pytest.raises(ValueError, match="expectation_alpha"):
            HouseholdAgent(id="h1", cash=100.0, expectation_alpha=1.5)

    def test_initial_cash_cannot_be_negative(self):
        """Initial cash must be non-negative"""
        with pytest.raises(ValueError, match="cash"):
            HouseholdAgent(id="h1", cash=-100.0)

    def test_zero_expected_price_produces_no_demand(self):
        """Good with zero or negative expected price should not be demanded"""
        household = HouseholdAgent(
            id="h1",
            cash=1000.0,
            budget_share=0.8,
            good_weights={"food": 0.5, "broken": 0.5},
            expected_prices={"food": 10.0, "broken": 0.0}
        )

        orders = household.plan_demand()

        # Should only have order for "food", not "broken"
        good_ids = [order["good_id"] for order in orders]
        assert "food" in good_ids
        assert "broken" not in good_ids

    def test_zero_weight_produces_no_demand(self):
        """Good with zero weight should not be demanded"""
        household = HouseholdAgent(
            id="h1",
            cash=1000.0,
            budget_share=0.8,
            good_weights={"food": 1.0, "unwanted": 0.0},
            expected_prices={"food": 10.0, "unwanted": 10.0}
        )

        orders = household.plan_demand()

        # Should only have order for "food", not "unwanted"
        good_ids = [order["good_id"] for order in orders]
        assert "food" in good_ids
        assert "unwanted" not in good_ids
