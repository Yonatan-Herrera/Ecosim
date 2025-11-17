"""
Unit tests for FirmAgent

Tests cover:
- Price adjustment based on sell-through
- Production planning under various constraints
- Production and sales execution
- Invariant enforcement
"""

import pytest
from agents import FirmAgent


class TestFirmAgent:
    """Test suite for FirmAgent functionality"""

    def test_price_moves_toward_target_with_stickiness(self):
        """Price should move partway toward target, not all the way (due to stickiness)"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=20.0,  # Current price
            markup=0.5,  # Target would be 10 * 1.5 = 15
            price_stickiness=0.3,
            last_sell_through=0.5,
            target_sell_through=0.5  # No adjustment, just move toward markup
        )

        initial_price = firm.price
        # Target is unit_cost * (1 + markup) = 10 * 1.5 = 15
        target_price = 15.0

        new_price = firm.plan_price()

        # Price should move toward target but not reach it (due to stickiness)
        # new_price = price + stickiness * (target - price)
        # new_price = 20 + 0.3 * (15 - 20) = 20 - 1.5 = 18.5
        expected_price = initial_price + firm.price_stickiness * (target_price - initial_price)

        assert abs(new_price - expected_price) < 1e-9
        assert new_price != target_price  # Should not reach target in one step
        assert new_price < initial_price  # Should move toward lower target

    def test_high_sell_through_increases_target_price(self):
        """When sell-through exceeds target, price should increase"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0,
            markup=0.5,
            price_stickiness=0.5,
            last_sell_through=0.95,  # High sell-through
            target_sell_through=0.8,
            price_adjust_step=1.0
        )

        initial_price = firm.price
        new_price = firm.plan_price()

        # Price should increase because demand was strong
        assert new_price > initial_price

    def test_low_sell_through_decreases_target_price(self):
        """When sell-through is below target, price should decrease"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0,
            markup=0.5,
            price_stickiness=0.5,
            last_sell_through=0.3,  # Low sell-through
            target_sell_through=0.8,
            price_adjust_step=1.0
        )

        initial_price = firm.price
        new_price = firm.plan_price()

        # Price should decrease because demand was weak
        assert new_price < initial_price

    def test_production_limited_by_capacity(self):
        """With huge expected sales but low capacity, production should be capped"""
        firm = FirmAgent(
            id="f1",
            cash=10000.0,  # Plenty of cash
            inventory=10.0,
            capacity=50.0,  # Limited capacity
            unit_cost=5.0,
            price=15.0,
            inventory_target_multiplier=2.0
        )

        expected_sales = 1000.0  # Huge demand
        production = firm.plan_production(expected_sales)

        # Production should be limited by capacity
        assert production <= firm.capacity
        assert production == firm.capacity  # Should use full capacity

    def test_production_limited_by_cash(self):
        """With huge expected sales but little cash, production should be limited"""
        firm = FirmAgent(
            id="f1",
            cash=100.0,  # Limited cash
            inventory=10.0,
            capacity=1000.0,  # Huge capacity
            unit_cost=10.0,
            price=15.0,
            inventory_target_multiplier=2.0
        )

        expected_sales = 500.0  # Huge demand
        production = firm.plan_production(expected_sales)

        # Production should be limited by cash
        max_affordable = firm.cash // firm.unit_cost  # 100 // 10 = 10
        assert production <= max_affordable
        assert production == max_affordable

    def test_apply_production_does_not_allow_negative_cash(self):
        """Producing more than cash allows should raise ValueError"""
        firm = FirmAgent(
            id="f1",
            cash=50.0,
            inventory=10.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        # Try to produce 10 units at cost 10 each = 100, but only have 50
        with pytest.raises(ValueError, match="Insufficient cash"):
            firm.apply_production(10.0)

    def test_apply_sales_results_updates_cash_inventory_and_sell_through(self):
        """Sales should correctly update cash, inventory, and sell-through"""
        initial_inventory = 100.0
        firm = FirmAgent(
            id="f1",
            cash=500.0,
            inventory=initial_inventory,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        sold_qty = 60.0
        unit_price = 15.0
        available_at_start = initial_inventory

        firm.apply_sales_results(sold_qty, unit_price, available_at_start)

        # Check inventory decreased
        assert firm.inventory == initial_inventory - sold_qty

        # Check cash increased
        assert firm.cash == 500.0 + (sold_qty * unit_price)

        # Check sell-through calculated correctly
        expected_sell_through = sold_qty / available_at_start
        assert abs(firm.last_sell_through - expected_sell_through) < 1e-9

    def test_price_never_falls_below_unit_cost(self):
        """Even with weak demand, price should not fall below unit cost"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=10.5,  # Just above cost
            markup=0.0,  # No markup
            price_stickiness=1.0,  # Full adjustment
            last_sell_through=0.1,  # Very weak demand
            target_sell_through=0.8,
            price_adjust_step=5.0  # Large adjustment
        )

        new_price = firm.plan_price()

        # Price should not go below unit cost
        assert new_price >= firm.unit_cost

    def test_apply_production_increases_inventory_decreases_cash(self):
        """Production should increase inventory and decrease cash"""
        initial_cash = 1000.0
        initial_inventory = 50.0
        firm = FirmAgent(
            id="f1",
            cash=initial_cash,
            inventory=initial_inventory,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        production_qty = 20.0
        firm.apply_production(production_qty)

        # Check inventory increased
        assert firm.inventory == initial_inventory + production_qty

        # Check cash decreased
        expected_cash = initial_cash - (production_qty * firm.unit_cost)
        assert abs(firm.cash - expected_cash) < 1e-9

    def test_negative_production_raises_error(self):
        """Attempting to produce negative quantity should raise ValueError"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        with pytest.raises(ValueError, match="non-negative"):
            firm.apply_production(-10.0)

    def test_selling_more_than_inventory_raises_error(self):
        """Cannot sell more than available inventory"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        with pytest.raises(ValueError, match="more than inventory"):
            firm.apply_sales_results(
                sold_qty=100.0,  # More than inventory
                unit_price=15.0,
                available_qty_at_sale_start=50.0
            )

    def test_negative_sold_qty_raises_error(self):
        """Negative sold quantity should raise ValueError"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        with pytest.raises(ValueError, match="non-negative"):
            firm.apply_sales_results(
                sold_qty=-10.0,
                unit_price=15.0,
                available_qty_at_sale_start=50.0
            )

    def test_initial_price_below_cost_raises_error(self):
        """Cannot initialize firm with price below unit cost"""
        with pytest.raises(ValueError, match="below unit_cost"):
            FirmAgent(
                id="f1",
                cash=1000.0,
                inventory=50.0,
                capacity=100.0,
                unit_cost=10.0,
                price=5.0  # Below cost
            )

    def test_negative_cash_raises_error(self):
        """Cannot initialize firm with negative cash"""
        with pytest.raises(ValueError, match="cash"):
            FirmAgent(
                id="f1",
                cash=-100.0,
                inventory=50.0,
                capacity=100.0,
                unit_cost=10.0,
                price=15.0
            )

    def test_negative_inventory_raises_error(self):
        """Cannot initialize firm with negative inventory"""
        with pytest.raises(ValueError, match="inventory"):
            FirmAgent(
                id="f1",
                cash=1000.0,
                inventory=-50.0,
                capacity=100.0,
                unit_cost=10.0,
                price=15.0
            )

    def test_invalid_price_stickiness_raises_error(self):
        """price_stickiness must be in (0, 1]"""
        with pytest.raises(ValueError, match="price_stickiness"):
            FirmAgent(
                id="f1",
                cash=1000.0,
                inventory=50.0,
                capacity=100.0,
                unit_cost=10.0,
                price=15.0,
                price_stickiness=0.0  # Invalid
            )

        with pytest.raises(ValueError, match="price_stickiness"):
            FirmAgent(
                id="f1",
                cash=1000.0,
                inventory=50.0,
                capacity=100.0,
                unit_cost=10.0,
                price=15.0,
                price_stickiness=1.5  # Invalid
            )

    def test_zero_expected_sales_produces_nothing(self):
        """With zero expected sales, should not plan any production"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=50.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0,
            inventory_target_multiplier=1.5
        )

        production = firm.plan_production(expected_sales=0.0)

        # Should produce nothing (inventory already exceeds target)
        assert production == 0.0

    def test_sell_through_clamped_to_valid_range(self):
        """Sell-through should be clamped to [0, 1]"""
        firm = FirmAgent(
            id="f1",
            cash=1000.0,
            inventory=100.0,
            capacity=100.0,
            unit_cost=10.0,
            price=15.0
        )

        # Normal case
        firm.apply_sales_results(
            sold_qty=50.0,
            unit_price=15.0,
            available_qty_at_sale_start=100.0
        )
        assert 0.0 <= firm.last_sell_through <= 1.0

    def test_production_with_sufficient_inventory_produces_nothing(self):
        """If inventory already exceeds target, should not produce"""
        firm = FirmAgent(
            id="f1",
            cash=10000.0,
            inventory=1000.0,  # Large existing inventory
            capacity=100.0,
            unit_cost=10.0,
            price=15.0,
            inventory_target_multiplier=1.5
        )

        expected_sales = 100.0
        # Target inventory = 1.5 * 100 = 150
        # Current inventory = 1000, which exceeds target
        production = firm.plan_production(expected_sales)

        assert production == 0.0
