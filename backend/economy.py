"""
Economy Simulation Engine

This module implements the main simulation coordinator that orchestrates
households, firms, and government through deterministic tick-based cycles.

All behavior is deterministic - no randomness, I/O, or side effects.

Performance optimizations:
- Caches household/firm lookups for O(1) access
- Uses NumPy vectorization for labor and goods market operations
- Batch operations to minimize Python loop overhead
"""

from typing import Dict, List, Tuple, Optional

from config import CONFIG
import numpy as np
import math
from agents import HouseholdAgent, FirmAgent, GovernmentAgent, _get_good_category


class Economy:
    """
    Main simulation coordinator for the economic model.

    Orchestrates all agents through a strict plan/apply cycle with
    deterministic labor and goods market clearing.
    """

    def __init__(
        self,
        households: List[HouseholdAgent],
        firms: List[FirmAgent],
        government: GovernmentAgent,
        queued_firms: Optional[List[FirmAgent]] = None
    ):
        """
        Initialize the economy with pre-constructed agents.

        Args:
            households: List of household agents
            firms: List of firm agents
            government: Government agent instance
        """
        self.households = households
        self.firms = firms
        self.government = government
        self.queued_firms: List[FirmAgent] = queued_firms or []
        self.target_total_firms = 0
        self._refresh_target_total_firms()
        self.large_market = len(self.households) >= CONFIG.firms.large_market_household_threshold

        # Track simulation progression and warm-up period state
        self.current_tick = 0
        self.in_warmup = True
        self.post_warmup_cooldown = 0

        # Performance optimization: Cache lookups for O(1) access
        self.household_lookup: Dict[int, HouseholdAgent] = {h.household_id: h for h in households}
        self.firm_lookup: Dict[int, FirmAgent] = {f.firm_id: f for f in firms}

        # Cache wage percentiles to avoid repeated sorting
        self.cached_wage_percentiles: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # low, mid, high
        self.wage_percentile_cache_tick: int = -1

        # Initialize tracking dictionaries with defaults
        self.last_tick_sales_units: Dict[int, float] = {}
        self.last_tick_revenue: Dict[int, float] = {}
        self.last_tick_sell_through_rate: Dict[int, float] = {}
        self.last_tick_prices: Dict[str, float] = {}

        # Set initial defaults
        for firm in firms:
            self.last_tick_sales_units[firm.firm_id] = 0.0
            self.last_tick_revenue[firm.firm_id] = 0.0
            self.last_tick_sell_through_rate[firm.firm_id] = 0.5  # neutral default
            self.last_tick_prices[firm.good_name] = firm.price

        # Misc firm: redistributes investment/R&D spending to random households
        self.misc_firm_revenue: float = 0.0  # Accumulated investment money
        self.misc_firm_beneficiaries: List[int] = []  # household_ids who receive payouts
        self._initialize_misc_firm_beneficiaries()
        self.post_warmup_stimulus_ticks: int = 0
        self.post_warmup_stimulus_duration: int = 0

    def _batch_plan_consumption(
        self,
        market_prices: Dict[str, float],
        category_market_snapshot: Dict[str, List[Dict[str, float]]],
        good_category_lookup: Optional[Dict[str, str]] = None,
        unemployment_rate: float = 0.0
    ) -> Dict[int, Dict]:
        """
        Vectorized batch consumption planning for all households.

        Replaces 10k individual calls to household.plan_consumption() with NumPy operations.
        Returns identical results to individual calls, but 10-20x faster.
        """
        def is_housing_good(good: str) -> bool:
            return _get_good_category(good, good_category_lookup) == "housing"

        # Extract household attributes as NumPy arrays
        cash_balances = np.array([h.cash_balance for h in self.households], dtype=np.float64)
        spending_tendencies = np.array([h.spending_tendency for h in self.households], dtype=np.float64)
        frugalities = np.array([max(h.frugality, 0.1) for h in self.households], dtype=np.float64)
        goods_values = np.array([sum(h.goods_inventory.values()) for h in self.households], dtype=np.float64)
        food_prefs = np.array([h.food_preference for h in self.households], dtype=np.float64)
        housing_prefs = np.array([h.housing_preference for h in self.households], dtype=np.float64)
        services_prefs = np.array([h.services_preference for h in self.households], dtype=np.float64)

        # H2: Subsistence vs discretionary spending with happiness modulation
        # Import config for new parameters
        from config import CONFIG

        # Macro confidence from unemployment
        macro_confidence = max(0.2, 1.0 - 0.6 * unemployment_rate)

        # Micro confidence from happiness (vectorized)
        happiness_arr = np.array([h.happiness for h in self.households], dtype=np.float64)
        micro_confidence = happiness_arr

        # Combined confidence
        confidence = 0.5 * macro_confidence + 0.5 * micro_confidence

        # Base spending rate as function of confidence
        base_spend = 0.5 + 0.3 * confidence  # 0.5-0.8 range

        # H5: Happiness modulates spending (±10% adjustment)
        happiness_multiplier = 0.9 + 0.2 * happiness_arr  # 0.9-1.1 range
        base_spend = base_spend * happiness_multiplier

        # Clamp to configured bounds
        base_spend = np.clip(base_spend, CONFIG.households.min_spend_fraction,
                           CONFIG.households.max_spend_fraction)

        # H3: Wealth and employment affect saving behavior
        net_worth_est = cash_balances + goods_values * 5.0

        # Normalized wealth score [0, 1]
        wealth_scores = np.clip(
            (net_worth_est - CONFIG.households.low_wealth_reference) /
            max(1.0, CONFIG.households.high_wealth_reference - CONFIG.households.low_wealth_reference),
            0.0, 1.0
        )

        # Wealth factor: richer households save more (0.8-1.2x multiplier)
        wealth_factor = 0.8 + 0.4 * wealth_scores

        # Trait factor
        trait_multiplier = np.clip(spending_tendencies / frugalities, 0.6, 1.4)

        # H3: Employment status adjustment
        # Employed: can save more, but also consume more if happy
        # Unemployed: forced dissaving if poor + long-term unemployed
        employment_status = np.array([h.is_employed for h in self.households], dtype=bool)
        unemployment_duration = np.array([h.unemployment_duration for h in self.households], dtype=np.float64)

        # Employed factor: slightly higher spending if happy
        employed_factor = 1.0 + 0.3 * wealth_scores - 0.2 * happiness_arr

        # Unemployed factor: forced dissaving if poor and long-term unemployed
        unemployed_poor = net_worth_est < CONFIG.households.unemployed_forced_dissaving_wealth
        unemployed_longterm = unemployment_duration > CONFIG.households.unemployed_forced_dissaving_duration
        forced_dissaving = unemployed_poor & unemployed_longterm
        unemployed_factor = np.where(forced_dissaving, 1.2, 1.0)  # Spend more when forced

        # Apply employment-specific factors
        employment_factor = np.where(employment_status, employed_factor, unemployed_factor)

        # Final spend fraction
        spend_fraction = base_spend * trait_multiplier * wealth_factor * employment_factor
        spend_fraction = np.clip(spend_fraction, 0.0, 1.0)

        # H2: Subsistence floor (always spend minimum if available)
        subsistence_min = CONFIG.households.subsistence_min_cash
        subsistence = np.minimum(cash_balances, subsistence_min)
        discretionary_cash = np.maximum(0.0, cash_balances - subsistence)
        discretionary_budget = discretionary_cash * spend_fraction
        budgets = subsistence + discretionary_budget

        # Precompute price caches per category for reuse
        price_cache: Dict[str, tuple] = {}
        category_option_cache: Dict[str, List[Dict[str, float]]] = {}
        for category, options in category_market_snapshot.items():
            affordable_opts = [opt for opt in options if opt.get("price", 0.0) > 0]
            if not affordable_opts:
                continue
            prices = [opt["price"] for opt in affordable_opts]
            if not prices:
                continue
            prices.sort()
            min_price = prices[0]
            max_price = prices[-1]
            median_price = prices[len(prices) // 2]
            price_cache[category] = (min_price, median_price, max_price)
            category_option_cache[category] = affordable_opts

        standard_categories = ["food", "housing", "services"]
        category_weights_matrix = np.array([
            [household.category_weights.get(cat, 0.0) for cat in standard_categories]
            for household in self.households
        ], dtype=np.float64)
        preference_matrix = np.column_stack((food_prefs, housing_prefs, services_prefs))
        biased_matrix = category_weights_matrix * preference_matrix
        precomputed_fractions = []
        for idx, household in enumerate(self.households):
            bias: Dict[str, float] = {}
            for cat_idx, cat in enumerate(standard_categories):
                val = biased_matrix[idx, cat_idx]
                if val > 0:
                    bias[cat] = val
            for cat, weight in household.category_weights.items():
                cat_lower = cat.lower()
                if cat_lower not in bias and weight > 0:
                    bias[cat_lower] = weight
            total_bias = sum(bias.values())
            if total_bias <= 0:
                precomputed_fractions.append({})
            else:
                fractions = {cat: weight / total_bias for cat, weight in bias.items() if weight > 0}
                precomputed_fractions.append(fractions)

        # Build consumption plans (fallback to Python loop for now due to complex logic)
        household_consumption_plans = {}

        for idx, household in enumerate(self.households):
            budget = budgets[idx]

            if budget <= 0:
                household_consumption_plans[household.household_id] = {
                    "household_id": household.household_id,
                    "category_budgets": {},
                    "planned_purchases": {},
                }
                continue

            # Use category weights if available
            if household.category_weights and sum(household.category_weights.values()) > 0 and category_market_snapshot:
                planned_purchases = household._plan_category_purchases(
                    budget,
                    category_market_snapshot,
                    price_cache,
                    category_fraction_override=precomputed_fractions[idx],
                    category_option_cache=category_option_cache
                )
                household_consumption_plans[household.household_id] = {
                    "household_id": household.household_id,
                    "category_budgets": {},
                    "planned_purchases": planned_purchases,
                }
            else:
                # Legacy good-based allocation
                local_beliefs = dict(household.price_beliefs)

                # Update beliefs with market prices
                for good, market_price in market_prices.items():
                    if good in local_beliefs:
                        old_belief = local_beliefs[good]
                        local_beliefs[good] = (
                            household.price_expectation_alpha * market_price +
                            (1.0 - household.price_expectation_alpha) * old_belief
                        )
                    else:
                        local_beliefs[good] = market_price

                # Normalize good weights
                total_weight = sum(household.good_weights.values())
                if total_weight <= 0:
                    all_goods = set(local_beliefs.keys()) | set(market_prices.keys())
                    if not all_goods:
                        normalized_weights = {}
                    else:
                        equal_weight = 1.0 / len(all_goods)
                        normalized_weights = {g: equal_weight for g in all_goods}
                else:
                    normalized_weights = {
                        g: w / total_weight for g, w in household.good_weights.items()
                    }

                # Plan purchases for each good
                planned_purchases = {}
                for good, weight in normalized_weights.items():
                    if weight <= 0:
                        continue

                    if good in local_beliefs:
                        expected_price = local_beliefs[good]
                    elif good in market_prices:
                        expected_price = market_prices[good]
                    else:
                        expected_price = household.default_price_level

                    if expected_price <= 0:
                        continue

                    good_budget = budget * weight
                    if is_housing_good(good):
                        planned_quantity = min(1.0, good_budget / expected_price)
                    else:
                        planned_quantity = good_budget / expected_price

                    if planned_quantity > 0:
                        planned_purchases[good] = planned_quantity

                household_consumption_plans[household.household_id] = {
                    "household_id": household.household_id,
                    "category_budgets": {},
                    "planned_purchases": planned_purchases,
                }

        return household_consumption_plans

    def _batch_apply_household_updates(
        self,
        transfer_plan: Dict[int, float],
        wage_taxes: Dict[int, float],
        per_household_purchases: Dict[int, Dict[str, Tuple[float, float]]],
        good_category_lookup: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Optimized batch update of all household states.

        Combines three separate loops into one for better cache locality.
        Eliminates method call overhead by inlining operations.
        """
        # Process loan repayments first (firms pay government)
        total_loan_repayments = 0.0
        for firm in self.firms:
            if firm.government_loan_remaining > 0:
                # Make weekly payment
                payment = min(firm.loan_payment_per_tick, firm.government_loan_remaining, firm.cash_balance)
                if payment > 0:
                    firm.cash_balance -= payment
                    firm.government_loan_remaining -= payment
                    total_loan_repayments += payment

        # Government receives loan repayments
        self.government.cash_balance += total_loan_repayments

        # Single pass through all households
        for household in self.households:
            hid = household.household_id
            household.met_housing_need = False

            # H4: Record starting cash for anomaly detection
            household.last_tick_cash_start = household.cash_balance

            # Apply income and taxes
            wage_income = household.wage if household.is_employed else 0.0

            # Add CEO salary if household is a CEO of any firm
            ceo_salary = 0.0
            for firm in self.firms:
                if firm.ceo_household_id == hid and firm.employees:
                    median_worker_wage = np.median([firm.actual_wages.get(e_id, firm.wage_offer) for e_id in firm.employees])
                    ceo_salary += median_worker_wage * 3.0  # CEO earns 3x median worker
                    firm.cash_balance -= ceo_salary  # Firm pays CEO

            transfers = transfer_plan.get(hid, 0.0)
            taxes_paid = wage_taxes.get(hid, 0.0)

            # H4: Track income components
            household.last_wage_income = wage_income + ceo_salary
            household.last_transfer_income = transfers
            household.last_other_income = -taxes_paid  # Taxes are negative income
            household.last_dividend_income = 0.0  # Will be set if dividends are distributed

            household.cash_balance += wage_income + ceo_salary + transfers - taxes_paid

            # Process medical loan payments (10% of wage per tick)
            medical_payment = household.make_medical_loan_payment()
            if medical_payment > 0:
                # Medical payments go to government (simplified - could go to healthcare provider)
                self.government.cash_balance += medical_payment

            # Apply purchases
            purchases = per_household_purchases.get(hid, {})
            total_spending = 0.0
            for good, (quantity, price_paid) in purchases.items():
                total_cost = quantity * price_paid
                total_spending += total_cost
                household.cash_balance -= total_cost
                category = _get_good_category(good, good_category_lookup)
                if category == "housing" and quantity > 0:
                    household.owns_housing = True
                    household.met_housing_need = True

                # Update inventory
                if good not in household.goods_inventory:
                    household.goods_inventory[good] = 0.0
                household.goods_inventory[good] += quantity

                # Update price beliefs
                if good in household.price_beliefs:
                    old_belief = household.price_beliefs[good]
                    household.price_beliefs[good] = (
                        household.price_expectation_alpha * price_paid +
                        (1.0 - household.price_expectation_alpha) * old_belief
                    )
                else:
                    household.price_beliefs[good] = price_paid

            # Consume goods from inventory
            consumption_rate = 0.1
            housing_usage = 1.0
            for good in list(household.goods_inventory.keys()):
                if household.goods_inventory[good] > 0:
                    category = _get_good_category(good, good_category_lookup)
                    current_qty = household.goods_inventory[good]
                    if category == "housing":
                        household.met_housing_need = household.met_housing_need or current_qty >= housing_usage
                        household.goods_inventory[good] = max(0.0, current_qty - housing_usage)
                        if household.goods_inventory[good] < 0.001 and household.owns_housing:
                            household.owns_housing = False
                    else:
                        consumed = current_qty * consumption_rate
                        household.goods_inventory[good] = max(0.0, current_qty - consumed)

                    if household.goods_inventory[good] < 0.001:
                        del household.goods_inventory[good]

            # H4: Record consumption spending and detect anomalies
            household.last_consumption_spending = total_spending

            # Anomaly detection: Flag large cash changes
            from config import CONFIG
            if CONFIG.debug.log_large_changes:
                net_change = (household.last_wage_income + household.last_transfer_income +
                             household.last_dividend_income + household.last_other_income -
                             household.last_consumption_spending)

                if abs(net_change) > CONFIG.debug.large_household_net_change:
                    print(f"[ANOMALY] HH {hid} tick {self.current_tick}: "
                          f"cash change ${net_change:+,.2f} "
                          f"(wage=${household.last_wage_income:.2f}, "
                          f"transfer=${household.last_transfer_income:.2f}, "
                          f"dividend=${household.last_dividend_income:.2f}, "
                          f"other=${household.last_other_income:.2f}, "
                          f"spending=${household.last_consumption_spending:.2f})")

    def step(self) -> None:
        """
        Execute one full simulation tick.

        Follows strict phase ordering:
        1. Firms plan production, labor, prices, wages
        2. Households plan labor supply and consumption
        3. Labor market matching
        4. Apply labor outcomes
        5. Firms apply production and costs
        6. Goods market clearing
        7. Government plans taxes
        8. Government plans transfers
        9. Apply sales, profits, taxes to firms
        10. Apply income, taxes, transfers, purchases to households
        11. Apply fiscal results to government
        12. Update world-level statistics
        """
        # Update warm-up flag for this tick (first 52 ticks are warm-up)
        was_in_warmup = self.in_warmup
        self.in_warmup = self.current_tick < 52
        if was_in_warmup and not self.in_warmup:
            self.post_warmup_cooldown = 8
            self.post_warmup_stimulus_ticks = 6
            self.post_warmup_stimulus_duration = 6
            self._sync_warmup_expectations(self.last_tick_prices)
            self._reset_post_warmup_expectations()
        self._refresh_target_total_firms()
        if not self.in_warmup:
            self._activate_queued_firms()
        if self.post_warmup_stimulus_ticks > 0:
            self._apply_post_warmup_stimulus()
        good_category_lookup = self._build_good_category_lookup()
        total_households = len(self.households)
        housing_private_inventory = 0.0
        housing_baseline_inventory = 0.0
        for firm in self.firms:
            if firm.good_category.lower() == "housing":
                if firm.is_baseline:
                    housing_baseline_inventory += firm.inventory_units
                else:
                    housing_private_inventory += firm.inventory_units
        housing_inventory_overhang = housing_private_inventory + housing_baseline_inventory
        unemployed_count = sum(1 for h in self.households if not h.is_employed)
        unemployment_rate = (unemployed_count / total_households) if total_households > 0 else 0.0

        gov_benefit = self.government.get_unemployment_benefit_level()

        # Update outstanding emergency-loan commitments before offering new aid
        self._update_loan_commitments()
        self._offer_emergency_loans(unemployment_rate)
        self._offer_inventory_liquidation_loans()  # No unemployment trigger
        self._ensure_public_works_capacity(unemployment_rate)

        # Phase 1: Firms plan
        firm_production_plans = {}
        firm_price_plans = {}
        firm_wage_plans = {}

        for firm in self.firms:
            # Plan production and labor
            production_plan = firm.plan_production_and_labor(
                self.last_tick_sales_units.get(firm.firm_id, 0.0),
                in_warmup=self.in_warmup,
                total_households=total_households,
                global_unsold_inventory=housing_inventory_overhang,
                private_housing_inventory=housing_private_inventory,
                large_market=self.large_market,
                post_warmup_cooldown=(self.post_warmup_cooldown > 0)
            )
            firm_production_plans[firm.firm_id] = production_plan

            # Plan pricing
            price_plan = firm.plan_pricing(
                self.last_tick_sell_through_rate.get(firm.firm_id, 0.5),
                unemployment_rate=unemployment_rate,
                in_warmup=self.in_warmup
            )
            firm_price_plans[firm.firm_id] = price_plan

            # Plan wage (pass unemployment rate for wage stabilization)
            wage_plan = firm.plan_wage(
                unemployment_rate=unemployment_rate,
                unemployment_benefit=gov_benefit
            )
            firm_wage_plans[firm.firm_id] = wage_plan

        # Enforce minimum wage floor (government policy)
        minimum_wage = self.government.get_minimum_wage()
        for wage_plan in firm_wage_plans.values():
            if wage_plan["wage_offer_next"] < minimum_wage:
                wage_plan["wage_offer_next"] = minimum_wage

        # Phase 2: Households plan
        household_labor_plans = {}

        category_market_snapshot = self._build_category_market_snapshot()

        for household in self.households:
            household.maybe_active_education()

        # Labor planning still uses loop (small overhead)
        for household in self.households:
            labor_plan = household.plan_labor_supply(gov_benefit)
            household_labor_plans[household.household_id] = labor_plan

        # Consumption planning now vectorized (major speedup)
        household_consumption_plans = self._batch_plan_consumption(
            self.last_tick_prices,
            category_market_snapshot,
            good_category_lookup,
            unemployment_rate
        )

        # Phase 3: Labor market matching
        firm_labor_outcomes, household_labor_outcomes = self._match_labor(
            firm_production_plans,
            firm_wage_plans,
            household_labor_plans
        )

        # Phase 4: Apply labor outcomes
        # Use cached wage percentiles (update every 5 ticks for performance)
        if self.current_tick - self.wage_percentile_cache_tick >= 5:
            market_paid_wages = []
            for outcome in firm_labor_outcomes.values():
                market_paid_wages.extend(outcome.get("actual_wages", {}).values())

            if market_paid_wages:
                # Use NumPy for fast percentile calculation
                wages_arr = np.array(market_paid_wages, dtype=np.float32)
                wage_anchor_low = float(np.percentile(wages_arr, 25))
                wage_anchor_mid = float(np.percentile(wages_arr, 50))
                wage_anchor_high = float(np.percentile(wages_arr, 75))
            else:
                wage_anchor_low = wage_anchor_mid = wage_anchor_high = None

            self.cached_wage_percentiles = (wage_anchor_low, wage_anchor_mid, wage_anchor_high)
            self.wage_percentile_cache_tick = self.current_tick
        else:
            wage_anchor_low, wage_anchor_mid, wage_anchor_high = self.cached_wage_percentiles

        for firm in self.firms:
            firm.apply_labor_outcome(firm_labor_outcomes[firm.firm_id])

        for household in self.households:
            anchor = None
            if household.skills_level < 0.4:
                anchor = wage_anchor_low
            elif household.skills_level > 0.7:
                anchor = wage_anchor_high
            else:
                anchor = wage_anchor_mid

            household.apply_labor_outcome(
                household_labor_outcomes[household.household_id],
                market_wage_anchor=anchor,
                current_tick=self.current_tick
            )

        # Update wages for continuing employees every 50 ticks (small 2-3% increases)
        if self.current_tick % 50 == 0:
            self._update_continuing_employee_wages()

        # Phase 5: Firms apply production and costs
        for firm in self.firms:
            production_plan = firm_production_plans[firm.firm_id]
            planned_production_units = production_plan["planned_production_units"]

            # Calculate actual production based on workforce experience and skills
            actual_production_units = self._calculate_experience_adjusted_production(
                firm, planned_production_units
            )

            firm.apply_production_and_costs({
                "realized_production_units": actual_production_units,
                "other_variable_costs": 0.0
            })

            # Update expectations
            firm.apply_updated_expectations(
                production_plan["updated_expected_sales"]
            )

        # Phase 6: Goods market clearing
        per_household_purchases, per_firm_sales = self._clear_goods_market(
            household_consumption_plans,
            self.firms
        )

        # Phase 6.5: Housing rental market clearing
        self._clear_housing_rental_market()
        self._apply_housing_repairs()

        # Phase 6.6: Housing firms consider unit expansion
        for firm in self.firms:
            if firm.good_category == "Housing":
                firm.invest_in_unit_expansion()

        # Phase 6.7: Misc firm operations
        self._misc_firm_add_beneficiary()  # Add 1 more random beneficiary
        self._misc_firm_redistribute_revenue()  # Pay out all accumulated revenue

        # Phase 6.8: Healthcare spending and medical loans
        self._process_healthcare_and_loans()

        # Phase 7: Government plans taxes
        household_tax_snapshots = self._build_household_tax_snapshots()
        firm_tax_snapshots = self._build_firm_tax_snapshots(per_firm_sales)

        tax_plan = self.government.plan_taxes(
            household_tax_snapshots,
            firm_tax_snapshots
        )

        # Phase 8: Government plans transfers
        household_transfer_snapshots = self._build_household_transfer_snapshots()
        transfer_plan = self.government.plan_transfers(household_transfer_snapshots)

        # Phase 9: Apply sales, profits, taxes to firms
        for firm in self.firms:
            sales_data = per_firm_sales.get(firm.firm_id, {"units_sold": 0.0, "revenue": 0.0})
            profit_tax = tax_plan["profit_taxes"].get(firm.firm_id, 0.0)
            property_tax = tax_plan["property_taxes"].get(firm.firm_id, 0.0)

            # Get price ceiling tax from firm snapshots
            price_ceiling_tax = 0.0
            for snapshot in firm_tax_snapshots:
                if snapshot["firm_id"] == firm.firm_id:
                    price_ceiling_tax = snapshot.get("price_ceiling_tax", 0.0)
                    break

            # Pay property tax if housing firm
            if property_tax > 0:
                firm.cash_balance -= property_tax

            firm.apply_sales_and_profit({
                "units_sold": sales_data["units_sold"],
                "revenue": sales_data["revenue"],
                "profit_taxes_paid": profit_tax + price_ceiling_tax
            })

            # Apply price and wage updates
            firm.apply_price_and_wage_updates(
                firm_price_plans[firm.firm_id],
                firm_wage_plans[firm.firm_id]
            )

        # Phase 10: Apply income, taxes, transfers, purchases to households
        self._batch_apply_household_updates(
            transfer_plan,
            tax_plan["wage_taxes"],
            per_household_purchases,
            good_category_lookup
        )

        # Phase 11: Apply government fiscal results
        total_wage_taxes = sum(tax_plan["wage_taxes"].values())
        total_profit_taxes = sum(tax_plan["profit_taxes"].values())
        total_property_taxes = sum(tax_plan["property_taxes"].values())
        total_price_ceiling_taxes = sum(snapshot.get("price_ceiling_tax", 0.0) for snapshot in firm_tax_snapshots)
        total_transfers = sum(transfer_plan.values())

        self.government.apply_fiscal_results(
            total_wage_taxes,
            total_profit_taxes + total_price_ceiling_taxes,  # Include price ceiling tax as profit tax
            total_transfers,
            total_property_taxes
        )

        # Phase 11.5: Government bond purchases with surplus (1 household per tick distribution)
        # Redirect bond spending to Misc firm
        govt_investments = self.government.make_investments()
        if govt_investments:
            for amount in govt_investments.values():
                self._collect_misc_revenue(amount)

        # Phase 11.6: Firm R&D spending (tax and redirect to Misc firm)
        total_investment_taxes = 0.0
        for firm in self.firms:
            revenue = per_firm_sales.get(firm.firm_id, {}).get("revenue", 0.0)
            if revenue > 0:
                rd_spending = firm.apply_rd_and_quality_update(revenue)
                # Apply investment tax
                investment_tax = rd_spending * self.government.investment_tax_rate
                after_tax_investment = rd_spending - investment_tax
                total_investment_taxes += investment_tax
                self._collect_misc_revenue(after_tax_investment)

        # Government collects investment taxes
        self.government.cash_balance += total_investment_taxes

        # Phase 11.75: Update household wellbeing (happiness, morale, health)
        if self.in_warmup:
            current_price_snapshot = {firm.good_name: firm.price for firm in self.firms}
            self._sync_warmup_expectations(current_price_snapshot)
        self._batch_update_wellbeing(
            happiness_multiplier=self.government.social_happiness_multiplier
        )

        # Phase 12: Handle firm bankruptcies and exits
        self._handle_firm_exits()

        # Phase 13: Potentially create new firms
        self._maybe_create_new_firms()

        # Phase 14: Government adjusts policies based on economic conditions
        self._adjust_government_policy()

        # Phase 15: Update world-level statistics
        self._update_statistics(per_firm_sales)

        # Phase 16: Distribute firm profits to owners (dividend payments)
        # This recycles wealth from firms back to households
        total_dividends_paid = 0.0
        for firm in self.firms:
            dividends = firm.distribute_profits(self.household_lookup)
            total_dividends_paid += dividends

        # Advance simulation clock after completing the tick
        self.current_tick += 1
        if self.post_warmup_cooldown > 0:
            self.post_warmup_cooldown -= 1

    def _match_labor(
        self,
        firm_production_plans: Dict[int, Dict],
        firm_wage_plans: Dict[int, Dict],
        household_labor_plans: Dict[int, Dict]
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Match firms and households in the labor market deterministically.

        Args:
            firm_production_plans: Production plans with hiring needs
            firm_wage_plans: Wage offers from firms
            household_labor_plans: Labor supply from households

        Returns:
            Tuple of (firm_labor_outcomes, household_labor_outcomes)
        """
        firm_labor_outcomes = {}
        household_labor_outcomes = {}
        assigned_households = set()

        # Track current employers to keep existing matches unless layoffs occur
        # Use cached firm_lookup instead of rebuilding
        planned_layoffs_set = set()
        for plan in firm_production_plans.values():
            planned_layoffs_set.update(plan.get("planned_layoffs_ids", []))

        for household in self.households:
            # Check if household is too sick to work (health < 40%)
            if not household.can_work:
                # Force unemployment due to health
                household_labor_outcomes[household.household_id] = {
                    "employer_id": None,
                    "wage": 0.0,
                    "employer_category": None
                }
                continue

            if household.is_employed and household.household_id not in planned_layoffs_set:
                employer_id = household.employer_id
                employer_category = None
                if employer_id is not None and employer_id in self.firm_lookup:
                    employer_category = self.firm_lookup[employer_id].good_category
                household_labor_outcomes[household.household_id] = {
                    "employer_id": employer_id,
                    "wage": household.wage,
                    "employer_category": employer_category
                }
                assigned_households.add(household.household_id)
            else:
                household_labor_outcomes[household.household_id] = {
                    "employer_id": None,
                    "wage": 0.0,
                    "employer_category": None
                }

        # Ensure all households present (even if not explicitly listed above)
        for household_id in household_labor_plans.keys():
            if household_id not in household_labor_outcomes:
                household_labor_outcomes[household_id] = {
                    "employer_id": None,
                    "wage": 0.0,
                    "employer_category": None
                }

        # Sort firms by firm_id for deterministic ordering
        sorted_firms = sorted(self.firms, key=lambda f: f.firm_id)

        for firm in sorted_firms:
            firm_id = firm.firm_id
            production_plan = firm_production_plans[firm_id]
            wage_plan = firm_wage_plans[firm_id]

            vacancies = production_plan["planned_hires_count"]
            wage_offer = wage_plan["wage_offer_next"]
            confirmed_layoffs = production_plan["planned_layoffs_ids"]

            # Initialize firm outcome
            firm_labor_outcomes[firm_id] = {
                "hired_households_ids": [],
                "confirmed_layoffs_ids": confirmed_layoffs,
                "actual_wages": {}
            }

            if vacancies <= 0:
                continue

            # Find eligible candidates (vectorized filtering)
            # Build arrays for unassigned job seekers
            unassigned_ids = []
            unassigned_skills = []
            unassigned_reservation = []

            for household_id, labor_plan in household_labor_plans.items():
                if household_id not in assigned_households and labor_plan["searching_for_job"]:
                    # Check if household is healthy enough to work
                    household = self.household_lookup.get(household_id)
                    if household and household.can_work:
                        unassigned_ids.append(household_id)
                        unassigned_skills.append(labor_plan["skills_level"])
                        unassigned_reservation.append(labor_plan["reservation_wage"])

            if not unassigned_ids:
                continue

            # Vectorized eligibility check
            unassigned_ids_arr = np.array(unassigned_ids, dtype=np.int32)
            unassigned_skills_arr = np.array(unassigned_skills, dtype=np.float32)
            unassigned_reservation_arr = np.array(unassigned_reservation, dtype=np.float32)

            # Filter by wage offer
            eligible_mask = wage_offer >= unassigned_reservation_arr
            eligible_ids = unassigned_ids_arr[eligible_mask]
            eligible_skills = unassigned_skills_arr[eligible_mask]

            if len(eligible_ids) == 0:
                continue

            # Sort by skills (descending), then by id (ascending)
            sort_keys = np.lexsort((eligible_ids, -eligible_skills))
            eligible_ids = eligible_ids[sort_keys]
            eligible_skills = eligible_skills[sort_keys]

            # Assign up to vacancies
            hired_count = min(vacancies, len(eligible_ids))
            for i in range(hired_count):
                household_id = int(eligible_ids[i])
                skills_level = float(eligible_skills[i])

                # Get household to check experience (O(1) lookup via cache)
                household = self.household_lookup[household_id]

                # Calculate skill premium (25% max for skill level 1.0)
                skill_premium = skills_level * 0.25

                # Calculate experience premium (3% per year, capped at 30%)
                # Assume 52 ticks per year
                experience_ticks = household.category_experience.get(firm.good_category, 0)
                experience_years = experience_ticks / 52.0
                experience_premium = min(experience_years * 0.03, 0.3)

                # Calculate actual wage with premiums
                actual_wage = wage_offer * (1.0 + skill_premium + experience_premium)

                # Record hire
                firm_labor_outcomes[firm_id]["hired_households_ids"].append(household_id)
                firm_labor_outcomes[firm_id]["actual_wages"][household_id] = actual_wage
                assigned_households.add(household_id)

                # Update household outcome
                household_labor_outcomes[household_id] = {
                    "employer_id": firm_id,
                    "wage": actual_wage,
                    "employer_category": firm.good_category
                }

        return firm_labor_outcomes, household_labor_outcomes

    def _update_continuing_employee_wages(self) -> None:
        """
        Update wages for continuing employees every 50 ticks.

        Applies a small 2-3% increase to existing employee wages to prevent
        massive wage increases within a single tick. Only updates employees
        who have been with the firm for at least 50 ticks.
        """
        import random

        for firm in self.firms:
            if not firm.employees:
                continue

            for employee_id in firm.employees:
                household = self.household_lookup.get(employee_id)
                if household is None:
                    continue

                # Only update if last wage update was at least 50 ticks ago
                if self.current_tick - household.last_wage_update_tick >= 50:
                    current_wage = firm.actual_wages.get(employee_id, firm.wage_offer)

                    # Apply 2-3% increase
                    increase_rate = random.uniform(0.02, 0.03)
                    new_wage = current_wage * (1.0 + increase_rate)

                    # Update the wage
                    firm.actual_wages[employee_id] = new_wage
                    household.last_wage_update_tick = self.current_tick

    def _clear_goods_market(
        self,
        household_consumption_plans: Dict[int, Dict],
        firms: List[FirmAgent]
    ) -> Tuple[Dict[int, Dict[str, Tuple[float, float]]], Dict[int, Dict[str, float]]]:
        """
        Clear the goods market deterministically.

        Args:
            household_consumption_plans: Desired purchases from households
            firms: List of firm agents with inventory

        Returns:
            Tuple of (per_household_purchases, per_firm_sales)
        """
        per_household_purchases: Dict[int, Dict[str, Tuple[float, float]]] = {}
        per_firm_sales: Dict[int, Dict[str, float]] = {}

        # Firm arrays for fast lookup
        firm_ids = [f.firm_id for f in firms]
        id_to_idx = {fid: idx for idx, fid in enumerate(firm_ids)}
        firm_prices = np.array([f.price for f in firms], dtype=np.float64)
        firm_goods = [f.good_name for f in firms]
        firm_remaining = np.array([f.inventory_units for f in firms], dtype=np.float64)

        for fid in firm_ids:
            per_firm_sales[fid] = {"units_sold": 0.0, "revenue": 0.0}

        # Group firm indices by good_name, sorted by price then id
        goods_to_indices: Dict[str, np.ndarray] = {}
        for idx, firm in enumerate(firms):
            goods_to_indices.setdefault(firm.good_name, []).append(idx)
        for good_name, idx_list in goods_to_indices.items():
            idx_list.sort(key=lambda i: (firm_prices[i], firm_ids[i]))
            goods_to_indices[good_name] = np.array(idx_list, dtype=np.int32)

        # Process households in ID order
        for household_id, consumption_plan in sorted(household_consumption_plans.items(), key=lambda x: x[0]):
            per_household_purchases[household_id] = {}
            planned = consumption_plan["planned_purchases"]

            for target, desired_qty in planned.items():
                if desired_qty <= 0:
                    continue

                # Direct firm id target (check for both int and np.integer)
                if isinstance(target, (int, np.integer)):
                    idx = id_to_idx.get(int(target))  # Convert np.int32 to Python int for dict lookup
                    if idx is None:
                        continue
                    available = firm_remaining[idx]
                    if available <= 0:
                        continue
                    qty = min(desired_qty, available)
                    if qty <= 0:
                        continue
                    firm_remaining[idx] -= qty
                    fid = firm_ids[idx]
                    price = firm_prices[idx]
                    per_firm_sales[fid]["units_sold"] += qty
                    per_firm_sales[fid]["revenue"] += qty * price

                    gname = firm_goods[idx]
                    prev_qty, prev_price = per_household_purchases[household_id].get(gname, (0.0, 0.0))
                    total_qty = prev_qty + qty
                    if total_qty > 0:
                        avg_price = ((prev_qty * prev_price) + (qty * price)) / total_qty
                        per_household_purchases[household_id][gname] = (total_qty, avg_price)
                    continue

                # Good-name target: spread across sorted firms
                good_name = target
                idx_list = goods_to_indices.get(good_name)
                if not idx_list:
                    continue

                remaining = desired_qty
                total_bought = 0.0
                price_sum = 0.0

                for idx in idx_list:
                    if remaining <= 0:
                        break
                    available = firm_remaining[idx]
                    if available <= 0:
                        continue
                    qty = min(remaining, available)
                    firm_remaining[idx] -= qty
                    fid = firm_ids[idx]
                    price = firm_prices[idx]
                    per_firm_sales[fid]["units_sold"] += qty
                    per_firm_sales[fid]["revenue"] += qty * price
                    total_bought += qty
                    price_sum += qty * price
                    remaining -= qty

                if total_bought > 0:
                    per_household_purchases[household_id][good_name] = (
                        total_bought,
                        price_sum / total_bought
                    )

        return per_household_purchases, per_firm_sales

    def _build_category_market_snapshot(self) -> Dict[str, List[Dict[str, float]]]:
        """Provide firms grouped by category for household consumption planning."""
        snapshot: Dict[str, List[Dict[str, float]]] = {}
        for firm in self.firms:
            category_key = firm.good_category.lower()
            if category_key not in snapshot:
                snapshot[category_key] = []
            snapshot[category_key].append({
                "firm_id": firm.firm_id,
                "good_name": firm.good_name,
                "price": firm.price,
                "quality": firm.quality_level,
                "inventory": firm.inventory_units,
            })
        return snapshot

    def _build_good_category_lookup(self) -> Dict[str, str]:
        """Map each good_name to its category (lowercased) for quick lookups."""
        return {firm.good_name: firm.good_category.lower() for firm in self.firms}

    def _next_firm_id(self) -> int:
        """Generate a unique firm_id across active and queued firms."""
        existing_ids = set(self.firm_lookup.keys())
        existing_ids.update(f.firm_id for f in self.queued_firms)
        return max(existing_ids) + 1 if existing_ids else 1

    def _refresh_target_total_firms(self) -> None:
        """Recalculate desired firm counts based on current population and pipeline size."""
        households = max(1, len(self.households))
        per_thousand = CONFIG.firms.target_firms_per_1000_households
        dynamic_target = int(math.ceil((households / 1000.0) * per_thousand))
        baseline_count = sum(1 for f in self.firms if f.is_baseline)
        dynamic_target = max(dynamic_target, baseline_count)
        pipeline = len(self.firms) + len(self.queued_firms)
        self.target_total_firms = max(dynamic_target, pipeline, self.target_total_firms)

    def _activate_queued_firms(self) -> None:
        """
        Activate queued firms gradually after warm-up (staggered entry).

        Instead of activating all firms at once, only activate up to
        max_new_firms_per_tick firms per tick to prevent labor market shocks.
        """
        if not self.queued_firms:
            return

        if len(self.firms) >= self.target_total_firms * 1.2:
            return

        max_new_firms = CONFIG.firms.max_new_firms_per_tick
        firms_to_activate = min(max_new_firms, len(self.queued_firms))

        allowed = max(0, int(self.target_total_firms * 1.2) - len(self.firms))
        firms_to_activate = min(firms_to_activate, allowed)

        for _ in range(firms_to_activate):
            firm = self.queued_firms.pop(0)
            self.firms.append(firm)
            self.firm_lookup[firm.firm_id] = firm
            self.last_tick_sales_units[firm.firm_id] = 0.0
            self.last_tick_revenue[firm.firm_id] = 0.0
            self.last_tick_sell_through_rate[firm.firm_id] = 0.5
            self.last_tick_prices[firm.good_name] = firm.price

    def _batch_update_wellbeing(self, happiness_multiplier: float) -> None:
        """Vectorized wellbeing update to reduce per-agent Python overhead."""
        if not self.households:
            return

        households = self.households
        n = len(households)

        employed = np.fromiter((h.is_employed for h in households), dtype=np.bool_, count=n)
        wages = np.fromiter((h.wage for h in households), dtype=np.float64, count=n)
        expected_wages = np.fromiter((h.expected_wage for h in households), dtype=np.float64, count=n)

        happiness = np.fromiter((h.happiness for h in households), dtype=np.float64, count=n)
        morale = np.fromiter((h.morale for h in households), dtype=np.float64, count=n)
        health = np.fromiter((h.health for h in households), dtype=np.float64, count=n)

        happiness_decay = np.fromiter((h.happiness_decay_rate for h in households), dtype=np.float64, count=n)
        morale_decay = np.fromiter((h.morale_decay_rate for h in households), dtype=np.float64, count=n)
        health_decay = np.fromiter((h.health_decay_rate for h in households), dtype=np.float64, count=n)

        total_goods = np.fromiter((sum(h.goods_inventory.values()) for h in households), dtype=np.float64, count=n)
        housing_met = np.fromiter((h.met_housing_need for h in households), dtype=np.bool_, count=n)

        # Happiness
        happiness_change = np.where(employed, 0.02, -0.03)
        happiness_change += np.where(total_goods > 10.0, 0.01, 0.0)
        happiness_change += np.where(total_goods < 2.0, -0.02, 0.0)
        happiness_change += np.where(~housing_met, -0.05, 0.0)

        # FIX: Wage adequacy affects happiness
        # Being employed at a poverty wage shouldn't make you happy
        cash_balances = np.fromiter((h.cash_balance for h in households), dtype=np.float64, count=n)
        poverty_threshold = 200.0  # If cash < $200, you're struggling
        in_poverty = cash_balances < poverty_threshold
        happiness_change -= np.where(in_poverty, 0.03, 0.0)  # -0.03 if in poverty

        # Extremely poor (cash < $100)
        extreme_poverty = cash_balances < 100.0
        happiness_change -= np.where(extreme_poverty, 0.05, 0.0)  # Additional -0.05

        if happiness_multiplier > 1.0:
            happiness_change += (happiness_multiplier - 1.0) * 0.05
        happiness_change -= happiness_decay
        happiness_next = np.clip(happiness + happiness_change, 0.0, 1.0)

        # Morale
        morale_change = np.zeros(n, dtype=np.float64)
        satisfied = employed & (wages >= expected_wages)
        morale_change += np.where(satisfied, 0.03, 0.0)

        underpaid = employed & (wages < expected_wages)
        wage_gap_ratio = np.zeros(n, dtype=np.float64)
        if underpaid.any():
            wage_gap_ratio[underpaid] = (expected_wages[underpaid] - wages[underpaid]) / np.maximum(
                expected_wages[underpaid], 1.0
            )
        morale_change -= wage_gap_ratio * 0.05

        morale_change -= np.where(~employed, 0.05, 0.0)
        morale_change -= morale_decay
        morale_next = np.clip(morale + morale_change, 0.0, 1.0)

        # Health
        health_change = np.zeros(n, dtype=np.float64)
        health_change += np.where(total_goods > 15.0, 0.01, 0.0)
        health_change += np.where(total_goods < 5.0, -0.02, 0.0)
        if happiness_multiplier > 1.0:
            health_change += (happiness_multiplier - 1.0) * 0.03
        health_change -= health_decay
        health_next = np.clip(health + health_change, 0.0, 1.0)

        # Write back
        for idx, household in enumerate(households):
            household.happiness = float(happiness_next[idx])
            household.morale = float(morale_next[idx])
            household.health = float(health_next[idx])

    def _sync_warmup_expectations(self, current_prices: Dict[str, float]) -> None:
        """During warm-up, force beliefs/expectations to current observed values."""
        for household in self.households:
            if household.is_employed:
                household.expected_wage = household.wage
            for good, price in current_prices.items():
                household.price_beliefs[good] = price

    def _reset_post_warmup_expectations(self) -> None:
        """Reset household wage expectations/reservations to align with post-warmup economy."""
        if not self.households:
            return

        wage_offers = [f.wage_offer for f in self.firms if f.wage_offer > 0]
        if wage_offers:
            wage_anchor = float(np.median(wage_offers))
        else:
            wage_anchor = 30.0

        for household in self.households:
            housing_price = household.price_beliefs.get("housing", household.default_price_level)
            food_price = household.price_beliefs.get("food", household.default_price_level)
            living_cost = 0.3 * housing_price + household.min_food_per_tick * food_price
            living_cost = max(living_cost, 25.0)

            household.expected_wage = wage_anchor

            # H1': Dynamic reservation wage with decay over unemployment duration
            unemployment_benefit = self.government.get_unemployment_benefit_level()
            wage_tax_rate = self.government.wage_tax_rate

            # Net unemployment benefit (after tax)
            benefit_net = unemployment_benefit * (1.0 - wage_tax_rate)

            if household.is_employed:
                # Employed: Update reservation wage toward current net wage
                current_net_wage = household.wage * (1.0 - wage_tax_rate)
                household.reservation_wage = 0.9 * household.reservation_wage + 0.1 * current_net_wage

                # Also ensure minimum wage floor for existing workers
                minimum_wage = self.government.get_minimum_wage()
                household.wage = max(household.wage, living_cost, minimum_wage)
            else:
                # Unemployed: Initialize or decay reservation wage
                if household.unemployment_duration == 1:
                    # First tick unemployed: Start 20% above benefits
                    household.reservation_wage = benefit_net * 1.2
                elif household.unemployment_duration > 1:
                    # Decay reservation wage toward 5% above benefits
                    decay_speed = 0.01  # 1% per tick
                    floor_factor = 1.05  # Long-run: 5% above benefits
                    target_reservation = benefit_net * floor_factor

                    household.reservation_wage = (
                        (1.0 - decay_speed) * household.reservation_wage +
                        decay_speed * target_reservation
                    )

                # Never go below living cost
                household.reservation_wage = max(household.reservation_wage, living_cost)

    def _build_household_transfer_snapshots(self) -> List[Dict[str, object]]:
        """
        Build snapshots for government transfer planning.

        Returns:
            List of dicts with household_id, is_employed, cash_balance
        """
        snapshots = []
        for household in self.households:
            snapshots.append({
                "household_id": household.household_id,
                "is_employed": household.is_employed,
                "cash_balance": household.cash_balance
            })
        return snapshots

    def _build_household_tax_snapshots(self) -> List[Dict[str, object]]:
        """
        Build snapshots for government tax planning (household part).

        Returns:
            List of dicts with household_id and wage_income
        """
        snapshots = []
        for household in self.households:
            wage_income = household.wage if household.is_employed else 0.0
            snapshots.append({
                "household_id": household.household_id,
                "wage_income": wage_income
            })
        return snapshots

    def _build_firm_tax_snapshots(
        self,
        per_firm_sales: Dict[int, Dict[str, float]]
    ) -> List[Dict[str, object]]:
        """
        Build snapshots for government tax planning (firm part).

        Args:
            per_firm_sales: Sales data from goods market clearing

        Returns:
            List of dicts with firm_id, profit_before_tax, and price_ceiling_tax
        """
        snapshots = []
        PRICE_CEILING = 50.0  # Price ceiling threshold
        PRICE_CEILING_TAX_RATE = 0.25  # 25% tax on revenue from sales above ceiling

        for firm in self.firms:
            sales_data = per_firm_sales.get(firm.firm_id, {"revenue": 0.0, "units_sold": 0.0})
            revenue = sales_data["revenue"]
            units_sold = sales_data["units_sold"]

            # Calculate price ceiling tax
            # If price > $50, firm pays 25% tax on the revenue from those sales
            price_ceiling_tax = 0.0
            if firm.price > PRICE_CEILING and units_sold > 0:
                # Tax applies to revenue from sales above the ceiling
                price_ceiling_tax = revenue * PRICE_CEILING_TAX_RATE

            # Compute costs
            wage_bill = sum(firm.actual_wages.get(e_id, firm.wage_offer) for e_id in firm.employees)

            # Add CEO salary if firm has a CEO (3x median worker wage)
            ceo_salary = 0.0
            if firm.ceo_household_id is not None and firm.employees:
                median_worker_wage = np.median([firm.actual_wages.get(e_id, firm.wage_offer) for e_id in firm.employees])
                ceo_salary = median_worker_wage * 3.0  # CEO earns 3x median worker
                wage_bill += ceo_salary

            # Note: Other variable costs would be included here if tracked

            # Profit = revenue - wage_bill - price_ceiling_tax (simplified)
            profit_before_tax = revenue - wage_bill - price_ceiling_tax

            snapshots.append({
                "firm_id": firm.firm_id,
                "profit_before_tax": profit_before_tax,
                "price_ceiling_tax": price_ceiling_tax
            })
        return snapshots

    def _calculate_experience_adjusted_production(
        self, firm: FirmAgent, planned_production_units: float
    ) -> float:
        """
        Calculate actual production based on workforce experience, skills, and wellbeing.

        Workers with more experience in the firm's category produce more.
        Workers with higher happiness/morale/health perform better.
        Government infrastructure investment boosts all productivity.

        Args:
            firm: The firm to calculate production for
            planned_production_units: Planned production from plan_production_and_labor

        Returns:
            Actual production units accounting for experience, wellbeing, and infrastructure
        """
        if len(firm.employees) == 0:
            return 0.0

        # Calculate average productivity multiplier for the workforce
        total_productivity_multiplier = 0.0
        for employee_id in firm.employees:
            # Find the household
            household = next((h for h in self.households if h.household_id == employee_id), None)
            if household is None:
                # Employee not found (shouldn't happen, but handle gracefully)
                total_productivity_multiplier += 1.0
                continue

            # Base multiplier is 1.0
            productivity_multiplier = 1.0

            # Add skill bonus (max 25% for skills_level = 1.0)
            skill_bonus = household.skills_level * 0.25

            # Add experience bonus (5% per year, capped at 50%)
            experience_ticks = household.category_experience.get(firm.good_category, 0)
            experience_years = experience_ticks / 52.0
            experience_bonus = min(experience_years * 0.05, 0.5)

            # Add wellbeing performance bonus (happiness/morale/health)
            # Performance multiplier ranges from 0.5x (low wellbeing) to 1.5x (high wellbeing)
            wellbeing_multiplier = household.get_performance_multiplier()

            # Combine all factors
            productivity_multiplier += skill_bonus + experience_bonus
            productivity_multiplier *= wellbeing_multiplier

            total_productivity_multiplier += productivity_multiplier

        # Calculate average productivity multiplier
        avg_productivity_multiplier = total_productivity_multiplier / len(firm.employees)

        # Apply government infrastructure multiplier
        # Government infrastructure investment boosts all productivity economy-wide
        avg_productivity_multiplier *= self.government.infrastructure_productivity_multiplier

        # Apply to planned production
        # Cap at production capacity
        worker_capacity = firm._capacity_for_workers(len(firm.employees))
        actual_production = min(
            planned_production_units * avg_productivity_multiplier,
            firm.production_capacity_units,
            worker_capacity
        )

        return actual_production

    def _handle_firm_exits(self) -> None:
        """
        Remove bankrupt firms from the economy.

        Firms with negative cash below a threshold are removed.
        Their employees are laid off.

        Mutates state.
        """
        bankruptcy_threshold = -1000.0  # Firms below this cash level exit
        zero_cash_max_streak = 12  # Ticks allowed at zero/negative cash before exit

        firms_to_remove = []
        for firm in self.firms:
            if firm.cash_balance < bankruptcy_threshold or firm.zero_cash_streak >= zero_cash_max_streak:
                # Protect government baseline firms at all times
                if self.government.is_baseline_firm(firm.firm_id):
                    continue

                # Firm is bankrupt - lay off all employees
                for employee_id in firm.employees:
                    # Find household and unemploy them (O(1) lookup via cache)
                    household = self.household_lookup.get(employee_id)
                    if household is not None:
                        household.employer_id = None
                        household.wage = 0.0

                firms_to_remove.append(firm)

        # Remove bankrupt firms
        for firm in firms_to_remove:
            self.firms.remove(firm)

            # Clean up tracking dictionaries
            if firm.firm_id in self.last_tick_sales_units:
                del self.last_tick_sales_units[firm.firm_id]
            if firm.firm_id in self.last_tick_revenue:
                del self.last_tick_revenue[firm.firm_id]
            if firm.firm_id in self.last_tick_sell_through_rate:
                del self.last_tick_sell_through_rate[firm.firm_id]

            # Clean up firm cache
            if firm.firm_id in self.firm_lookup:
                del self.firm_lookup[firm.firm_id]

    def _maybe_create_new_firms(self) -> None:
        """
        Potentially create new firms to replace bankrupt ones.

        New firms are created if:
        1. Total number of firms is below a minimum threshold
        2. There's demand in the economy (some households have cash)

        Mutates state.
        """
        from agents import FirmAgent

        # Warm-up period blocks private firm creation to avoid destabilizing start
        if self.in_warmup:
            return

        # Check if there's economic activity (households have cash)
        total_household_cash = sum(h.cash_balance for h in self.households)
        if total_household_cash < 1000.0:
            return  # Not enough demand to support new firms

        desired_pipeline = int(self.target_total_firms * 1.1) if self.target_total_firms else len(self.firms) + 1
        if len(self.firms) + len(self.queued_firms) >= max(desired_pipeline, self.target_total_firms):
            return

        # Create a new firm
        existing_ids = [f.firm_id for f in self.firms]
        existing_ids.extend(firm.firm_id for firm in self.queued_firms)
        new_firm_id = max(existing_ids, default=0) + 1

        categories = ["Food", "Housing", "Services"]
        total_units = sum(
            f.max_rental_units for f in self.firms
            if f.good_category == "Housing"
        )
        if total_units < len(self.households):
            chosen_category = "Housing"
        else:
            if self.firms:
                category_counts = {}
                for cat in categories:
                    category_counts[cat] = sum(1 for f in self.firms if f.good_category == cat)
                chosen_category = min(category_counts, key=category_counts.get)
            else:
                chosen_category = "Food"

        # Determine firm personality (deterministic based on firm_id)
        # Use modulo to cycle through personalities
        personality_index = new_firm_id % 3
        if personality_index == 0:
            personality = "aggressive"
        elif personality_index == 1:
            personality = "conservative"
        else:
            personality = "moderate"

        # Base quality level affected by government technology investment
        base_quality = 5.0 * self.government.technology_quality_multiplier

        # Determine median quality in category for seeding
        category_qualities = [f.quality_level for f in self.firms if f.good_category == chosen_category]
        median_quality = np.median(category_qualities) if category_qualities else 5.0

        # Government seed loan if available
        seed_cash = min(250000.0, max(50000.0, total_household_cash * 0.02))
        if self.government.cash_balance > seed_cash:
            self.government.cash_balance -= seed_cash
        else:
            seed_cash = 50000.0

        # Assign a random household as CEO owner (after warmup, tick > 52)
        ceo_id = None
        if self.current_tick > 52 and self.households:
            # Pick a random household to be the CEO
            ceo_id = np.random.choice([h.household_id for h in self.households])

        # Housing-specific initialization
        max_rental_units = 0
        property_tax_rate = 0.0
        if chosen_category == "Housing":
            max_rental_units = np.random.randint(0, 51)
            seed_cash = max(50000.0, max_rental_units * 10000.0)
            if self.government.cash_balance > seed_cash:
                self.government.cash_balance -= seed_cash
            else:
                seed_cash = 50000.0
            property_tax_rate = 0.005 * max_rental_units

        # Calculate loan repayment: 1% annual interest, 3-year payment plan
        # Interest rate: 1% per year = 1/52% per week
        # Total amount to repay: principal * (1 + 0.01 * 3) = principal * 1.03
        # Payment per tick: total / (52 weeks * 3 years) = total / 156
        total_repayment = seed_cash * 1.03
        weekly_payment = total_repayment / 156.0

        new_firm = FirmAgent(
            firm_id=new_firm_id,
            good_name=f"{chosen_category}Product{new_firm_id}",
            cash_balance=seed_cash,
            inventory_units=100.0 if chosen_category != "Housing" else 0.0,  # Housing has no inventory
            good_category=chosen_category,
            quality_level=min(10.0, max(1.0, median_quality + np.random.uniform(-1.0, 1.0))),
            wage_offer=35.0,
            price=150.0 if chosen_category == "Housing" else 8.0,  # Rent starts at $150/week
            expected_sales_units=50.0 if chosen_category != "Housing" else float(max_rental_units),
            production_capacity_units=500.0 if chosen_category != "Housing" else float(max_rental_units),
            productivity_per_worker=10.0,
            units_per_worker=18.0,
            personality=personality,
            government_loan_principal=seed_cash,
            government_loan_remaining=total_repayment,
            loan_payment_per_tick=weekly_payment,
            ceo_household_id=ceo_id,
            max_rental_units=max_rental_units,
            property_tax_rate=property_tax_rate
        )

        # Set personality-based behavior parameters
        new_firm.set_personality(personality)

        self.firms.append(new_firm)

        # Initialize tracking for new firm
        self.last_tick_sales_units[new_firm_id] = 0.0
        self.last_tick_revenue[new_firm_id] = 0.0
        self.last_tick_sell_through_rate[new_firm_id] = 0.5
        self.last_tick_prices[new_firm.good_name] = new_firm.price

        # Add to firm cache
        self.firm_lookup[new_firm_id] = new_firm

    def _update_loan_commitments(self) -> None:
        """Tick down hiring commitments tied to emergency loans and reclaim aid if ignored."""
        config = CONFIG.government
        for firm in self.firms:
            if firm.loan_required_headcount <= 0:
                continue

            if firm.loan_support_ticks > 0:
                firm.loan_support_ticks = max(0, firm.loan_support_ticks - 1)
                # If the firm has already met the requirement, clear the commitment early
                if len(firm.employees) >= firm.loan_required_headcount:
                    firm.loan_required_headcount = 0
                    firm.loan_support_ticks = 0
                continue

            # Commitment window expired. If requirement still unmet, claw back remaining aid.
            if len(firm.employees) >= firm.loan_required_headcount:
                firm.loan_required_headcount = 0
                firm.loan_support_ticks = 0
                continue

            reclaimable = min(
                firm.cash_balance,
                firm.government_loan_remaining * config.emergency_loan_penalty_reclaim_fraction
            )
            if reclaimable > 0:
                firm.cash_balance -= reclaimable
                firm.government_loan_remaining = max(0.0, firm.government_loan_remaining - reclaimable)
                self.government.cash_balance += reclaimable

            firm.loan_required_headcount = 0
            firm.loan_support_ticks = 0

    def _offer_emergency_loans(self, unemployment_rate: float) -> None:
        """
        Provide temporary low-interest loans to cash-strapped firms
        whenever unemployment breaches the configured trigger.
        """
        config = CONFIG.government
        if unemployment_rate < config.emergency_loan_trigger:
            return

        reserve_floor = config.investment_reserve_threshold
        available_cash = self.government.cash_balance - reserve_floor
        if available_cash <= 0:
            return

        per_tick_budget = max(
            config.emergency_loan_amount,
            available_cash * config.emergency_loan_fraction_of_cash
        )
        per_tick_budget = min(per_tick_budget, available_cash)

        candidate_firms = [
            f for f in self.firms
            if (not f.is_baseline) and (
                f.cash_balance < config.emergency_loan_cash_threshold
                or len(f.employees) < config.emergency_loan_min_headcount
            )
        ]
        if not candidate_firms or per_tick_budget <= 0:
            return

        candidate_firms.sort(key=lambda firm: firm.cash_balance)
        term_ticks = max(
            1,
            int(CONFIG.time.ticks_per_year * config.emergency_loan_term_years)
        )
        interest_multiplier = 1.0 + config.emergency_loan_interest

        for firm in candidate_firms:
            if per_tick_budget <= 0 or self.government.cash_balance <= reserve_floor:
                break

            desired = max(
                config.emergency_loan_amount,
                config.emergency_loan_cash_threshold - firm.cash_balance
            )
            capacity = self.government.cash_balance - reserve_floor
            loan_amount = min(desired, per_tick_budget, capacity)
            if loan_amount <= 0:
                continue

            firm.cash_balance += loan_amount
            total_repayment = loan_amount * interest_multiplier
            firm.government_loan_principal += loan_amount
            firm.government_loan_remaining += total_repayment
            firm.loan_payment_per_tick += total_repayment / term_ticks
            enforcement_ticks = max(1, config.emergency_loan_enforcement_ticks)
            required_headcount = max(
                config.emergency_loan_min_headcount,
                int(
                    math.ceil(
                        max(1, len(firm.employees)) *
                        config.emergency_loan_required_headcount_multiplier
                    )
                )
            )
            firm.loan_required_headcount = max(firm.loan_required_headcount, required_headcount)
            firm.loan_support_ticks = max(firm.loan_support_ticks, enforcement_ticks)
            self.government.cash_balance -= loan_amount
            per_tick_budget -= loan_amount

    def _offer_inventory_liquidation_loans(self) -> None:
        """
        Provide loans to firms with poor sales to help with R&D and inventory liquidation.

        Loans are offered to firms where:
        - Items sold < items produced (underselling)
        - Inventory > 2× production rate (inventory buildup)

        Loan amount: 10% of firm's current cash balance (scaled, not fixed)
        Interest rate: 2% annual (2/52 = 0.0385% per tick)
        Term: 3 years (156 ticks)
        """
        # Check government has funds available (minimum 1% reserve)
        min_reserve = self.government.cash_balance * 0.01
        if self.government.cash_balance < min_reserve:
            return

        # Calculate per-tick budget (20% of available cash above reserve)
        available_cash = max(0.0, self.government.cash_balance - min_reserve)
        per_tick_budget = available_cash * 0.2

        # Find candidate firms (poor sales performance)
        candidate_firms = [
            f for f in self.firms
            if (not f.is_baseline) and (
                (f.last_units_produced > 0 and f.last_units_sold < f.last_units_produced)
                or (f.last_units_produced > 0 and f.inventory_units > 2.0 * f.last_units_produced)
            ) and f.cash_balance > 0
        ]

        if not candidate_firms or per_tick_budget <= 0:
            return

        # Prioritize firms with worst sales performance
        candidate_firms.sort(key=lambda f: f.last_units_sold / max(f.last_units_produced, 1.0))

        # Loan parameters
        term_ticks = 156  # 3 years (52 ticks/year × 3)
        annual_interest_rate = 0.02  # 2% annual
        interest_multiplier = 1.0 + annual_interest_rate

        for firm in candidate_firms:
            if per_tick_budget <= 0 or self.government.cash_balance <= min_reserve:
                break

            # Loan amount: 10% of firm's current cash balance (scaled)
            loan_amount = firm.cash_balance * 0.10

            # Don't give loans if firm already has outstanding balance > 50% of cash
            if firm.government_loan_remaining > firm.cash_balance * 0.5:
                continue

            # Cap loan at available budget and government reserves
            actual_loan = min(loan_amount, per_tick_budget, self.government.cash_balance - min_reserve)
            if actual_loan <= 100:  # Minimum viable loan
                continue

            # Grant loan
            firm.cash_balance += actual_loan
            total_repayment = actual_loan * interest_multiplier
            firm.government_loan_principal += actual_loan
            firm.government_loan_remaining += total_repayment
            firm.loan_payment_per_tick += total_repayment / term_ticks

            self.government.cash_balance -= actual_loan
            per_tick_budget -= actual_loan

    def _ensure_public_works_capacity(self, unemployment_rate: float) -> None:
        """Stand up or scale public works firms to absorb excess labor."""
        config = CONFIG.government
        if unemployment_rate < config.public_works_unemployment_threshold:
            return

        target_jobs = max(1, int(len(self.households) * config.public_works_job_fraction))
        public_firms = [f for f in self.firms if f.good_category == "PublicWorks"]

        if not public_firms:
            new_firm_id = self._next_firm_id()
            capacity = float(target_jobs * 2)
            public_firm = FirmAgent(
                firm_id=new_firm_id,
                good_name=f"PublicWorks{new_firm_id}",
                cash_balance=CONFIG.government.public_works_job_fraction * 1_000_000.0,
                inventory_units=0.0,
                good_category="PublicWorks",
                quality_level=1.0,
                wage_offer=config.public_works_wage,
                price=config.public_works_price,
                expected_sales_units=float(target_jobs),
                production_capacity_units=capacity,
                units_per_worker=15.0,
                productivity_per_worker=15.0,
                personality="conservative",
                is_baseline=True,
                baseline_production_quota=float(target_jobs)
            )
            public_firm.set_personality("conservative")
            self.firms.append(public_firm)
            self.firm_lookup[new_firm_id] = public_firm
            self.last_tick_sales_units[new_firm_id] = 0.0
            self.last_tick_revenue[new_firm_id] = 0.0
            self.last_tick_sell_through_rate[new_firm_id] = 0.5
            self.last_tick_prices[public_firm.good_name] = public_firm.price
            public_firms = [public_firm]

        per_firm_quota = max(
            config.emergency_loan_min_headcount,
            int(math.ceil(target_jobs / len(public_firms)))
        )
        for firm in public_firms:
            firm.baseline_production_quota = max(float(per_firm_quota), firm.baseline_production_quota)
            firm.expected_sales_units = max(float(per_firm_quota), firm.expected_sales_units)
            firm.production_capacity_units = max(float(per_firm_quota * 2), firm.production_capacity_units)
            firm.price = config.public_works_price
            firm.wage_offer = config.public_works_wage

    def _apply_post_warmup_stimulus(self) -> None:
        """
        Temporary demand boost once the market opens to private firms.

        For a few ticks after warm-up the government sends a per-household
        transfer that decays each tick. This keeps demand alive long enough
        for new firms to record sales and justify hiring.
        """
        if self.post_warmup_stimulus_ticks <= 0 or not self.households:
            return

        duration = max(1, self.post_warmup_stimulus_duration)
        decay_ratio = self.post_warmup_stimulus_ticks / duration
        base_transfer = 40.0  # roughly one week of basic goods
        per_household_transfer = base_transfer * decay_ratio
        total_transfer = per_household_transfer * len(self.households)

        # Governments can run deficits; deduct directly from cash balance
        self.government.cash_balance -= total_transfer
        for household in self.households:
            household.cash_balance += per_household_transfer

        self.post_warmup_stimulus_ticks -= 1

    def _clear_housing_rental_market(self) -> None:
        """
        Match households with housing firms for rental agreements.

        HOUSING RENTAL RULES:
        1. Each household needs exactly 1 housing rental
        2. Households without housing seek rentals
        3. Housing firms try to fill all units
        4. Rent is paid weekly (not one-time purchase)
        5. Households can be evicted if they can't afford rent
        6. Housing firms adjust rent based on occupancy rate

        Mutates state.
        """
        # Get all housing firms
        housing_firms = [f for f in self.firms if f.good_category == "Housing"]

        if not housing_firms:
            return  # No housing available

        # Phase 1: Check affordability and evict households who can't pay
        for household in self.households:
            if household.renting_from_firm_id is not None:
                # Find the housing firm
                housing_firm = next((f for f in housing_firms if f.firm_id == household.renting_from_firm_id), None)

                if housing_firm is None:
                    # Firm no longer exists, evict household
                    household.renting_from_firm_id = None
                    household.monthly_rent = 0.0
                    household.owns_housing = False
                    continue

                # Check if household can afford rent (rent <= 30% of income)
                income = household.wage if household.is_employed else 0.0
                max_affordable_rent = income * 0.30

                if household.monthly_rent > max_affordable_rent or household.cash_balance < household.monthly_rent:
                    # EVICTION: Can't afford rent
                    housing_firm.current_tenants.remove(household.household_id)
                    household.renting_from_firm_id = None
                    household.monthly_rent = 0.0
                    household.owns_housing = False
                    household.happiness = max(0.0, household.happiness - 0.3)  # Happiness penalty for eviction
                else:
                    # Pay rent
                    household.cash_balance -= household.monthly_rent
                    housing_firm.cash_balance += household.monthly_rent
                    household.owns_housing = True

        # Phase 2: Match homeless households with available units
        homeless_households = [h for h in self.households if h.renting_from_firm_id is None]

        # Sort homeless by income (higher income gets priority)
        homeless_households.sort(key=lambda h: h.wage if h.is_employed else 0.0, reverse=True)

        for household in homeless_households:
            income = household.wage if household.is_employed else 0.0
            max_affordable_rent = income * 0.30

            # Find cheapest housing firm with available units that household can afford
            affordable_housing = [
                (f, f.price) for f in housing_firms
                if len(f.current_tenants) < f.max_rental_units and f.price <= max_affordable_rent
            ]

            if affordable_housing:
                # Sort by price (cheapest first)
                affordable_housing.sort(key=lambda x: x[1])
                chosen_firm, rent = affordable_housing[0]

                # Sign rental agreement
                household.renting_from_firm_id = chosen_firm.firm_id
                household.monthly_rent = rent
                household.owns_housing = True
                chosen_firm.current_tenants.append(household.household_id)

                # Pay first month's rent
                household.cash_balance -= rent
                chosen_firm.cash_balance += rent

        total_units = sum(f.max_rental_units for f in housing_firms)
        shortage = total_units < len(self.households)

        # Phase 3: Housing firms adjust rent based on occupancy
        for firm in housing_firms:
            occupancy_rate = len(firm.current_tenants) / max(firm.max_rental_units, 1)

            # Seek equilibrium: raise rent if fully occupied, lower if vacant
            if occupancy_rate >= 0.95:
                # Nearly full - raise rent by 2%
                firm.price *= 1.02
            elif occupancy_rate >= 0.80:
                # Good occupancy - raise rent by 1%
                firm.price *= 1.01
            elif occupancy_rate < 0.50:
                # High vacancy - lower rent by 5%
                firm.price *= 0.95
            elif occupancy_rate < 0.70:
                # Moderate vacancy - lower rent by 2%
                firm.price *= 0.98

            # Ensure rent doesn't drop below $50/week
            firm.price = max(50.0, firm.price)

            if shortage and occupancy_rate >= 0.95 and self.current_tick % 13 == 0:
                firm.price *= 1.05

    def _apply_housing_repairs(self) -> None:
        """Apply random weekly repair costs to housing firms."""
        for firm in self.firms:
            if firm.good_category.lower() != "housing":
                continue
            if firm.max_rental_units <= 0 or firm.price <= 0:
                continue
            repair_rate = np.random.uniform(0.01, 0.05)
            repair_cost = firm.price * firm.max_rental_units * repair_rate
            if repair_cost <= 0:
                continue
            payment = min(firm.cash_balance, repair_cost)
            if payment <= 0:
                continue
            firm.cash_balance -= payment
            self._collect_misc_revenue(payment)

    def _initialize_misc_firm_beneficiaries(self) -> None:
        """Initialize Misc firm with 10-20 random household beneficiaries."""
        if self.households:
            num_beneficiaries = np.random.randint(10, 21)
            num_beneficiaries = min(num_beneficiaries, len(self.households))
            self.misc_firm_beneficiaries = np.random.choice(
                [h.household_id for h in self.households],
                size=num_beneficiaries,
                replace=False
            ).tolist()

    def _misc_firm_add_beneficiary(self) -> None:
        """Each tick, potentially add 1 more random beneficiary."""
        if not self.households:
            return

        # Don't add if we already have 50+ beneficiaries
        if len(self.misc_firm_beneficiaries) >= 50:
            return

        # Find households not already beneficiaries
        non_beneficiaries = [
            h.household_id for h in self.households
            if h.household_id not in self.misc_firm_beneficiaries
        ]

        if non_beneficiaries:
            new_beneficiary = np.random.choice(non_beneficiaries)
            self.misc_firm_beneficiaries.append(new_beneficiary)

    def _misc_firm_redistribute_revenue(self) -> None:
        """
        Distribute all accumulated Misc firm revenue to beneficiaries.

        The Misc firm collects:
        - R&D spending from firms
        - Investment spending from government
        - Other "dead money" that would leave the economy

        It then redistributes ALL revenue equally to beneficiaries.
        """
        if self.misc_firm_revenue <= 0 or not self.misc_firm_beneficiaries:
            return

        # Distribute equally among beneficiaries
        payout_per_household = self.misc_firm_revenue / len(self.misc_firm_beneficiaries)

        for hid in self.misc_firm_beneficiaries:
            household = self.household_lookup.get(hid)
            if household:
                household.cash_balance += payout_per_household

        # Reset revenue to 0 after payout
        self.misc_firm_revenue = 0.0

    def _collect_misc_revenue(self, amount: float) -> None:
        """
        Route spending into the misc pool with a variable tax skim.

        A random fraction (0-20%) is collected as tax, the rest
        is accumulated as misc_firm_revenue for redistribution.
        """
        if amount <= 0:
            return

        import random
        rng = random.Random(self.current_tick + 1234)
        tax_rate = rng.uniform(0.0, 0.20)
        tax = amount * tax_rate
        net = amount - tax
        if tax > 0:
            self.government.cash_balance += tax
        if net > 0:
            self.misc_firm_revenue += net

    def _process_healthcare_and_loans(self) -> None:
        """
        Process healthcare spending decisions and medical loans for households.

        Each household evaluates if they need healthcare:
        - If health < 70%: probabilistically seek care
        - If cost > cash: take medical loan (1-3% annual interest, 10% wage repayment)
        - Healthcare spending goes to Services category firms

        Mutates state.
        """
        for household in self.households:
            # Check if household needs healthcare
            should_spend, amount, needs_loan = household.should_spend_on_healthcare()

            if not should_spend:
                continue

            # If household needs a loan, grant it
            if needs_loan:
                # Calculate loan amount (shortfall between cost and cash)
                affordable_cash = max(0.0, household.cash_balance - household.cash_balance * 0.10)
                loan_amount = max(0.0, amount - affordable_cash)

                if loan_amount > 0:
                    household.take_medical_loan(loan_amount)

            # Spend on healthcare (deduct from cash, add to misc firm revenue)
            if amount > 0 and household.cash_balance >= amount:
                household.cash_balance -= amount
                self._collect_misc_revenue(amount)

                # Improve health based on spending
                # Cost was calculated for specific recovery, so apply that recovery
                # Recovery is roughly: amount / (base_cost_per_percent * 100 * exponential_factor)
                # Simplified: Each $200 recovers ~1% health
                health_recovery = min(0.30, amount / 20000.0)  # Cap at 30% recovery
                household.health = min(1.0, household.health + health_recovery)

    def _adjust_government_policy(self) -> None:
        """
        Calculate economic indicators and adjust government policy.

        Mutates state.
        """
        # Calculate unemployment rate
        total_households = len(self.households)
        if total_households == 0:
            return

        unemployed = sum(1 for h in self.households if not h.is_employed)
        unemployment_rate = unemployed / total_households

        # Calculate inflation (change in average price level)
        if self.last_tick_prices:
            avg_current_price = sum(self.last_tick_prices.values()) / len(self.last_tick_prices)
            # Store previous for inflation calc (simplified - would track over time)
            inflation_rate = 0.0  # Placeholder - would need historical prices
        else:
            inflation_rate = 0.0

        # Calculate deficit ratio
        total_gdp = sum(self.last_tick_revenue.values()) if self.last_tick_revenue else 1.0
        deficit_ratio = abs(self.government.cash_balance) / max(total_gdp, 1.0)

        bankruptcies = sum(1 for f in self.firms if f.cash_balance < 0.0)
        total_tax_revenue = total_gdp * self.government.profit_tax_rate

        self.government.adjust_policies(
            unemployment_rate,
            inflation_rate,
            deficit_ratio,
            num_unemployed=unemployed,
            gdp=total_gdp,
            total_tax_revenue=total_tax_revenue,
            num_bankrupt_firms=bankruptcies
        )

    def _update_statistics(self, per_firm_sales: Dict[int, Dict[str, float]]) -> None:
        """
        Update world-level statistics for next tick.

        Args:
            per_firm_sales: Sales data from this tick
        """
        # Update firm-level stats
        for firm in self.firms:
            sales_data = per_firm_sales.get(firm.firm_id, {"units_sold": 0.0, "revenue": 0.0})

            self.last_tick_sales_units[firm.firm_id] = sales_data["units_sold"]
            self.last_tick_revenue[firm.firm_id] = sales_data["revenue"]

            # Compute sell-through rate
            units_sold = sales_data["units_sold"]
            ending_inventory = firm.inventory_units
            total_available = max(units_sold + ending_inventory, 1.0)
            sell_through_rate = units_sold / total_available

            self.last_tick_sell_through_rate[firm.firm_id] = sell_through_rate

        # Update prices by good (simple approach: use current firm prices)
        # Could be quantity-weighted if multiple firms per good
        good_prices: Dict[str, List[float]] = {}
        for firm in self.firms:
            if firm.good_name not in good_prices:
                good_prices[firm.good_name] = []
            good_prices[firm.good_name].append(firm.price)

        # Average price per good (deterministic)
        for good_name, prices in good_prices.items():
            self.last_tick_prices[good_name] = sum(prices) / len(prices)

    def get_economic_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive economic metrics for monitoring and display.

        Returns:
            Dictionary with economic indicators including GDP, unemployment,
            wages, firm metrics, household metrics, and government finances.
        """
        metrics = {}

        # Household metrics
        if self.households:
            employed_households = [h for h in self.households if h.is_employed]
            unemployed_households = [h for h in self.households if not h.is_employed]

            metrics["total_households"] = len(self.households)
            metrics["employed_count"] = len(employed_households)
            metrics["unemployed_count"] = len(unemployed_households)
            metrics["unemployment_rate"] = len(unemployed_households) / len(self.households)

            # Wage statistics
            if employed_households:
                wages = [h.wage for h in employed_households]
                metrics["mean_wage"] = sum(wages) / len(wages)
                metrics["median_wage"] = sorted(wages)[len(wages) // 2]
                metrics["min_wage"] = min(wages)
                metrics["max_wage"] = max(wages)
            else:
                metrics["mean_wage"] = 0.0
                metrics["median_wage"] = 0.0
                metrics["min_wage"] = 0.0
                metrics["max_wage"] = 0.0

            # Household cash/wealth
            household_cash = [h.cash_balance for h in self.households]
            metrics["total_household_cash"] = sum(household_cash)
            metrics["mean_household_cash"] = sum(household_cash) / len(household_cash)
            metrics["median_household_cash"] = sorted(household_cash)[len(household_cash) // 2]

            # Wellbeing metrics
            metrics["mean_happiness"] = sum(h.happiness for h in self.households) / len(self.households)
            metrics["mean_morale"] = sum(h.morale for h in self.households) / len(self.households)
            metrics["mean_health"] = sum(h.health for h in self.households) / len(self.households)

            # Skills
            metrics["mean_skills"] = sum(h.skills_level for h in self.households) / len(self.households)
        else:
            metrics.update({
                "total_households": 0, "employed_count": 0, "unemployed_count": 0,
                "unemployment_rate": 0.0, "mean_wage": 0.0, "median_wage": 0.0,
                "min_wage": 0.0, "max_wage": 0.0, "total_household_cash": 0.0,
                "mean_household_cash": 0.0, "median_household_cash": 0.0,
                "mean_happiness": 0.0, "mean_morale": 0.0, "mean_health": 0.0,
                "mean_skills": 0.0
            })

        # Firm metrics
        if self.firms:
            firm_cash = [f.cash_balance for f in self.firms]
            metrics["total_firms"] = len(self.firms)
            metrics["total_firm_cash"] = sum(firm_cash)
            metrics["mean_firm_cash"] = sum(firm_cash) / len(firm_cash)
            metrics["median_firm_cash"] = sorted(firm_cash)[len(firm_cash) // 2]

            # Inventory
            total_inventory = sum(f.inventory_units for f in self.firms)
            metrics["total_firm_inventory"] = total_inventory

            # Employees
            total_employees = sum(len(f.employees) for f in self.firms)
            metrics["total_employees"] = total_employees

            # Prices
            prices = [f.price for f in self.firms]
            metrics["mean_price"] = sum(prices) / len(prices)
            metrics["median_price"] = sorted(prices)[len(prices) // 2]

            # Quality
            qualities = [f.quality_level for f in self.firms]
            metrics["mean_quality"] = sum(qualities) / len(qualities)
        else:
            metrics.update({
                "total_firms": 0, "total_firm_cash": 0.0, "mean_firm_cash": 0.0,
                "median_firm_cash": 0.0, "total_firm_inventory": 0.0,
                "total_employees": 0, "mean_price": 0.0, "median_price": 0.0,
                "mean_quality": 0.0
            })

        # GDP calculation (sum of all firm revenues this tick)
        metrics["gdp_this_tick"] = sum(self.last_tick_revenue.values())

        # Government metrics
        metrics["government_cash"] = self.government.cash_balance
        metrics["wage_tax_rate"] = self.government.wage_tax_rate
        metrics["profit_tax_rate"] = self.government.profit_tax_rate
        metrics["unemployment_benefit"] = self.government.unemployment_benefit_level
        metrics["transfer_budget"] = self.government.transfer_budget

        # Infrastructure multipliers
        metrics["infrastructure_productivity"] = self.government.infrastructure_productivity_multiplier
        metrics["technology_quality"] = self.government.technology_quality_multiplier
        metrics["social_happiness"] = self.government.social_happiness_multiplier

        # Total wealth in economy
        metrics["total_economy_cash"] = (
            metrics["total_household_cash"] +
            metrics["total_firm_cash"] +
            metrics["government_cash"]
        )

        # Current tick
        metrics["current_tick"] = self.current_tick

        return metrics
