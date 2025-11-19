"""
Economy Simulation Engine

This module implements the main simulation coordinator that orchestrates
households, firms, and government through deterministic tick-based cycles.

All behavior is deterministic - no randomness, I/O, or side effects.
"""

from typing import Dict, List, Tuple
from agents import HouseholdAgent, FirmAgent, GovernmentAgent


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
        government: GovernmentAgent
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

        # Track simulation progression and warm-up period state
        self.current_tick = 0
        self.in_warmup = True

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
        self.in_warmup = self.current_tick < 52

        # Phase 1: Firms plan
        firm_production_plans = {}
        firm_price_plans = {}
        firm_wage_plans = {}

        for firm in self.firms:
            # Plan production and labor
            production_plan = firm.plan_production_and_labor(
                self.last_tick_sales_units.get(firm.firm_id, 0.0),
                in_warmup=self.in_warmup
            )
            firm_production_plans[firm.firm_id] = production_plan

            # Plan pricing
            price_plan = firm.plan_pricing(
                self.last_tick_sell_through_rate.get(firm.firm_id, 0.5),
                in_warmup=self.in_warmup
            )
            firm_price_plans[firm.firm_id] = price_plan

            # Plan wage
            wage_plan = firm.plan_wage()
            firm_wage_plans[firm.firm_id] = wage_plan

        # Phase 2: Households plan
        gov_benefit = self.government.get_unemployment_benefit_level()
        household_labor_plans = {}
        household_consumption_plans = {}

        category_market_snapshot = self._build_category_market_snapshot()

        for household in self.households:
            # Plan labor supply
            labor_plan = household.plan_labor_supply(gov_benefit)
            household_labor_plans[household.household_id] = labor_plan

            # Plan consumption
            consumption_plan = household.plan_consumption(
                self.last_tick_prices,
                firm_market_info=category_market_snapshot
            )
            household_consumption_plans[household.household_id] = consumption_plan

        # Phase 3: Labor market matching
        firm_labor_outcomes, household_labor_outcomes = self._match_labor(
            firm_production_plans,
            firm_wage_plans,
            household_labor_plans
        )

        # Phase 4: Apply labor outcomes
        for firm in self.firms:
            firm.apply_labor_outcome(firm_labor_outcomes[firm.firm_id])

        for household in self.households:
            household.apply_labor_outcome(household_labor_outcomes[household.household_id])

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

            # Get price ceiling tax from firm snapshots
            price_ceiling_tax = 0.0
            for snapshot in firm_tax_snapshots:
                if snapshot["firm_id"] == firm.firm_id:
                    price_ceiling_tax = snapshot.get("price_ceiling_tax", 0.0)
                    break

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
        for household in self.households:
            # Compute wage income
            wage_income = household.wage if household.is_employed else 0.0

            # Get transfer and tax amounts
            transfers = transfer_plan.get(household.household_id, 0.0)
            taxes_paid = tax_plan["wage_taxes"].get(household.household_id, 0.0)

            # Apply income and taxes
            household.apply_income_and_taxes({
                "wage_income": wage_income,
                "transfers": transfers,
                "taxes_paid": taxes_paid
            })

            # Apply purchases
            purchases = per_household_purchases.get(household.household_id, {})
            household.apply_purchases(purchases)

            # Consume goods from inventory
            household.consume_goods()

        # Phase 11: Apply government fiscal results
        total_wage_taxes = sum(tax_plan["wage_taxes"].values())
        total_profit_taxes = sum(tax_plan["profit_taxes"].values())
        total_price_ceiling_taxes = sum(snapshot.get("price_ceiling_tax", 0.0) for snapshot in firm_tax_snapshots)
        total_transfers = sum(transfer_plan.values())

        self.government.apply_fiscal_results(
            total_wage_taxes,
            total_profit_taxes + total_price_ceiling_taxes,  # Include price ceiling tax as profit tax
            total_transfers
        )

        # Phase 11.5: Government makes investments in infrastructure, technology, and social programs
        self.government.make_investments()

        # Phase 11.75: Update household wellbeing (happiness, morale, health)
        for household in self.households:
            household.update_wellbeing(
                government_happiness_multiplier=self.government.social_happiness_multiplier
            )

        # Phase 12: Handle firm bankruptcies and exits
        self._handle_firm_exits()

        # Phase 13: Potentially create new firms
        self._maybe_create_new_firms()

        # Phase 14: Government adjusts policies based on economic conditions
        self._adjust_government_policy()

        # Phase 15: Update world-level statistics
        self._update_statistics(per_firm_sales)

        # Advance simulation clock after completing the tick
        self.current_tick += 1

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
        firm_lookup = {firm.firm_id: firm for firm in self.firms}
        planned_layoffs_set = set()
        for plan in firm_production_plans.values():
            planned_layoffs_set.update(plan.get("planned_layoffs_ids", []))

        for household in self.households:
            if household.is_employed and household.household_id not in planned_layoffs_set:
                employer_id = household.employer_id
                employer_category = None
                if employer_id is not None and employer_id in firm_lookup:
                    employer_category = firm_lookup[employer_id].good_category
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

            # Find eligible candidates
            eligible_candidates = []
            for household_id, labor_plan in household_labor_plans.items():
                # Skip if already assigned
                if household_id in assigned_households:
                    continue

                # Check eligibility
                if (labor_plan["searching_for_job"] and
                    wage_offer >= labor_plan["reservation_wage"]):
                    eligible_candidates.append({
                        "household_id": household_id,
                        "skills_level": labor_plan["skills_level"]
                    })

            # Sort candidates by skills (descending), then household_id (ascending)
            eligible_candidates.sort(
                key=lambda c: (-c["skills_level"], c["household_id"])
            )

            # Assign up to vacancies
            hired_count = min(vacancies, len(eligible_candidates))
            for i in range(hired_count):
                household_id = eligible_candidates[i]["household_id"]
                skills_level = eligible_candidates[i]["skills_level"]

                # Get household to check experience
                household = next(h for h in self.households if h.household_id == household_id)

                # Calculate skill premium (50% max for skill level 1.0)
                skill_premium = skills_level * 0.5

                # Calculate experience premium (5% per year, capped at 50%)
                # Assume 52 ticks per year
                experience_ticks = household.category_experience.get(firm.good_category, 0)
                experience_years = experience_ticks / 52.0
                experience_premium = min(experience_years * 0.05, 0.5)

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
        per_household_purchases = {}
        per_firm_sales = {}

        # Initialize firm sales
        for firm in firms:
            per_firm_sales[firm.firm_id] = {
                "units_sold": 0.0,
                "revenue": 0.0
            }

        # Group firms by good_name and build lookup by id
        goods_to_firms: Dict[str, List[FirmAgent]] = {}
        firm_lookup_by_id: Dict[int, FirmAgent] = {}
        for firm in firms:
            firm_lookup_by_id[firm.firm_id] = firm
            if firm.good_name not in goods_to_firms:
                goods_to_firms[firm.good_name] = []
            goods_to_firms[firm.good_name].append(firm)

        # Sort firms within each good by price (ascending), then firm_id (ascending)
        for good_name in goods_to_firms:
            goods_to_firms[good_name].sort(
                key=lambda f: (f.price, f.firm_id)
            )

        # Track remaining inventory per firm
        firm_remaining_inventory = {
            firm.firm_id: firm.inventory_units for firm in firms
        }

        # Process households in sorted order by household_id
        sorted_households = sorted(
            household_consumption_plans.items(),
            key=lambda item: item[0]
        )

        for household_id, consumption_plan in sorted_households:
            per_household_purchases[household_id] = {}
            planned_purchases = consumption_plan["planned_purchases"]

            # Process each desired purchase (firm_id or good name)
            for purchase_target, desired_quantity in planned_purchases.items():
                if desired_quantity <= 0:
                    continue

                if isinstance(purchase_target, int):
                    firm = firm_lookup_by_id.get(purchase_target)
                    if firm is None:
                        continue

                    available = firm_remaining_inventory.get(firm.firm_id, 0.0)
                    if available <= 0:
                        continue

                    quantity_to_buy = min(desired_quantity, available)
                    if quantity_to_buy <= 0:
                        continue

                    firm_remaining_inventory[firm.firm_id] -= quantity_to_buy
                    per_firm_sales[firm.firm_id]["units_sold"] += quantity_to_buy
                    per_firm_sales[firm.firm_id]["revenue"] += quantity_to_buy * firm.price

                    prev_qty, prev_price = per_household_purchases[household_id].get(
                        firm.good_name,
                        (0.0, 0.0)
                    )
                    total_qty = prev_qty + quantity_to_buy
                    if total_qty > 0:
                        avg_price = (
                            (prev_qty * prev_price) + (quantity_to_buy * firm.price)
                        ) / total_qty
                        per_household_purchases[household_id][firm.good_name] = (
                            total_qty,
                            avg_price
                        )
                    continue

                good_name = purchase_target
                if good_name not in goods_to_firms:
                    continue

                remaining_demand = desired_quantity
                total_quantity_bought = 0.0
                weighted_price_sum = 0.0

                for firm in goods_to_firms[good_name]:
                    if remaining_demand <= 0:
                        break

                    available = firm_remaining_inventory[firm.firm_id]
                    if available <= 0:
                        continue

                    quantity_to_buy = min(remaining_demand, available)
                    firm_remaining_inventory[firm.firm_id] -= quantity_to_buy
                    per_firm_sales[firm.firm_id]["units_sold"] += quantity_to_buy
                    per_firm_sales[firm.firm_id]["revenue"] += quantity_to_buy * firm.price

                    total_quantity_bought += quantity_to_buy
                    weighted_price_sum += quantity_to_buy * firm.price
                    remaining_demand -= quantity_to_buy

                if total_quantity_bought > 0:
                    effective_price = weighted_price_sum / total_quantity_bought
                    per_household_purchases[household_id][good_name] = (
                        total_quantity_bought,
                        effective_price
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
        actual_production = min(
            planned_production_units * avg_productivity_multiplier,
            firm.production_capacity_units
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

        firms_to_remove = []
        for firm in self.firms:
            if firm.cash_balance < bankruptcy_threshold:
                # Protect government baseline firms during warm-up
                if self.in_warmup and self.government.is_baseline_firm(firm.firm_id):
                    continue

                # Firm is bankrupt - lay off all employees
                for employee_id in firm.employees:
                    # Find household and unemploy them
                    household = next((h for h in self.households if h.household_id == employee_id), None)
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

        baseline_count = len(self.government.baseline_firm_ids)
        max_firms = baseline_count + 5  # Allow a handful of private competitors
        if len(self.firms) >= max_firms:
            return

        # Check if there's economic activity (households have cash)
        total_household_cash = sum(h.cash_balance for h in self.households)
        if total_household_cash < 1000.0:
            return  # Not enough demand to support new firms

        # Create a new firm
        new_firm_id = max([f.firm_id for f in self.firms], default=0) + 1

        # Choose a random category from existing firms or default categories
        categories = ["Food", "Housing", "Services"]
        if self.firms:
            # Prefer categories that are underrepresented
            category_counts = {}
            for cat in categories:
                category_counts[cat] = sum(1 for f in self.firms if f.good_category == cat)
            # Pick category with fewest firms
            chosen_category = min(category_counts, key=category_counts.get)
        else:
            chosen_category = "Food"  # Default for first firm

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

        new_firm = FirmAgent(
            firm_id=new_firm_id,
            good_name=f"{chosen_category}Product{new_firm_id}",
            cash_balance=2000.0,  # Minimal starting capital
            inventory_units=25.0,
            good_category=chosen_category,
            quality_level=min(10.0, base_quality),  # Cap at 10.0
            wage_offer=35.0,  # Lower initial wage
            price=8.0,
            expected_sales_units=20.0,
            production_capacity_units=200.0,
            productivity_per_worker=8.0,
            units_per_worker=15.0,
            personality=personality
        )

        # Set personality-based behavior parameters
        new_firm.set_personality(personality)

        self.firms.append(new_firm)

        # Initialize tracking for new firm
        self.last_tick_sales_units[new_firm_id] = 0.0
        self.last_tick_revenue[new_firm_id] = 0.0
        self.last_tick_sell_through_rate[new_firm_id] = 0.5
        self.last_tick_prices[new_firm.good_name] = new_firm.price

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

        # Adjust policies
        self.government.adjust_policies(unemployment_rate, inflation_rate, deficit_ratio)

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
