"""
EcoSim Agent System

This module defines the autonomous agents that will run the economic simulation.
These agents will eventually replace the current recommendation system and actively
make decisions to drive the simulation forward.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _get_good_category(good_name: str, good_categories: Optional[Dict[str, str]] = None) -> str:
    """Best-effort inference of a good's category (defaults to lowercased name)."""
    if good_categories:
        category = good_categories.get(good_name)
        if category:
            return category.lower()

    lowered = good_name.lower()
    if "housing" in lowered:
        return "housing"

    return lowered


@dataclass(slots=True)
class HouseholdAgent:
    """
    Represents a household in the economic simulation.

    Households work for firms, consume goods, and form expectations
    about prices and wages. All behavior is deterministic.
    """

    # Identification and traits
    household_id: int
    skills_level: float  # 0.0 to 1.0, used in hiring
    age: int

    # Economic state
    cash_balance: float
    goods_inventory: Dict[str, float] = field(default_factory=dict)
    employer_id: Optional[int] = None
    wage: float = 0.0
    ticks: int=1
    owns_housing: bool = False  # Track if household already owns housing
    # Preferences and heuristics
    consumption_budget_share: float = 0.7  # Legacy field (overridden by savings_rate_target if set)
    good_weights: Dict[str, float] = field(default_factory=dict)  # DEPRECATED: use category_weights
    category_weights: Dict[str, float] = field(default_factory=dict)  # category -> share of budget
    savings_rate_target: Optional[float] = None  # long-run desired savings share [0,1]
    default_purchase_style: str = "value"
    purchase_styles: Dict[str, str] = field(default_factory=dict)  # category -> cheap/value/quality

    # Quality/price preferences
    quality_preference_weight: float = 1.0  # elasticity for quality in purchase decisions
    price_sensitivity: float = 1.0  # elasticity for price in purchase decisions

    # Experience tracking
    category_experience: Dict[str, int] = field(default_factory=dict)  # category -> ticks worked

    # Expectations and beliefs
    price_beliefs: Dict[str, float] = field(default_factory=dict)
    expected_wage: float = 10.0  # initial default wage expectation
    reservation_wage: float = 8.0  # minimum acceptable wage
    #wage_increase: float=5*ticks

    # Config / tuning parameters
    price_expectation_alpha: float = 0.3  # [0,1] for price smoothing
    wage_expectation_alpha: float = 0.2  # [0,1] for wage smoothing
    reservation_markup_over_benefit: float = 1.1  # reservation = benefit * markup
    default_price_level: float = 10.0  # fallback when no price history
    min_cash_for_aggressive_job_search: float = 100.0  # threshold for wage flexibility

    # Skill development
    skill_growth_rate: float = 0.001  # base skill improvement per tick when employed
    education_cost_per_skill_point: float = 1000.0  # cost to improve skill by 0.1

    # Wellbeing and performance factors
    happiness: float = 0.7  # 0-1 scale, affects productivity and consumption
    morale: float = 0.7  # 0-1 scale, affects work performance
    health: float = 1.0  # 0-1 scale, affects productivity and skill development
    unemployment_duration: int = 0  # consecutive ticks without employment

    # Wellbeing dynamics
    happiness_decay_rate: float = 0.01  # Happiness naturally decays without maintenance
    morale_decay_rate: float = 0.02  # Morale decays faster than happiness
    health_decay_rate: float = 0.005  # Health decays slowly over time

    # Minimum consumption requirements per tick
    min_food_per_tick: float = 2.0  # Minimum food units needed per tick
    min_services_per_tick: float = 1.0  # Minimum services units needed per tick
    met_housing_need: bool = False  # Track if housing service was consumed this tick
    spending_tendency: float = 1.0  # Multiplier for overall spend appetite
    food_preference: float = 1.0
    services_preference: float = 1.0
    housing_preference: float = 1.0
    quality_lavishness: float = 1.0
    frugality: float = 1.0  # Higher = saves more
    saving_tendency: float = 0.5  # Innate thriftiness [0.0, 1.0], initialized randomly in __post_init__

    def __post_init__(self):
        """Validate invariants after initialization."""
        if not (0.0 <= self.consumption_budget_share <= 1.0):
            raise ValueError(
                f"consumption_budget_share must be in [0,1], got {self.consumption_budget_share}"
            )
        self._initialize_personality_preferences()

        if not (0.0 <= self.savings_rate_target <= 1.0):
            raise ValueError(
                f"savings_rate_target must be in [0,1], got {self.savings_rate_target}"
            )
        if not (0.0 <= self.price_expectation_alpha <= 1.0):
            raise ValueError(
                f"price_expectation_alpha must be in [0,1], got {self.price_expectation_alpha}"
            )
        if not (0.0 <= self.wage_expectation_alpha <= 1.0):
            raise ValueError(
                f"wage_expectation_alpha must be in [0,1], got {self.wage_expectation_alpha}"
            )
        if not (0.0 <= self.skills_level <= 1.0):
            raise ValueError(f"skills_level must be in [0,1], got {self.skills_level}")
        if self.age < 0:
            raise ValueError(f"age cannot be negative, got {self.age}")
        for good, quantity in self.goods_inventory.items():
            if quantity > 0 and _get_good_category(good) == "housing":
                self.owns_housing = True
                break

    def _initialize_personality_preferences(self) -> None:
        """Deterministically assign savings, weights, and purchase styles."""
        if self.savings_rate_target is None:
            bucket = (self.household_id % 6) / 10
            self.savings_rate_target = 0.1 + bucket
        self.savings_rate_target = max(0.1, min(0.6, self.savings_rate_target))

        # Traits: deterministic pseudo-random
        rng = random.Random(self.household_id * 9973)
        self.spending_tendency = rng.uniform(0.7, 1.3)
        self.food_preference = rng.uniform(0.8, 1.2)
        self.services_preference = rng.uniform(0.8, 1.2)
        self.housing_preference = rng.uniform(0.9, 1.1)
        self.quality_lavishness = rng.uniform(0.8, 1.3)
        self.frugality = rng.uniform(0.7, 1.3)
        # Initialize saving_tendency as innate thriftiness [0.0, 1.0]
        self.saving_tendency = rng.random()  # Uniform [0.0, 1.0]

        if not self.category_weights:
            base_categories = ["food", "housing", "services"]
            self.category_weights = {cat: 1.0 / len(base_categories) for cat in base_categories}
        biased_weights = {
            "food": self.category_weights.get("food", 0.0) * self.food_preference,
            "housing": self.category_weights.get("housing", 0.0) * self.housing_preference,
            "services": self.category_weights.get("services", 0.0) * self.services_preference,
        }
        self.category_weights = self._normalize_category_weights(biased_weights)

        if not self.purchase_styles:
            style_options = ["cheap", "value", "quality"]
            base_offset = self.household_id % len(style_options)
            for idx, category in enumerate(sorted(self.category_weights.keys())):
                style = style_options[(base_offset + idx) % len(style_options)]
                self.purchase_styles[category] = style
        self.purchase_styles = {
            category.lower(): self.purchase_styles[category].lower()
            for category in self.purchase_styles
        }
        self.default_purchase_style = self.default_purchase_style.lower()

    def _normalize_category_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        total = 0.0
        for category, weight in weights.items():
            weight = max(0.0, weight)
            if weight <= 0:
                continue
            category_key = category.lower()
            normalized[category_key] = normalized.get(category_key, 0.0) + weight
            total += weight

        if total <= 0:
            fallback_categories = ["food", "housing", "services"]
            normalized = {cat: 1.0 / len(fallback_categories) for cat in fallback_categories}
            total = 1.0

        return {category: weight / total for category, weight in normalized.items()}

    def _get_affordability_score(self) -> float:
        """
        Calculate a normalized affordability score based on skills, cash, and wages.

        Returns:
            Float in [0.1, 4.0] representing how flexible the household can be on prices.
        """
        wage_basis = self.wage if self.wage > 0 else self.expected_wage
        skill_component = self.skills_level * 1.5
        cash_component = min(3.0, self.cash_balance / 400.0)
        wage_component = min(3.0, wage_basis / 40.0)

        score = 0.3 * skill_component + 0.35 * cash_component + 0.35 * wage_component
        return max(0.1, min(4.0, score))

    def _get_category_price_cap(
        self,
        category: str,
        options: List[Dict[str, float]]
    ) -> float:
        """
        Determine the maximum acceptable price for a category this tick.
        """
        prices = [opt.get("price", 0.0) for opt in options if opt.get("price", 0.0) > 0]
        if not prices:
            return 0.0

        prices.sort()
        min_price = prices[0]
        max_price = prices[-1]
        median_price = prices[len(prices) // 2]

        affordability = self._get_affordability_score()
        wage_basis = self.wage if self.wage > 0 else self.expected_wage
        liquid_cash = max(25.0, self.cash_balance * 0.2 + wage_basis)

        base_cap = min_price * (1.2 + 2.5 * affordability)
        median_cap = median_price * (0.8 + affordability)
        premium_cap = max_price * min(affordability, 2.5)

        price_cap = max(base_cap, median_cap, premium_cap)
        price_cap = min(price_cap, liquid_cash)

        if affordability > 2.0:
            price_cap = max(price_cap, min(liquid_cash * 1.2, max_price))

        price_cap *= self.quality_lavishness

        return max(min_price * 1.1, price_cap)

    def _plan_category_purchases(
        self,
        budget: float,
        firm_market_info: Dict[str, List[Dict[str, float]]]
    ) -> Dict[int, float]:
        """
        Plan purchases using budget allocations influenced by preferences/traits.
        """
        planned: Dict[int, float] = {}

        biased = {
            "food": self.category_weights.get("food", 0.0) * self.food_preference,
            "housing": self.category_weights.get("housing", 0.0) * self.housing_preference,
            "services": self.category_weights.get("services", 0.0) * self.services_preference,
        }
        total_bias = sum(biased.values())
        if total_bias <= 0:
            return planned

        for category, bias_weight in biased.items():
            if bias_weight <= 0:
                continue
            options = firm_market_info.get(category, [])
            if not options:
                continue

            price_cap = self._get_category_price_cap(category, options)
            if price_cap <= 0:
                continue

            affordable_options = [
                option for option in options
                if 0 < option.get("price", 0.0) <= price_cap
            ]
            if not affordable_options:
                continue

            style = self.purchase_styles.get(category, self.default_purchase_style)
            chosen = self._choose_firm_based_on_style(affordable_options, style)
            if chosen is None or chosen.get("price", 0.0) <= 0:
                continue

            category_budget = budget * (bias_weight / total_bias)
            quantity = category_budget / chosen["price"]

            if quantity <= 0:
                continue

            cap_ratio = chosen["price"] / price_cap
            if cap_ratio > 0.85:
                sensitivity = max(0.2, min(1.5, self.price_sensitivity))
                scale = 1.0 - sensitivity * (cap_ratio - 0.85) * 3.0
                quantity *= max(0.15, scale)

            firm_id = chosen["firm_id"]
            planned[firm_id] = planned.get(firm_id, 0.0) + quantity

        return planned


    def _choose_firm_based_on_style(
        self,
        options: List[Dict[str, float]],
        style: str
    ) -> Optional[Dict[str, float]]:
        if not options:
            return None

        style = style.lower()
        cheapest = None
        quality_best = None
        value_best = None
        best_value_ratio = -1.0

        for firm in options:
            price = firm.get("price", 0.0)
            quality = firm.get("quality", 0.0)
            if price <= 0:
                continue

            if (cheapest is None or price < cheapest.get("price", float("inf")) or (
                price == cheapest.get("price", float("inf")) and quality > cheapest.get("quality", 0.0)
            )):
                cheapest = firm

            if (quality_best is None or quality > quality_best.get("quality", 0.0) or (
                quality == quality_best.get("quality", 0.0) and price < quality_best.get("price", float("inf"))
            )):
                quality_best = firm

            value_ratio = quality / price if price > 0 else 0.0
            if value_ratio > best_value_ratio:
                best_value_ratio = value_ratio
                value_best = firm

        if style == "cheap" and cheapest is not None:
            return cheapest
        if style == "quality" and quality_best is not None:
            return quality_best
        if style == "value" and value_best is not None:
            return value_best

        return cheapest or quality_best or value_best or (options[0] if options else None)

    @property
    def is_employed(self) -> bool:
        """Check if household is currently employed."""
        return self.employer_id is not None

    def to_dict(self) -> Dict[str, object]:
        """
        Serialize all fields to basic Python types.

        Returns:
            Dictionary representation of the household state
        """
        return {
            "household_id": self.household_id,
            "skills_level": self.skills_level,
            "age": self.age,
            "cash_balance": self.cash_balance,
            "goods_inventory": dict(self.goods_inventory),
            "employer_id": self.employer_id,
            "wage": self.wage,
            "owns_housing": self.owns_housing,
            "met_housing_need": self.met_housing_need,
            "spending_tendency": self.spending_tendency,
            "food_preference": self.food_preference,
            "services_preference": self.services_preference,
            "housing_preference": self.housing_preference,
            "quality_lavishness": self.quality_lavishness,
            "frugality": self.frugality,
            "consumption_budget_share": self.consumption_budget_share,
            "good_weights": dict(self.good_weights),
            "category_weights": dict(self.category_weights),
            "savings_rate_target": self.savings_rate_target,
            "purchase_styles": dict(self.purchase_styles),
            "quality_preference_weight": self.quality_preference_weight,
            "price_sensitivity": self.price_sensitivity,
            "category_experience": dict(self.category_experience),
            "price_beliefs": dict(self.price_beliefs),
            "expected_wage": self.expected_wage,
            "reservation_wage": self.reservation_wage,
            "price_expectation_alpha": self.price_expectation_alpha,
            "wage_expectation_alpha": self.wage_expectation_alpha,
            "reservation_markup_over_benefit": self.reservation_markup_over_benefit,
            "default_price_level": self.default_price_level,
            "min_cash_for_aggressive_job_search": self.min_cash_for_aggressive_job_search,
            "min_food_per_tick": self.min_food_per_tick,
            "min_services_per_tick": self.min_services_per_tick,
        }

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        """
        Apply external overrides to household state.

        Useful for UI or script-driven state modifications.

        Args:
            overrides: Dictionary of attribute names to new values
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def plan_labor_supply(self, unemployment_benefit: float) -> Dict[str, object]:
        """
        Decide whether to search for job and what wage to require.

        Does not mutate state; returns a plan dict.

        Args:
            unemployment_benefit: Government support for unemployed

        Returns:
            Dict with household_id, searching_for_job, reservation_wage, skills_level
        """
        market_value = self.skills_level * 20.0

        # Baseline reservation from government support
        baseline_reservation = unemployment_benefit * self.reservation_markup_over_benefit

        # Minimum living cost: housing + food
        housing_price = self.price_beliefs.get("housing", self.default_price_level)
        food_price = self.price_beliefs.get("food", self.default_price_level)
        living_cost = housing_price + self.min_food_per_tick * food_price

        # Adjust based on cash position
        if self.cash_balance < self.min_cash_for_aggressive_job_search:
            # Desperate: willing to accept less
            adjustment_factor = 0.85  # deterministic downward adjustment
            reservation_wage_for_tick = baseline_reservation * adjustment_factor
        else:
            # Comfortable: can be pickier, nudge toward expected wage
            reservation_wage_for_tick = 0.7 * baseline_reservation + 0.3 * self.expected_wage

        if self.cash_balance < 200:
            desperation = 0.6
            reservation_wage_for_tick = min(self.expected_wage, market_value) * desperation
            reservation_wage_for_tick = max(reservation_wage_for_tick, 1.0)

        # Ensure ability to cover living costs when cash is low
        if self.cash_balance < living_cost:
            reservation_wage_for_tick = max(reservation_wage_for_tick, living_cost)
            searching_for_job = True
        else:
            searching_for_job = not self.is_employed

        # Decide if searching
        return {
            "household_id": self.household_id,
            "searching_for_job": searching_for_job,
            "reservation_wage": reservation_wage_for_tick,
            "skills_level": self.skills_level,
        }

    def compute_saving_rate(self) -> float:
        """
        Compute the saving rate as a fraction of income (0.0 to 0.15).

        The saving rate is based on:
        1. Household's innate saving_tendency (thriftiness)
        2. Current wealth relative to typical wealth range
        3. Very low-wealth households save less (paycheck-to-paycheck)

        Returns:
            Float in [0.0, 0.15] representing fraction of income to save
        """
        from config import CONFIG

        # Get wealth reference points from config
        low_w = CONFIG.households.low_wealth_reference
        high_w = CONFIG.households.high_wealth_reference

        # Ensure valid range
        if high_w <= low_w:
            high_w = low_w + 1.0

        # Compute wealth_score in [0, 1]
        # Use cash_balance as a proxy for wealth (could also include goods_inventory value)
        wealth = self.cash_balance
        wealth_score = (wealth - low_w) / (high_w - low_w)
        wealth_score = max(0.0, min(1.0, wealth_score))

        # Combine saving_tendency and wealth_score
        # Thrifty + wealthy households save more
        mix = 0.5 * self.saving_tendency + 0.5 * wealth_score
        raw_saving_share = 0.01 + 0.14 * mix  # Range: 1% to 15%

        # Adjustment: very low-wealth households save even less
        # Poor households are paycheck-to-paycheck, can't afford to save
        adjustment_low_wealth = 1.0 - 0.7 * (1.0 - wealth_score) ** 2
        adjusted_saving_share = raw_saving_share * adjustment_low_wealth

        # Clamp to [0.0, 0.15]
        saving_rate = max(0.0, min(0.15, adjusted_saving_share))

        return saving_rate

    def plan_consumption(
        self,
        market_prices: Dict[str, float],
        firm_qualities: Dict[str, float] = None,
        firm_categories: Dict[str, str] = None,
        firm_market_info: Optional[Dict[str, List[Dict[str, float]]]] = None,
        unemployment_rate: float = 0.0
    ) -> Dict[str, object]:
        """
        Decide desired budget allocation across categories.

        NEW APPROACH: Budget is based on income minus savings, not cash balance.
        - Employed households save 0-15% of wage (based on saving_tendency and wealth)
        - Unemployed households dissave (spend transfers + draw down cash reserves)

        Does not mutate state; returns a plan dict with category budgets.
        Market clearing will handle firm selection within categories.

        Args:
            market_prices: Current market prices for goods (good_name -> price)
            firm_qualities: Quality levels for goods (good_name -> quality) - optional
            firm_categories: Category mappings (good_name -> category) - optional

        Returns:
            Dict with household_id, category_budgets, and legacy planned_purchases
        """
        # Calculate income for this tick (wage if employed, or transfers if unemployed)
        # Note: Transfers are already added to cash_balance in earlier phase
        # So we treat wage as "new income" and existing cash as "savings"

        if self.is_employed:
            # Employed: Save a fraction of wage, spend the rest
            income = self.wage
            saving_rate = self.compute_saving_rate()
            amount_saved = saving_rate * income
            amount_for_consumption = max(0.0, income - amount_saved)

            # Can also dip into cash reserves if needed (up to 20% of balance)
            # This allows consumption smoothing
            max_cash_drawdown = self.cash_balance * 0.2
            budget = amount_for_consumption + max_cash_drawdown
        else:
            # Unemployed: No wage income, must draw down cash reserves
            # Spend more aggressively to maintain consumption (dissaving)
            # Use unemployment rate to modulate spending anxiety
            confidence = 1.0 / (1.0 + max(unemployment_rate, 0.0))

            # Unemployed households spend 50-80% of cash balance per tick
            # Higher unemployment → more fear → less spending
            spend_fraction = 0.5 + (0.3 * confidence)
            budget = self.cash_balance * spend_fraction

        if budget <= 0:
            return {
                "household_id": self.household_id,
                "category_budgets": {},
                "planned_purchases": {},  # legacy field for backward compatibility
            }

        # Use category weights if available, otherwise fall back to good weights
        if self.category_weights and sum(self.category_weights.values()) > 0 and firm_market_info:
            planned_purchases = self._plan_category_purchases(budget, firm_market_info)
            return {
                "household_id": self.household_id,
                "category_budgets": {},
                "planned_purchases": planned_purchases,
            }
        else:
            # Legacy good-based allocation (backward compatibility)
            # Create local copy of price beliefs for planning (don't mutate state)
            local_beliefs = dict(self.price_beliefs)

            # Update local beliefs with market prices
            for good, market_price in market_prices.items():
                if good in local_beliefs:
                    # Exponentially smooth toward market price
                    old_belief = local_beliefs[good]
                    local_beliefs[good] = (
                        self.price_expectation_alpha * market_price +
                        (1.0 - self.price_expectation_alpha) * old_belief
                    )
                else:
                    # Initialize belief to market price
                    local_beliefs[good] = market_price

            # Normalize good weights
            total_weight = sum(self.good_weights.values())
            if total_weight <= 0:
                # No weights specified: treat all goods equally
                all_goods = set(local_beliefs.keys()) | set(market_prices.keys())
                if not all_goods:
                    # No goods available
                    normalized_weights = {}
                else:
                    equal_weight = 1.0 / len(all_goods)
                    normalized_weights = {g: equal_weight for g in all_goods}
            else:
                normalized_weights = {
                    g: w / total_weight for g, w in self.good_weights.items()
                }

            # Plan purchases for each good
            planned_purchases = {}
            for good, weight in normalized_weights.items():
                if weight <= 0:
                    continue

                # Determine expected price
                if good in local_beliefs:
                    expected_price = local_beliefs[good]
                elif good in market_prices:
                    expected_price = market_prices[good]
                else:
                    expected_price = self.default_price_level

                if expected_price <= 0:
                    continue

                # Allocate budget to this good
                good_budget = budget * weight
                if is_housing_good(good):
                    planned_quantity = min(1.0, good_budget / expected_price)
                else:
                    planned_quantity = good_budget / expected_price

                if planned_quantity > 0:
                    planned_purchases[good] = planned_quantity

            return {
                "household_id": self.household_id,
                "category_budgets": {},  # Empty for legacy mode
                "planned_purchases": planned_purchases,
            }

    def apply_labor_outcome(
        self,
        outcome: Dict[str, object],
        market_wage_anchor: Optional[float] = None
    ) -> None:
        """
        Update employment status and wage beliefs based on labor market outcome.

        Mutates state.

        Args:
            outcome: Dict with employer_id (int | None), wage (float), and employer_category (str | None)
            market_wage_anchor: Optional market-paid wage to nudge expectations toward
        """
        self.employer_id = outcome["employer_id"]
        self.wage = outcome["wage"]
        employer_category = outcome.get("employer_category", None)

        # Track experience in category (increment by 1 tick if employed)
        if self.is_employed:
            self.unemployment_duration = 0
            if employer_category is not None:
                if employer_category not in self.category_experience:
                    self.category_experience[employer_category] = 0
                self.category_experience[employer_category] += 1

            # Passive skill growth through work experience (diminishing returns)
            skill_improvement = self.skill_growth_rate * (1.0 - self.skills_level)
            self.skills_level = min(1.0, self.skills_level + skill_improvement)

        # Update wage expectations
        if self.is_employed and self.wage > 0:
            # Employed: update expected wage toward actual wage
            self.expected_wage = (
                self.wage_expectation_alpha * self.wage +
                (1.0 - self.wage_expectation_alpha) * self.expected_wage
            )
        else:
            self.unemployment_duration += 1
            # Unemployed: gently nudge expected wage downward
            duration_pressure = min(0.35, self.unemployment_duration * 0.01)
            happiness_gap = max(0.0, 0.7 - self.happiness)
            happiness_pressure = min(0.3, happiness_gap * 0.5)
            base_decay = 0.97  # slightly faster baseline decay
            decay_factor = max(
                0.5,
                base_decay - duration_pressure - happiness_pressure
            )
            decayed_expectation = max(self.expected_wage * decay_factor, 10.0)

            if market_wage_anchor is not None:
                anchor_weight = 0.4  # stronger pull toward market when unemployed
                self.expected_wage = (
                    (1.0 - anchor_weight) * decayed_expectation
                    + anchor_weight * market_wage_anchor
                )
            else:
                self.expected_wage = decayed_expectation

        # Update reservation wage toward expected wage (slow adjustment)
        reservation_adjustment_rate = 0.1
        self.reservation_wage = (
            reservation_adjustment_rate * self.expected_wage +
            (1.0 - reservation_adjustment_rate) * self.reservation_wage
        )

    def apply_income_and_taxes(self, flows: Dict[str, float]) -> None:
        """
        Update cash balance based on income, transfers, and taxes.

        Mutates state.

        Args:
            flows: Dict with wage_income, transfers, and taxes_paid
        """
        wage_income = flows.get("wage_income", 0.0)
        transfers = flows.get("transfers", 0.0)
        taxes_paid = flows.get("taxes_paid", 0.0)

        self.cash_balance += wage_income + transfers - taxes_paid

    def apply_purchases(self, purchases: Dict[str, tuple[float, float]],
                        firm_categories: Optional[Dict[str, str]] = None) -> None:
        """
        Update inventory, cash, and price beliefs based on executed purchases.

        Mutates state.

        Args:
            purchases: Dict mapping good_name -> (quantity, price_paid)
            firm_categories: Optional dict mapping good_name -> category (to detect housing purchases)
        """
        for good, (quantity, price_paid) in purchases.items():
            # Update cash
            total_cost = quantity * price_paid
            self.cash_balance -= total_cost

            # Check if this is a housing purchase
            category = _get_good_category(good, firm_categories)
            if category == "housing" and quantity > 0:
                self.owns_housing = True
                self.met_housing_need = True

            # Update inventory
            if good not in self.goods_inventory:
                self.goods_inventory[good] = 0.0
            self.goods_inventory[good] += quantity

            # Update price beliefs
            if good in self.price_beliefs:
                old_belief = self.price_beliefs[good]
                self.price_beliefs[good] = (
                    self.price_expectation_alpha * price_paid +
                    (1.0 - self.price_expectation_alpha) * old_belief
                )
            else:
                self.price_beliefs[good] = price_paid

        # Safety check: detect serious bugs
        if self.cash_balance < -1e6:  # Allow some float tolerance but catch serious errors
            raise ValueError(
                f"Household {self.household_id} cash balance became extremely negative: "
                f"{self.cash_balance}. This indicates a configuration or market clearing bug."
            )

    def invest_in_education(self, investment_amount: float) -> bool:
        """
        Invest cash in education to improve skills.

        Returns True if investment was made, False if insufficient cash.

        Args:
            investment_amount: Amount of cash to invest

        Returns:
            bool: True if investment successful, False otherwise
        """
        if self.cash_balance >= investment_amount and investment_amount > 0:
            self.cash_balance -= investment_amount

            # Diminishing returns: harder to improve at higher skill levels
            skill_gain_rate = 0.0001  # 0.1 skill points per $1000 invested at low skills
            skill_gain = investment_amount * skill_gain_rate * (1.0 - self.skills_level)
            self.skills_level = min(1.0, self.skills_level + skill_gain)

            return True
        return False

    def maybe_active_education(self) -> bool:
        """
        Actively invest in education when unemployed and below median skill.

        Trigger: skills < 0.5, cash > 300, unemployed.
        Cost: $100, Skill gain: +0.005
        """
        if self.is_employed:
            return False
        if self.skills_level >= 0.5 or self.cash_balance <= 300.0:
            return False

        cost = 100.0
        if self.cash_balance >= cost:
            self.cash_balance -= cost
            self.skills_level = min(1.0, self.skills_level + 0.005)
            return True
        return False

    def consume_goods(self, good_categories: Optional[Dict[str, str]] = None) -> None:
        """
        Consume goods from inventory each tick.

        Households consume a fraction of their goods inventory each tick
        to represent using up food, services, housing, etc.

        Mutates state.
        """
        consumption_rate = 0.1  # Consume 10% of inventory per tick
        housing_usage = 1.0  # Housing treated as a service: consume need each tick
        self.met_housing_need = False

        for good in list(self.goods_inventory.keys()):
            if self.goods_inventory[good] > 0:
                category = _get_good_category(good, good_categories)
                current_qty = self.goods_inventory[good]
                if category == "housing":
                    self.met_housing_need = current_qty >= housing_usage
                    new_qty = max(0.0, current_qty - housing_usage)
                    self.goods_inventory[good] = new_qty
                    if new_qty < 0.001 and self.owns_housing:
                        self.owns_housing = False
                else:
                    consumed = current_qty * consumption_rate
                    self.goods_inventory[good] = max(0.0, current_qty - consumed)

                # Remove from dict if depleted
                if self.goods_inventory[good] < 0.001:
                    del self.goods_inventory[good]

    def update_wellbeing(self, government_happiness_multiplier: float = 1.0) -> None:
        """
        Update happiness, morale, and health each tick.

        Happiness is affected by:
        - Employment status (employed = +boost)
        - Consumption (having goods = +boost)
        - Government social programs (multiplier)
        - Natural decay

        Morale is affected by:
        - Wage satisfaction (wage vs expected)
        - Employment status
        - Natural decay (faster than happiness)

        Health is affected by:
        - Consumption (goods = healthcare)
        - Government social programs
        - Natural decay (slowest)

        Mutates state.

        Args:
            government_happiness_multiplier: Multiplier from government social investment
        """
        # Happiness updates
        happiness_change = 0.0

        # Employment boosts happiness
        if self.is_employed:
            happiness_change += 0.02
        else:
            happiness_change -= 0.03  # Unemployment hurts happiness

        # Having goods boosts happiness
        total_goods = sum(self.goods_inventory.values())
        if total_goods > 10.0:
            happiness_change += 0.01
        elif total_goods < 2.0:
            happiness_change -= 0.02
        if not self.met_housing_need:
            happiness_change -= 0.05

        # Government social programs boost happiness
        if government_happiness_multiplier > 1.0:
            happiness_change += (government_happiness_multiplier - 1.0) * 0.05

        # Natural decay
        happiness_change -= self.happiness_decay_rate

        # Apply change and clamp
        self.happiness = max(0.0, min(1.0, self.happiness + happiness_change))

        # Morale updates
        morale_change = 0.0

        # Wage satisfaction affects morale
        if self.is_employed:
            if self.wage >= self.expected_wage:
                morale_change += 0.03  # Satisfied with wage
            else:
                wage_gap_ratio = (self.expected_wage - self.wage) / max(self.expected_wage, 1.0)
                morale_change -= wage_gap_ratio * 0.05  # Dissatisfied

        # Unemployment hurts morale more than happiness
        if not self.is_employed:
            morale_change -= 0.05

        # Natural decay (faster than happiness)
        morale_change -= self.morale_decay_rate

        # Apply change and clamp
        self.morale = max(0.0, min(1.0, self.morale + morale_change))

        # Health updates
        health_change = 0.0

        # Consumption supports health (goods = food, healthcare)
        if total_goods > 15.0:
            health_change += 0.01
        elif total_goods < 5.0:
            health_change -= 0.02

        # Government social programs (healthcare) boost health
        if government_happiness_multiplier > 1.0:
            health_change += (government_happiness_multiplier - 1.0) * 0.03

        # Natural decay (slowest)
        health_change -= self.health_decay_rate

        # Apply change and clamp
        self.health = max(0.0, min(1.0, self.health + health_change))

    def get_performance_multiplier(self) -> float:
        """
        Calculate overall performance multiplier based on wellbeing.

        Performance affects:
        - Productivity (how much worker produces)
        - Skill development rate
        - Effective skills in labor market

        Returns:
            Multiplier in range [0.5, 1.5]
            - Low wellbeing = 0.5x performance
            - Perfect wellbeing = 1.5x performance
        """
        # Weighted average of wellbeing factors
        # Morale affects day-to-day performance most
        # Health affects capacity
        # Happiness affects engagement
        wellbeing_score = (
            self.morale * 0.5 +
            self.health * 0.3 +
            self.happiness * 0.2
        )

        # Convert to multiplier: 0.0 wellbeing = 0.5x, 1.0 wellbeing = 1.5x
        performance_multiplier = 0.5 + (wellbeing_score * 1.0)

        return performance_multiplier


@dataclass(slots=True)
class FirmAgent:
    """
    Represents a firm in the economic simulation.

    Firms produce goods, hire workers, set prices and wages,
    and respond to market conditions. All behavior is deterministic.
    """

    # Identity & product (required fields first)
    firm_id: int
    good_name: str
    cash_balance: float
    inventory_units: float

    # Identity & product (optional fields with defaults)
    good_category: str = "Generic"  # e.g., "Food", "Housing", "Services"
    quality_level: float = 5.0  # 0-10 scale, affects market share
    employees: List[int] = field(default_factory=list)  # household_ids
    owners: List[int] = field(default_factory=list)  # household_ids who own this firm

    # Production & technology
    expected_sales_units: float = 100.0  # moving average
    production_capacity_units: float = 200.0  # max units per tick
    productivity_per_worker: float = 10.0  # units per worker per tick
    units_per_worker: float = 20.0  # hiring heuristic target

    # Labour market state
    wage_offer: float = 50.0
    planned_headcount: int = 0
    planned_hires_count: int = 0
    planned_layoffs_ids: List[int] = field(default_factory=list)
    last_tick_planned_hires: int = 0
    last_tick_actual_hires: int = 0

    # Pricing & costs
    unit_cost: float = 5.0  # cost per unit produced
    markup: float = 0.3  # markup over unit_cost
    price: float = 6.5  # current price

    # Quality and R&D
    rd_spending_rate: float = 0.05  # fraction of revenue spent on R&D each tick
    quality_improvement_per_rd_dollar: float = 0.01  # quality points per $ of R&D
    quality_decay_rate: float = 0.02  # quality degradation per tick without maintenance
    accumulated_rd_investment: float = 0.0  # total R&D spending lifetime

    # Config / tuning
    sales_expectation_alpha: float = 0.3  # [0,1] for smoothing sales
    price_adjustment_rate: float = 0.05  # small positive adjustment rate
    wage_adjustment_rate: float = 0.1  # small positive adjustment rate
    target_inventory_multiplier: float = 1.5  # desired inventory as multiple of expected sales
    min_price: float = 5.0  # hard floor on price
    max_hires_per_tick: int = 2
    max_fires_per_tick: int = 2
    target_inventory_weeks: float = 2.0  # desired weeks of supply buffer
    price_pressure: float = 0.0  # accumulator for pricing control

    # Firm personality & strategy
    # "aggressive": High risk, high reward - invests heavily, adjusts prices aggressively
    # "conservative": Low risk, stable - minimal investment, gradual adjustments
    personality: str = "moderate"  # "aggressive", "moderate", or "conservative"
    investment_propensity: float = 0.05  # Fraction of profits to invest (varies by personality)
    risk_tolerance: float = 0.5  # 0-1 scale, affects pricing and hiring decisions
    is_baseline: bool = False
    baseline_production_quota: float = 500.0
    actual_wages: Dict[int, float] = field(default_factory=dict)
    last_tick_total_costs: float = 0.0  # Track costs for dividend calculation
    payout_ratio: float = 0.0  # Fraction of net profit paid as dividends
    net_profit: float = 0.0  # Track last tick net profit

    def __post_init__(self):
        """Validate invariants after initialization."""
        if self.production_capacity_units < 0:
            raise ValueError(
                f"production_capacity_units cannot be negative, got {self.production_capacity_units}"
            )
        if self.productivity_per_worker < 0:
            raise ValueError(
                f"productivity_per_worker cannot be negative, got {self.productivity_per_worker}"
            )
        if not (0.0 <= self.quality_level <= 10.0):
            raise ValueError(f"quality_level must be in [0,10], got {self.quality_level}")
        if not (0.0 <= self.sales_expectation_alpha <= 1.0):
            raise ValueError(
                f"sales_expectation_alpha must be in [0,1], got {self.sales_expectation_alpha}"
            )
        if self.price_adjustment_rate < 0:
            raise ValueError(
                f"price_adjustment_rate must be non-negative, got {self.price_adjustment_rate}"
            )
        if self.wage_adjustment_rate < 0:
            raise ValueError(
                f"wage_adjustment_rate must be non-negative, got {self.wage_adjustment_rate}"
            )
        if self.markup < 0:
            raise ValueError(f"markup cannot be negative, got {self.markup}")
        if self.target_inventory_multiplier < 0:
            raise ValueError(
                f"target_inventory_multiplier cannot be negative, got {self.target_inventory_multiplier}"
            )
        if self.rd_spending_rate < 0:
            raise ValueError(f"rd_spending_rate cannot be negative, got {self.rd_spending_rate}")
        if self.quality_decay_rate < 0:
            raise ValueError(f"quality_decay_rate cannot be negative, got {self.quality_decay_rate}")
        if self.payout_ratio <= 0:
            rng = random.Random(self.firm_id)
            self.payout_ratio = rng.uniform(0.0, 0.5)

    def to_dict(self) -> Dict[str, object]:
        """
        Serialize all fields to basic Python types.

        Returns:
            Dictionary representation of the firm state
        """
        return {
            "firm_id": self.firm_id,
            "good_name": self.good_name,
            "good_category": self.good_category,
            "quality_level": self.quality_level,
            "cash_balance": self.cash_balance,
            "inventory_units": self.inventory_units,
            "employees": list(self.employees),
            "employers":list(self.employers),
            "expected_sales_units": self.expected_sales_units,
            "production_capacity_units": self.production_capacity_units,
            "productivity_per_worker": self.productivity_per_worker,
            "units_per_worker": self.units_per_worker,
            "wage_offer": self.wage_offer,
            "planned_headcount": self.planned_headcount,
            "planned_hires_count": self.planned_hires_count,
            "planned_layoffs_ids": list(self.planned_layoffs_ids),
            "last_tick_planned_hires": self.last_tick_planned_hires,
            "last_tick_actual_hires": self.last_tick_actual_hires,
            "unit_cost": self.unit_cost,
            "markup": self.markup,
            "price": self.price,
            "rd_spending_rate": self.rd_spending_rate,
            "quality_improvement_per_rd_dollar": self.quality_improvement_per_rd_dollar,
            "quality_decay_rate": self.quality_decay_rate,
            "accumulated_rd_investment": self.accumulated_rd_investment,
            "sales_expectation_alpha": self.sales_expectation_alpha,
            "price_adjustment_rate": self.price_adjustment_rate,
            "wage_adjustment_rate": self.wage_adjustment_rate,
            "target_inventory_multiplier": self.target_inventory_multiplier,
            "min_price": self.min_price,
            "max_hires_per_tick": self.max_hires_per_tick,
            "max_fires_per_tick": self.max_fires_per_tick,
            "is_baseline": self.is_baseline,
            "baseline_production_quota": self.baseline_production_quota,
            "target_inventory_weeks": self.target_inventory_weeks,
            "price_pressure": self.price_pressure,
            "payout_ratio": self.payout_ratio,
            "net_profit": self.net_profit,
        }

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        """
        Apply external overrides to firm state.

        Useful for UI or script-driven state modifications.

        Args:
            overrides: Dictionary of attribute names to new values
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_personality(self, personality: str) -> None:
        """
        Set firm personality and adjust behavior parameters accordingly.

        Aggressive firms: High investment, aggressive pricing, higher risk
        Conservative firms: Low investment, gradual pricing, lower risk
        Moderate firms: Balanced approach

        Mutates state.

        Args:
            personality: "aggressive", "conservative", or "moderate"
        """
        self.personality = personality

        if personality == "aggressive":
            # Aggressive firms invest heavily and adjust prices rapidly
            self.investment_propensity = 0.15  # Invest 15% of profits
            self.risk_tolerance = 0.9  # High risk tolerance
            self.price_adjustment_rate = 0.10  # Rapid price changes
            self.wage_adjustment_rate = 0.15  # Aggressive wage bidding
            self.rd_spending_rate = 0.08  # Heavy R&D investment
            self.max_hires_per_tick = 3
            self.max_fires_per_tick = 3
            self.units_per_worker = 18.0
        elif personality == "conservative":
            # Conservative firms play it safe
            self.investment_propensity = 0.02  # Minimal investment
            self.risk_tolerance = 0.2  # Low risk tolerance
            self.price_adjustment_rate = 0.02  # Gradual price changes
            self.wage_adjustment_rate = 0.05  # Conservative wage bidding
            self.rd_spending_rate = 0.02  # Minimal R&D
            self.max_hires_per_tick = 1
            self.max_fires_per_tick = 1
            self.units_per_worker = 25.0
        else:  # moderate (default)
            # Balanced approach
            self.investment_propensity = 0.05  # Moderate investment
            self.risk_tolerance = 0.5  # Moderate risk
            self.price_adjustment_rate = 0.05  # Standard price adjustment
            self.wage_adjustment_rate = 0.1  # Standard wage adjustment
            self.rd_spending_rate = 0.05  # Standard R&D
            self.max_hires_per_tick = 2
            self.max_fires_per_tick = 2
            self.units_per_worker = 20.0

    def plan_production_and_labor(
        self,
        last_tick_sales_units: float,
        in_warmup: bool = False,
        total_households: int = 0,
        global_unsold_inventory: float = 0.0
    ) -> Dict[str, object]:
        """
        Decide how much to produce and how many workers are needed.

        NEW ECONOMIC LOGIC:
        1. Goal: Sell EVERYTHING (current production + existing inventory)
        2. Hiring decision: Will additional workers generate more revenue than cost?
        3. Pricing: Lower price aggressively to clear inventory
        4. Wage cuts: Only as last resort when revenue can't cover payroll

        FIRM THINKING:
        - "If I hire X more workers, they produce Y more units"
        - "If I sell all Y units at price P, I get revenue R"
        - "Does R > (wage × X)? If yes, hire them!"
        - "I want to sell ALL inventory, not just new production"

        Does not mutate state; returns a plan dict.

        Args:
            last_tick_sales_units: Actual units sold in the previous tick

        Returns:
            Dict with firm_id, planned_production_units, planned_hires_count, planned_layoffs_ids
        """
        # Update expected sales (local computation, don't mutate yet)
        updated_expected_sales = (
            self.sales_expectation_alpha * last_tick_sales_units +
            (1.0 - self.sales_expectation_alpha) * self.expected_sales_units
        )

        import math

        is_housing_producer = self.good_category.lower() == "housing"
        # HOUSING MARKET FIX: Stop production if inventory exceeds 110% of households
        # This prevents the housing glut (was 1.7M units for 10k households)
        housing_market_saturated = (
            is_housing_producer
            and total_households > 0
            and global_unsold_inventory > (total_households * 1.10)
        )

        base_productivity = max(self.units_per_worker, 1.0)

        def capacity_for_workers(worker_count: int) -> float:
            if worker_count <= 0:
                return 0.0
            return base_productivity * (worker_count ** 0.9)

        def workers_needed_for_output(target_output: float) -> int:
            if target_output <= 0:
                return 0
            required = (target_output / base_productivity) ** (1.0 / 0.9)
            return max(1, math.ceil(required))

        current_workers = len(self.employees)
        planned_hires = 0
        planned_layoffs: List[int] = []

        # DYNAMIC SCALING: Allow firms to change workforce by 10% per tick, or at least 5 people
        scaling_limit = max(5, int(current_workers * 0.10))

        if housing_market_saturated:
            planned_production_units = 0.0
        elif self.is_baseline and in_warmup:
            planned_hires = 1_000_000_000  # effectively infinite during warm-up
            revenue_per_worker = self.price * self.productivity_per_worker
            self.wage_offer = revenue_per_worker * 0.95
            planned_production_units = min(self.production_capacity_units, max(self.baseline_production_quota, self.production_capacity_units))
        else:
            # ===== NEW FIRM DECISION LOGIC =====
            # Goal: Maximize revenue by selling ALL inventory (not just new production)

            # Step 1: Calculate how much we COULD produce with more workers
            base_productivity = max(self.units_per_worker, 1.0)

            # Step 2: Project revenue if we hire more workers
            # "If I hire X more workers, they produce Y units, I sell at price P"
            # Total units to sell = current inventory + new production
            total_units_to_sell = self.inventory_units + capacity_for_workers(current_workers)

            # Step 3: Estimate demand at current price
            # Use recent sales as demand proxy (with some growth assumption)
            estimated_demand_at_current_price = updated_expected_sales * 1.1  # Assume 10% growth potential

            # Step 4: Check if we have excess inventory
            inventory_ratio = self.inventory_units / max(updated_expected_sales, 1.0)
            has_excess_inventory = inventory_ratio > 1.5  # More than 1.5 ticks of sales in stock

            # Step 5: Decide hiring based on revenue projection
            if has_excess_inventory:
                # We have inventory piling up - focus on clearing it, not producing more
                # Don't hire, but don't fire either (workers might be needed after clearance)
                target_workers = current_workers
                planned_production_units = capacity_for_workers(current_workers) * 0.5  # Reduce production
            else:
                # Inventory is reasonable - consider hiring
                # Calculate marginal revenue from hiring one more worker
                additional_units_from_hire = base_productivity * ((current_workers + 1) ** 0.9) - capacity_for_workers(current_workers)
                marginal_revenue = additional_units_from_hire * self.price
                marginal_cost = self.wage_offer

                # Hire if marginal revenue > marginal cost
                if marginal_revenue > marginal_cost:
                    # Calculate how many workers we should hire
                    # Target: Produce enough to meet projected demand
                    target_production = min(estimated_demand_at_current_price, self.production_capacity_units)
                    target_workers = workers_needed_for_output(target_production)
                else:
                    # Workers aren't profitable at current price - don't expand
                    target_workers = current_workers

                planned_production_units = min(
                    capacity_for_workers(max(current_workers, target_workers)),
                    self.production_capacity_units
                )

            # Step 6: Execute hiring/firing decisions
            delta = target_workers - current_workers
            if delta > 0:
                planned_hires = min(delta, scaling_limit)
            elif delta < 0:
                # Only fire if absolutely necessary (firm philosophy: avoid firing)
                # Fire only if we have massive overcapacity (>200% of demand)
                if inventory_ratio > 3.0:
                    layoff_count = min(-delta, scaling_limit)
                    planned_layoffs = self.employees[:layoff_count]

        self.planned_hires_count = planned_hires
        self.planned_layoffs_ids = planned_layoffs
        self.last_tick_planned_hires = planned_hires

        return {
            "firm_id": self.firm_id,
            "planned_production_units": planned_production_units,
            "planned_hires_count": planned_hires,
            "planned_layoffs_ids": planned_layoffs,
            "updated_expected_sales": updated_expected_sales,  # include for later apply
        }

    def plan_pricing(self, sell_through_rate: float, in_warmup: bool = False) -> Dict[str, float]:
        """
        AGGRESSIVE INVENTORY CLEARANCE PRICING

        NEW PHILOSOPHY:
        1. Goal: Sell ALL inventory, not maintain margins
        2. If inventory isn't selling → lower price aggressively
        3. Keep lowering until everything sells (even down to $0.01)
        4. Only constraint: Must still afford to pay workers
        5. Price floor: wage_bill / total_production (break-even on labor)

        FIRM THINKING:
        - "I have 1000 units sitting unsold"
        - "Lower price 10% → if still unsold → lower 10% more"
        - "Keep going until it all sells"
        - "Better to sell at low margin than not sell at all"
        """
        if self.is_baseline and in_warmup:
            labor_cost_per_unit = self.wage_offer / max(self.productivity_per_worker, 1.0)
            target_price = labor_cost_per_unit * 1.05
            return {
                "price_next": target_price,
                "markup_next": (target_price / self.unit_cost - 1.0) if self.unit_cost > 0 else self.markup,
            }

        # Calculate inventory pressure (how much isn't selling)
        average_weekly_sales = max(self.expected_sales_units * 7.0, 1e-3)
        weeks_of_supply = self.inventory_units / average_weekly_sales
        inventory_ratio = self.inventory_units / max(self.expected_sales_units, 1.0)

        # Calculate labor cost floor (can't price below what we pay workers)
        current_workers = max(len(self.employees), 1)
        total_wage_bill = current_workers * self.wage_offer
        expected_production = self.units_per_worker * (current_workers ** 0.9)
        labor_cost_per_unit = total_wage_bill / max(expected_production, 1.0)
        absolute_floor = max(labor_cost_per_unit * 0.95, 0.01)  # 95% of labor cost or $0.01

        # AGGRESSIVE CLEARANCE PRICING
        # If inventory is building up, cut price aggressively
        if inventory_ratio > 3.0:
            # Severe glut (>3 ticks of inventory) - FIRE SALE
            # Cut price 30-50% to clear everything
            price_next = self.price * 0.50
        elif inventory_ratio > 2.0:
            # Major glut (>2 ticks) - aggressive discounting
            price_next = self.price * 0.70
        elif inventory_ratio > 1.5:
            # Moderate glut (>1.5 ticks) - significant discount
            price_next = self.price * 0.85
        elif inventory_ratio > 1.0:
            # Slight glut (>1 tick) - small discount
            price_next = self.price * 0.95
        elif inventory_ratio < 0.5:
            # Inventory too low - raise price to rebuild margin
            price_next = self.price * 1.05
        else:
            # Inventory in good range - maintain price
            price_next = self.price

        # Additional sell-through pressure
        # If we're not selling even at reduced prices, cut more
        if sell_through_rate < 0.1:
            # Almost nothing selling - DESPERATE clearance
            price_next *= 0.70
        elif sell_through_rate < 0.5:
            # Weak sales - more discounting
            price_next *= 0.90

        # Absolute floor: Can't price below labor cost (would lose money on every sale)
        price_next = max(price_next, absolute_floor)

        # Calculate markup
        if self.unit_cost > 0:
            markup_next = (price_next / self.unit_cost) - 1.0
        else:
            markup_next = self.markup

        return {
            "price_next": price_next,
            "markup_next": markup_next,
        }

    def plan_wage(self) -> Dict[str, float]:
        """
        Adjust wages based on revenue and hiring success.

        NEW PHILOSOPHY - Wage cuts are LAST RESORT:
        1. Firms prefer NOT to cut wages (bad for morale/retention)
        2. First try: Lower prices to increase revenue
        3. Second try: Reduce production (don't hire)
        4. Last resort: Cut wages ONLY if revenue can't cover payroll at ANY price

        WAGE CUT CONDITIONS:
        - Revenue at rock-bottom prices (labor cost floor) still can't cover payroll
        - Cash balance is negative (firm is insolvent)
        - No other option except bankruptcy

        Otherwise: Maintain or raise wages based on hiring success

        Does not mutate state; returns a plan dict.

        Returns:
            Dict with wage_offer_next
        """
        if self.is_baseline and self.last_tick_planned_hires >= 1_000_000:
            return {"wage_offer_next": self.wage_offer}

        # Calculate revenue per worker (value created by each worker)
        current_workers = max(len(self.employees), 1)

        # Revenue per worker = price × productivity (with diminishing returns)
        base_productivity = max(self.units_per_worker, 1.0)
        actual_productivity_per_worker = base_productivity * (current_workers ** -0.1)
        revenue_per_worker = self.price * actual_productivity_per_worker

        # Calculate total revenue and wage bill
        total_wage_bill = current_workers * self.wage_offer
        projected_production = base_productivity * (current_workers ** 0.9)
        projected_revenue = projected_production * self.price

        # Check if firm is in financial distress
        revenue_covers_payroll = projected_revenue >= total_wage_bill
        is_insolvent = self.cash_balance < -10000.0  # Deep in debt

        # DECISION TREE:
        if not revenue_covers_payroll and is_insolvent:
            # CRISIS MODE: Revenue can't cover payroll even at current prices
            # AND firm is insolvent
            # Last resort: Cut wages by 10%
            wage_offer_next = self.wage_offer * 0.90
            wage_offer_next = max(wage_offer_next, 1.0)  # Floor at $1
        else:
            # NORMAL MODE: Adjust wages based on hiring success and revenue
            # Firms target paying workers 60-70% of revenue they generate
            target_wage_share = 0.65
            fundamental_wage = revenue_per_worker * target_wage_share

            # Adjust based on hiring pressure (if can't hire, raise wages)
            if self.last_tick_planned_hires > 0:
                hiring_denominator = max(self.last_tick_planned_hires, 1)
                hiring_success = self.last_tick_actual_hires / hiring_denominator

                if hiring_success < 0.5:
                    # Desperate for workers - raise wages toward fundamental value
                    adjustment = 1.0 + (0.1 * (1.0 - hiring_success))
                elif hiring_success < 1.0:
                    # Moderate hiring difficulty - small wage increase
                    adjustment = 1.0 + (0.05 * (1.0 - hiring_success))
                else:
                    # Hired everyone we wanted - can lower wages slightly
                    adjustment = 0.98
            else:
                # Not hiring - gradually drift toward fundamental wage
                adjustment = 0.99

            # Blend current wage with fundamental wage (smooth adjustment)
            wage_offer_next = self.wage_offer * adjustment
            wage_offer_next = 0.7 * wage_offer_next + 0.3 * fundamental_wage

            # Floor: wage must be at least $1
            # Ceiling: wage cannot exceed 90% of revenue per worker (firm needs profit)
            wage_offer_next = max(1.0, min(wage_offer_next, revenue_per_worker * 0.90))

        return {
            "wage_offer_next": wage_offer_next,
        }

    def apply_labor_outcome(self, outcome: Dict[str, object]) -> None:
        """
        Update workforce based on labor market outcome.

        Mutates state.

        Args:
            outcome: Dict with hired_households_ids and confirmed_layoffs_ids
        """
        hired_households_ids = outcome.get("hired_households_ids", [])
        confirmed_layoffs_ids = outcome.get("confirmed_layoffs_ids", [])

        wage_map = outcome.get("actual_wages", {})

        # Remove laid-off workers
        for worker_id in confirmed_layoffs_ids:
            if worker_id in self.employees:
                self.employees.remove(worker_id)
            if worker_id in self.actual_wages:
                del self.actual_wages[worker_id]

        # Add new hires (avoid duplicates)
        for worker_id in hired_households_ids:
            if worker_id not in self.employees:
                self.employees.append(worker_id)
            self.actual_wages[worker_id] = wage_map.get(worker_id, self.wage_offer)

        # Track hiring for next planning cycle
        # Note: These should be set from the plan, but we update actual hires here
        self.last_tick_actual_hires = len(hired_households_ids)

    def apply_production_and_costs(self, result: Dict[str, float]) -> None:
        """
        Update inventory, cash, and costs based on production.

        Mutates state.

        Args:
            result: Dict with realized_production_units and optionally other_variable_costs
        """
        realized_production_units = result.get("realized_production_units", 0.0)
        other_variable_costs = result.get("other_variable_costs", 0.0)

        # Update inventory
        self.inventory_units += realized_production_units

        # Compute wage bill based on actual wages paid
        wage_bill = 0.0
        for employee_id in self.employees:
            wage_bill += self.actual_wages.get(employee_id, self.wage_offer)

        # Update cash (pay wages and costs)
        self.cash_balance -= wage_bill
        self.cash_balance -= other_variable_costs

        # Update unit cost
        total_production_cost = wage_bill + other_variable_costs

        # Track total costs for dividend calculation
        self.last_tick_total_costs = total_production_cost

        if realized_production_units > 0:
            self.unit_cost = total_production_cost / realized_production_units
        else:
            # No production - keep previous unit cost or set to a default
            # To avoid division by zero, we keep the existing unit_cost
            pass

    def apply_sales_and_profit(self, result: Dict[str, float]) -> None:
        """
        Update inventory and cash based on sales.

        Mutates state.

        Args:
            result: Dict with units_sold, revenue, and profit_taxes_paid
        """
        units_sold = result.get("units_sold", 0.0)
        revenue = result.get("revenue", 0.0)
        profit_taxes_paid = result.get("profit_taxes_paid", 0.0)

        # Update inventory (clamp at zero)
        self.inventory_units = max(0.0, self.inventory_units - units_sold)

        # Update cash
        self.cash_balance += revenue
        self.cash_balance -= profit_taxes_paid

        # Track net profit for dividend policy
        self.net_profit = revenue - profit_taxes_paid - self.last_tick_total_costs

    def apply_price_and_wage_updates(
        self,
        price_plan: Dict[str, float],
        wage_plan: Dict[str, float]
    ) -> None:
        """
        Update price, markup, and wage offer from plans.

        Mutates state.

        Args:
            price_plan: Dict with price_next and markup_next
            wage_plan: Dict with wage_offer_next
        """
        # Update price and markup
        self.price = max(price_plan["price_next"], self.min_price)
        self.markup = max(0.0, price_plan["markup_next"])

        # Update wage offer
        self.wage_offer = wage_plan["wage_offer_next"]

    def apply_updated_expectations(self, updated_expected_sales: float) -> None:
        """
        Update sales expectations from production planning.

        Mutates state.

        Args:
            updated_expected_sales: New expected sales value
        """
        self.expected_sales_units = updated_expected_sales

    def apply_rd_and_quality_update(self, revenue: float) -> None:
        """
        Invest in R&D and update quality level.

        Mutates state.

        Args:
            revenue: Revenue from this tick (used to compute R&D spending)
        """
        # Compute R&D spending
        rd_spending = revenue * self.rd_spending_rate
        self.accumulated_rd_investment += rd_spending

        # Deduct R&D spending from cash
        self.cash_balance -= rd_spending

        # Improve quality based on R&D investment
        quality_gain = rd_spending * self.quality_improvement_per_rd_dollar

        # Apply quality decay (degradation over time)
        quality_loss = self.quality_decay_rate

        # Update quality (clamped to [0, 10])
        self.quality_level = max(
            0.0,
            min(10.0, self.quality_level + quality_gain - quality_loss)
        )

    def distribute_profits(self, household_lookup: Dict[int, 'HouseholdAgent']) -> float:
        """
        Distribute excess cash to firm owners as dividends.

        This prevents wealth from accumulating in firms and ensures
        profits flow back to households who own the firms.

        Args:
            household_lookup: Dict mapping household_id -> HouseholdAgent

        Returns:
            Total dividends distributed

        Mutates state:
            - Reduces firm cash_balance
            - Increases owner household cash_balance
        """
        if not self.owners or len(self.owners) == 0:
            return 0.0  # No owners, no dividends

        if self.net_profit <= 0:
            return 0.0

        target_dividend = self.net_profit * self.payout_ratio

        # Keep six weeks of operating costs as reserve
        safety_buffer = self.last_tick_total_costs * 6.0
        available_cash = self.cash_balance - safety_buffer
        actual_dividend = min(target_dividend, max(0.0, available_cash))

        if actual_dividend <= 0:
            return 0.0

        dividend_per_owner = actual_dividend / len(self.owners)

        total_distributed = 0.0
        for owner_id in self.owners:
            if owner_id in household_lookup:
                household = household_lookup[owner_id]
                household.cash_balance += dividend_per_owner
                total_distributed += dividend_per_owner

        self.cash_balance -= total_distributed

        return total_distributed


@dataclass
class GovernmentAgent:
    """
    Represents a government in the economic simulation.

    The government collects taxes on wages and profits,
    and distributes unemployment benefits and transfers to households.
    All behavior is deterministic.
    """

    # Financial state
    cash_balance: float = 0.0

    # Policy parameters
    wage_tax_rate: float = 0.15  # [0,1]
    profit_tax_rate: float = 0.20  # [0,1]
    unemployment_benefit_level: float = 30.0  # per-tick payment to unemployed
    min_cash_threshold: float = 100.0  # safety net threshold
    transfer_budget: float = 10000.0  # max total transfers per tick

    # Government baseline firm tracking (category -> firm_id)
    baseline_firm_ids: Dict[str, int] = field(default_factory=dict)

    # Government investment capabilities
    infrastructure_investment_budget: float = 1000.0  # Budget for infrastructure per tick
    technology_investment_budget: float = 500.0  # Budget for technology per tick
    social_investment_budget: float = 750.0  # Budget for social programs per tick

    # Economic multipliers from government investment
    infrastructure_productivity_multiplier: float = 1.0  # Affects all worker productivity
    technology_quality_multiplier: float = 1.0  # Affects all firm quality
    social_happiness_multiplier: float = 1.0  # Affects worker happiness/performance
    wage_bracket_scalers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate invariants after initialization."""
        if not (0.0 <= self.wage_tax_rate <= 1.0):
            raise ValueError(f"wage_tax_rate must be in [0,1], got {self.wage_tax_rate}")
        if not (0.0 <= self.profit_tax_rate <= 1.0):
            raise ValueError(f"profit_tax_rate must be in [0,1], got {self.profit_tax_rate}")
        if self.unemployment_benefit_level < 0:
            raise ValueError(
                f"unemployment_benefit_level cannot be negative, got {self.unemployment_benefit_level}"
            )
        if self.min_cash_threshold < 0:
            raise ValueError(
                f"min_cash_threshold cannot be negative, got {self.min_cash_threshold}"
            )
        if self.transfer_budget < 0:
            raise ValueError(f"transfer_budget cannot be negative, got {self.transfer_budget}")
        if not self.wage_bracket_scalers:
            rng = random.Random(12345)
            self.wage_bracket_scalers = {
                "low": rng.uniform(0.5, 0.9),
                "median": 1.0,
                "p60": rng.uniform(1.05, 1.15),
                "p70": rng.uniform(1.10, 1.20),
                "p90": rng.uniform(1.15, 1.25),
            }

    def to_dict(self) -> Dict[str, object]:
        """
        Serialize all fields to basic Python types.

        Returns:
            Dictionary representation of the government state
        """
        return {
            "cash_balance": self.cash_balance,
            "wage_tax_rate": self.wage_tax_rate,
            "profit_tax_rate": self.profit_tax_rate,
            "unemployment_benefit_level": self.unemployment_benefit_level,
            "min_cash_threshold": self.min_cash_threshold,
            "transfer_budget": self.transfer_budget,
            "baseline_firm_ids": dict(self.baseline_firm_ids),
        }

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        """
        Apply external overrides to government state.

        Useful for UI or script-driven state modifications.

        Args:
            overrides: Dictionary of attribute names to new values
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def register_baseline_firm(self, category: str, firm_id: int) -> None:
        """Record the firm id for a government baseline firm."""
        self.baseline_firm_ids[category.lower()] = firm_id

    def is_baseline_firm(self, firm_id: int) -> bool:
        """Check if a firm belongs to the government baseline set."""
        return firm_id in self.baseline_firm_ids.values()

    def get_unemployment_benefit_level(self) -> float:
        """
        Get the current unemployment benefit level.

        This is used by households to anchor their reservation wage.

        Returns:
            Unemployment benefit amount per tick
        """
        return self.unemployment_benefit_level

    def plan_transfers(self, households: List[Dict[str, object]]) -> Dict[int, float]:
        """
        Plan transfers to unemployed households and those below cash threshold.

        REALISTIC GOVERNMENT BEHAVIOR:
        - Governments can run deficits (borrow money) to fund transfers during recessions
        - Transfer budget is dynamic, not a hard cap
        - During high unemployment, governments increase spending (counter-cyclical policy)

        Does not mutate state; returns a plan dict.

        Args:
            households: List of dicts with household_id, is_employed, cash_balance

        Returns:
            Dict mapping household_id -> transfer_amount
        """
        transfers = {}

        # First pass: baseline unemployment benefits
        unemployed_households = [
            h for h in households
            if not h.get("is_employed", False)
        ]

        # REALISTIC: Governments pay full unemployment benefits even if it creates deficits
        # The transfer_budget is more of a soft constraint than a hard cap
        # In real life, governments borrow during recessions to fund unemployment insurance
        baseline_transfers_total = 0.0
        for household in unemployed_households:
            household_id = household["household_id"]
            baseline_transfer = self.unemployment_benefit_level
            transfers[household_id] = baseline_transfer
            baseline_transfers_total += baseline_transfer

        # Second pass: additional gap-filling for households below min_cash_threshold
        # Calculate gaps for all unemployed households
        gaps = {}
        total_gap = 0.0

        for household in unemployed_households:
            household_id = household["household_id"]
            cash_balance = household.get("cash_balance", 0.0)

            # Gap after receiving baseline transfer
            future_cash = cash_balance + transfers[household_id]
            gap = max(self.min_cash_threshold - future_cash, 0.0)

            if gap > 0:
                gaps[household_id] = gap
                total_gap += gap

        # Allocate additional transfers to close gaps, subject to budget
        remaining_budget = max(self.transfer_budget - baseline_transfers_total, 0.0)

        if total_gap > 0 and remaining_budget > 0:
            # Determine how much we can afford to close gaps
            if total_gap <= remaining_budget:
                # Can fully close all gaps
                scale_factor = 1.0
            else:
                # Must scale down gap-filling to fit budget
                scale_factor = remaining_budget / total_gap

            # Add gap-filling transfers
            for household_id, gap in gaps.items():
                additional_transfer = gap * scale_factor
                transfers[household_id] += additional_transfer

        return transfers

    def plan_taxes(
        self,
        households: List[Dict[str, object]],
        firms: List[Dict[str, object]]
    ) -> Dict[str, Dict[int, float]]:
        """
        Plan taxes on wages and profits.

        Does not mutate state; returns a plan dict.

        Args:
            households: List of dicts with household_id and wage_income
            firms: List of dicts with firm_id and profit_before_tax

        Returns:
            Dict with "wage_taxes" and "profit_taxes", each mapping ID -> tax amount
        """
        wage_taxes: Dict[int, float] = {}
        profit_taxes: Dict[int, float] = {}

        wages = [h.get("wage_income", 0.0) for h in households]
        if wages:
            wages_sorted = sorted(wages)
            p25 = wages_sorted[int(0.25 * (len(wages_sorted) - 1))]
            p50 = wages_sorted[int(0.50 * (len(wages_sorted) - 1))]
            p60 = wages_sorted[int(0.60 * (len(wages_sorted) - 1))]
            p70 = wages_sorted[int(0.70 * (len(wages_sorted) - 1))]
            p90 = wages_sorted[int(0.90 * (len(wages_sorted) - 1))]
        else:
            p25 = p50 = p60 = p70 = p90 = 0.0

        for household in households:
            household_id = household["household_id"]
            wage_income = household.get("wage_income", 0.0)

            if wage_income <= p25:
                rate = self.wage_tax_rate * self.wage_bracket_scalers.get("low", 0.8)
            elif wage_income <= p50:
                rate = self.wage_tax_rate * self.wage_bracket_scalers.get("median", 1.0)
            elif wage_income <= p60:
                rate = self.wage_tax_rate * self.wage_bracket_scalers.get("p60", 1.1)
            elif wage_income <= p70:
                rate = self.wage_tax_rate * self.wage_bracket_scalers.get("p70", 1.15)
            elif wage_income <= p90:
                rate = self.wage_tax_rate * self.wage_bracket_scalers.get("p90", 1.2)
            else:
                rate = self.wage_tax_rate * (self.wage_bracket_scalers.get("p90", 1.2) + 0.03)

            wage_taxes[household_id] = max(wage_income * rate, 0.0)

        # Profit tax with oligarchy surcharge based on cash concentration
        firm_cash = [f.get("cash_balance", 0.0) for f in firms]
        oligarchy = False
        cash_threshold = 0.0
        if firm_cash:
            firm_cash_sorted = sorted(firm_cash)
            cash_threshold = firm_cash_sorted[int(0.9 * (len(firm_cash_sorted) - 1))]
            top_sum = sum(c for c in firm_cash if c >= cash_threshold)
            total_sum = sum(firm_cash) or 1.0
            oligarchy = (top_sum / total_sum) > 0.5

        for firm in firms:
            firm_id = firm["firm_id"]
            profit_before_tax = firm.get("profit_before_tax", 0.0)
            base_tax = max(profit_before_tax * self.profit_tax_rate, 0.0)
            surcharge = 0.0
            if oligarchy and firm.get("cash_balance", 0.0) >= cash_threshold:
                surcharge = base_tax * 0.1
            profit_taxes[firm_id] = base_tax + surcharge

        return {
            "wage_taxes": wage_taxes,
            "profit_taxes": profit_taxes,
        }

    def apply_fiscal_results(
        self,
        total_wage_taxes: float,
        total_profit_taxes: float,
        total_transfers: float
    ) -> None:
        """
        Update government cash based on fiscal operations.

        Mutates state.

        Args:
            total_wage_taxes: Sum of all wage taxes collected
            total_profit_taxes: Sum of all profit taxes collected
            total_transfers: Sum of all transfers paid out
        """
        # Collect taxes
        self.cash_balance += total_wage_taxes
        self.cash_balance += total_profit_taxes

        # Pay transfers
        self.cash_balance -= total_transfers

    def adjust_policies(self, unemployment_rate: float, inflation_rate: float, deficit_ratio: float, num_unemployed: int = 0) -> None:
        """
        Dynamically adjust government policies based on economic conditions.

        REALISTIC GOVERNMENT BEHAVIOR:
        - Governments tolerate deficits during recessions (counter-cyclical fiscal policy)
        - Tax rates increase when deficits become unsustainable (debt-to-GDP > 100%)
        - Benefits increase during high unemployment, decrease during booms
        - Transfer budgets scale with number of unemployed (not fixed)

        Mutates state.

        Args:
            unemployment_rate: Current unemployment rate (0-1)
            inflation_rate: Current inflation rate (can be negative for deflation)
            deficit_ratio: Government deficit as ratio of total economic activity
            num_unemployed: Actual count of unemployed households (for budget calculation)
        """
        # COUNTER-CYCLICAL UNEMPLOYMENT BENEFITS
        # During recessions, governments increase benefits to stabilize demand
        if unemployment_rate > 0.30:  # Severe recession (>30%)
            # Aggressive stimulus - increase benefits substantially
            self.unemployment_benefit_level = min(60.0, self.unemployment_benefit_level * 1.10)
        elif unemployment_rate > 0.15:  # High unemployment (>15%)
            # Moderate increase in benefits
            self.unemployment_benefit_level = min(50.0, self.unemployment_benefit_level * 1.03)
        elif unemployment_rate < 0.03:  # Very low unemployment (<3%)
            # Can reduce benefits slightly during boom
            self.unemployment_benefit_level = max(20.0, self.unemployment_benefit_level * 0.99)

        # REALISTIC DEFICIT TOLERANCE
        # Governments can run deficits, but must address them if debt becomes unsustainable
        # In real life, deficits of 3-5% of GDP are normal; >10% requires action

        # Only raise taxes if deficit is EXTREME (not just negative)
        # Assume deficit_ratio is calculated as abs(gov_cash) / GDP
        if self.cash_balance < -50000000.0:  # Catastrophic deficit (>$50M)
            # Crisis mode - increase taxes aggressively
            self.wage_tax_rate = min(0.35, self.wage_tax_rate * 1.05)
            self.profit_tax_rate = min(0.40, self.profit_tax_rate * 1.05)
        elif self.cash_balance < -10000000.0:  # Large deficit (>$10M)
            # Moderate tax increases
            self.wage_tax_rate = min(0.30, self.wage_tax_rate * 1.02)
            self.profit_tax_rate = min(0.35, self.profit_tax_rate * 1.02)
        elif self.cash_balance > 50000000.0:  # Large surplus (>$50M)
            # Return surplus to economy via tax cuts
            self.wage_tax_rate = max(0.05, self.wage_tax_rate * 0.98)
            self.profit_tax_rate = max(0.10, self.profit_tax_rate * 0.98)

        # DYNAMIC TRANSFER BUDGET (scales with unemployment)
        # Real governments don't have fixed transfer budgets - spending rises during recessions
        # Set transfer_budget to cover expected unemployment benefits + 50% buffer for gap-filling
        if num_unemployed > 0:
            expected_baseline = num_unemployed * self.unemployment_benefit_level
            self.transfer_budget = expected_baseline * 1.5  # 50% extra for gap-filling
        else:
            # Fallback if num_unemployed not provided (backward compatibility)
            self.transfer_budget = max(10000.0, self.transfer_budget)

    def invest_in_infrastructure(self) -> float:
        """
        Government invests in infrastructure to boost productivity.

        Infrastructure investment increases the productivity multiplier,
        which affects all workers in the economy.

        Mutates state.

        Returns:
            Amount invested in infrastructure
        """
        if self.cash_balance >= self.infrastructure_investment_budget:
            investment = self.infrastructure_investment_budget
            self.cash_balance -= investment

            # Each $1000 invested increases productivity by 0.5%
            productivity_gain = (investment / 1000.0) * 0.005
            self.infrastructure_productivity_multiplier += productivity_gain

            return investment
        return 0.0

    def invest_in_technology(self) -> float:
        """
        Government invests in technology to improve quality.

        Technology investment increases the quality multiplier,
        which affects all firms' product quality.

        Mutates state.

        Returns:
            Amount invested in technology
        """
        if self.cash_balance >= self.technology_investment_budget:
            investment = self.technology_investment_budget
            self.cash_balance -= investment

            # Each $500 invested increases quality by 0.5%
            quality_gain = (investment / 500.0) * 0.005

            # Cap quality multiplier at 1.05 (max 5% quality improvement)
            # This prevents quality from reaching 10.0 (perfection)
            self.technology_quality_multiplier = min(
                1.05,
                self.technology_quality_multiplier + quality_gain
            )

            return investment
        return 0.0

    def invest_in_social_programs(self) -> float:
        """
        Government invests in social programs to improve happiness.

        Social investment increases healthcare, amenities, and other
        quality-of-life factors that boost worker happiness and performance.

        Mutates state.

        Returns:
            Amount invested in social programs
        """
        if self.cash_balance >= self.social_investment_budget:
            investment = self.social_investment_budget
            self.cash_balance -= investment

            # Each $750 invested increases happiness by 0.5%
            happiness_gain = (investment / 750.0) * 0.005
            self.social_happiness_multiplier += happiness_gain

            return investment
        return 0.0

    def make_investments(self) -> Dict[str, float]:
        """
        Execute all government investments if cash is available.

        This should be called each tick to allow government to invest
        surplus funds into the economy.

        Mutates state.

        Returns:
            Dict with investment amounts for each category
        """
        investments = {
            "infrastructure": 0.0,
            "technology": 0.0,
            "social": 0.0,
        }

        # Only invest if government has surplus (positive cash balance)
        if self.cash_balance > 10000.0:  # Keep some reserve
            investments["infrastructure"] = self.invest_in_infrastructure()
            investments["technology"] = self.invest_in_technology()
            investments["social"] = self.invest_in_social_programs()

        return investments
