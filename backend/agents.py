"""
EcoSim Agent System

This module defines the autonomous agents that will run the economic simulation.
These agents will eventually replace the current recommendation system and actively
make decisions to drive the simulation forward.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import CONFIG


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
    renting_from_firm_id: Optional[int] = None  # Firm ID of housing provider (rental)
    monthly_rent: float = 0.0  # Current rent amount paid per tick

    # H4: Income breakdown tracking (for debugging and anomaly detection)
    last_wage_income: float = 0.0
    last_transfer_income: float = 0.0
    last_dividend_income: float = 0.0
    last_other_income: float = 0.0
    last_consumption_spending: float = 0.0
    last_tick_cash_start: float = 0.0  # Cash at start of tick for change calculation
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
    last_skill_update_tick: int = 0  # Tick when skills were last increased (for rate limiting)
    last_wage_update_tick: int = 0  # Tick when wage premiums were last increased (for rate limiting)

    # Wellbeing and performance factors
    happiness: float = 0.7  # 0-1 scale, affects productivity and consumption
    morale: float = 0.7  # 0-1 scale, affects work performance
    health: float = 1.0  # 0-1 scale, affects productivity and skill development
    unemployment_duration: int = 0  # consecutive ticks without employment

    # Wellbeing dynamics
    happiness_decay_rate: float = 0.01  # Happiness naturally decays without maintenance
    morale_decay_rate: float = 0.02  # Morale decays faster than happiness
    health_decay_rate: float = 0.0  # Dynamic per-tick health decay (set in __post_init__)
    health_decay_per_year: float = 0.0  # Annual health decay characteristic (set in __post_init__)

    # Medical loan tracking
    medical_loan_principal: float = 0.0  # Original medical loan amount
    medical_loan_remaining: float = 0.0  # Remaining balance with interest
    medical_loan_payment_per_tick: float = 0.0  # Payment per tick (10% of wage)

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

        # Initialize health decay characteristic (annual health loss)
        # Distribution: majority lose 0-20 per year, some 20-30, very few 30-50
        rand_val = rng.random()
        if rand_val < 0.70:  # 70% of people: 0-20 health loss per year
            self.health_decay_per_year = rng.uniform(0.0, 0.20)
        elif rand_val < 0.95:  # 25% of people: 20-30 health loss per year
            self.health_decay_per_year = rng.uniform(0.20, 0.30)
        else:  # 5% of people: 30-50 health loss per year (chronic conditions)
            self.health_decay_per_year = rng.uniform(0.30, 0.50)

        # Convert annual decay to per-tick decay (52 ticks per year)
        self.health_decay_rate = self.health_decay_per_year / 52.0

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
        options: List[Dict[str, float]],
        precomputed_prices: Optional[tuple] = None
    ) -> float:
        """
        Determine the maximum acceptable price for a category this tick.
        """
        if precomputed_prices:
            min_price, median_price, max_price = precomputed_prices
        else:
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
        firm_market_info: Dict[str, List[Dict[str, float]]],
        price_cache: Optional[Dict[str, tuple]] = None,
        biased_weights_override: Optional[Dict[str, float]] = None,
        category_fraction_override: Optional[Dict[str, float]] = None,
        category_option_cache: Optional[Dict[str, List[Dict[str, float]]]] = None
    ) -> Dict[int, float]:
        """
        Plan purchases using budget allocations influenced by preferences/traits.
        """
        planned: Dict[int, float] = {}

        if category_fraction_override is not None:
            fractions = {k: v for k, v in category_fraction_override.items() if v > 0}
        elif biased_weights_override is not None:
            biased = dict(biased_weights_override)
            total_bias = sum(biased.values())
            if total_bias <= 0:
                return planned
            fractions = {cat: weight / total_bias for cat, weight in biased.items() if weight > 0}
        else:
            biased = {
                "food": self.category_weights.get("food", 0.0) * self.food_preference,
                "housing": self.category_weights.get("housing", 0.0) * self.housing_preference,
                "services": self.category_weights.get("services", 0.0) * self.services_preference,
            }
            total_bias = sum(biased.values())
            if total_bias <= 0:
                return planned
            fractions = {cat: weight / total_bias for cat, weight in biased.items() if weight > 0}

        housing_share = fractions.pop("housing", 0.0)
        housing_budget_cap = max(0.0, budget * housing_share)
        remaining_budget = budget
        housing_qty_remaining = 1.0

        if housing_budget_cap > 0 and remaining_budget > 0:
            options = category_option_cache.get("housing") if category_option_cache else firm_market_info.get("housing", [])
            if options:
                precomputed = price_cache.get("housing") if price_cache else None
                price_cap = self._get_category_price_cap("housing", options, precomputed_prices=precomputed)
                if price_cap > 0:
                    style = self.purchase_styles.get("housing", self.default_purchase_style)
                    chosen = self._choose_firm_based_on_style(options, style)
                    if chosen and chosen.get("price", 0.0) > 0:
                        price = chosen["price"]
                        allowed_budget = min(remaining_budget, housing_budget_cap)
                        qty = min(housing_qty_remaining, allowed_budget / price)
                        if qty > 0:
                            cost = qty * price
                            remaining_budget = max(0.0, remaining_budget - cost)
                            housing_qty_remaining -= qty
                            firm_id = chosen["firm_id"]
                            planned[firm_id] = planned.get(firm_id, 0.0) + qty

        total_other_share = sum(fractions.values())
        weights_remaining = total_other_share

        for category, share in fractions.items():
            if share <= 0 or remaining_budget <= 0 or weights_remaining <= 0:
                continue
            options = category_option_cache.get(category) if category_option_cache else firm_market_info.get(category, [])
            if not options:
                continue

            precomputed = price_cache.get(category) if price_cache else None
            price_cap = self._get_category_price_cap(category, options, precomputed_prices=precomputed)
            if price_cap <= 0:
                continue

            affordable_options: List[Dict[str, float]] = options
            category_budget = remaining_budget * (share / weights_remaining)
            weights_remaining -= share
            if category_budget <= 0:
                continue

            firm_ids = np.array([opt["firm_id"] for opt in affordable_options], dtype=np.int32)
            prices = np.array([opt["price"] for opt in affordable_options], dtype=np.float64)
            qualities = np.array([opt["quality"] for opt in affordable_options], dtype=np.float64)
            valid_mask = prices > 0
            firm_ids = firm_ids[valid_mask]
            prices = prices[valid_mask]
            qualities = qualities[valid_mask]
            if firm_ids.size == 0:
                continue

            # Utilities and softmax weights
            utilities = (
                self.quality_lavishness * qualities -
                self.price_sensitivity * (prices / max(price_cap, 1e-6))
            )
            rng = random.Random(hash((self.household_id, category)))
            utilities += np.array([rng.uniform(-0.25, 0.25) for _ in range(len(utilities))])
            max_u = utilities.max()
            weights = np.exp(utilities - max_u)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                continue
            shares = weights / weight_sum
            firm_budgets = category_budget * shares
            quantities = firm_budgets / prices
            cap_ratio = prices / price_cap
            sensitivity = max(0.2, min(1.5, self.price_sensitivity))
            adjustments = np.where(
                cap_ratio > 0.85,
                np.maximum(0.15, 1.0 - sensitivity * (cap_ratio - 0.85) * 3.0),
                1.0
            )
            quantities *= adjustments
            spent = 0.0
            for fid, qty, price, adj in zip(firm_ids, quantities, prices, adjustments):
                if qty <= 0:
                    continue
                actual_qty = max(0.0, qty)
                cost = actual_qty * price
                if cost <= 0:
                    continue
                planned[fid] = planned.get(fid, 0.0) + actual_qty
                spent += cost
            remaining_budget = max(0.0, remaining_budget - min(spent, remaining_budget))

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

    @property
    def can_work(self) -> bool:
        """Households can work regardless of health (health affects productivity, not eligibility)."""
        return True

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

    def plan_labor_supply(self, unemployment_benefit: float = 0.0) -> Dict[str, object]:
        """
        Decide whether to search for job and what wage to require.

        H1': Uses the household's reservation_wage field which decays over unemployment duration.
        The reservation wage is set by the economy's _sync_household_expectations method.

        Does not mutate state; returns a plan dict.

        Args:
            unemployment_benefit: Government support (unused - kept for API compatibility)

        Returns:
            Dict with household_id, searching_for_job, reservation_wage, skills_level
        """
        # Note: unemployment_benefit parameter kept for backward compatibility but not used
        # Reservation wage is now managed by the economy's _sync_household_expectations
        # Use the household's reservation wage (set by economy based on employment status)
        # This reservation wage decays from 1.2× benefits (day 1) to 1.05× benefits (long-term)
        reservation_wage_for_tick = self.reservation_wage

        # Desperation adjustment: very low cash makes households accept lower wages
        if self.cash_balance < 200:
            desperation_factor = 0.85  # Accept 15% less when desperate
            reservation_wage_for_tick *= desperation_factor

        # Ensure minimum to survive (living cost floor)
        housing_price = self.price_beliefs.get("housing", self.default_price_level)
        food_price = self.price_beliefs.get("food", self.default_price_level)
        living_cost = 0.3 * housing_price + self.min_food_per_tick * food_price
        living_cost = max(living_cost, 25.0)
        reservation_wage_for_tick = max(reservation_wage_for_tick, living_cost)

        # Search if unemployed or desperate (very low cash)
        if self.cash_balance < living_cost:
            searching_for_job = True
        else:
            searching_for_job = not self.is_employed

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

        NEW APPROACH: Budget scales with total liquid wealth (cash + this tick's wage).
        - Fraction of wealth spent grows with confidence and wealth
        - High-income households (CEOs) now deploy far more capital each tick

        Does not mutate state; returns a plan dict with category budgets.
        Market clearing will handle firm selection within categories.

        Args:
            market_prices: Current market prices for goods (good_name -> price)
            firm_qualities: Quality levels for goods (good_name -> quality) - optional
            firm_categories: Category mappings (good_name -> category) - optional

        Returns:
            Dict with household_id, category_budgets, and legacy planned_purchases
        """
        config = CONFIG.households
        confidence = 1.0 / (1.0 + max(unemployment_rate, 0.0))

        # Spendable base includes cash on hand plus this tick's wage if employed.
        # This ensures CEOs/owners with large balances deploy more capital each tick.
        resource_base = max(0.0, self.cash_balance)
        if self.is_employed:
            resource_base += max(0.0, self.wage)

        if resource_base <= 0.0:
            return {
                "household_id": self.household_id,
                "category_budgets": {},
                "planned_purchases": {},
            }

        wealth_ratio = min(1.0, resource_base / max(1.0, config.high_wealth_reference))
        spend_fraction = config.min_spend_fraction + (config.confidence_multiplier * confidence)

        if self.is_employed:
            saving_rate = self.compute_saving_rate()
            spend_fraction += 0.1  # Stability bonus for steady income
            spend_fraction *= max(0.5, 1.0 - saving_rate)
        else:
            spend_fraction -= 0.05  # Unemployed households stay cautious

        spend_fraction += 0.3 * wealth_ratio  # Wealthy households spend a larger share
        panic_factor = min(1.0, unemployment_rate * config.unemployment_spend_sensitivity)
        spend_fraction *= max(0.2, 1.0 - panic_factor)
        spend_fraction = max(config.min_spend_fraction, min(config.max_spend_fraction, spend_fraction))

        budget = resource_base * spend_fraction
        subsistence_floor = min(resource_base, config.subsistence_min_cash)
        budget = max(budget, subsistence_floor)

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
            housing_infos = []
            other_infos = []
            for good, weight in normalized_weights.items():
                if weight <= 0:
                    continue
                if good in local_beliefs:
                    expected_price = local_beliefs[good]
                elif good in market_prices:
                    expected_price = market_prices[good]
                else:
                    expected_price = self.default_price_level
                if expected_price <= 0:
                    continue
                category = _get_good_category(good, good_category_lookup)
                if category == "housing":
                    housing_infos.append((good, weight, expected_price))
                else:
                    other_infos.append((good, weight, expected_price))

            remaining_budget = budget
            housing_needed = 1.0
            housing_infos.sort(key=lambda item: item[2])
            for good, weight, expected_price in housing_infos:
                if remaining_budget <= 0 or housing_needed <= 0:
                    break
                target_budget = budget * weight if weight > 0 else remaining_budget
                allowed_budget = min(remaining_budget, target_budget)
                if allowed_budget <= 0:
                    continue
                qty = min(housing_needed, allowed_budget / expected_price)
                if qty <= 0:
                    continue
                cost = qty * expected_price
                planned_purchases[good] = planned_purchases.get(good, 0.0) + qty
                remaining_budget = max(0.0, remaining_budget - cost)
                housing_needed -= qty

            weights_remaining = sum(weight for _, weight, _ in other_infos if weight > 0)
            for good, weight, expected_price in other_infos:
                if remaining_budget <= 0 or weight <= 0 or weights_remaining <= 0:
                    break
                share = weight / weights_remaining
                weights_remaining -= weight
                good_budget = remaining_budget * share
                if good_budget <= 0:
                    continue
                qty = good_budget / expected_price
                if qty <= 0:
                    continue
                cost = qty * expected_price
                planned_purchases[good] = planned_purchases.get(good, 0.0) + qty
                remaining_budget = max(0.0, remaining_budget - cost)

            return {
                "household_id": self.household_id,
                "category_budgets": {},  # Empty for legacy mode
                "planned_purchases": planned_purchases,
            }

    def should_spend_on_healthcare(self) -> tuple[bool, float, bool]:
        """
        Determine if household should spend on healthcare and how much.

        Returns:
            (should_spend, amount, needs_loan): Whether to spend, budget, and if loan needed

        Healthcare spending logic:
        - Health < 70%: Should prioritize healthcare
        - Probability of spending increases as health decreases
        - Health < 20%: Always seek care and borrow if needed
        - Cost scales with health recovery needed (exponential)
        - If cost > cash: take medical loan for the shortfall
        """
        import random

        if self.health >= 0.70:
            return False, 0.0, False

        # Probability of seeking healthcare (higher when health is lower)
        # At 70%: ~10% chance, At 40%: ~60% chance, At 20%: ~90% chance
        health_urgency = max(0.0, 0.70 - self.health)  # 0.0 to 0.70
        base_probability = min(0.95, health_urgency / 0.70 * 0.9)  # Scale to 0-90%

        critical = self.health < 0.40
        if critical:
            base_probability = max(base_probability, 0.80)

        if self.health < 0.20:
            base_probability = 1.0

        if random.random() > base_probability:
            return False, 0.0, False

        # Calculate cost of healthcare (exponential scaling)
        # Recovery amount: bring health up to 70% (or current +0.30, whichever is less)
        desired_recovery = min(0.30, max(0.0, 0.70 - self.health))

        # Cost formula: Exponential scaling - recovering more health is much more expensive
        # 10% recovery ≈ 50-100 cash, 30% recovery ≈ 500-1000 cash, 50% recovery ≈ 2000-4000 cash
        base_cost_per_percent = 200.0  # Base cost for 1% health recovery
        exponential_factor = 1.5  # Exponential growth factor
        total_cost = base_cost_per_percent * (desired_recovery * 100) * (1.0 + desired_recovery * exponential_factor)

        # Check if we need a medical loan
        emergency_reserve = self.cash_balance * 0.10
        max_affordable = max(0.0, self.cash_balance - emergency_reserve)

        if total_cost > max_affordable:
            # Need a medical loan for the shortfall
            if self.health < 0.20:
                needs_loan = True
                actual_budget = total_cost
            else:
                needs_loan = True
                actual_budget = total_cost
        else:
            # Can afford it without loan
            needs_loan = False
            actual_budget = total_cost

        # Only proceed if we can afford at least some minimum
        if actual_budget < 20.0:
            return False, 0.0, False

        return True, actual_budget, needs_loan

    def take_medical_loan(self, loan_amount: float) -> None:
        """
        Take out a medical loan to cover healthcare costs.

        Loan terms:
        - Interest rate: 1-3% annually (random, scaled by 52 ticks/year)
        - Repayment: 10% of wage per tick
        - Only available to employed households

        Args:
            loan_amount: Amount to borrow for medical expenses
        """
        import random

        # Random interest rate between 1-3% annually
        annual_interest_rate = random.uniform(0.01, 0.03)

        # Calculate total repayment with interest (simple interest)
        # Total = principal × (1 + annual_rate)
        total_repayment = loan_amount * (1.0 + annual_interest_rate)

        # Set loan terms
        self.medical_loan_principal = loan_amount
        self.medical_loan_remaining = total_repayment
        self.medical_loan_payment_per_tick = 0.0

        # Grant the loan (add to cash balance)
        self.cash_balance += loan_amount

    def make_medical_loan_payment(self) -> float:
        """
        Make a medical loan payment based on minimum wage.

        Returns:
            Amount paid toward loan this tick

        Mutates state by deducting payment from cash and reducing loan balance.
        """
        if self.medical_loan_remaining <= 0:
            return 0.0

        from config import CONFIG
        min_wage = CONFIG.government.default_unemployment_benefit * CONFIG.government.wage_floor_multiplier
        base_payment = 0.10 * min_wage
        payment_amount = min(base_payment, self.medical_loan_remaining, self.cash_balance)

        if payment_amount <= 0:
            return 0.0

        self.cash_balance -= payment_amount
        self.medical_loan_remaining -= payment_amount

        if self.medical_loan_remaining <= 0:
            self.medical_loan_payment_per_tick = 0.0
            self.medical_loan_principal = 0.0

        return payment_amount

    def apply_labor_outcome(
        self,
        outcome: Dict[str, object],
        market_wage_anchor: Optional[float] = None,
        current_tick: int = 0
    ) -> None:
        """
        Update employment status and wage beliefs based on labor market outcome.

        Mutates state.

        Args:
            outcome: Dict with employer_id (int | None), wage (float), and employer_category (str | None)
            market_wage_anchor: Optional market-paid wage to nudge expectations toward
            current_tick: Current simulation tick (for rate-limiting skill/wage growth)
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
            # Only update skills once every 52 ticks (yearly)
            if current_tick - self.last_skill_update_tick >= 52:
                skill_improvement = self.skill_growth_rate * (1.0 - self.skills_level)
                # Apply 52 ticks worth of growth at once
                total_improvement = skill_improvement * 52
                self.skills_level = min(1.0, self.skills_level + total_improvement)
                self.last_skill_update_tick = current_tick

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
            duration_pressure = min(0.45, self.unemployment_duration * 0.02)
            happiness_gap = max(0.0, 0.7 - self.happiness)
            happiness_pressure = min(0.3, happiness_gap * 0.5)
            base_decay = 0.95
            decay_factor = max(
                0.5,
                base_decay - duration_pressure - happiness_pressure
            )
            decayed_expectation = max(self.expected_wage * decay_factor, 5.0)
            if self.unemployment_duration > 52:
                decayed_expectation = min(decayed_expectation, self.expected_wage * 0.85)

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
    quality_improvement_per_rd_dollar: float = 0.0002  # quality points per $ of R&D (slowed 50x)
    quality_decay_rate: float = 0.0  # quality decay removed
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

    # Loan tracking (for government startup loans)
    government_loan_principal: float = 0.0  # Original loan amount
    government_loan_remaining: float = 0.0  # Remaining balance
    loan_payment_per_tick: float = 0.0  # Weekly payment amount
    loan_support_ticks: int = 0  # Ticks remaining to meet hiring commitment
    loan_required_headcount: int = 0  # Target headcount promised when accepting aid
    ceo_household_id: Optional[int] = None  # CEO owner (gets high salary)

    # Housing-specific properties (only for housing firms)
    max_rental_units: int = 0  # Maximum number of tenants (0-50 for housing firms)
    current_tenants: List[int] = field(default_factory=list)  # household_ids renting
    property_tax_rate: float = 0.0  # Annual property tax rate based on units
    age_in_ticks: int = 0
    burn_mode: bool = False
    high_inventory_streak: int = 0
    low_inventory_streak: int = 0
    last_units_sold: float = 0.0
    last_units_produced: float = 0.0  # Track production for pricing decisions
    last_revenue: float = 0.0
    last_profit: float = 0.0
    burn_mode_active: bool = False  # Track whether firm is in inventory burn mode
    zero_cash_streak: int = 0  # Consecutive ticks with zero or negative cash

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
            "owners": list(self.owners),
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
            "price": self.price,
            "unit_cost": self.unit_cost,
            "markup": self.markup,
            "min_price": self.min_price,
            "max_hires_per_tick": self.max_hires_per_tick,
            "max_fires_per_tick": self.max_fires_per_tick,
            "is_baseline": self.is_baseline,
            "baseline_production_quota": self.baseline_production_quota,
            "personality": self.personality,
            "investment_propensity": self.investment_propensity,
            "risk_tolerance": self.risk_tolerance,
            "target_inventory_weeks": self.target_inventory_weeks,
            "price_pressure": self.price_pressure,
            "payout_ratio": self.payout_ratio,
            "net_profit": self.net_profit,
            "last_revenue": self.last_revenue,
            "last_profit": self.last_profit,
            "last_units_sold": self.last_units_sold,
            "government_loan_remaining": self.government_loan_remaining,
            "loan_payment_per_tick": self.loan_payment_per_tick,
            "age_in_ticks": self.age_in_ticks,
            "burn_mode": self.burn_mode,
            "high_inventory_streak": self.high_inventory_streak,
            "low_inventory_streak": self.low_inventory_streak,
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

    # --- Capacity / productivity helpers ---
    def _firm_config(self):
        return CONFIG.firms

    def _capacity_for_workers(self, worker_count: float) -> float:
        """Diminishing-returns capacity frontier for a given workforce."""
        config = self._firm_config()
        if worker_count <= 0:
            return 0.0
        units = max(self.units_per_worker, config.min_base_productivity)
        alpha = max(0.1, min(0.99, config.diminishing_returns_exponent))
        return units * (worker_count ** alpha)

    def _productivity_per_worker(self, worker_count: float) -> float:
        """Average worker productivity implied by the frontier."""
        if worker_count <= 0:
            return 0.0
        return self._capacity_for_workers(worker_count) / worker_count

    def _workers_for_sales(self, target_output: float) -> int:
        """Inverse of the capacity function to meet desired output."""
        config = self._firm_config()
        if target_output <= 0:
            return config.min_target_workers
        units = max(self.units_per_worker, config.min_base_productivity)
        alpha = max(0.1, min(0.99, config.diminishing_returns_exponent))
        required = (target_output / max(units, 1e-6)) ** (1.0 / alpha)
        return max(config.min_target_workers, math.ceil(required))

    def _expected_skill_premium(self) -> float:
        """Baseline expectation for skill + experience wage premia."""
        return self._firm_config().expected_skill_premium

    def _profit_optimal_workers(
        self,
        current_workers: int,
        expected_sales: float,
        effective_wage_cost: float
    ) -> int:
        """Search a small neighborhood for the most profitable staffing level."""
        config = self._firm_config()
        candidate_workers = set()
        for delta in range(-2, 3):
            candidate_workers.add(max(config.min_target_workers, current_workers + delta))
        candidate_workers.add(self._workers_for_sales(expected_sales))

        best_workers = max(config.min_target_workers, current_workers)
        best_profit = -float("inf")
        fixed_cost = getattr(self, "fixed_cost", 0.0)
        for workers in sorted(candidate_workers):
            capacity = self._capacity_for_workers(workers)
            expected_output = min(capacity, expected_sales)
            expected_revenue = expected_output * max(self.price, 0.0)
            expected_wage_bill = workers * effective_wage_cost
            expected_profit = expected_revenue - expected_wage_bill - fixed_cost
            if expected_profit > best_profit:
                best_profit = expected_profit
                best_workers = workers
        return best_workers

    def plan_production_and_labor(
        self,
        last_tick_sales_units: float,
        in_warmup: bool = False,
        total_households: int = 0,
        global_unsold_inventory: float = 0.0,
        private_housing_inventory: float = 0.0,
        large_market: bool = False,
        post_warmup_cooldown: bool = False
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
        firm_config = self._firm_config()
        self.age_in_ticks += 1
        self.last_units_sold = last_tick_sales_units

        smoothed_sales = (
            self.sales_expectation_alpha * last_tick_sales_units +
            (1.0 - self.sales_expectation_alpha) * self.expected_sales_units
        )
        self.expected_sales_units = max(firm_config.min_expected_sales, smoothed_sales)

        if (
            last_tick_sales_units < firm_config.min_expected_sales
            and self.inventory_units < firm_config.inventory_exit_epsilon
        ):
            self.expected_sales_units = max(
                firm_config.min_expected_sales,
                self.expected_sales_units * 0.9
            )

        is_housing_producer = self.good_category.lower() == "housing"
        if is_housing_producer:
            self.expected_sales_units = max(
                firm_config.min_expected_sales,
                float(max(1, self.max_rental_units))
            )
            planned_layoffs = list(self.employees) if self.employees else []
            return {
                "firm_id": self.firm_id,
                "planned_production_units": 0.0,
                "planned_hires_count": 0,
                "planned_layoffs_ids": planned_layoffs,
                "updated_expected_sales": self.expected_sales_units,
            }

        housing_market_saturated = False
        if total_households > 0:
            if private_housing_inventory > total_households * firm_config.housing_private_saturation_multiplier:
                firm_high_inventory = self.inventory_units > 2.0 * max(1.0, self.expected_sales_units)
                if firm_high_inventory:
                    housing_market_saturated = True

        expected_baseline = max(firm_config.min_expected_sales, self.expected_sales_units)
        demand_workers = max(
            firm_config.min_target_workers,
            self._workers_for_sales(min(expected_baseline, self.production_capacity_units))
        )

        if large_market:
            high_inventory_factor = firm_config.high_inventory_factor_large * firm_config.large_market_inventory_relief
            trigger_streak_threshold = (
                firm_config.burn_mode_trigger_streak_large +
                firm_config.large_market_burn_mode_buffer
            )
        else:
            high_inventory_factor = firm_config.high_inventory_factor_small
            trigger_streak_threshold = firm_config.burn_mode_trigger_streak_small

        high_inventory = self.inventory_units > high_inventory_factor * expected_baseline
        low_sellthrough = self.last_units_sold < 0.5 * expected_baseline

        if high_inventory and low_sellthrough:
            self.high_inventory_streak += 1
            self.low_inventory_streak = 0
        else:
            relief = max(1, firm_config.burn_mode_relief_rate)
            self.high_inventory_streak = max(0, self.high_inventory_streak - relief)
            if self.last_units_sold >= 0.8 * expected_baseline:
                self.low_inventory_streak += 1
            else:
                self.low_inventory_streak = max(0, self.low_inventory_streak - relief)

        if (
            not self.burn_mode
            and self.age_in_ticks >= firm_config.burn_mode_grace_period
            and self.high_inventory_streak >= trigger_streak_threshold
        ):
            self.burn_mode = True

        if self.burn_mode and (
            self.low_inventory_streak >= firm_config.burn_mode_exit_streak
            or self.inventory_units < firm_config.inventory_exit_epsilon
        ):
            self.burn_mode = False
            self.high_inventory_streak = 0
            self.low_inventory_streak = 0

        current_workers = len(self.employees)
        planned_hires = 0
        planned_layoffs: List[int] = []
        expected_skill_premium = self._expected_skill_premium()
        effective_wage_cost = self.wage_offer * (1.0 + expected_skill_premium)
        minimum_private_staff = max(10, firm_config.min_target_workers) if not self.is_baseline else firm_config.min_target_workers
        skeleton_min = max(firm_config.min_skeleton_workers, firm_config.min_target_workers)
        if self.loan_required_headcount > 0:
            minimum_private_staff = max(minimum_private_staff, self.loan_required_headcount)
            skeleton_min = max(skeleton_min, min(self.loan_required_headcount, minimum_private_staff))

        scaling_limit = max(5, int(current_workers * 0.10))
        self.burn_mode_active = self.burn_mode

        target_workers = max(current_workers, firm_config.min_target_workers)
        planned_production_units = 0.0

        needs_bootstrap = (not self.is_baseline) and (current_workers < minimum_private_staff) and not self.burn_mode
        if needs_bootstrap:
            target_workers = minimum_private_staff
            planned_hires = min(target_workers - current_workers, scaling_limit)
            planned_production_units = min(
                self._capacity_for_workers(target_workers),
                self.production_capacity_units
            )
        elif self.burn_mode:
            reduction_factor = firm_config.burn_mode_staff_reduction_factor
            reduced_workers = int(math.ceil(max(1, current_workers) * reduction_factor))
            target_workers = max(skeleton_min, reduced_workers)
            idle_fraction = max(0.0, firm_config.burn_mode_idle_production_fraction)
            if idle_fraction > 0:
                planned_production_units = min(
                    self._capacity_for_workers(target_workers),
                    self.production_capacity_units * idle_fraction
                )
            else:
                planned_production_units = 0.0
        elif housing_market_saturated:
            planned_production_units = 0.0
            target_workers = max(skeleton_min, int(current_workers * 0.5))
        elif self.is_baseline:
            if in_warmup:
                planned_hires = 1_000_000_000  # effectively infinite during warm-up
                revenue_per_worker = self.price * self.productivity_per_worker
                self.wage_offer = min(revenue_per_worker * 0.95, 40.0)
                planned_production_units = min(
                    self.production_capacity_units,
                    max(self.baseline_production_quota, self.production_capacity_units)
                )
            else:
                support_ratio = 0.6 if post_warmup_cooldown else 0.25
                support_output = self.baseline_production_quota * support_ratio
                target_output = min(
                    self.production_capacity_units,
                    max(support_output, expected_baseline * 0.8)
                )
                target_workers = self._workers_for_sales(target_output)

                delta = target_workers - current_workers
                if delta > 0:
                    planned_hires = min(delta, scaling_limit)
                elif delta < 0:
                    layoff_count = min(-delta, scaling_limit)
                    if layoff_count > 0:
                        planned_layoffs = self.employees[:layoff_count]

                planned_production_units = min(
                    self._capacity_for_workers(max(target_workers, firm_config.min_target_workers)),
                    target_output
                )
        else:
            additional_output = (
                    self._capacity_for_workers(current_workers + 1) -
                    self._capacity_for_workers(current_workers)
                )
            delta_profit = additional_output * self.price - effective_wage_cost
            demand_target = self._workers_for_sales(min(expected_baseline, self.production_capacity_units))
            profit_target = self._profit_optimal_workers(
                max(current_workers, firm_config.min_target_workers),
                expected_baseline,
                effective_wage_cost
            )

            if delta_profit <= 0:
                target_workers = min(current_workers, profit_target)
            else:
                target_workers = min(demand_target, profit_target)
                target_workers = max(target_workers, minimum_private_staff)

            planned_production_units = min(
                self._capacity_for_workers(max(current_workers, target_workers)),
                self.production_capacity_units
            )

        target_workers = max(target_workers, demand_workers)

        if (not self.is_baseline) and (not is_housing_producer):
            target_workers = max(target_workers, current_workers + 1)

        if self.loan_required_headcount > 0:
            target_workers = max(target_workers, self.loan_required_headcount)

        if not (self.is_baseline and in_warmup):
            delta = target_workers - current_workers
            if delta > 0:
                planned_hires = min(delta, scaling_limit)
            elif delta < 0:
                layoff_count = min(-delta, scaling_limit)
                if layoff_count > 0:
                    planned_layoffs = self.employees[:layoff_count]

        self.planned_hires_count = planned_hires
        self.planned_layoffs_ids = planned_layoffs
        self.last_tick_planned_hires = planned_hires

        return {
            "firm_id": self.firm_id,
            "planned_production_units": planned_production_units,
            "planned_hires_count": planned_hires,
            "planned_layoffs_ids": planned_layoffs,
            "updated_expected_sales": self.expected_sales_units,  # include for later apply
        }

    def plan_pricing(
        self,
        sell_through_rate: float,
        unemployment_rate: float,
        in_warmup: bool = False
    ) -> Dict[str, float]:
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

        import random

        capacity = self._capacity_for_workers(max(len(self.employees), 1))
        sold_ratio = (self.last_units_sold / capacity) if capacity > 0 else 0.0
        inv_ratio = self.inventory_units / max(1.0, self.expected_sales_units)

        up_factor = max(0.2, 1.0 - unemployment_rate)
        down_factor = 1.0 + unemployment_rate

        price_change = 1.0

        # AGGRESSIVE INVENTORY LIQUIDATION LOGIC
        # If inventory > 3× production rate: drastic price cuts (20-30% reduction)
        if self.last_units_produced > 0 and self.inventory_units > 3.0 * self.last_units_produced:
            price_change *= random.uniform(0.70, 0.80)  # 20-30% price cut

        # If items sold < items produced: decrease price by random 0-5%
        elif self.last_units_produced > 0 and self.last_units_sold < self.last_units_produced:
            reduction = random.uniform(0.0, 0.05)
            price_change *= (1.0 - reduction)

        # Original pricing logic (for normal conditions)
        elif sold_ratio < 0.3:
            price_change *= (1.0 - 0.03 * down_factor)
        elif sold_ratio > 0.8 and inv_ratio < 0.5:
            price_change *= (1.0 + 0.02 * up_factor)

        price_change = max(0.5, min(1.1, price_change))  # Allow deeper cuts (down to 50%)
        price_next = max(self.min_price, self.price * price_change)

        # Calculate markup
        if self.unit_cost > 0:
            markup_next = (price_next / self.unit_cost) - 1.0
        else:
            markup_next = self.markup

        return {
            "price_next": price_next,
            "markup_next": markup_next,
        }

    def plan_wage(self, unemployment_rate: float = 0.0, unemployment_benefit: float = 0.0) -> Dict[str, float]:
        firm_config = self._firm_config()
        if self.is_baseline and self.last_tick_planned_hires >= 1_000_000:
            return {"wage_offer_next": self.wage_offer}

        expected_skill_premium = self._expected_skill_premium()
        current_workers = max(len(self.employees), firm_config.min_target_workers)

        if current_workers > 0 and self.last_revenue > 0:
            realized_rev_per_worker = self.last_revenue / current_workers
        else:
            realized_rev_per_worker = self.price * self._productivity_per_worker(max(current_workers, 1))

        margin = 0.0
        if self.last_revenue > 0:
            margin = self.last_profit / max(1.0, self.last_revenue)

        slack_factor = max(0.2, 1.0 - unemployment_rate)
        fundamental_wage = realized_rev_per_worker * firm_config.target_labor_share * slack_factor
        wage_offer_next = self.wage_offer
        raise_damp = max(0.2, 1.0 - 0.8 * unemployment_rate)
        floor_wage = max(firm_config.minimum_wage_floor, unemployment_benefit * 1.5)

        if self.last_revenue <= 1e-3:
            wage_target = min(self.wage_offer, max(floor_wage, fundamental_wage))
            wage_offer_next = 0.9 * self.wage_offer + 0.1 * wage_target
        elif margin <= 0.0:
            wage_target = min(self.wage_offer, fundamental_wage)
            wage_offer_next = 0.9 * self.wage_offer + 0.1 * wage_target
        elif margin < firm_config.margin_low:
            wage_offer_next = 0.95 * self.wage_offer + 0.05 * fundamental_wage
        elif margin < firm_config.margin_high:
            wage_target = 0.9 * self.wage_offer + 0.1 * fundamental_wage
            wage_offer_next = self.wage_offer + (wage_target - self.wage_offer) * raise_damp
        else:
            wage_target = 0.8 * self.wage_offer + 0.2 * fundamental_wage
            wage_offer_next = self.wage_offer + (wage_target - self.wage_offer) * raise_damp

        max_increase = self.wage_offer * 1.15
        max_decrease = self.wage_offer * 0.85
        wage_offer_next = max(max_decrease, min(max_increase, wage_offer_next))

        wage_offer_next = max(floor_wage, wage_offer_next)

        if realized_rev_per_worker > 0:
            max_wage = firm_config.max_labor_share * realized_rev_per_worker
            wage_offer_next = min(wage_offer_next, max_wage)

            min_wage = firm_config.min_labor_share * realized_rev_per_worker
            if wage_offer_next < min_wage and margin > firm_config.margin_low:
                wage_offer_next = min(max_wage, max(min_wage, wage_offer_next))

        if self.last_revenue <= 1e-3:
            wage_offer_next = min(wage_offer_next, self.wage_offer)

        if self.cash_balance <= 0.0:
            wage_offer_next = max(floor_wage, wage_offer_next * 0.95)

        if not self.is_baseline and len(self.employees) == 0:
            wage_offer_next = min(wage_offer_next, 40.0)

        return {"wage_offer_next": wage_offer_next}

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

        # FIX: Update existing workers' wages to meet minimum wage floor
        # This prevents grandfathering of old low wages
        # Minimum wage is set at firm level via wage_offer enforcement
        # But we also need to ensure actual_wages dict is updated
        for worker_id in self.employees:
            if worker_id in self.actual_wages:
                # Ensure existing workers get at least the current wage_offer
                # (which has minimum wage floor already enforced)
                self.actual_wages[worker_id] = max(self.actual_wages[worker_id], self.wage_offer)

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

        # Track production for pricing decisions
        self.last_units_produced = realized_production_units

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

        self.last_units_sold = units_sold
        self.last_revenue = revenue
        profit = revenue - profit_taxes_paid - self.last_tick_total_costs
        self.last_profit = profit
        # Track net profit for dividend policy
        self.net_profit = profit

        if self.cash_balance <= 0.0:
            self.zero_cash_streak += 1
        else:
            self.zero_cash_streak = 0

        # Adjust wages if they exceed 80% of revenue
        self.adjust_wages_to_revenue_ratio(revenue)

    def adjust_wages_to_revenue_ratio(self, revenue: float) -> None:
        """
        Adjust wages if wage bill exceeds 80% of revenue.

        Firms target 70-80% of revenue as wages. If wages exceed 80% of revenue,
        reduce all wages by 10% (floored at minimum wage of $20).

        Args:
            revenue: Revenue from this tick
        """
        if revenue <= 0 or not self.employees:
            return

        # Calculate current wage bill
        wage_bill = 0.0
        for employee_id in self.employees:
            wage_bill += self.actual_wages.get(employee_id, self.wage_offer)

        # Check if wages exceed 80% of revenue
        wage_ratio = wage_bill / revenue
        if wage_ratio > 0.80:
            # Reduce all wages by 10%, floored at minimum wage
            minimum_wage = 20.0
            for employee_id in self.employees:
                current_wage = self.actual_wages.get(employee_id, self.wage_offer)
                reduced_wage = current_wage * 0.9  # 10% reduction
                new_wage = max(reduced_wage, minimum_wage)
                self.actual_wages[employee_id] = new_wage

            # Also reduce wage_offer for new hires
            self.wage_offer = max(self.wage_offer * 0.9, minimum_wage)

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

    def invest_in_unit_expansion(self) -> bool:
        """
        Housing firms can invest in adding more rental units.

        INVESTMENT RULES:
        - Only housing firms can do this
        - Cost increases with each additional unit (diminishing returns)
        - Base cost: $15,000 per unit
        - Cost multiplier: 1.2 ^ (current_units / 10)
        - Firm must have at least 2x the cost in cash

        Returns:
            True if investment was made, False otherwise

        Mutates state.
        """
        if self.good_category != "Housing":
            return False

        # Check if we should expand (high occupancy rate)
        occupancy_rate = len(self.current_tenants) / max(self.max_rental_units, 1)

        if occupancy_rate < 0.85:
            # Not enough demand to justify expansion
            return False

        # Calculate cost with diminishing returns
        base_cost = 15000.0
        cost_multiplier = 1.2 ** (self.max_rental_units / 10.0)
        total_cost = base_cost * cost_multiplier

        # Check if firm can afford it (needs 2x the cost in cash)
        if self.cash_balance < total_cost * 2.0:
            return False

        # Make the investment
        self.cash_balance -= total_cost
        self.max_rental_units += 1
        self.production_capacity_units += 1.0
        self.expected_sales_units += 1.0

        # Property tax increases slightly
        self.property_tax_rate += 0.005  # +0.5% per new unit

        return True

    def apply_rd_and_quality_update(self, revenue: float) -> float:
        """
        Invest in R&D and update quality level.

        Mutates state.

        Args:
            revenue: Revenue from this tick (used to compute R&D spending)

        Returns:
            Amount spent on R&D (to be redirected to Misc firm)
        """
        # Increase R&D when sales are poor (items sold < items produced)
        # This encourages quality improvement to boost sales
        if self.last_units_produced > 0 and self.last_units_sold < self.last_units_produced:
            # Boost R&D spending by 20-30% when underselling
            rd_rate = min(0.15, self.rd_spending_rate * 1.25)  # Cap at 15% of revenue
        else:
            rd_rate = self.rd_spending_rate

        # Compute R&D spending
        rd_spending = revenue * rd_rate
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

        return rd_spending

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


@dataclass(slots=True)
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
    investment_tax_rate: float = 0.10  # [0,1] - Tax on R&D and investments
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

    def get_minimum_wage(self) -> float:
        """
        Calculate the minimum wage floor based on unemployment benefit.

        Minimum wage = unemployment_benefit × wage_floor_multiplier
        Default multiplier is 1.2, so minimum wage is 20% above unemployment benefit.

        This prevents situations where firms pay 1 worker $400 instead of
        hiring 5 workers at $80 each, and ensures jobs are worthwhile vs. unemployment.

        Returns:
            Minimum wage that firms must pay
        """
        # Default multiplier if config not available
        multiplier = 1.2
        return self.unemployment_benefit_level * multiplier

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

        # WEALTH-BASED PROGRESSIVE TAXATION FOR FIRMS
        # Use quartiles of firm cash balance to determine tax brackets
        # This ensures consistent progressive taxation regardless of absolute wealth levels

        firm_cash = [f.get("cash_balance", 0.0) for f in firms]

        if firm_cash and len(firm_cash) >= 4:
            firm_cash_sorted = sorted(firm_cash)
            n = len(firm_cash_sorted)

            # Calculate percentile thresholds
            q1 = firm_cash_sorted[int(0.25 * (n - 1))]  # 25th percentile (poor)
            q2 = firm_cash_sorted[int(0.50 * (n - 1))]  # 50th percentile (average)
            q3 = firm_cash_sorted[int(0.75 * (n - 1))]  # 75th percentile (rich)
            p90 = firm_cash_sorted[int(0.90 * (n - 1))]  # 90th percentile (very rich)
            p99 = firm_cash_sorted[int(0.99 * (n - 1))]  # 99th percentile (ultra rich - TOP 1%)

            # Initialize random tax rate modifiers (deterministic per simulation)
            import random
            rng = random.Random(54321)  # Fixed seed for consistency

            # Base profit tax rate (for average firms in Q2-Q3 range)
            base_rate = self.profit_tax_rate

            # Random additional tax for each bracket
            # Top 1%: base + (20-35% extra) - MASSIVE wealth tax on ultra-rich
            top_1_extra = rng.uniform(0.20, 0.35)
            top_1_rate = min(0.60, base_rate + top_1_extra)  # Cap at 60%

            # Very rich (top 10%): base + (10-20% extra)
            very_rich_extra = rng.uniform(0.10, 0.20)
            very_rich_rate = base_rate + very_rich_extra

            # Rich (top 25%): base + (5% to very_rich_extra - 1%)
            rich_extra = rng.uniform(0.05, max(0.06, very_rich_extra - 0.01))
            rich_rate = base_rate + rich_extra

            # Average: base rate (Q2-Q3)
            average_rate = base_rate

            # Poor: base - (0-5%)
            poor_discount = rng.uniform(0.0, 0.05)
            poor_rate = max(0.01, base_rate - poor_discount)
        else:
            # Not enough firms for quartiles, use base rate
            q1 = q2 = q3 = p90 = p99 = 0.0
            poor_rate = average_rate = rich_rate = very_rich_rate = top_1_rate = self.profit_tax_rate

        # Apply wealth-based tax rates to each firm
        for firm in firms:
            firm_id = firm["firm_id"]
            profit_before_tax = firm.get("profit_before_tax", 0.0)
            cash_balance = firm.get("cash_balance", 0.0)

            # Determine tax rate based on wealth percentile
            if cash_balance <= q1:
                # Poor firms (bottom 25%)
                rate = poor_rate
            elif cash_balance <= q2:
                # Below average firms (25-50%)
                rate = average_rate * 0.9  # Slight discount
            elif cash_balance <= q3:
                # Above average firms (50-75%)
                rate = average_rate
            elif cash_balance <= p90:
                # Rich firms (75-90%)
                rate = rich_rate
            elif cash_balance <= p99:
                # Very rich firms (90-99%)
                rate = very_rich_rate
            else:
                # TOP 1% - Ultra rich firms get hit with massive wealth tax
                rate = top_1_rate

            # Calculate total profit tax
            profit_taxes[firm_id] = max(profit_before_tax * rate, 0.0)

        # Calculate property taxes for housing firms
        property_taxes = {}
        for firm in firms:
            firm_id = firm["firm_id"]
            if firm.get("good_category") == "Housing" and firm.get("property_tax_rate", 0.0) > 0:
                # Property tax is annual rate, so divide by 52 for weekly payment
                weekly_property_tax = firm["property_tax_rate"] * firm["cash_balance"] / 52.0
                property_taxes[firm_id] = max(weekly_property_tax, 0.0)

        return {
            "wage_taxes": wage_taxes,
            "profit_taxes": profit_taxes,
            "property_taxes": property_taxes,
        }

    def apply_fiscal_results(
        self,
        total_wage_taxes: float,
        total_profit_taxes: float,
        total_transfers: float,
        total_property_taxes: float = 0.0
    ) -> None:
        """
        Update government cash based on fiscal operations.

        Mutates state.

        Args:
            total_wage_taxes: Sum of all wage taxes collected
            total_profit_taxes: Sum of all profit taxes collected
            total_transfers: Sum of all transfers paid out
            total_property_taxes: Sum of all property taxes collected (from housing firms)
        """
        # Collect taxes
        self.cash_balance += total_wage_taxes
        self.cash_balance += total_profit_taxes
        self.cash_balance += total_property_taxes

        # Pay transfers
        self.cash_balance -= total_transfers

    def adjust_policies(self, unemployment_rate: float, inflation_rate: float, deficit_ratio: float, num_unemployed: int = 0, gdp: float = 0.0, total_tax_revenue: float = 0.0, num_bankrupt_firms: int = 0) -> None:
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

        # Dynamic tax policy based on cash flow and GDP
        from config import CONFIG
        gov_cfg = CONFIG.government

        if gdp > 0.0 and total_tax_revenue < -gdp:
            rng = random.Random(777)
            bump = rng.uniform(0.0, 0.08)
            self.wage_tax_rate = min(gov_cfg.deficit_wage_tax_max, self.wage_tax_rate * (1.0 + bump))
            self.profit_tax_rate = min(gov_cfg.deficit_profit_tax_max, self.profit_tax_rate * (1.0 + bump))
        elif self.cash_balance > 0.0:
            rng = random.Random(778)
            cut = rng.uniform(0.0, 0.08)
            self.wage_tax_rate = max(gov_cfg.surplus_wage_tax_min, self.wage_tax_rate * (1.0 - cut))
            self.profit_tax_rate = max(gov_cfg.surplus_profit_tax_min, self.profit_tax_rate * (1.0 - cut))

        if num_bankrupt_firms > 0:
            rng = random.Random(779)
            relief = rng.uniform(0.0, 0.04)
            self.wage_tax_rate = max(gov_cfg.surplus_wage_tax_min, self.wage_tax_rate * (1.0 - relief))
            self.profit_tax_rate = max(gov_cfg.surplus_profit_tax_min, self.profit_tax_rate * (1.0 - relief))

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
        Government bond purchases with surplus funds (removed infrastructure/social programs).

        When government has surplus, it purchases bonds from Misc firm,
        which distributes the money to households (1 person per tick).

        Mutates state.

        Returns:
            Dict with "bonds" key containing amount spent on bond purchases
        """
        investments = {"bonds": 0.0}

        # Define surplus threshold as percentage of cash balance
        surplus_threshold_pct = 0.20  # Consider 20%+ above baseline as surplus
        baseline_reserve = 50000.0  # Minimum reserve to maintain

        # Calculate surplus (scaled, not fixed)
        if self.cash_balance < baseline_reserve:
            return investments

        surplus = max(0.0, self.cash_balance - baseline_reserve)

        # Spend 10-15% of surplus on bonds each tick (scaled)
        if surplus > baseline_reserve * surplus_threshold_pct:
            bond_purchase_rate = 0.12  # 12% of surplus per tick
            bond_spend = surplus * bond_purchase_rate

            self.cash_balance -= bond_spend
            investments["bonds"] = bond_spend

        return investments
