"""
EcoSim Agent System

This module defines the autonomous agents that will run the economic simulation.
These agents will eventually replace the current recommendation system and actively
make decisions to drive the simulation forward.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
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
    expected_wage: float = 50.0  # initial default wage expectation
    reservation_wage: float = 40.0  # minimum acceptable wage

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

    def _initialize_personality_preferences(self) -> None:
        """Deterministically assign savings, weights, and purchase styles."""
        if self.savings_rate_target is None:
            bucket = (self.household_id % 6) / 10
            self.savings_rate_target = 0.1 + bucket
        self.savings_rate_target = max(0.1, min(0.6, self.savings_rate_target))

        if not self.category_weights:
            base_categories = ["food", "housing", "services"]
            self.category_weights = {cat: 1.0 / len(base_categories) for cat in base_categories}
        self.category_weights = self._normalize_category_weights(self.category_weights)

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

        return max(min_price * 1.1, price_cap)

    def _plan_category_purchases(
        self,
        budget: float,
        firm_market_info: Dict[str, List[Dict[str, float]]]
    ) -> Dict[int, float]:
        planned: Dict[int, float] = {}
        for category, weight in self.category_weights.items():
            if weight <= 0:
                continue

            category_budget = budget * weight
            if category_budget <= 0:
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
        # Baseline reservation from government support
        baseline_reservation = unemployment_benefit * self.reservation_markup_over_benefit

        # Adjust based on cash position
        if self.cash_balance < self.min_cash_for_aggressive_job_search:
            # Desperate: willing to accept less
            adjustment_factor = 0.85  # deterministic downward adjustment
            reservation_wage_for_tick = baseline_reservation * adjustment_factor
        else:
            # Comfortable: can be pickier, nudge toward expected wage
            reservation_wage_for_tick = 0.7 * baseline_reservation + 0.3 * self.expected_wage

        # Decide if searching
        searching_for_job = not self.is_employed

        return {
            "household_id": self.household_id,
            "searching_for_job": searching_for_job,
            "reservation_wage": reservation_wage_for_tick,
            "skills_level": self.skills_level,
        }

    def plan_consumption(
        self,
        market_prices: Dict[str, float],
        firm_qualities: Dict[str, float] = None,
        firm_categories: Dict[str, str] = None,
        firm_market_info: Optional[Dict[str, List[Dict[str, float]]]] = None
    ) -> Dict[str, object]:
        """
        Decide desired budget allocation across categories.

        Does not mutate state; returns a plan dict with category budgets.
        Market clearing will handle firm selection within categories.

        Args:
            market_prices: Current market prices for goods (good_name -> price)
            firm_qualities: Quality levels for goods (good_name -> quality) - optional
            firm_categories: Category mappings (good_name -> category) - optional

        Returns:
            Dict with household_id, category_budgets, and legacy planned_purchases
        """
        # Compute consumption budget based on savings target
        spend_fraction = 1.0 - (self.savings_rate_target or self.consumption_budget_share)
        spend_fraction = max(0.0, min(1.0, spend_fraction))
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
                planned_quantity = good_budget / expected_price

                if planned_quantity > 0:
                    planned_purchases[good] = planned_quantity

            return {
                "household_id": self.household_id,
                "category_budgets": {},  # Empty for legacy mode
                "planned_purchases": planned_purchases,
            }

    def apply_labor_outcome(self, outcome: Dict[str, object]) -> None:
        """
        Update employment status and wage beliefs based on labor market outcome.

        Mutates state.

        Args:
            outcome: Dict with employer_id (int | None), wage (float), and employer_category (str | None)
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
            self.expected_wage = max(self.expected_wage * decay_factor, 10.0)

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

    def apply_purchases(self, purchases: Dict[str, tuple[float, float]]) -> None:
        """
        Update inventory, cash, and price beliefs based on executed purchases.

        Mutates state.

        Args:
            purchases: Dict mapping good_name -> (quantity, price_paid)
        """
        for good, (quantity, price_paid) in purchases.items():
            # Update cash
            total_cost = quantity * price_paid
            self.cash_balance -= total_cost

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

    def consume_goods(self) -> None:
        """
        Consume goods from inventory each tick.

        Households consume a fraction of their goods inventory each tick
        to represent using up food, services, housing, etc.

        Mutates state.
        """
        consumption_rate = 0.1  # Consume 10% of inventory per tick

        for good in list(self.goods_inventory.keys()):
            if self.goods_inventory[good] > 0:
                consumed = self.goods_inventory[good] * consumption_rate
                self.goods_inventory[good] = max(0.0, self.goods_inventory[good] - consumed)

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


@dataclass
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

    # Firm personality & strategy
    # "aggressive": High risk, high reward - invests heavily, adjusts prices aggressively
    # "conservative": Low risk, stable - minimal investment, gradual adjustments
    personality: str = "moderate"  # "aggressive", "moderate", or "conservative"
    investment_propensity: float = 0.05  # Fraction of profits to invest (varies by personality)
    risk_tolerance: float = 0.5  # 0-1 scale, affects pricing and hiring decisions
    is_baseline: bool = False
    baseline_production_quota: float = 500.0
    actual_wages: Dict[int, float] = field(default_factory=dict)

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
        in_warmup: bool = False
    ) -> Dict[str, object]:
        """
        Decide how much to produce and how many workers are needed.

        REDESIGNED: Just-in-time production - produce to replace what was sold,
        not to build inventory. This creates continuous employment demand.

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

        current_workers = len(self.employees)
        planned_hires = 0
        planned_layoffs: List[int] = []

        if self.is_baseline and in_warmup:
            planned_production_units = min(
                self.baseline_production_quota,
                self.production_capacity_units
            )
            target_workers = max(1, math.ceil(
                planned_production_units / max(self.units_per_worker, 1.0)
            ))
            if target_workers > current_workers:
                planned_hires = min(target_workers - current_workers, self.max_hires_per_tick)
            elif target_workers < current_workers:
                layoff_count = min(current_workers - target_workers, self.max_fires_per_tick)
                planned_layoffs = self.employees[:layoff_count]
        else:
            small_positive_floor = 10.0
            smoothed_sales = max(updated_expected_sales, small_positive_floor)
            target_workers = max(1, math.ceil(
                smoothed_sales / max(self.units_per_worker, 1.0)
            ))
            delta = target_workers - current_workers
            if delta > 0:
                planned_hires = min(delta, self.max_hires_per_tick)
            elif delta < 0:
                layoff_count = min(-delta, self.max_fires_per_tick)
                planned_layoffs = self.employees[:layoff_count]

            expected_workforce = max(0, current_workers + planned_hires - len(planned_layoffs))
            planned_production_units = min(
                max(expected_workforce * self.units_per_worker, small_positive_floor),
                self.production_capacity_units
            )

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
        REDESIGNED: Demand-based pricing that responds to what people are willing to buy.

        Price adjusts based on sales velocity relative to expected sales.
        Higher demand = higher prices, lower demand = lower prices.

        Does not mutate state; returns a plan dict.

        Args:
            sell_through_rate: Fraction of available inventory that was sold [0,1]

        Returns:
            Dict with price_next and markup_next
        """
        if self.is_baseline and in_warmup:
            return {
                "price_next": self.price,
                "markup_next": self.markup,
            }

        # Aggressive demand-based pricing
        # Sell-through > 0.9: Very high demand, raise prices significantly
        # Sell-through 0.7-0.9: Good demand, raise prices moderately
        # Sell-through 0.5-0.7: Moderate demand, keep prices stable
        # Sell-through 0.3-0.5: Weak demand, lower prices moderately
        # Sell-through < 0.3: Very weak demand, lower prices significantly

        if sell_through_rate > 0.9:
            # Very high demand - aggressive price increase
            price_change_factor = 1.0 + (self.price_adjustment_rate * 3.0)
        elif sell_through_rate > 0.7:
            # Good demand - moderate price increase
            price_change_factor = 1.0 + (self.price_adjustment_rate * 1.5)
        elif sell_through_rate > 0.5:
            # Moderate demand - small price increase
            price_change_factor = 1.0 + (self.price_adjustment_rate * 0.5)
        elif sell_through_rate > 0.3:
            # Weak demand - moderate price decrease
            price_change_factor = 1.0 - (self.price_adjustment_rate * 1.0)
        else:
            # Very weak demand - aggressive price decrease
            price_change_factor = 1.0 - (self.price_adjustment_rate * 2.0)

        # Apply demand-based pricing
        price_next = self.price * price_change_factor

        # Enforce minimum price floor (must cover some costs)
        price_next = max(price_next, self.min_price)

        # Calculate implied markup for tracking
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
        Adjust wage offer based on past hiring success.

        Does not mutate state; returns a plan dict.

        Returns:
            Dict with wage_offer_next
        """
        if self.last_tick_planned_hires > 0:
            hiring_denominator = max(self.last_tick_planned_hires, 1)
            hiring_success = self.last_tick_actual_hires / hiring_denominator
            if hiring_success < 1.0:
                pressure = 1.0 - hiring_success
                wage_offer_next = self.wage_offer * (1.0 + self.wage_adjustment_rate * pressure)
            else:
                wage_offer_next = self.wage_offer * (1.0 - 0.05 * self.wage_adjustment_rate)
        else:
            # Gradual downward drift if no hiring pressure
            wage_offer_next = self.wage_offer * (1.0 - 0.02 * self.wage_adjustment_rate)

        wage_offer_next = max(wage_offer_next, 1.0)

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
        wage_taxes = {}
        profit_taxes = {}

        # Compute wage taxes for each household
        for household in households:
            household_id = household["household_id"]
            wage_income = household.get("wage_income", 0.0)
            wage_tax = max(wage_income * self.wage_tax_rate, 0.0)
            wage_taxes[household_id] = wage_tax

        # Compute profit taxes for each firm
        for firm in firms:
            firm_id = firm["firm_id"]
            profit_before_tax = firm.get("profit_before_tax", 0.0)
            profit_tax = max(profit_before_tax * self.profit_tax_rate, 0.0)
            profit_taxes[firm_id] = profit_tax

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

    def adjust_policies(self, unemployment_rate: float, inflation_rate: float, deficit_ratio: float) -> None:
        """
        Dynamically adjust government policies based on economic conditions.

        Mutates state.

        Args:
            unemployment_rate: Current unemployment rate (0-1)
            inflation_rate: Current inflation rate (can be negative for deflation)
            deficit_ratio: Government deficit as ratio of total economic activity
        """
        # Adjust unemployment benefits based on unemployment rate
        if unemployment_rate > 0.15:  # High unemployment (>15%)
            # Increase benefits to support unemployed
            self.unemployment_benefit_level = min(50.0, self.unemployment_benefit_level * 1.05)
            self.transfer_budget = min(20000.0, self.transfer_budget * 1.1)
        elif unemployment_rate < 0.03:  # Very low unemployment (<3%)
            # Can reduce benefits slightly
            self.unemployment_benefit_level = max(20.0, self.unemployment_benefit_level * 0.98)

        # Adjust tax rates based on fiscal health
        if self.cash_balance < -10000.0:  # Large deficit
            # Increase taxes to reduce deficit
            self.wage_tax_rate = min(0.30, self.wage_tax_rate * 1.02)
            self.profit_tax_rate = min(0.35, self.profit_tax_rate * 1.02)
        elif self.cash_balance > 50000.0:  # Large surplus
            # Can afford to reduce taxes
            self.wage_tax_rate = max(0.05, self.wage_tax_rate * 0.98)
            self.profit_tax_rate = max(0.10, self.profit_tax_rate * 0.98)

        # Counter-cyclical transfer budget
        if unemployment_rate > 0.10 and self.cash_balance > 0:
            # Recession - increase transfers
            self.transfer_budget = min(30000.0, self.transfer_budget * 1.05)
        elif unemployment_rate < 0.05 and self.cash_balance > 20000.0:
            # Boom - can reduce transfers
            self.transfer_budget = max(5000.0, self.transfer_budget * 0.98)

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
