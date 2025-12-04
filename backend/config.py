"""
Simulation Configuration

Centralizes all tunable parameters for the economic simulation.
This replaces scattered "magic numbers" throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TimeConfig:
    """Time-related constants."""
    ticks_per_year: int = 52  # One tick = one week
    warmup_ticks: int = 52  # First year is warmup period


@dataclass
class HouseholdBehaviorConfig:
    """Household behavioral parameters."""

    # Consumption & Savings
    min_savings_rate: float = 0.1  # 10% minimum savings
    max_savings_rate: float = 0.6  # 60% maximum savings
    personality_buckets: int = 6  # Number of savings personality types
    unemployment_spend_sensitivity: float = 0.8

    # Wealth-based Saving Rate (NEW - for compute_saving_rate method)
    low_wealth_reference: float = 0.0  # Minimum wealth for saving calculation
    high_wealth_reference: float = 10000.0  # Typical high wealth (e.g., 90th percentile)

    # Unemployment-based spending (NEW)
    min_spend_fraction: float = 0.3  # Spend 30% when scared
    confidence_multiplier: float = 0.5  # Up to 80% when confident

    # Subsistence spending floor (prevents hoarding during crisis)
    subsistence_min_cash: float = 50.0  # Always spend at least this much if available
    max_spend_fraction: float = 0.9  # Upper bound on discretionary spending fraction

    # Job acceptance (H1 - no job worse than benefits)
    min_job_premium_over_unemployment: float = 1.05  # Jobs must pay 5% more than unemployment benefits

    # Marginal propensity to consume (H2 - spend from income vs cash)
    mpc_from_wages_employed: float = 0.6  # Employed: save 40% of wages
    mpc_from_transfers_unemployed: float = 0.8  # Unemployed: spend 80% of transfers

    # Saving behavior (H3 - employed vs unemployed)
    max_saving_rate_absolute: float = 0.2  # Hard cap on saving rate
    unemployed_forced_dissaving_wealth: float = 1000.0  # Below this, long-term unemployed can't save
    unemployed_forced_dissaving_duration: float = 50.0  # Ticks before forced dissaving kicks in

    # Job search intensity (H5 - happiness link)
    base_search_intensity: float = 1.0  # Baseline job search effort
    min_search_intensity: float = 0.5  # Minimum (happy & comfortable)
    max_search_intensity: float = 2.0  # Maximum (desperate)

    # Price Elasticity (NEW - Prompt 4)
    food_elasticity: float = 0.5  # Inelastic (necessity)
    services_elasticity: float = 0.8  # Somewhat elastic
    housing_elasticity: float = 1.5  # Elastic (luxury)

    # Substitution Effect (NEW - Prompt 4)
    max_food_budget_share: float = 0.6  # Can use 60% of budget for food if needed

    # Minimum Consumption (Base needs before elasticity)
    min_food_per_tick: float = 2.0
    min_services_per_tick: float = 1.0

    # Price/Wage Expectations
    initial_expected_wage: float = 10.0
    initial_reservation_wage: float = 8.0
    price_expectation_alpha: float = 0.3  # Price belief smoothing
    wage_expectation_alpha: float = 0.2  # Wage belief smoothing
    reservation_markup_over_benefit: float = 1.1  # Reservation = benefit * 1.1
    default_price_level: float = 10.0  # Fallback when no price history
    min_cash_for_aggressive_job_search: float = 100.0  # Desperation threshold

    # Labor Supply Planning
    desperate_wage_adjustment: float = 0.85  # Accept 15% less when desperate
    comfortable_reservation_mix_baseline: float = 0.7
    comfortable_reservation_mix_expected: float = 0.3

    # Wage Expectation Decay (Unemployed)
    duration_pressure_cap: float = 0.35
    duration_pressure_rate: float = 0.01
    happiness_pressure_cap: float = 0.3
    happiness_pressure_rate: float = 0.5
    happiness_threshold: float = 0.7
    base_wage_decay: float = 0.97
    min_decay_factor: float = 0.5
    wage_floor: float = 10.0
    unemployed_market_anchor_weight: float = 0.4
    reservation_adjustment_rate: float = 0.1

    # Skill Development
    skill_growth_rate: float = 0.001  # Passive improvement per tick
    education_cost_per_skill_point: float = 1000.0
    education_skill_gain_rate: float = 0.0001  # 0.1 skill per $1000

    # Affordability Scoring
    affordability_skill_component: float = 1.5
    affordability_cash_divisor: float = 400.0
    affordability_wage_divisor: float = 40.0
    affordability_skill_weight: float = 0.3
    affordability_cash_weight: float = 0.35
    affordability_wage_weight: float = 0.35
    affordability_min_score: float = 0.1
    affordability_max_score: float = 4.0

    # Price Cap Calculation
    min_liquid_cash: float = 25.0
    cash_liquidity_factor: float = 0.2
    base_cap_multiplier_1: float = 1.2
    base_cap_multiplier_2: float = 2.5
    median_cap_factor: float = 0.8
    affordability_premium_threshold: float = 2.0
    premium_liquid_multiplier: float = 1.2
    min_price_cap_buffer: float = 1.1

    # Consumption Adjustment
    price_cap_threshold: float = 0.85
    min_price_sensitivity: float = 0.2
    max_price_sensitivity: float = 1.5
    cap_ratio_scaling: float = 3.0
    min_quantity_scale: float = 0.15

    # Wellbeing Dynamics
    happiness_decay_rate: float = 0.01
    morale_decay_rate: float = 0.02
    health_decay_rate: float = 0.005

    # Wellbeing Updates - Employment
    employed_happiness_boost: float = 0.02
    unemployed_happiness_penalty: float = 0.03

    # Wellbeing Updates - Goods
    high_goods_threshold: float = 10.0
    low_goods_threshold: float = 2.0
    high_goods_happiness_boost: float = 0.01
    low_goods_happiness_penalty: float = 0.02

    # Wellbeing Updates - Government
    government_happiness_scaling: float = 0.05
    government_health_scaling: float = 0.03

    # Wellbeing Updates - Morale
    wage_satisfaction_boost: float = 0.03
    wage_dissatisfaction_scaling: float = 0.05
    unemployment_morale_penalty: float = 0.05

    # Wellbeing Updates - Health
    health_high_goods_threshold: float = 15.0
    health_low_goods_threshold: float = 5.0
    health_high_goods_boost: float = 0.01
    health_low_goods_penalty: float = 0.02

    # Performance Multiplier
    performance_morale_weight: float = 0.5
    performance_health_weight: float = 0.3
    performance_happiness_weight: float = 0.2
    performance_min_multiplier: float = 0.5
    performance_max_multiplier: float = 1.5

    # Goods Consumption
    consumption_rate: float = 0.1  # 10% per tick
    housing_maintenance_rate: float = 0.01  # 1% per tick (10x faster - fixes housing glut)
    inventory_depletion_threshold: float = 0.001

    # Safety Checks
    extreme_negative_cash_threshold: float = -1e6


@dataclass
class FirmBehaviorConfig:
    """Firm behavioral parameters."""

    # Production & Technology
    default_expected_sales: float = 100.0
    default_production_capacity: float = 200.0
    default_productivity_per_worker: float = 10.0
    default_units_per_worker: float = 20.0

    # Diminishing Returns / Productivity
    production_scaling_exponent: float = 0.9  # Legacy exponent (deprecated)
    diminishing_returns_exponent: float = 0.82  # Unified exponent for capacity/productivity (0.80-0.85)
    min_base_productivity: float = 1.0
    min_target_workers: int = 1
    expected_skill_premium: float = 0.3  # Anticipated wage premium over offer
    min_skeleton_workers: int = 3
    minimum_wage_floor: float = 20.0
    min_labor_share: float = 0.5
    max_labor_share: float = 0.8
    burn_mode_grace_period: int = 15
    high_inventory_factor_small: float = 1.5
    high_inventory_factor_large: float = 2.5
    burn_mode_trigger_streak_small: int = 5
    burn_mode_trigger_streak_large: int = 15
    burn_mode_exit_streak: int = 2
    inventory_exit_epsilon: float = 5.0
    min_expected_sales: float = 10.0
    large_market_inventory_relief: float = 1.3
    large_market_burn_mode_buffer: int = 5
    burn_mode_relief_rate: int = 2
    burn_mode_staff_reduction_factor: float = 0.65
    burn_mode_idle_production_fraction: float = 0.05
    target_labor_share: float = 0.65
    margin_low: float = 0.05
    margin_high: float = 0.20
    target_firms_per_1000_households: int = 30
    max_new_firms_per_tick: int = 10
    large_market_household_threshold: int = 2000
    housing_private_saturation_multiplier: float = 3.0

    # Pricing & Costs
    default_unit_cost: float = 5.0
    default_markup: float = 0.3
    default_price: float = 6.5
    default_min_price: float = 5.0

    # PID Pricing (NEW - Prompt 3)
    target_inventory_weeks: float = 2.0  # Target weeks of supply
    days_per_week: float = 7.0
    pid_pressure_decay: float = 0.7  # Integral decay
    pid_integral_gain: float = 0.1  # Integral coefficient
    pid_control_scaling: float = 0.05  # Overall gain
    pid_adjustment_min: float = 0.80  # Max -20% price change
    pid_adjustment_max: float = 1.20  # Max +20% price change
    pid_min_margin: float = 1.05  # 5% above cost minimum
    pid_safety_epsilon: float = 1e-3

    # R&D and Quality
    default_rd_spending_rate: float = 0.05  # 5% of revenue
    quality_improvement_per_rd_dollar: float = 0.0002  # Slowed down 50x (was 0.01)
    quality_decay_rate: float = 0.0  # Quality decay removed
    quality_min: float = 0.0
    quality_max: float = 10.0

    # Adjustment Rates
    sales_expectation_alpha: float = 0.3
    price_adjustment_rate: float = 0.05
    wage_adjustment_rate: float = 0.1
    target_inventory_multiplier: float = 1.5

    # Hiring/Firing Constraints
    default_max_hires_per_tick: int = 2
    default_max_fires_per_tick: int = 2

    # Production Planning
    housing_saturation_threshold: float = 0.2  # 20% of households
    small_positive_production_floor: float = 10.0

    # Wage Planning
    hiring_success_threshold: float = 1.0
    successful_hiring_wage_reduction: float = 0.05
    no_pressure_wage_drift: float = 0.02
    minimum_wage: float = 1.0

    # Wage Stabilization (prevent explosive wage growth)
    max_wage_increase_per_tick: float = 1.15  # Max +15% per tick
    max_wage_decrease_per_tick: float = 0.85  # Max -15% per tick
    unemployment_damping_min: float = 0.3  # Minimum damping factor at high unemployment
    unemployment_damping_rate: float = 0.8  # How much unemployment reduces wage pressure

    # Dividend Distribution
    dividend_cost_reserve_ticks: float = 3.0  # Retain 3 ticks of costs
    dividend_min_safety_reserve: float = 10000.0

    # Personality: Aggressive
    aggressive_investment_propensity: float = 0.15
    aggressive_risk_tolerance: float = 0.9
    aggressive_price_adjustment: float = 0.10
    aggressive_wage_adjustment: float = 0.15
    aggressive_rd_spending: float = 0.08
    aggressive_max_hires: int = 3
    aggressive_max_fires: int = 3
    aggressive_units_per_worker: float = 18.0

    # Personality: Conservative
    conservative_investment_propensity: float = 0.02
    conservative_risk_tolerance: float = 0.2
    conservative_price_adjustment: float = 0.02
    conservative_wage_adjustment: float = 0.05
    conservative_rd_spending: float = 0.02
    conservative_max_hires: int = 1
    conservative_max_fires: int = 1
    conservative_units_per_worker: float = 25.0

    # Personality: Moderate
    moderate_investment_propensity: float = 0.05
    moderate_risk_tolerance: float = 0.5
    moderate_price_adjustment: float = 0.05
    moderate_wage_adjustment: float = 0.1
    moderate_rd_spending: float = 0.05
    moderate_max_hires: int = 2
    moderate_max_fires: int = 2
    moderate_units_per_worker: float = 20.0


@dataclass
class GovernmentPolicyConfig:
    """Government policy parameters."""

    # Tax & Transfer Rates
    default_wage_tax_rate: float = 0.15
    default_profit_tax_rate: float = 0.20
    default_investment_tax_rate: float = 0.10  # Tax on R&D and capital investments
    default_unemployment_benefit: float = 30.0
    default_min_cash_threshold: float = 100.0
    default_transfer_budget: float = 10000.0

    # Wage Floor Policy (minimum wage tied to unemployment benefit)
    wage_floor_multiplier: float = 1.2  # Minimum wage = unemployment_benefit × 1.2

    # Investment Budgets
    infrastructure_investment_budget: float = 1000.0
    technology_investment_budget: float = 500.0
    social_investment_budget: float = 750.0
    investment_reserve_threshold: float = 10000.0  # Don't invest below this

    # Economic Multipliers (Initial)
    initial_infrastructure_multiplier: float = 1.0
    initial_technology_multiplier: float = 1.0
    initial_social_multiplier: float = 1.0
    infra_multiplier_target: float = 1.5
    social_multiplier_target: float = 1.5
    infra_multiplier_hard_cap: float = 2.0
    social_multiplier_hard_cap: float = 2.0
    investment_speed: float = 0.02
    unemployment_cutoff_for_investment: float = 0.3
    emergency_loan_trigger: float = 0.08
    emergency_loan_amount: float = 75000.0
    emergency_loan_cash_threshold: float = 40000.0
    emergency_loan_interest: float = 0.01
    emergency_loan_fraction_of_cash: float = 0.05
    emergency_loan_term_years: float = 2.0
    emergency_loan_required_headcount_multiplier: float = 1.3
    emergency_loan_min_headcount: int = 12
    emergency_loan_enforcement_ticks: int = 26
    emergency_loan_penalty_reclaim_fraction: float = 1.0
    public_works_unemployment_threshold: float = 0.25
    public_works_job_fraction: float = 0.2
    public_works_wage: float = 45.0
    public_works_price: float = 1.0

    # Investment Effects
    infrastructure_gain_divisor: float = 1000.0  # $1000 → 0.5% productivity
    infrastructure_gain_rate: float = 0.005
    technology_gain_divisor: float = 500.0  # $500 → 0.5% quality
    technology_gain_rate: float = 0.005
    technology_max_multiplier: float = 1.05  # Cap at 5% improvement
    social_gain_divisor: float = 750.0  # $750 → 0.5% happiness
    social_gain_rate: float = 0.005

    # Policy Adjustment - Unemployment Thresholds
    high_unemployment_threshold: float = 0.15  # 15%
    low_unemployment_threshold: float = 0.03  # 3%
    high_unemployment_benefit_max: float = 50.0
    high_unemployment_transfer_max: float = 20000.0
    high_unemployment_benefit_increase: float = 1.05
    high_unemployment_transfer_increase: float = 1.1
    low_unemployment_benefit_min: float = 20.0
    low_unemployment_benefit_decrease: float = 0.98

    # Policy Adjustment - Deficit Thresholds
    large_deficit_threshold: float = -10000.0
    large_surplus_threshold: float = 50000.0
    deficit_wage_tax_max: float = 0.30
    deficit_profit_tax_max: float = 0.35
    deficit_tax_increase: float = 1.02
    surplus_wage_tax_min: float = 0.05
    surplus_profit_tax_min: float = 0.10
    surplus_tax_decrease: float = 0.98

    # Policy Adjustment - Counter-Cyclical Transfers
    recession_unemployment_threshold: float = 0.10
    recession_transfer_max: float = 30000.0
    recession_transfer_increase: float = 1.05
    boom_unemployment_threshold: float = 0.05
    boom_cash_threshold: float = 20000.0
    boom_transfer_min: float = 5000.0
    boom_transfer_decrease: float = 0.98


@dataclass
class LaborMarketConfig:
    """Labor market matching parameters."""

    # Wage Percentile Caching
    wage_percentile_cache_interval: int = 5  # Refresh every 5 ticks
    wage_percentile_low: int = 25
    wage_percentile_mid: int = 50
    wage_percentile_high: int = 75

    # Skill-Based Wage Anchoring
    low_skill_threshold: float = 0.4
    high_skill_threshold: float = 0.7

    # Experience & Skill Premiums
    max_skill_premium: float = 0.5  # 50% max for skills
    max_experience_premium: float = 0.5  # 50% max for experience
    experience_premium_per_year: float = 0.05  # 5% per year

    # Production Experience Bonuses
    max_skill_bonus: float = 0.25  # 25% productivity from skills
    experience_bonus_per_year: float = 0.05  # 5% per year


@dataclass
class MarketMechanicsConfig:
    """Market clearing and pricing mechanisms."""

    # Price Ceiling Tax
    price_ceiling: float = 50.0
    price_ceiling_tax_rate: float = 0.25  # 25% on excess revenue

    # Firm Exit/Entry
    bankruptcy_threshold: float = -1000.0
    max_private_competitors: int = 5
    new_firm_demand_threshold: float = 1000.0  # Min household cash

    # New Firm Initialization
    new_firm_initial_cash: float = 2000.0
    new_firm_initial_inventory: float = 25.0
    new_firm_initial_wage: float = 35.0
    new_firm_initial_price: float = 8.0
    new_firm_initial_expected_sales: float = 20.0
    new_firm_initial_capacity: float = 200.0
    new_firm_initial_productivity: float = 8.0
    new_firm_initial_units_per_worker: float = 15.0


@dataclass
class DebugConfig:
    """Debug and anomaly detection settings."""

    # H4: Household income anomaly detection
    large_household_net_change: float = 10000.0  # Flag cash changes above this threshold
    enable_income_tracking: bool = True  # Track wage/transfer/dividend breakdown
    log_large_changes: bool = False  # Log anomalous household income changes


@dataclass
class SimulationConfig:
    """Master configuration for the entire simulation."""

    # Sub-configurations
    time: TimeConfig = field(default_factory=TimeConfig)
    households: HouseholdBehaviorConfig = field(default_factory=HouseholdBehaviorConfig)
    firms: FirmBehaviorConfig = field(default_factory=FirmBehaviorConfig)
    government: GovernmentPolicyConfig = field(default_factory=GovernmentPolicyConfig)
    labor_market: LaborMarketConfig = field(default_factory=LaborMarketConfig)
    market: MarketMechanicsConfig = field(default_factory=MarketMechanicsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Simulation Scale (Legacy - kept for backward compatibility)
    num_households: int = 10000
    num_firms: int = 100

    # Initial Distributions (Legacy)
    initial_cash_min: float = 1000.0
    initial_cash_max: float = 2000.0
    initial_skills_min: float = 0.1
    initial_skills_max: float = 0.9

    def __post_init__(self):
        """Validation and derived values."""
        # Validate time parameters
        if self.time.ticks_per_year <= 0:
            raise ValueError("ticks_per_year must be positive")
        if self.time.warmup_ticks < 0:
            raise ValueError("warmup_ticks cannot be negative")

        # Validate bounds
        if not (0.0 <= self.households.min_savings_rate <= 1.0):
            raise ValueError("min_savings_rate must be in [0, 1]")
        if not (0.0 <= self.households.max_savings_rate <= 1.0):
            raise ValueError("max_savings_rate must be in [0, 1]")

        # Validate elasticities (should be positive)
        if self.households.food_elasticity < 0:
            raise ValueError("food_elasticity must be non-negative")
        if self.households.services_elasticity < 0:
            raise ValueError("services_elasticity must be non-negative")


# Global configuration instance
CONFIG = SimulationConfig()
