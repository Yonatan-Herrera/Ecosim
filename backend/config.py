"""
Simulation Configuration

Centralizes all tunable parameters for the economic simulation.
"""

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SimulationConfig:
    # Simulation dimensions
    num_households: int = 10000
    num_firms: int = 100  # Will be adjusted dynamically or set initially
    
    # Household parameters
    initial_cash_min: float = 1000.0
    initial_cash_max: float = 2000.0
    initial_skills_min: float = 0.1
    initial_skills_max: float = 0.9
    
    # Wellbeing decay rates
    happiness_decay_rate: float = 0.01
    morale_decay_rate: float = 0.02
    health_decay_rate: float = 0.005
    
    # Firm parameters
    price_adjustment_rate: float = 0.05
    wage_adjustment_rate: float = 0.05
    
    # Government parameters
    wage_tax_rate: float = 0.15
    profit_tax_rate: float = 0.20
    unemployment_benefit: float = 40.0
    
    # Market parameters
    price_expectation_alpha: float = 0.3
    wage_expectation_alpha: float = 0.2
    
    # Derived/Advanced config
    production_capacity_per_worker: float = 10.0
    
    def __post_init__(self):
        # Validation or derived values can go here
        pass
