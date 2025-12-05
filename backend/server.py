import asyncio
import json
import logging
import sys
import os
import random

# Add current directory to path so we can import backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CONFIG
from run_large_simulation import create_large_economy, compute_household_stats, compute_firm_stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationManager:
    def __init__(self):
        self.economy = None
        self.is_running = False
        self.tick = 0
        self.logs = []
        self.active_websocket = None
        self.tracked_household_ids: List[int] = []
        self.tracked_firm_ids: List[int] = []
        self.subject_histories: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
        self.firm_histories: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
        self.stabilizer_state = {
            "households": True,
            "firms": True,
            "government": True
        }

    def initialize(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
            
        num_households = config.get("num_households", 1000)
        num_firms = config.get("num_firms", 5)
        
        logger.info(f"Initializing economy with {num_households} households and {num_firms} firms/cat...")
        self.economy = create_large_economy(
            num_households=num_households, 
            num_firms_per_category=num_firms
        )

        disabled_agents: List[str] = []
        if config.get("disable_stabilizers"):
            disabled_agents = config.get("disabled_agents", [])
            if not disabled_agents:
                disabled_agents = ["households", "firms", "government"]
        elif config.get("disabled_agents"):
            disabled_agents = config["disabled_agents"]

        if disabled_agents:
            self.economy.apply_stabilization_overrides(disabled_agents)
            households = not ("all" in disabled_agents or "households" in disabled_agents)
            firms = not ("all" in disabled_agents or "firms" in disabled_agents)
            government = not ("all" in disabled_agents or "government" in disabled_agents)
            self.stabilizer_state = {
                "households": households,
                "firms": firms,
                "government": government
            }
        else:
            self.economy.configure_stabilizers(
                households=True,
                firms=True,
                government=True
            )
            self.stabilizer_state = {
                "households": True,
                "firms": True,
                "government": True
            }
        
        # Apply initial tax rates if provided
        if "wage_tax" in config:
            self.economy.government.wage_tax_rate = config["wage_tax"]
        if "profit_tax" in config:
            self.economy.government.profit_tax_rate = config["profit_tax"]
            
        self.tick = 0
        self.logs = []
        self.gdp_history = [] 
        self.unemployment_history = []
        self.wage_history = []
        self.median_wage_history = []
        self.happiness_history = []
        self.health_history = []
        self.gov_debt_history = []
        self.gov_profit_history = []
        self.firm_count_history = []
        self.net_worth_history = []
        self.gini_history = []
        self.top10_share_history = []
        self.bottom50_share_history = []
        
        # Consolidated histories
        self.price_history = {"food": [], "housing": [], "services": []}
        self.supply_history = {"food": [], "housing": [], "services": []}

        # Select 12 random households to track (more diverse sample)
        import random
        if self.economy.households:
            self.tracked_household_ids = [h.household_id for h in random.sample(self.economy.households, min(12, len(self.economy.households)))]
        else:
            self.tracked_household_ids = []

        # Track historical data for each subject
        self.subject_histories = {hid: {
            "cash": [],
            "wage": [],
            "happiness": [],
            "health": [],
            "netWorth": [],
            "events": []  # Life events (job changes, medical, etc.)
        } for hid in self.tracked_household_ids}

        # Initialize tracked firms (filled below)
        self.tracked_firm_ids = []
        self.firm_histories = {}
        self._select_tracked_firms()
            
        logger.info("Economy initialized")

    def update_stabilizers(self, disable_flag: bool, disabled_agents: List[str]):
        if not self.economy:
            return

        if not disable_flag:
            self.economy.configure_stabilizers(True, True, True)
            self.stabilizer_state = {
                "households": True,
                "firms": True,
                "government": True
            }
            return

        disabled = {agent.lower() for agent in disabled_agents}
        if not disabled:
            disabled = {"all"}
        disable_all = "all" in disabled
        households_enabled = not (disable_all or "households" in disabled)
        firms_enabled = not (disable_all or "firms" in disabled)
        government_enabled = not (disable_all or "government" in disabled)
        self.economy.configure_stabilizers(
            households=households_enabled,
            firms=firms_enabled,
            government=government_enabled
        )
        self.stabilizer_state = {
            "households": households_enabled,
            "firms": firms_enabled,
            "government": government_enabled
        }

    def _select_tracked_firms(self):
        """Ensure tracked firm list highlights top private performers plus baselines."""
        firms = getattr(self.economy, "firms", [])
        if not firms:
            self.tracked_firm_ids = []
            self.firm_histories = {}
            return

        private_firms = [f for f in firms if not getattr(f, "is_baseline", False)]
        baseline_firms = [f for f in firms if getattr(f, "is_baseline", False)]

        private_firms.sort(key=lambda f: f.cash_balance, reverse=True)
        baseline_firms.sort(key=lambda f: f.cash_balance, reverse=True)

        selected = []
        for f in private_firms[:5]:
            selected.append(f.firm_id)
        for f in baseline_firms[:2]:
            if f.firm_id not in selected:
                selected.append(f.firm_id)

        # Fallback to additional private firms if we still have slots
        if len(selected) < 7:
            extra = [f.firm_id for f in private_firms[5:]] + [f.firm_id for f in baseline_firms[2:]]
            for fid in extra:
                if fid not in selected:
                    selected.append(fid)
                if len(selected) >= 7:
                    break

        selected = selected[:7]
        self.tracked_firm_ids = selected

        # Ensure histories exist for new selections
        for fid in selected:
            if fid not in self.firm_histories:
                self.firm_histories[fid] = {
                    "cash": [],
                    "price": [],
                    "wageOffer": [],
                    "inventory": [],
                    "employees": [],
                    "profit": [],
                    "revenue": [],
                    "events": []
                }

    async def run_loop(self):
        if not self.economy:
            logger.warning("Attempted to run loop without economy. Waiting for SETUP.")
            return
        
        logger.info("Starting simulation loop")
        try:
            while self.is_running and self.active_websocket:
                start_time = asyncio.get_event_loop().time()
                
                # Run one step
                self.economy.step()
                self.tick += 1
                
                # Collect metrics using the economy's built-in method
                econ_metrics = self.economy.get_economic_metrics()
                stats = compute_household_stats(self.economy.households)
                firm_stats = compute_firm_stats(self.economy.firms)

                # Calculate GDP (sum of revenue)
                gdp = sum(self.economy.last_tick_revenue.values())
                
                # Calculate Fiscal Balance (Gov Profit)
                current_gov_cash = self.economy.government.cash_balance
                if not hasattr(self, 'prev_gov_cash'):
                    self.prev_gov_cash = current_gov_cash
                fiscal_balance = current_gov_cash - self.prev_gov_cash
                self.prev_gov_cash = current_gov_cash
                
                # Market Metrics (Prices & Supply)
                prices = {"Food": [], "Housing": [], "Services": []}
                supplies = {"Food": 0.0, "Housing": 0.0, "Services": 0.0}
                
                for f in self.economy.firms:
                    if f.good_category in prices:
                        prices[f.good_category].append(f.price)
                        supplies[f.good_category] += f.inventory_units
                        # Add tenants to housing supply (occupied units)
                        if f.good_category == "Housing":
                            supplies["Housing"] += len(f.current_tenants)
                
                # Add household owned housing to supply
                for h in self.economy.households:
                    for good, qty in h.goods_inventory.items():
                        # Simple check if good name contains housing or we assume category
                        if "housing" in good.lower():
                             supplies["Housing"] += qty

                mean_prices = {
                    k: sum(v)/len(v) if v else 0.0 for k, v in prices.items()
                }

                # Calculate Total Net Worth (Cash + Inventory Value)
                total_net_worth = 0.0
                for h in self.economy.households:
                    total_net_worth += h.cash_balance
                    for good, qty in h.goods_inventory.items():
                        # Infer category to get price
                        cat = "Food" # Default
                        lower_good = good.lower()
                        if "housing" in lower_good: cat = "Housing"
                        elif "service" in lower_good: cat = "Services"
                        
                        price = mean_prices.get(cat, 0.0)
                        total_net_worth += qty * price

                # Gather Tracked Subjects Data
                self._select_tracked_firms()

                tracked_subjects = []
                for hid in self.tracked_household_ids:
                    h = self.economy.household_lookup.get(hid)
                    if h:
                        # Infer State
                        state = "IDLE"
                        if h.employer_id:
                            state = "WORKING"
                        elif h.cash_balance < 100:
                            state = "STRESSED"
                        elif h.happiness > 0.8:
                            state = "THRIVING"

                        # Get Employer Name & Category
                        employer_name = "Unemployed"
                        employer_category = None
                        if h.employer_id:
                            employer = self.economy.firm_lookup.get(h.employer_id)
                            if employer:
                                employer_name = employer.good_name
                                employer_category = employer.good_category

                        # Calculate Personal Net Worth
                        personal_net_worth = h.cash_balance
                        for good, qty in h.goods_inventory.items():
                            cat = "Food"
                            if "housing" in good.lower(): cat = "Housing"
                            elif "service" in good.lower(): cat = "Services"
                            personal_net_worth += qty * mean_prices.get(cat, 0.0)

                        # Track history every 50 ticks
                        if self.tick % 50 == 0:
                            if hid in self.subject_histories:
                                self.subject_histories[hid]["cash"].append({"tick": self.tick, "value": h.cash_balance})
                                self.subject_histories[hid]["wage"].append({"tick": self.tick, "value": h.wage})
                                self.subject_histories[hid]["happiness"].append({"tick": self.tick, "value": h.happiness * 100})
                                self.subject_histories[hid]["health"].append({"tick": self.tick, "value": h.health * 100})
                                self.subject_histories[hid]["netWorth"].append({"tick": self.tick, "value": personal_net_worth})

                        # Get recent events
                        recent_events = self.subject_histories.get(hid, {}).get("events", [])[-5:] if hid in self.subject_histories else []

                        tracked_subjects.append({
                            "id": h.household_id,
                            "name": f"Subject-{h.household_id}",
                            "age": h.age,
                            "state": state,
                            "employer": employer_name,
                            "employerCategory": employer_category,
                            "wage": h.wage,
                            "cash": h.cash_balance,
                            "netWorth": personal_net_worth,
                            "happiness": h.happiness,
                            "health": h.health,
                            "morale": h.morale,
                            "skills": h.skills_level,
                            "medicalDebt": h.medical_loan_remaining,
                            "needs": {
                                "food": h.goods_inventory.get("Food", 0) + h.goods_inventory.get("food", 0),
                                "housing": 1 if h.owns_housing or h.renting_from_firm_id else 0
                            },
                            "history": {
                                "cash": self.subject_histories.get(hid, {}).get("cash", []),
                                "wage": self.subject_histories.get(hid, {}).get("wage", []),
                                "happiness": self.subject_histories.get(hid, {}).get("happiness", []),
                                "health": self.subject_histories.get(hid, {}).get("health", []),
                                "netWorth": self.subject_histories.get(hid, {}).get("netWorth", [])
                            },
                            "recentEvents": recent_events
                        })

                tracked_firms = []
                for fid in self.tracked_firm_ids:
                    firm = self.economy.firm_lookup.get(fid)
                    if not firm:
                        continue

                    if firm.cash_balance <= 0 or getattr(firm, "zero_cash_streak", 0) > 2:
                        firm_state = "DISTRESS"
                    elif getattr(firm, "burn_mode", False):
                        firm_state = "BURN"
                    elif firm.planned_hires_count > 0:
                        firm_state = "SCALING"
                    else:
                        firm_state = "STABLE"

                    revenue = getattr(firm, "last_revenue", 0.0)
                    profit = getattr(firm, "last_profit", 0.0)

                    if self.tick % 50 == 0 and fid in self.firm_histories:
                        history = self.firm_histories[fid]
                        history["cash"].append({"tick": self.tick, "value": firm.cash_balance})
                        history["price"].append({"tick": self.tick, "value": firm.price})
                        history["wageOffer"].append({"tick": self.tick, "value": firm.wage_offer})
                        history["inventory"].append({"tick": self.tick, "value": firm.inventory_units})
                        history["employees"].append({"tick": self.tick, "value": len(firm.employees)})
                        history["profit"].append({"tick": self.tick, "value": profit})
                        history["revenue"].append({"tick": self.tick, "value": revenue})

                    tracked_firms.append({
                        "id": firm.firm_id,
                        "name": firm.good_name,
                        "category": firm.good_category,
                        "cash": firm.cash_balance,
                        "inventory": firm.inventory_units,
                        "employees": len(firm.employees),
                        "price": firm.price,
                        "wageOffer": firm.wage_offer,
                        "quality": firm.quality_level,
                        "lastRevenue": revenue,
                        "lastProfit": profit,
                        "state": firm_state,
                        "history": self.firm_histories.get(fid, {})
                    })

                # Update history every 50 ticks
                if self.tick % 50 == 0:
                    self.gdp_history.append({"tick": self.tick, "value": gdp / 1000000.0})
                    self.unemployment_history.append({"tick": self.tick, "value": stats["unemployment_rate"] * 100})
                    self.wage_history.append({"tick": self.tick, "value": stats["mean_wage"]})
                    self.median_wage_history.append({"tick": self.tick, "value": stats["median_wage"]})
                    self.happiness_history.append({"tick": self.tick, "value": stats["mean_happiness"] * 100})
                    self.health_history.append({"tick": self.tick, "value": stats["mean_health"] * 100})
                    self.gov_profit_history.append({"tick": self.tick, "value": fiscal_balance / 1000000.0})
                    self.gov_debt_history.append({"tick": self.tick, "value": -current_gov_cash / 1000000.0 if current_gov_cash < 0 else 0})
                    self.firm_count_history.append({"tick": self.tick, "value": len(self.economy.firms)})
                    self.net_worth_history.append({"tick": self.tick, "value": total_net_worth / 1000000.0})

                    # Wealth inequality metrics
                    self.gini_history.append({"tick": self.tick, "value": econ_metrics.get("gini_coefficient", 0.0)})
                    self.top10_share_history.append({"tick": self.tick, "value": econ_metrics.get("top_10_percent_share", 0.0) * 100})
                    self.bottom50_share_history.append({"tick": self.tick, "value": econ_metrics.get("bottom_50_percent_share", 0.0) * 100})
                    
                    # Consolidated Histories
                    self.price_history["food"].append({"tick": self.tick, "value": mean_prices["Food"]})
                    self.price_history["housing"].append({"tick": self.tick, "value": mean_prices["Housing"]})
                    self.price_history["services"].append({"tick": self.tick, "value": mean_prices["Services"]})
                    
                    self.supply_history["food"].append({"tick": self.tick, "value": supplies["Food"]})
                    self.supply_history["housing"].append({"tick": self.tick, "value": supplies["Housing"]})
                    self.supply_history["services"].append({"tick": self.tick, "value": supplies["Services"]})
                
                # Generate logs (mock/real)
                new_logs = []
                if self.tick % 10 == 0:
                     new_logs.append({"type": "SYS", "txt": f"Tick {self.tick} completed."})
                
                # Construct state update
                state = {
                    "tick": self.tick,
                    "metrics": {
                        "unemployment": stats["unemployment_rate"] * 100,
                        "gdp": gdp / 1000000.0,
                        "govDebt": -self.economy.government.cash_balance / 1000000.0 if self.economy.government.cash_balance < 0 else 0,
                        "govProfit": fiscal_balance / 1000000.0,
                        "happiness": stats["mean_happiness"] * 100,
                        "avgWage": stats["mean_wage"],
                        "netWorth": total_net_worth / 1000000.0,
                        "giniCoefficient": econ_metrics.get("gini_coefficient", 0.0),
                        "top10Share": econ_metrics.get("top_10_percent_share", 0.0) * 100,
                        "bottom50Share": econ_metrics.get("bottom_50_percent_share", 0.0) * 100,
                        "gdpHistory": self.gdp_history,
                        "unemploymentHistory": self.unemployment_history,
                        "wageHistory": self.wage_history,
                        "medianWageHistory": self.median_wage_history,
                        "happinessHistory": self.happiness_history,
                        "healthHistory": self.health_history,
                        "govProfitHistory": self.gov_profit_history,
                        "govDebtHistory": self.gov_debt_history,
                        "firmCountHistory": self.firm_count_history,
                        "netWorthHistory": self.net_worth_history,
                        "giniHistory": self.gini_history,
                        "top10ShareHistory": self.top10_share_history,
                        "bottom50ShareHistory": self.bottom50_share_history,
                        "priceHistory": self.price_history,
                        "supplyHistory": self.supply_history,
                        "trackedSubjects": tracked_subjects,
                        "trackedFirms": tracked_firms
                    },
                    "logs": new_logs,
                    "firm_stats": firm_stats
                }
                
                # Send update
                await self.active_websocket.send_json(state)
                
                # Throttle
                elapsed = asyncio.get_event_loop().time() - start_time
                await asyncio.sleep(max(0.05, 0.1 - elapsed)) # Slightly faster updates
                
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            self.is_running = False
            if self.active_websocket:
                await self.active_websocket.send_json({"error": str(e)})

    def update_config(self, config_data):
        if not self.economy:
            return

        # Tax rates
        if "wageTax" in config_data:
            self.economy.government.wage_tax_rate = config_data["wageTax"]
        if "profitTax" in config_data:
            self.economy.government.profit_tax_rate = config_data["profitTax"]

        # Minimum wage - enforce on all firms
        if "minimumWage" in config_data:
            min_wage = config_data["minimumWage"]
            self.economy.config.labor_market.minimum_wage_floor = min_wage
            # Update all firm wage offers to meet minimum
            for firm in self.economy.firms:
                if firm.wage_offer < min_wage:
                    firm.wage_offer = min_wage

        # Unemployment benefits (percentage of average wage)
        if "unemploymentBenefitRate" in config_data:
            rate = config_data["unemploymentBenefitRate"]
            # Calculate average wage
            total_wages = sum(h.wage for h in self.economy.households if h.is_employed)
            employed_count = sum(1 for h in self.economy.households if h.is_employed)
            avg_wage = total_wages / employed_count if employed_count > 0 else 30.0
            # Set benefit as percentage of average wage
            self.economy.government.unemployment_benefit_level = avg_wage * rate

        # Universal Basic Income (flat payment to all)
        if "universalBasicIncome" in config_data:
            ubi_amount = config_data["universalBasicIncome"]
            # Store UBI amount (will be distributed in economy step)
            if not hasattr(self.economy.government, 'ubi_amount'):
                self.economy.government.ubi_amount = ubi_amount
            else:
                self.economy.government.ubi_amount = ubi_amount

        # Wealth tax threshold
        if "wealthTaxThreshold" in config_data:
            threshold = config_data["wealthTaxThreshold"]
            if not hasattr(self.economy.government, 'wealth_tax_threshold'):
                self.economy.government.wealth_tax_threshold = threshold
            else:
                self.economy.government.wealth_tax_threshold = threshold

        # Wealth tax rate
        if "wealthTaxRate" in config_data:
            rate = config_data["wealthTaxRate"]
            if not hasattr(self.economy.government, 'wealth_tax_rate'):
                self.economy.government.wealth_tax_rate = rate
            else:
                self.economy.government.wealth_tax_rate = rate

        # Inflation rate (annual rate applied gradually)
        if "inflationRate" in config_data:
            inflation = config_data["inflationRate"]
            if not hasattr(self.economy.government, 'target_inflation_rate'):
                self.economy.government.target_inflation_rate = inflation
            else:
                self.economy.government.target_inflation_rate = inflation

        # Birth rate (population growth every 36 ticks)
        if "birthRate" in config_data:
            birth_rate = config_data["birthRate"]
            if not hasattr(self.economy.government, 'birth_rate'):
                self.economy.government.birth_rate = birth_rate
            else:
                self.economy.government.birth_rate = birth_rate

manager = SimulationManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_websocket = websocket
    logger.info("WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            
            if command == "SETUP":
                config = data.get("config", {})
                manager.initialize(config)
                await websocket.send_json({"type": "SETUP_COMPLETE"})
            elif command == "START":
                if not manager.economy:
                     # Auto-initialize if not done yet (fallback)
                     manager.initialize()
                
                if not manager.is_running:
                    manager.is_running = True
                    asyncio.create_task(manager.run_loop())
            elif command == "STOP":
                manager.is_running = False
            elif command == "RESET":
                manager.is_running = False
                # Don't re-initialize immediately, let user go back to setup if they want
                # Or just reset with same config? Let's just reset state for now.
                # Actually, RESET usually means "Stop and clear".
                # If we want to re-configure, we might need a different flow.
                # For now, RESET stops and clears.
                manager.tick = 0
                await websocket.send_json({"type": "RESET", "tick": 0})
            elif command == "CONFIG":
                config_data = data.get("config", {})
                manager.update_config(config_data)
            elif command == "STABILIZERS":
                disable_flag = data.get("disable_stabilizers", False)
                disabled_agents = data.get("disabled_agents", [])
                manager.update_stabilizers(disable_flag, disabled_agents)
                await websocket.send_json({
                    "type": "STABILIZERS_UPDATED",
                    "state": manager.stabilizer_state
                })

    except WebSocketDisconnect:
        manager.is_running = False
        manager.active_websocket = None
        logger.info("Client disconnected")
