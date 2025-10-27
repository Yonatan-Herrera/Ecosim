# from typing import Optional, Dict, Any, Tuple
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field, conint, confloat
# from ai import call_llm
# import json

# app = FastAPI(title="OP EcoSim Backend", version="1.0.0")

# # ---------- Request/Response Models ----------

# class CorporateInput(BaseModel):
#     current_revenue: confloat(ge=0) = Field(..., description="Current revenue in USD")

# class GovernmentInput(BaseModel):
#     current_popularity: confloat(ge=0, le=100) = Field(..., description="Popularity index (0–100)")
#     budget: confloat(ge=0) = Field(..., description="Budget in USD")

# class WorkerInput(BaseModel):
#     compensation: confloat(ge=0) = Field(..., description="Compensation in USD")
#     happiness_level: conint(ge=0, le=10) = Field(..., description="Happiness rating 0–10")

# class RequestModel(BaseModel):
#     corporate: Optional[CorporateInput] = None
#     government: Optional[GovernmentInput] = None
#     workers: Optional[WorkerInput] = None
#     scenario_id: Optional[str] = None
#     notes: Optional[str] = None

# class ResponseBlock(BaseModel):
#     rationale: str
#     next_actions: Dict[str, Any]

# class ResponseModel(BaseModel):
#     scenario_id: Optional[str] = None
#     corporate: Optional[ResponseBlock] = None
#     government: Optional[ResponseBlock] = None
#     workers: Optional[ResponseBlock] = None


# # ---------- Helper ----------

# def _split_rationale_actions(text: str) -> Tuple[str, Dict[str, Any]]:
#     """
#     Extracts a JSON block from the model's text, if present.
#     Returns (rationale_text, parsed_json_dict).
#     """
#     try:
#         start = text.index("{")
#         end = text.rindex("}") + 1
#         actions = json.loads(text[start:end])
#         rationale = text[:start].strip()
#         return rationale, actions
#     except Exception:
#         return text.strip(), {}


# # ---------- Advisor Functions (call the LLM) ----------

# async def advise_corporate(current_revenue: float) -> ResponseBlock:
#     system = (
#         "You are a corporate strategy assistant. Provide concise, actionable, step-by-step recommendations. "
#         "Return a short rationale and a compact JSON of next actions."
#     )
#     user = (
#         f"Current revenue: ${current_revenue:,.2f}.\n"
#         "Advise next steps for the next two quarters. Include: growth levers, KPI targets, and risks.\n"
#         "Respond in two parts:\n"
#         "1) RATIONALE (3–5 sentences)\n"
#         "2) ACTIONS as JSON with keys: plan, kpis, risks"
#     )
#     text = await call_llm(system, user)
#     rationale, actions = _split_rationale_actions(text)
#     return ResponseBlock(rationale=rationale, next_actions=actions)


# async def advise_government(current_popularity: float, budget: float) -> ResponseBlock:
#     system = (
#         "You are a government policy advisor optimizing welfare and political feasibility."
#     )
#     user = (
#         f"Popularity: {current_popularity:.1f}/100\nBudget: ${budget:,.2f}\n"
#         "Recommend next 6-month policy initiatives. Include: cost, expected popularity impact, and comms plan.\n"
#         "Respond in two parts:\n"
#         "1) RATIONALE (3–5 sentences)\n"
#         "2) ACTIONS as JSON with keys: initiatives, impact, comms"
#     )
#     text = await call_llm(system, user)
#     rationale, actions = _split_rationale_actions(text)
#     return ResponseBlock(rationale=rationale, next_actions=actions)


# async def advise_workers(compensation: float, happiness_level: int) -> ResponseBlock:
#     system = "You are a workplace well-being and productivity coach."
#     user = (
#         f"Compensation: ${compensation:,.2f}\nHappiness: {happiness_level}/10\n"
#         "Provide next steps to improve well-being and productivity. Include: habits, negotiation/benefits, and metrics.\n"
#         "Respond in two parts:\n"
#         "1) RATIONALE (2–4 sentences)\n"
#         "2) ACTIONS as JSON with keys: habits, negotiation, metrics"
#     )
#     text = await call_llm(system, user)
#     rationale, actions = _split_rationale_actions(text)
#     return ResponseBlock(rationale=rationale, next_actions=actions)


# # ---------- Main API Endpoint ----------

# @app.post("/recommendations", response_model=ResponseModel)
# async def recommendations(req: RequestModel):
#     """
#     Accepts a JSON payload with any combination of:
#     corporate, government, workers → returns combined LLM recommendations.
#     """
#     if not any([req.corporate, req.government, req.workers]):
#         raise HTTPException(status_code=400, detail="Provide at least one of: corporate, government, or workers.")

#     out = ResponseModel(scenario_id=req.scenario_id)

#     if req.corporate:
#         out.corporate = await advise_corporate(req.corporate.current_revenue)
#     if req.government:
#         out.government = await advise_government(req.government.current_popularity, req.government.budget)
#     if req.workers:
#         out.workers = await advise_workers(req.workers.compensation, req.workers.happiness_level)

#     return out


from typing import Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from ai import call_llm
import json

app = FastAPI(title="OP EcoSim Backend", version="1.0.0")

# ---------- Request/Response Models ----------

class CorporateInput(BaseModel):
    current_revenue: confloat(ge=0)

class GovernmentInput(BaseModel):
    current_popularity: confloat(ge=0, le=100)
    budget: confloat(ge=0)

class WorkerInput(BaseModel):
    compensation: confloat(ge=0)
    happiness_level: conint(ge=0, le=10)

class RequestModel(BaseModel):
    corporate: Optional[CorporateInput] = None
    government: Optional[GovernmentInput] = None
    workers: Optional[WorkerInput] = None
    scenario_id: Optional[str] = None
    notes: Optional[str] = None

class ResponseBlock(BaseModel):
    rationale: str
    next_actions: Dict[str, Any]

class ResponseModel(BaseModel):
    scenario_id: Optional[str] = None
    corporate: Optional[ResponseBlock] = None
    government: Optional[ResponseBlock] = None
    workers: Optional[ResponseBlock] = None

# ---------- Helper ----------

def _split_rationale_actions(text: str) -> Tuple[str, Dict[str, Any]]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        actions = json.loads(text[start:end])
        rationale = text[:start].strip()
        return rationale, actions
    except Exception:
        return text.strip(), {}

# ---------- Advisor Functions ----------

async def advise_corporate(current_revenue: float) -> ResponseBlock:
    system = "You are a corporate strategy assistant. Provide concise, actionable recommendations."
    user = (
        f"Current revenue: ${current_revenue:,.2f}.\n"
        "Advise next steps for the next two quarters in two parts:\n"
        "1) RATIONALE (3–5 sentences)\n2) ACTIONS as JSON with keys: plan, kpis, risks"
    )
    text = await call_llm(system, user)
    rationale, actions = _split_rationale_actions(text)
    return ResponseBlock(rationale=rationale, next_actions=actions)

async def advise_government(current_popularity: float, budget: float) -> ResponseBlock:
    system = "You are a government policy advisor optimizing welfare and political feasibility."
    user = (
        f"Popularity: {current_popularity:.1f}/100\nBudget: ${budget:,.2f}\n"
        "Recommend next 6-month policy initiatives in two parts:\n"
        "1) RATIONALE\n2) ACTIONS as JSON with keys: initiatives, impact, comms"
    )
    text = await call_llm(system, user)
    rationale, actions = _split_rationale_actions(text)
    return ResponseBlock(rationale=rationale, next_actions=actions)

async def advise_workers(compensation: float, happiness_level: int) -> ResponseBlock:
    system = "You are a workplace well-being and productivity coach."
    user = (
        f"Compensation: ${compensation:,.2f}\nHappiness: {happiness_level}/10\n"
        "Provide next steps in two parts:\n1) RATIONALE\n2) ACTIONS as JSON with keys: habits, negotiation, metrics"
    )
    text = await call_llm(system, user)
    rationale, actions = _split_rationale_actions(text)
    return ResponseBlock(rationale=rationale, next_actions=actions)

# ---------- API Endpoint ----------

@app.post("/recommendations", response_model=ResponseModel)
async def recommendations(req: RequestModel):
    if not any([req.corporate, req.government, req.workers]):
        raise HTTPException(status_code=400, detail="Provide at least one of: corporate, government, or workers.")

    out = ResponseModel(scenario_id=req.scenario_id)
    if req.corporate:
        out.corporate = await advise_corporate(req.corporate.current_revenue)
    if req.government:
        out.government = await advise_government(req.government.current_popularity, req.government.budget)
    if req.workers:
        out.workers = await advise_workers(req.workers.compensation, req.workers.happiness_level)

    return out
