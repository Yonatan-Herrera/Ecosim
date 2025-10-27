# import os
# import httpx 
# import dotenv 
# from dotenv import load_dotenv

# load_dotenv ()
# openrounter_api_key = os.getenv ("openrounter_api_key")
# model = os.getenv ("openrounter_motor","nvidia/nemotron-nano-9b-v2:free" )
# base_url = "https://openrouter.ai/api/v1"

# if not openrounter_api_key:
#     raise ValueError ("openrounter_api_key not found in .env file")

# #http headers for authorization (OpenRouter)
# headers = {
#   "authorization": f"Bearer {openrounter_api_key}",
#   "Content-Type": "application/json",
#   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
# }

# def build_payload(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
#     """Create JSON body for OpenRouter request."""
#     return {
#         "model": MODEL,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": temperature,
#     }

# async def send_request(payload: dict) -> dict:
#     """Send async POST request to OpenRouter and return parsed JSON."""
#     async with httpx.AsyncClient(timeout=60) as client:
#         r = await client.post(BASE_URL, headers=HEADERS, json=payload)
#         r.raise_for_status()
#         return r.json()

# def extract_text(response_json: dict) -> str:
#     """Extract the assistant’s text content from the OpenRouter response."""
#     return response_json["choices"][0]["message"]["content"]

# # --- Main callable used by other files ---

# async def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
#     """Builds payload, sends to OpenRouter, and returns model’s text output."""

#     payload = build_payload(system_prompt, user_prompt, temperature)
#     response_json = await send_request(payload)
#     return extract_text(response_json)


import os
import httpx
import dotenv
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-9b-v2:free")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

def build_payload(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

async def send_request(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(BASE_URL, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()

def extract_text(response_json: dict) -> str:
    return response_json["choices"][0]["message"]["content"]

async def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    payload = build_payload(system_prompt, user_prompt, temperature)
    response_json = await send_request(payload)
    return extract_text(response_json)

    