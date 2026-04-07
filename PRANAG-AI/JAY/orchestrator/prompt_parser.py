# orchestrator/prompt_parser.py
# Converts a raw user prompt into a structured ParsedPrompt object using an LLM.
#
# Flow:
#   raw text → LLM (DeepSeek-R1 via Ollama) → JSON string → ParsedPrompt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import requests
from typing import Optional
from JAY.shared.models import ParsedPrompt, CropType
from JAY.shared.config import OLLAMA_BASE_URL, OLLAMA_MODEL


SYSTEM_PROMPT = """
You are an agricultural science expert. Extract crop parameters from the user prompt.
Respond ONLY with a valid JSON object — no preamble, no explanation, no markdown.

Required keys:
{
  "crop_type": one of [wheat, rice, maize, sorghum, barley, cotton, soybean, sugarcane, unknown],
  "location_raw": "city or region name",
  "temp_celsius": number or null,
  "stress_hints": ["heat", "drought", ...],
  "variety_hint": "variety name" or null,
  "raw_entities": {}
}
"""


def call_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Send prompt to local Ollama server and return the raw text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9}
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        print("[PARSER] Ollama not running — using fallback parser.")
        return _fallback_parse(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def _fallback_parse(prompt: str) -> str:
    """
    Regex-based fallback parser used when no LLM server is available.
    Handles demos and testing without a GPU.
    """
    p = prompt.lower()

    crop = "unknown"
    for c in ["wheat", "rice", "maize", "sorghum", "barley", "cotton", "soybean"]:
        if c in p:
            crop = c
            break

    temp_match = re.search(r"(\d+)\s*(?:°c|°|celsius|degrees?)", p)
    temp = float(temp_match.group(1)) if temp_match else None

    loc_match = re.search(r"(?:for|in)\s+([A-Z][a-z]+)", prompt)
    location  = loc_match.group(1) if loc_match else "Unknown"

    stresses = []
    if temp and temp > 38:
        stresses.append("heat")
    if "drought" in p or "dry" in p:
        stresses.append("drought")

    return json.dumps({
        "crop_type":    crop,
        "location_raw": location,
        "temp_celsius": temp,
        "stress_hints": stresses,
        "variety_hint": None,
        "raw_entities": {"source": "fallback_parser"}
    })


def extract_json(text: str) -> dict:
    """Strip markdown fences and parse the first JSON object found in text."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def parse_prompt(user_prompt: str) -> ParsedPrompt:
    """
    Main entry point — converts a raw agricultural prompt into a ParsedPrompt.

    Example:
        Input:  "wheat for Jodhpur at 48°C"
        Output: ParsedPrompt(crop_type=WHEAT, location_raw="Jodhpur",
                             temp_celsius=48.0, stress_hints=["heat"])
    """
    raw_response = call_llm(user_prompt)
    data         = extract_json(raw_response)
    return ParsedPrompt(**data)


if __name__ == "__main__":
    for prompt in [
        "wheat for Jodhpur at 48°C",
        "rice cultivation in Punjab under drought stress",
    ]:
        print(f"\nInput:  {prompt}")
        result = parse_prompt(prompt)
        print(f"Output: {result.model_dump_json(indent=2)}")
