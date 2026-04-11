# orchestrator/prompt_parser.py
# Converts a raw user prompt into a structured ParsedPrompt object using an LLM.
#
# Flow:
#   raw text → LLM (DeepSeek-R1 via Ollama) → JSON string → ParsedPrompt

import logging

import json
import re
import requests
from typing import Optional
from JAY.shared.models import ParsedPrompt, CropType
from JAY.shared.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an agricultural science expert. Extract crop parameters from the user prompt.
Respond ONLY with a valid JSON object — no preamble, no explanation, no markdown.

Required keys:
{
  "crop_type": one of [wheat, rice, maize, sorghum, barley, cotton, soybean, sugarcane, potato, tomato, onion, garlic, peanut, chickpea, lentil, mustard, sunflower, sesame, millet, banana, apple, unknown],
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
    except requests.exceptions.RequestException as e:
        logger.warning("[PARSER] Ollama request failed: %s — using fallback parser.", e)
        return _fallback_parse(prompt)
    except Exception as e:
        logger.exception("[PARSER] Unexpected parsing error — using fallback parser.")
        return _fallback_parse(prompt)


def _fallback_parse(prompt: str) -> str:
    """
    Enhanced regex-based fallback parser with Indian crop and location bias.
    Used when LLM is unavailable.
    """
    p = prompt.lower()

    # Expanded crop list with Indian focus
    crop = "unknown"
    crop_aliases = {
        # Cereals
        "wheat": "wheat", "गेहूं": "wheat",
        "rice": "rice", "धान": "rice", "paddy": "rice",
        "maize": "maize", "मक्का": "maize", "corn": "maize",
        "sorghum": "sorghum", "ज्वार": "sorghum",
        "barley": "barley", "जौ": "barley",
        "millet": "millet", "बाजरा": "millet",
        
        # Pulses
        "chickpea": "chickpea", "चना": "chickpea", "chickpeas": "chickpea",
        "lentil": "lentil", "मसूर": "lentil", "lentils": "lentil",
        "peanut": "peanut", "मूंगफली": "peanut", "groundnut": "peanut",
        
        # Oilseeds
        "mustard": "mustard", "सरसों": "mustard",
        "sunflower": "sunflower", "सूरजमुखी": "sunflower",
        "sesame": "sesame", "तिल": "sesame",
        "soybean": "soybean", "सोयाबीन": "soybean",
        
        # Vegetables
        "potato": "potato", "आलू": "potato", "potatoes": "potato",
        "tomato": "tomato", "टमाटर": "tomato", "tomatoes": "tomato",
        "onion": "onion", "प्याज": "onion", "onions": "onion",
        "garlic": "garlic", "लहसुन": "garlic",
        
        # Others
        "cotton": "cotton", "कपास": "cotton",
        "sugarcane": "sugarcane", "गन्ना": "sugarcane",
        "banana": "banana", "केला": "banana",
        "apple": "apple", "सेब": "apple",
    }

    for alias, canonical in crop_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", p):
            crop = canonical
            break

    # Temperature extraction
    temp_match = re.search(r"(\d+)\s*(?:°c|°|celsius|degrees?|सेल्सियस)", p)
    temp = float(temp_match.group(1)) if temp_match else None

    # Location extraction with Indian bias
    location = "Unknown"
    indian_locations = [
        "jodhpur", "jaipur", "bikaner", "ludhiana", "amritsar", "nagpur", "hyderabad", "bhopal",
        "delhi", "mumbai", "chennai", "kolkata", "bangalore", "pune", "ahmedabad", "surat",
        "rajasthan", "punjab", "haryana", "maharashtra", "tamil nadu", "telangana", "madhya pradesh",
        "uttar pradesh", "karnataka", "gujarat", "bihar", "west bengal", "andhra pradesh"
    ]
    for loc in indian_locations:
        if loc in p:
            location = loc.title()
            break
    
    # Enhanced stress detection
    stresses = []
    stress_keywords = {
        "heat": ["heat", "hot", "temperature", "गर्मी", "उष्मा"],
        "drought": ["drought", "dry", "water stress", "सूखा", "अनावृष्टि"],
        "flood": ["flood", "waterlogging", "बाढ़"],
        "salinity": ["salinity", "salt", "नमकीन"],
        "cold": ["cold", "chill", "ठंड"],
    }
    for stress, keywords in stress_keywords.items():
        if any(kw in p for kw in keywords):
            stresses.append(stress)
    
    # If temp > 38 and not already heat, add heat
    if temp and temp > 38 and "heat" not in stresses:
        stresses.append("heat")

    return json.dumps({
        "crop_type": crop,
        "location_raw": location,
        "temp_celsius": temp,
        "stress_hints": stresses,
        "variety_hint": None,
        "raw_entities": {"source": "enhanced_fallback_parser", "confidence": 0.7}
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
    try:
        raw_response = call_llm(user_prompt)
        data = extract_json(raw_response)
        return ParsedPrompt(**data)
    except Exception:
        fallback = _fallback_parse(user_prompt)
        return ParsedPrompt(**json.loads(fallback))


if __name__ == "__main__":
    for prompt in [
        "wheat for Jodhpur at 48°C",
        "rice cultivation in Punjab under drought stress",
    ]:
        print(f"\nInput:  {prompt}")
        result = parse_prompt(prompt)
        print(f"Output: {result.model_dump_json(indent=2)}")
