"""Claude temperature predictor — asks Claude to predict today's high.

Replaces the old spread-analysis LLM call. Instead of asking Claude
to approve/reject a spread bet, we give it ALL the weather data and ask
for a structured prediction: what will the high be, when, and which
Kalshi bucket will win.

This is the human-in-the-loop intelligence layer — Claude sees patterns
across data sources that simple math can't.
"""

from __future__ import annotations

import json
import logging

import httpx

from weather_bets.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a meteorological analyst specializing in daily high temperature prediction for Austin, Texas (Austin-Bergstrom International Airport, KAUS).

You are given comprehensive weather data and Kalshi prediction market prices. Your job is to:

1. Analyze ALL the data provided — current temperature, historical patterns, multiple forecast models, hourly forecasts, and market prices.
2. Predict the official daily high temperature that will be recorded.
3. Identify which Kalshi bucket will win.
4. Assess whether the market is correctly priced.

Rules:
- The "official high" is the NWS Climatological Report value for KAUS, which uses the standard meteorological day (midnight to midnight local time).
- The temperature typically peaks between 3-5 PM CT in Austin.
- The CLEAR_DAY_PATTERNS data shows the VERIFIED historical average remaining rise from a given hour. This is the most directly useful signal — it tells you how much higher the temp will go from the current reading.
- If the current temp is 76°F at noon and the historical pattern says +5.5°F avg remaining rise, the predicted high is ~82°F. Forecasts saying 87°F would need strong justification.
- Pay attention to disagreements between data sources — they reveal uncertainty.
- The market favorite (highest YES price) is often right, but our historical data shows the market systematically underprices at certain levels.

You MUST respond with valid JSON only, no markdown formatting. Use this exact structure:
{
    "predicted_high_f": <integer>,
    "predicted_time_ct": "<HH:MM>",
    "confidence_low_f": <integer>,
    "confidence_high_f": <integer>,
    "winning_bucket": "<bucket label>",
    "reasoning": "<2-3 sentences explaining your prediction>",
    "data_conflicts": "<any disagreements between data sources>",
    "market_assessment": "<is the market correctly priced? which buckets are mispriced?>",
    "trade_suggestion": "<what would you buy/sell if anything, based on the data?>"
}"""


async def predict_high(weather_summary: str) -> dict | None:
    """Send weather summary to Claude and get a structured prediction.

    Returns:
        dict with keys: predicted_high_f, predicted_time_ct, confidence_low_f,
        confidence_high_f, winning_bucket, reasoning, data_conflicts,
        market_assessment, trade_suggestion
        Or None if the call fails.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("[Claude] No ANTHROPIC_API_KEY — skipping prediction")
        return None

    user_prompt = f"""Here is today's weather data. Analyze it and predict the daily high.

{weather_summary}

Respond with JSON only."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                CLAUDE_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 1024,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text content from Claude response
        content = data.get("content", [])
        if not content:
            logger.error("[Claude] Empty response")
            return None

        text = content[0].get("text", "")

        # Parse JSON from response (handle markdown code blocks if present)
        text = text.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        prediction = json.loads(text)

        # Log the prediction
        logger.info(f"[Claude] 🌡️  PREDICTION: {prediction.get('predicted_high_f')}°F")
        logger.info(f"[Claude]    Time: {prediction.get('predicted_time_ct', '?')} CT")
        logger.info(
            f"[Claude]    Range: {prediction.get('confidence_low_f')}°F – "
            f"{prediction.get('confidence_high_f')}°F"
        )
        logger.info(f"[Claude]    Winning bucket: {prediction.get('winning_bucket')}")
        logger.info(f"[Claude]    Reasoning: {prediction.get('reasoning')}")
        logger.info(f"[Claude]    Conflicts: {prediction.get('data_conflicts')}")
        logger.info(f"[Claude]    Market: {prediction.get('market_assessment')}")
        logger.info(f"[Claude]    Trade: {prediction.get('trade_suggestion')}")

        return prediction

    except json.JSONDecodeError as e:
        logger.error(f"[Claude] Failed to parse JSON response: {e}")
        logger.error(f"[Claude] Raw text: {text[:500]}")
        return None
    except Exception as e:
        logger.error(f"[Claude] Prediction failed: {e}", exc_info=True)
        return None
