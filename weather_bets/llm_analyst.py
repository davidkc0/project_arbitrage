"""Claude LLM analyst — makes bet/skip decisions on spread bets."""

from __future__ import annotations

import json
import logging

import httpx

from weather_bets.config import ANTHROPIC_API_KEY, BET_SIZE_USD
from weather_bets.models import (
    BetDecision,
    BetRecommendation,
    EdgeOpportunity,
    SpreadBet,
    SpreadRecommendation,
)

logger = logging.getLogger(__name__)

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


async def analyze_spread(
    spread: SpreadBet,
    all_buckets_summary: str,
    hourly_forecast: list[dict] | None = None,
) -> SpreadRecommendation:
    """
    Ask Claude to analyze a spread bet and decide whether to place it.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("[LLM] No ANTHROPIC_API_KEY — using fallback")
        return _fallback_spread(spread)

    hourly_str = ""
    if hourly_forecast:
        hourly_str = "\n**Hourly Temperature Forecast:**\n"
        for h in hourly_forecast:
            hourly_str += f"  {h['hour']}: {h['temp_f']}°F ({h['forecast']}, {h['wind']})\n"

    bucket_labels = " + ".join(b.label for b in spread.buckets)
    bucket_detail = "\n".join(
        f"  - {b.label}: YES ${b.yes_price:.2f} (our prob: {p:.1%})"
        for b, p in zip(spread.buckets, spread.bucket_probabilities)
    )

    prompt = f"""You are a conservative weather betting analyst focused on STEADY PROFITS with high certainty.

**Critical Context:** Kalshi settles based on the **NWS Climatological Report (Daily)** for the airport station. We are using the **same NWS forecast** — so we're essentially betting on whether the NWS forecast will be approximately correct.

**Strategy:** We're using a SPREAD approach — buying YES on multiple adjacent temperature buckets to cover the likely range. Like an options iron condor, we profit as long as the actual temp lands anywhere in our range.

**Market:** Kalshi temperature bet for {spread.city} on {spread.date}
**NWS Forecast:** {spread.forecast_high}°F
{hourly_str}

**Proposed Spread:** {bucket_labels}
{bucket_detail}

**Spread Economics:**
- Combined probability: {spread.total_probability:.1%}
- Total cost: ${spread.total_cost:.2f}
- Profit if any bucket hits: ${spread.profit_if_hit:.2f}
- Expected profit: ${spread.expected_profit:.3f}
- ROI: {spread.roi_percent:+.1f}%

**All Market Buckets:**
{all_buckets_summary}

**Decision Framework:**
1. The NWS says {spread.forecast_high}°F. Does our spread cover the realistic range?
2. Is the combined probability ({spread.total_probability:.0%}) high enough for the cost (${spread.total_cost:.2f})?
3. We want STEADY PROFITS — is this a bet we'd make every day?
4. Consider: NWS accuracy is typically ±2-3°F. Does our spread cover that range?
5. A good spread should have 60%+ combined probability with positive expected value.

Respond with a JSON object (no markdown, just raw JSON):
{{
    "decision": "bet" or "skip",
    "confidence": 0.0 to 1.0,
    "bet_size_usd": total bet (max ${BET_SIZE_USD}), will be split across buckets proportionally,
    "reasoning": "2-3 sentences on why this spread is good/bad for steady profit"
}}"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                CLAUDE_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

            if resp.status_code != 200:
                logger.error(f"[LLM] Claude API error {resp.status_code}: {resp.text[:500]}")
                return _fallback_spread(spread)

            result = resp.json()

        text = result.get("content", [{}])[0].get("text", "")
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0] if "```" in text else text
            text = text.strip()

        data = json.loads(text)

        decision = BetDecision.BET if data.get("decision") == "bet" else BetDecision.SKIP
        confidence = float(data.get("confidence", 0))
        bet_size = min(float(data.get("bet_size_usd", 0)), BET_SIZE_USD)
        reasoning = data.get("reasoning", "")

        # Allocate proportionally to probability
        total_prob = sum(spread.bucket_probabilities)
        allocations = [
            bet_size * (p / total_prob) if total_prob > 0 else 0
            for p in spread.bucket_probabilities
        ]

        logger.info(
            f"[LLM] Spread [{bucket_labels}]: {decision.value} "
            f"(confidence={confidence:.0%}, size=${bet_size:.0f}) — {reasoning}"
        )

        return SpreadRecommendation(
            decision=decision,
            spread=spread,
            confidence=confidence,
            bet_size_usd=bet_size,
            allocations=allocations,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"[LLM] Claude error: {e}", exc_info=True)
        return _fallback_spread(spread)


def _fallback_spread(spread: SpreadBet) -> SpreadRecommendation:
    """Fallback: bet if combined probability >= 50% and expected profit > 0."""
    if spread.total_probability >= 0.50 and spread.expected_profit > 0:
        total_prob = sum(spread.bucket_probabilities)
        allocations = [
            BET_SIZE_USD * (p / total_prob) if total_prob > 0 else 0
            for p in spread.bucket_probabilities
        ]
        return SpreadRecommendation(
            decision=BetDecision.BET,
            spread=spread,
            confidence=0.6,
            bet_size_usd=BET_SIZE_USD,
            allocations=allocations,
            reasoning=f"Auto-approved: {spread.total_probability:.0%} combined prob with +${spread.expected_profit:.3f} EV (Claude unavailable)",
        )
    return SpreadRecommendation(
        decision=BetDecision.SKIP,
        spread=spread,
        confidence=0.3,
        bet_size_usd=0,
        allocations=[0] * len(spread.buckets),
        reasoning=f"Auto-skipped: {spread.total_probability:.0%} combined prob or negative EV (Claude unavailable)",
    )


# Keep backward-compatible single-bucket analysis
async def analyze_opportunity(
    opportunity: EdgeOpportunity,
    all_buckets_summary: str,
    hourly_forecast: list[dict] | None = None,
) -> BetRecommendation:
    """Legacy single-bucket analysis — wraps into spread logic."""
    return _fallback_single(opportunity)


def _fallback_single(opp: EdgeOpportunity) -> BetRecommendation:
    if opp.edge >= 0.15:
        return BetRecommendation(
            decision=BetDecision.BET, opportunity=opp, confidence=0.6,
            bet_size_usd=BET_SIZE_USD,
            reasoning=f"Auto-approved: {opp.edge_percent:.1f}% edge",
            side="yes", ticker=opp.bucket.ticker,
        )
    return BetRecommendation(
        decision=BetDecision.SKIP, opportunity=opp, confidence=0.3,
        bet_size_usd=0,
        reasoning=f"Auto-skipped: {opp.edge_percent:.1f}% edge below threshold",
        side="yes", ticker=opp.bucket.ticker,
    )
