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

    # Pre-flight: directional conflict check
    # If hourly forecast peak is significantly above the highest bucket we're betting on,
    # skip immediately — no point asking the LLM about a fundamentally bad bet.
    if hourly_forecast:
        hourly_peak = max((h["temp_f"] for h in hourly_forecast), default=None)
        if hourly_peak:
            # Find the ceiling of our spread (highest bucket's upper bound)
            # For "≤X" buckets, high_bound is X. For open-ended "≥X" buckets, no ceiling.
            # Also check: if our spread includes a "≤X" bucket, the spread_ceiling is that X.
            spread_ceiling = max(
                (b.high_bound for b in spread.buckets if b.high_bound is not None),
                default=None
            )
            # Also check: if hourly peak is well above the consensus mean we bet on, flag it
            spread_mean = spread.forecast_high
            conflict = False
            conflict_reason = ""
            if spread_ceiling and hourly_peak > spread_ceiling + 2:
                conflict = True
                conflict_reason = (
                    f"Hourly forecast peaks at {hourly_peak}°F but our spread ceiling is "
                    f"{spread_ceiling}°F — temperature is likely to overshoot."
                )
            elif hourly_peak > spread_mean + 4:
                # Even without a hard ceiling, if hourly is 4°F+ above our target mean, skip
                conflict = True
                conflict_reason = (
                    f"Hourly forecast peaks at {hourly_peak}°F vs our consensus of "
                    f"{spread_mean}°F — too much upside risk for this spread."
                )
            if conflict:
                logger.warning(f"[LLM] ⚠️ Directional conflict: {conflict_reason}")
                return SpreadRecommendation(
                    decision=BetDecision.SKIP,
                    spread=spread,
                    confidence=0.9,
                    bet_size_usd=0,
                    allocations=[0] * len(spread.buckets),
                    reasoning=f"Directional conflict: {conflict_reason}",
                )

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

    prompt = f"""You are a weather betting analyst focused on CONSISTENT RETURNS over many bets.

**Critical Context:** Kalshi settles based on the **NWS Climatological Report (Daily)** for the airport station. We are using the **same NWS forecast** — so we're essentially betting on whether the NWS forecast will be approximately correct.

**Strategy:** We're using a SPREAD approach — buying YES on multiple adjacent temperature buckets to cover the likely range. We want to bet FREQUENTLY on positive-EV opportunities. Small, consistent profits compound over time. We'd rather win $0.10 on 8 out of 10 bets than swing for $1.00 on a coin flip.

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

**Decision Framework — BALANCED APPROACH:**
1. Only bet when probability is 50%+ AND expected value is positive.
2. NWS is the settlement source for Kalshi — if NWS forecast falls inside a different bucket than our spread, that's a WARNING. We need strong multi-model agreement to bet against NWS.
3. For spreads with 55%+ probability AND positive EV AND NWS agreement: BET with confidence.
4. For spreads where our consensus disagrees with NWS by 2°F+: be cautious. NWS may still be right.
5. NEVER bet against the market AND NWS simultaneously — if both disagree with our spread, that's a SKIP.
6. We'd rather miss a good bet than take a bad one. Capital preservation > volume.
7. Scale bet size with confidence: high conviction = full ${BET_SIZE_USD}, moderate = ${BET_SIZE_USD * 0.5:.0f}.

**CRITICAL — Directional integrity:**
- If the hourly forecast shows a peak ABOVE our spread's upper bound, that is a SKIP. We should never bet "under X" if hourly says it'll exceed X.
- Read the hourly data carefully: if temps are forecast to hit 94°F, do NOT bet on ≤90° or ≤91° buckets. That is throwing money away.
- Hourly NWS tends to run slightly hot, but a 4°F+ gap between hourly peak and our spread ceiling is a hard skip.

Respond with a JSON object (no markdown, just raw JSON):
{{
    "decision": "bet" or "skip",
    "confidence": 0.0 to 1.0,
    "bet_size_usd": total bet (max ${BET_SIZE_USD}), will be split across buckets proportionally,
    "reasoning": "2-3 sentences on why"
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
