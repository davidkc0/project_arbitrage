"""Arbitrage engine — calculates cross-platform spreads and scores opportunities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import settings
from src.models import (
    ArbitrageOpportunity,
    MarketEvent,
    MatchedPair,
    OutcomeSide,
    Platform,
)

logger = logging.getLogger(__name__)


def _get_price(event: MarketEvent, side: OutcomeSide, use_ask: bool = True) -> float:
    """Get the best available price for a side.

    When buying, we use the ask price (what we pay).
    The ask represents the cheapest seller is willing to sell at.
    If ask is 0 (no data), fall back to last_price.
    """
    for outcome in event.outcomes:
        if outcome.side == side:
            if use_ask:
                return outcome.best_ask if outcome.best_ask > 0 else outcome.last_price
            else:
                return outcome.best_bid if outcome.best_bid > 0 else outcome.last_price
    return 0.0


def _get_depth(event: MarketEvent, side: OutcomeSide) -> float:
    """Get the available depth (USD) on the ask side for a given outcome."""
    for outcome in event.outcomes:
        if outcome.side == side:
            return outcome.ask_depth
    return 0.0


def calculate_edge(
    event_a: MarketEvent,
    event_b: MarketEvent,
) -> ArbitrageOpportunity | None:
    """
    Given two matched events, find the best arbitrage opportunity.

    Strategy: Buy YES on one platform + NO on the other.
    Try both directions and pick the one with the higher edge.
    """
    # Get all four prices
    yes_a = _get_price(event_a, OutcomeSide.YES, use_ask=True)
    no_a = _get_price(event_a, OutcomeSide.NO, use_ask=True)
    yes_b = _get_price(event_b, OutcomeSide.YES, use_ask=True)
    no_b = _get_price(event_b, OutcomeSide.NO, use_ask=True)

    # ── Guard: skip if ANY price is zero (illiquid / no data) ───────────
    if yes_a <= 0 or no_a <= 0 or yes_b <= 0 or no_b <= 0:
        return None

    # ── Direction 1: YES on A, NO on B ──────────────────────────────────
    cost_1 = yes_a + no_b
    edge_1 = 1.0 - cost_1

    # ── Direction 2: YES on B, NO on A ──────────────────────────────────
    cost_2 = yes_b + no_a
    edge_2 = 1.0 - cost_2

    # Pick the better direction
    if edge_1 >= edge_2 and edge_1 > 0:
        buy_yes_platform = event_a.platform
        buy_no_platform = event_b.platform
        buy_yes_price = yes_a
        buy_no_price = no_b
        total_cost = cost_1
        gross_edge = edge_1
        depth_a = _get_depth(event_a, OutcomeSide.YES)
        depth_b = _get_depth(event_b, OutcomeSide.NO)
    elif edge_2 > 0:
        buy_yes_platform = event_b.platform
        buy_no_platform = event_a.platform
        buy_yes_price = yes_b
        buy_no_price = no_a
        total_cost = cost_2
        gross_edge = edge_2
        depth_a = _get_depth(event_b, OutcomeSide.YES)
        depth_b = _get_depth(event_a, OutcomeSide.NO)
    else:
        return None  # No positive edge

    # ── Fee estimation ──────────────────────────────────────────────────
    # Fees are charged on WINNINGS (payout - cost), not on the full payout.
    # Winning payout for 1 share = $1.00.
    # Winning leg profit = payout - leg_cost = 1.0 - leg_price
    winning_profit_yes = 1.0 - buy_yes_price
    winning_profit_no = 1.0 - buy_no_price

    # We always win exactly one leg, but we don't know which one.
    # Worst case fee = max of the two.
    fee_rate_a = (
        settings.polymarket_fee_rate
        if buy_yes_platform == Platform.POLYMARKET
        else settings.kalshi_fee_rate
    )
    fee_rate_b = (
        settings.polymarket_fee_rate
        if buy_no_platform == Platform.POLYMARKET
        else settings.kalshi_fee_rate
    )

    fee_on_yes_win = winning_profit_yes * fee_rate_a
    fee_on_no_win = winning_profit_no * fee_rate_b

    # Conservative: use the higher fee scenario
    fee_estimate = max(fee_on_yes_win, fee_on_no_win)

    net_edge = gross_edge - fee_estimate
    net_edge_percent = (net_edge / total_cost * 100) if total_cost > 0 else 0.0

    # Sanity cap: real prediction market arb is typically 1-10%.
    # Anything above 50% is almost certainly bad data (zero prices,
    # stale orderbooks, wrong market matched).
    if net_edge_percent > 50:
        return None

    # Max bet size limited by the shallower book
    max_bet = min(depth_a, depth_b) if depth_a > 0 and depth_b > 0 else 0.0

    # ── Time-value: days until the earlier expiry ────────────────────────
    now = datetime.now(timezone.utc)
    end_dates = [d for d in [event_a.end_date, event_b.end_date] if d]
    if end_dates:
        earliest = min(end_dates)
        days_to_expiry = max((earliest - now).total_seconds() / 86400, 1)
    else:
        days_to_expiry = 365.0  # Unknown → assume 1 year

    annualized_edge = net_edge_percent * (365.0 / days_to_expiry)

    return ArbitrageOpportunity(
        event_a=event_a,
        event_b=event_b,
        buy_yes_platform=buy_yes_platform,
        buy_yes_price=buy_yes_price,
        buy_no_platform=buy_no_platform,
        buy_no_price=buy_no_price,
        total_cost=round(total_cost, 6),
        gross_edge=round(gross_edge, 6),
        fee_estimate=round(fee_estimate, 6),
        net_edge=round(net_edge, 6),
        net_edge_percent=round(net_edge_percent, 4),
        max_bet_size=round(max_bet, 2),
        available_depth_a=depth_a,
        available_depth_b=depth_b,
        # Side-by-side prices for display
        yes_a=round(yes_a, 4),
        yes_b=round(yes_b, 4),
        no_a=round(no_a, 4),
        no_b=round(no_b, 4),
        yes_spread=round(abs(yes_a - yes_b), 4),
        no_spread=round(abs(no_a - no_b), 4),
        # Time-value
        days_to_expiry=round(days_to_expiry, 1),
        annualized_edge=round(annualized_edge, 2),
    )


def find_opportunities(
    polymarket_events: list[MarketEvent],
    kalshi_events: list[MarketEvent],
    matched_pairs: list[MatchedPair],
    min_edge_pct: float | None = None,
) -> list[ArbitrageOpportunity]:
    """
    Scan all matched pairs for arbitrage opportunities.
    Returns opportunities sorted by net edge (best first).
    """
    if min_edge_pct is None:
        min_edge_pct = settings.min_edge_percent

    pm_by_id = {e.platform_id: e for e in polymarket_events}
    k_by_id = {e.platform_id: e for e in kalshi_events}

    opportunities: list[ArbitrageOpportunity] = []

    for pair in matched_pairs:
        pm_event = pm_by_id.get(pair.polymarket_id)
        k_event = k_by_id.get(pair.kalshi_ticker)

        if not pm_event or not k_event:
            continue

        opp = calculate_edge(pm_event, k_event)
        if opp and opp.net_edge_percent >= min_edge_pct:
            opp.match_confidence = pair.match_confidence
            opportunities.append(opp)

    # Sort by annualized edge (time-value weighted) descending
    opportunities.sort(key=lambda o: o.annualized_edge, reverse=True)

    logger.info(
        f"[Arbitrage] Found {len(opportunities)} opportunities "
        f"above {min_edge_pct}% edge from {len(matched_pairs)} matched pairs"
    )

    return opportunities
