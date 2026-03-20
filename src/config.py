"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configurable parameters, loaded from .env file."""

    # ── Polymarket ──────────────────────────────────────────────────────
    polymarket_private_key: str = ""
    polymarket_chain_id: int = 137  # Polygon mainnet
    polymarket_funder_address: str = ""
    polymarket_host: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # ── Kalshi ──────────────────────────────────────────────────────────
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    # ── Risk Management ────────────────────────────────────────────────
    max_position_size: float = 50.0       # Max USD per leg
    max_total_exposure: float = 500.0     # Max total USD at risk
    max_concurrent_positions: int = 10
    min_edge_percent: float = 1.0         # Minimum net edge to surface

    # ── Scanner ─────────────────────────────────────────────────────────
    scan_interval_seconds: int = 10

    # ── Execution ───────────────────────────────────────────────────────
    execution_mode: str = "semi"  # "semi" or "auto"

    # ── Database ────────────────────────────────────────────────────────
    db_path: str = "arbitrage.db"

    # ── Server ──────────────────────────────────────────────────────────
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # ── Fee Estimates (fraction, e.g. 0.02 = 2%) ───────────────────────
    polymarket_fee_rate: float = 0.02     # ~2% on winnings
    kalshi_fee_rate: float = 0.03         # ~3% on winnings (varies)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton
settings = Settings()
