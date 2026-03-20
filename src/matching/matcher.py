"""Fuzzy event matcher — pairs equivalent events across platforms."""

from __future__ import annotations

import logging
import re
from datetime import timedelta

from rapidfuzz import fuzz

from src.models import MarketEvent, MatchedPair, Platform

logger = logging.getLogger(__name__)

# Minimum fuzzy score to consider a potential match
MATCH_THRESHOLD = 55
# Bonus score if events share the same category
CATEGORY_BONUS = 10
# Maximum date gap between event end dates (days) to allow matching
MAX_DATE_GAP_DAYS = 30
# Minimum keyword overlap to trigger expensive fuzzy matching
MIN_KEYWORD_OVERLAP = 2

# ── Team name normalization ─────────────────────────────────────────────
# Maps abbreviations, short names, and alternate names to a canonical form.
# This is critical for matching Kalshi ("BKN", "Golden State") to
# Polymarket ("Brooklyn Nets", "Warriors").

TEAM_ALIASES: dict[str, str] = {
    # NBA
    "atl": "atlanta hawks", "hawks": "atlanta hawks", "atlanta": "atlanta hawks",
    "bos": "boston celtics", "celtics": "boston celtics", "boston": "boston celtics",
    "bkn": "brooklyn nets", "nets": "brooklyn nets", "brooklyn": "brooklyn nets",
    "cha": "charlotte hornets", "hornets": "charlotte hornets", "charlotte": "charlotte hornets",
    "chi": "chicago bulls", "bulls": "chicago bulls",
    "cle": "cleveland cavaliers", "cavaliers": "cleveland cavaliers", "cavs": "cleveland cavaliers", "cleveland": "cleveland cavaliers",
    "dal": "dallas mavericks", "mavericks": "dallas mavericks", "mavs": "dallas mavericks", "dallas": "dallas mavericks",
    "den": "denver nuggets", "nuggets": "denver nuggets", "denver": "denver nuggets",
    "det": "detroit pistons", "pistons": "detroit pistons", "detroit": "detroit pistons",
    "gsw": "golden state warriors", "warriors": "golden state warriors", "golden state": "golden state warriors",
    "hou": "houston rockets", "rockets": "houston rockets", "houston": "houston rockets",
    "ind": "indiana pacers", "pacers": "indiana pacers", "indiana": "indiana pacers",
    "lac": "los angeles clippers", "clippers": "los angeles clippers", "la clippers": "los angeles clippers",
    "los angeles c": "los angeles clippers",
    "lal": "los angeles lakers", "lakers": "los angeles lakers", "la lakers": "los angeles lakers",
    "mem": "memphis grizzlies", "grizzlies": "memphis grizzlies", "memphis": "memphis grizzlies",
    "mia": "miami heat", "heat": "miami heat",
    "mil": "milwaukee bucks", "bucks": "milwaukee bucks", "milwaukee": "milwaukee bucks",
    "min": "minnesota timberwolves", "timberwolves": "minnesota timberwolves", "wolves": "minnesota timberwolves", "minnesota": "minnesota timberwolves",
    "nop": "new orleans pelicans", "pelicans": "new orleans pelicans", "new orleans": "new orleans pelicans",
    "nyk": "new york knicks", "knicks": "new york knicks",
    "okc": "oklahoma city thunder", "thunder": "oklahoma city thunder", "oklahoma city": "oklahoma city thunder",
    "orl": "orlando magic", "magic": "orlando magic", "orlando": "orlando magic",
    "phi": "philadelphia 76ers", "76ers": "philadelphia 76ers", "sixers": "philadelphia 76ers", "philadelphia": "philadelphia 76ers",
    "phx": "phoenix suns", "suns": "phoenix suns", "phoenix": "phoenix suns",
    "por": "portland trail blazers", "blazers": "portland trail blazers", "trail blazers": "portland trail blazers", "portland": "portland trail blazers",
    "sac": "sacramento kings", "kings": "sacramento kings", "sacramento": "sacramento kings",
    "sas": "san antonio spurs", "spurs": "san antonio spurs", "san antonio": "san antonio spurs",
    "tor": "toronto raptors", "raptors": "toronto raptors", "toronto": "toronto raptors",
    "uta": "utah jazz", "jazz": "utah jazz", "utah": "utah jazz",
    "was": "washington wizards", "wizards": "washington wizards", "washington": "washington wizards",
    # Soccer
    "fc barcelona": "barcelona", "fcb": "barcelona", "barca": "barcelona",
    "atletico madrid": "atletico", "club atletico de madrid": "atletico", "atm": "atletico",
    "real madrid": "real madrid", "rma": "real madrid",
    "bv borussia 09 dortmund": "dortmund", "borussia dortmund": "dortmund", "bvb": "dortmund",
    "manchester city fc": "manchester city", "man city": "manchester city",
    "manchester united fc": "manchester united", "man utd": "manchester united", "man united": "manchester united",
    "chelsea fc": "chelsea",
    "arsenal fc": "arsenal",
    "liverpool fc": "liverpool",
    "tottenham hotspur fc": "tottenham", "tottenham hotspur": "tottenham",
    "rc strasbourg alsace": "strasbourg", "strasbourg alsace": "strasbourg",
    "athletic bilbao": "athletic bilbao", "athletic club": "athletic bilbao",
    "olympique de marseille": "marseille", "om": "marseille",
    "paris saint-germain": "psg", "paris sg": "psg",
    "juventus fc": "juventus", "juve": "juventus",
    "ac milan": "milan", "inter milan": "inter",
    "bayern munich": "bayern", "fc bayern": "bayern",
    "udinese calcio": "udinese",
    # NFL (common)
    "ne": "new england patriots", "patriots": "new england patriots",
    "nyg": "new york giants", "giants": "new york giants",
    "nyj": "new york jets", "jets": "new york jets",
    "sf": "san francisco 49ers", "49ers": "san francisco 49ers", "niners": "san francisco 49ers",
    "kc": "kansas city chiefs", "chiefs": "kansas city chiefs",
    "buf": "buffalo bills", "bills": "buffalo bills",
    # MLB (common)
    "nyy": "new york yankees", "yankees": "new york yankees",
    "nym": "new york mets", "mets": "new york mets",
    "lad": "los angeles dodgers", "dodgers": "los angeles dodgers",
}


def _normalize_team_names(q: str) -> str:
    """Replace team abbreviations and alternate names with canonical forms."""
    q_lower = q.lower()
    # Sort by longest key first to match "golden state" before "state"
    for alias, canonical in sorted(TEAM_ALIASES.items(), key=lambda x: -len(x[0])):
        # Use word boundary matching to avoid partial replacements
        pattern = r'\b' + re.escape(alias) + r'\b'
        q_lower = re.sub(pattern, canonical, q_lower)
    return q_lower


def _normalize_question(q: str) -> str:
    """Normalize a question string for better fuzzy matching."""
    q = q.lower().strip()

    # Apply team name normalization
    q = _normalize_team_names(q)

    # Remove common prefixes
    for prefix in ["will ", "is ", "does ", "can ", "should ", "when will "]:
        if q.startswith(prefix):
            q = q[len(prefix):]

    # Normalize whitespace
    q = re.sub(r"\s+", " ", q)

    # Remove trailing punctuation and "winner?"
    q = q.rstrip("?!.")
    q = re.sub(r"\s*winner\s*$", "", q)

    # Normalize "at" to "vs" for matchups
    q = re.sub(r"\bat\b", "vs", q)

    # Strip specific dates like "on 2026-03-07" or "on March 7"
    q = re.sub(r"\bon\s+\d{4}-\d{2}-\d{2}\b", "", q)
    q = re.sub(r"\bon\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}", "", q, flags=re.IGNORECASE)

    # Synonym normalization
    synonyms = {
        "federal reserve": "fed",
        "the fed": "fed",
        "federal funds rate": "fed rate",
        "upper bound of the federal funds rate": "fed rate",
        "cut rates": "decrease interest rates",
        "cut rate": "decrease interest rates",
        "hike rates": "increase interest rates",
        "hike rate": "increase interest rates",
        "basis points": "bps",
        "25bps": "25 bps",
        "50bps": "50 bps",
        "bitcoin": "btc",
        "ethereum": "eth",
        "president": "pres",
        "united states": "us",
        "u.s.": "us",
        "donald trump": "trump",
        "president trump": "trump",
        "strait of hormuz": "hormuz strait",
        "$": "",
        "—": "-",
        "–": "-",
        "2025–26": "2025-26",
    }
    for old, new in synonyms.items():
        q = q.replace(old, new)

    return q.strip()


# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "of", "by", "for", "and", "or",
    "be", "is", "are", "was", "were", "will", "would", "do", "does", "did",
    "have", "has", "had", "this", "that", "it", "its", "from", "with", "as",
    "not", "no", "yes", "before", "after", "any", "more", "than", "above",
    "below", "their", "there", "over", "under", "between", "following",
    "vs", "win", "pro",
})


def _extract_keywords(q: str) -> set[str]:
    """Extract meaningful keywords from a normalized question."""
    normalized = _normalize_question(q)
    words = set(re.findall(r"[a-z0-9]+", normalized))
    return words - _STOP_WORDS


def score_match(event_a: MarketEvent, event_b: MarketEvent) -> float:
    """
    Score how well two events from different platforms match.
    Returns 0–100; higher = more confident match.
    """
    q_a = _normalize_question(event_a.question)
    q_b = _normalize_question(event_b.question)

    # Compute multiple fuzzy metrics
    token_sort = fuzz.token_sort_ratio(q_a, q_b)
    token_set = fuzz.token_set_ratio(q_a, q_b)
    partial = fuzz.partial_ratio(q_a, q_b)

    # Sanity check: if the raw word-order similarity is very low,
    # the questions are probably about different things even if they
    # share some keywords. E.g., "GTA VI released before June" vs
    # "What will the price of GTA VI be?" share 'gta' and 'vi' but
    # are completely different questions.
    if token_sort < 35:
        return min(token_sort, 40)  # Cap at 40 — never passes threshold

    # Use the BEST score as the primary signal.
    # token_set_ratio handles "A is a subset of B" perfectly, which is
    # the most common case (e.g., "FC Barcelona win" ⊂ "FC Barcelona win on March 8")
    # If token_set is very high, trust it — don't let token_sort drag it down.
    if token_set >= 95:
        # Near-exact match (one may just have extra words)
        score = max(90, token_set)
    elif token_set >= 80:
        # Very good set match — weight it heavily
        score = (token_set * 0.5) + (partial * 0.3) + (token_sort * 0.2)
    else:
        # Moderate match — use balanced mix
        score = (token_sort * 0.35) + (token_set * 0.35) + (partial * 0.3)

    # Category bonus
    if event_a.category and event_b.category:
        cat_a = event_a.category.lower().strip()
        cat_b = event_b.category.lower().strip()
        if cat_a == cat_b or fuzz.ratio(cat_a, cat_b) > 70:
            score = min(100, score + CATEGORY_BONUS)

    # Date proximity bonus/penalty
    if event_a.end_date and event_b.end_date:
        gap = abs((event_a.end_date - event_b.end_date).total_seconds())
        gap_days = gap / 86400
        if gap_days > MAX_DATE_GAP_DAYS:
            score *= 0.5  # Heavy penalty for mismatched dates
        elif gap_days < 1:
            score = min(100, score + 5)  # Small bonus for same-day

    return score


class EventMatcher:
    """Matches events across Polymarket and Kalshi."""

    def __init__(self):
        self._confirmed_pairs: dict[str, MatchedPair] = {}
        self._manual_overrides: list[tuple[str, str]] = []

    @property
    def confirmed_pairs(self) -> list[MatchedPair]:
        return list(self._confirmed_pairs.values())

    def add_manual_override(self, polymarket_id: str, kalshi_ticker: str) -> None:
        """Manually pair two events that fuzzy matching misses."""
        self._manual_overrides.append((polymarket_id, kalshi_ticker))

    def load_confirmed_pairs(self, pairs: list[MatchedPair]) -> None:
        """Load previously confirmed pairs from the database."""
        self._confirmed_pairs = {p.polymarket_id: p for p in pairs}

    def find_matches(
        self,
        polymarket_events: list[MarketEvent],
        kalshi_events: list[MarketEvent],
        min_score: float = MATCH_THRESHOLD,
    ) -> list[MatchedPair]:
        """
        Find matching events between Polymarket and Kalshi.
        Uses keyword pre-filter with inverted index for speed.
        """
        new_matches: list[MatchedPair] = []

        override_map = {pm_id: k_ticker for pm_id, k_ticker in self._manual_overrides}
        kalshi_by_ticker = {e.platform_id: e for e in kalshi_events}
        matched_kalshi = {p.kalshi_ticker for p in self._confirmed_pairs.values()}

        # ── Pre-compute keyword sets and inverted index ─────────────────
        kalshi_keywords: dict[str, set[str]] = {}
        for k_event in kalshi_events:
            if k_event.platform_id not in matched_kalshi:
                kalshi_keywords[k_event.platform_id] = _extract_keywords(k_event.question)

        keyword_index: dict[str, set[str]] = {}
        for k_id, kw_set in kalshi_keywords.items():
            for kw in kw_set:
                if kw not in keyword_index:
                    keyword_index[kw] = set()
                keyword_index[kw].add(k_id)

        near_misses: list[tuple[float, str, str]] = []
        total_comparisons = 0

        for pm_event in polymarket_events:
            if pm_event.platform_id in self._confirmed_pairs:
                continue

            # Manual overrides
            if pm_event.platform_id in override_map:
                k_ticker = override_map[pm_event.platform_id]
                k_event = kalshi_by_ticker.get(k_ticker)
                if k_event:
                    pair = MatchedPair(
                        polymarket_id=pm_event.platform_id,
                        kalshi_ticker=k_ticker,
                        polymarket_question=pm_event.question,
                        kalshi_question=k_event.question,
                        match_confidence=100.0,
                        confirmed_by_user=True,
                    )
                    new_matches.append(pair)
                    self._confirmed_pairs[pm_event.platform_id] = pair
                    continue

            # ── Keyword pre-filter ──────────────────────────────────────
            pm_keywords = _extract_keywords(pm_event.question)

            candidate_counts: dict[str, int] = {}
            for kw in pm_keywords:
                for k_id in keyword_index.get(kw, []):
                    candidate_counts[k_id] = candidate_counts.get(k_id, 0) + 1

            candidates = [
                k_id for k_id, count in candidate_counts.items()
                if count >= MIN_KEYWORD_OVERLAP and k_id not in matched_kalshi
            ]

            # ── Fuzzy match against filtered candidates ─────────────────
            best_score = 0.0
            best_kalshi: MarketEvent | None = None

            for k_id in candidates:
                k_event = kalshi_by_ticker.get(k_id)
                if not k_event:
                    continue

                total_comparisons += 1
                score = score_match(pm_event, k_event)
                if score > best_score:
                    best_score = score
                    best_kalshi = k_event

            if best_score >= min_score and best_kalshi:
                pair = MatchedPair(
                    polymarket_id=pm_event.platform_id,
                    kalshi_ticker=best_kalshi.platform_id,
                    polymarket_question=pm_event.question,
                    kalshi_question=best_kalshi.question,
                    match_confidence=best_score,
                    confirmed_by_user=False,
                )
                new_matches.append(pair)
                self._confirmed_pairs[pm_event.platform_id] = pair
                matched_kalshi.add(best_kalshi.platform_id)

                logger.info(
                    f"[Matcher] Pair (score={best_score:.1f}): "
                    f"PM: {pm_event.question[:50]} ↔ "
                    f"K: {best_kalshi.question[:50]}"
                )
            elif best_score >= 40 and best_kalshi:
                near_misses.append((
                    best_score,
                    pm_event.question[:60],
                    best_kalshi.question[:60],
                ))

        logger.info(
            f"[Matcher] {total_comparisons} fuzzy comparisons "
            f"(pre-filtered from {len(polymarket_events) * len(kalshi_events):,}), "
            f"{len(new_matches)} new matches"
        )

        if near_misses:
            near_misses.sort(reverse=True)
            logger.info(f"[Matcher] Top near-misses below {min_score}:")
            for score, pm_q, k_q in near_misses[:5]:
                logger.info(f"  {score:.1f}: PM={pm_q} ↔ K={k_q}")

        return new_matches
