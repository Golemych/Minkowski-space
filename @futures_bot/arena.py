"""
Finam Arena REST client + MOEX ISS data provider.
Async, auto JWT refresh, OrderResult dataclass.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd

log = logging.getLogger("futures_combined.arena")

SIDE_BUY = "SIDE_BUY"
SIDE_SELL = "SIDE_SELL"

_BASE = "https://arena.finam.ru"
_TOKEN_TTL = 14 * 60


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    quantity: float
    execution_price: float
    commission: float


# ── MOEX ISS helpers (sync, blocking) ──────────────────────────────────────

MOEX_BASE = "https://iss.moex.com/iss"
MOEX_SUPPORTED_INTERVALS = {1, 10, 60, 24, 7, 31}

TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}

FUTURES_MAP = {
    "SBER": "SRU6", "SBERP": "SPU6", "GAZP": "GZU6",
    "ROSN": "RNU6", "LKOH": "LKU6", "VTBR": "VBU6",
    "GMKN": "GKU6", "ALRS": "ALU6", "AFLT": "AFU6", "MGNT": "MNU6",
}


def _is_futures_symbol(symbol: str) -> bool:
    return bool(re.search(r'-\d{1,2}\.\d{2}$', symbol.upper()))


def _get_base_ticker(symbol: str) -> str:
    if _is_futures_symbol(symbol):
        return symbol.rsplit('-', 1)[0]
    return symbol


def _clean_symbol(symbol: str) -> str:
    return symbol.split('@')[0].upper()


def _resolve_moex_secid(symbol: str) -> str:
    clean = _clean_symbol(symbol)
    if not _is_futures_symbol(clean):
        return clean
    base = _get_base_ticker(clean)
    return FUTURES_MAP.get(base, clean)


def _get_moex_market_info(symbol: str) -> dict:
    clean = _clean_symbol(symbol)
    if _is_futures_symbol(clean):
        return {"engine": "futures", "market": "forts", "board": "RFUD"}
    return {"engine": "stock", "market": "shares", "board": "TQBR"}


def _fetch_1m_candles_sync(symbol: str, days: int,
                            cache_dir: Path) -> Optional[pd.DataFrame]:
    """Blocking MOEX ISS 1m fetch with parquet cache."""
    import requests as sync_requests

    cache_path = cache_dir / f"{_clean_symbol(symbol)}_1m.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from cache first
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            last_ts = df.index[-1]
            hours_since = (datetime.now() - last_ts).total_seconds() / 3600
            refresh_days = max(1, int(hours_since / 24 + 0.5))
            if refresh_days < min(days, 30):
                log.info(f"Cache {cache_path.name}: +{refresh_days}d (have {len(df)} bars)")
                fresh = _do_fetch_1m(symbol, refresh_days)
                if fresh is not None and not fresh.empty:
                    df = pd.concat([df, fresh])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    cutoff = datetime.now() - timedelta(days=min(days, 30))
                    df = df[df.index >= cutoff]
                    df.to_parquet(cache_path, index=True)
                return df
        except Exception as e:
            log.warning(f"Cache read error: {e}")

    log.info(f"MOEX: full fetch {_clean_symbol(symbol)} 1m ({days}d)")
    df = _do_fetch_1m(symbol, days)
    if df is not None and not df.empty:
        df.to_parquet(cache_path, index=True)
    return df


def _do_fetch_1m(symbol: str, days: int) -> Optional[pd.DataFrame]:
    import requests as sync_requests

    secid = _resolve_moex_secid(symbol)
    market = _get_moex_market_info(symbol)
    end = datetime.now()
    start = end - timedelta(days=days)

    url = (f"{MOEX_BASE}/engines/{market['engine']}/markets/{market['market']}/"
           f"boards/{market['board']}/securities/{secid}/candles.json")
    params = {"from": start.strftime("%Y-%m-%d"), "till": end.strftime("%Y-%m-%d"),
              "interval": 1, "start": 0}

    all_candles = []
    page = 0
    try:
        while True:
            params["start"] = page
            resp = sync_requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                log.error(f"MOEX {resp.status_code} for {secid}")
                return None
            data = resp.json()
            candles = data.get("candles", {})
            cols = candles.get("columns", [])
            vals = candles.get("data", [])
            if not vals:
                break
            try:
                oi = cols.index("open"); hi = cols.index("high")
                li = cols.index("low"); ci = cols.index("close")
                vi = cols.index("volume"); bi = cols.index("begin")
            except ValueError:
                break
            for row in vals:
                try:
                    dt = datetime.strptime(row[bi], "%Y-%m-%d %H:%M:%S")
                    all_candles.append({
                        "datetime": dt, "open": float(row[oi]),
                        "high": float(row[hi]), "low": float(row[li]),
                        "close": float(row[ci]), "volume": float(row[vi]),
                    })
                except Exception:
                    continue
            if len(vals) < 500 or page > 10000:
                break
            page += 500
    except Exception as e:
        log.error(f"MOEX fetch error: {e}")
        return None

    if not all_candles:
        return None
    df = pd.DataFrame(all_candles).set_index("datetime")
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df


def _aggregate_1m(df_1m: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    rule = f"{target_minutes}min"
    return df_1m.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


# ── Async Arena client ─────────────────────────────────────────────────────

class ArenaClient:
    """Async Arena REST client with auto JWT refresh + MOEX ISS data."""

    def __init__(self, api_token: str, account_id: int,
                 cache_dir: Optional[Path] = None) -> None:
        self._api_token = api_token
        self.account_id = account_id
        self._jwt: Optional[str] = None
        self._jwt_expires_at: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache_dir = cache_dir or Path(__file__).parent / "cache"

    async def _ensure_session(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(_BASE)
            await self._refresh_token()

    async def __aenter__(self) -> "ArenaClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _refresh_token(self) -> None:
        async with self._session.post("/v1/sessions",
                                       json={"secret": self._api_token}) as resp:
            if not resp.ok:
                text = await resp.text()
                raise RuntimeError(f"Arena auth failed [{resp.status}]: {text}")
            data = await resp.json()
        self._jwt = data["token"]
        self._jwt_expires_at = time.monotonic() + _TOKEN_TTL
        log.debug("Arena JWT refreshed")

    async def _ensure_token(self) -> None:
        await self._ensure_session()
        if not self._jwt or time.monotonic() >= self._jwt_expires_at:
            await self._refresh_token()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._jwt}"}

    async def place_order(self, symbol: str, side: str, qty: int) -> Optional[OrderResult]:
        await self._ensure_token()
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": {"value": str(qty)},
        }
        try:
            async with self._session.post(
                f"/v1/accounts/{self.account_id}/orders",
                json=payload, headers=self._headers(),
            ) as resp:
                if not resp.ok:
                    text = await resp.text()
                    log.error(f"Arena order failed [{resp.status}] {side} {symbol} ×{qty}: {text}")
                    return None
                data = await resp.json()
        except Exception as exc:
            log.error(f"Arena order error {side} {symbol} ×{qty}: {exc}")
            return None

        o = data["order"]
        result = OrderResult(
            order_id=data["order_id"],
            symbol=o["symbol"],
            side=o["side"],
            quantity=float(o["quantity"]["value"]),
            execution_price=float(o["execution_price"]["value"]),
            commission=float(o["commission"]["value"]),
        )
        log.info(
            f"Arena  {result.side:<10} {result.symbol:<16} ×{result.quantity:.0f}"
            f"  price={result.execution_price:.4f}  comm={result.commission:.4f}"
            f"  id={result.order_id}"
        )
        return result

    async def get_account(self) -> dict:
        await self._ensure_token()
        async with self._session.get(
            f"/v1/accounts/{self.account_id}", headers=self._headers(),
        ) as resp:
            if not resp.ok:
                text = await resp.text()
                raise RuntimeError(f"Arena get_account failed [{resp.status}]: {text}")
            return await resp.json()

    async def get_balance(self) -> float:
        try:
            acc = await self.get_account()
            return float(acc.get("available_cash", {}).get("value", 0))
        except Exception:
            return 0.0

    async def get_equity(self) -> float:
        try:
            acc = await self.get_account()
            return float(acc.get("equity", {}).get("value", 0))
        except Exception:
            return 0.0

    async def get_positions(self) -> list:
        try:
            acc = await self.get_account()
        except Exception:
            return []
        raw = acc.get("positions", [])
        positions = []
        for pos in raw:
            try:
                qty = float(pos.get("quantity", {}).get("value", 0))
                positions.append({
                    "symbol": pos.get("symbol"),
                    "quantity": abs(qty),
                    "side": "BUY" if qty > 0 else "SELL",
                    "avg_price": float(pos.get("average_price", {}).get("value", 0)),
                    "unrealized_pnl": float(pos.get("unrealized_pnl", {}).get("value", 0)),
                })
            except Exception:
                continue
        return positions

    async def get_bars(self, symbol: str, timeframe: str = "15m",
                       days: int = 30) -> Optional[pd.DataFrame]:
        target_min = TIMEFRAME_MINUTES.get(timeframe)
        if not target_min:
            log.error(f"Unknown timeframe: {timeframe}")
            return None

        fetch_days = min(days, 30)

        # Run blocking MOEX ISS call in thread pool
        loop = asyncio.get_running_loop()
        df_1m = await loop.run_in_executor(
            None, _fetch_1m_candles_sync, symbol, fetch_days, self._cache_dir
        )
        if df_1m is None or df_1m.empty:
            return None

        if target_min in MOEX_SUPPORTED_INTERVALS:
            df = _aggregate_1m(df_1m, target_min)
        else:
            df = _aggregate_1m(df_1m, target_min)

        if df is None or df.empty:
            return None

        log.info(f"Bars {symbol} {timeframe}: {len(df)} candles, last={df['close'].iloc[-1]:.2f}")
        return df
