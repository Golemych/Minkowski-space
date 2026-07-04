import asyncio
import aiohttp
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path
class OrderResult:
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: int
    price: float

def _ensure_session(session):
    if session is None or not session.closed:
        return session
    else:
        raise ValueError("Session is closed")

@dataclass
class ArenaClient:
    api_token: str
    account_id: int
    cache_dir: Path
    _session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_account(self) -> Dict:
        session = await self._ensure_session()
        url = f"https://api.arena.finance/v1/accounts/{self.account_id}"