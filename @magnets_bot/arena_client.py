import aiohttp
from typing import Optional, List, Dict

class ArenaClient:
    def __init__(self, api_token: str, account_id: int, cache_dir: str):
        self.api_token = api_token
        self.account_id = account_id
        self.cache_dir = cache_dir
        self.session = aiohttp.ClientSession()

    async def get_account(self) -> Optional[Dict]:
        url = f"https://api.arena.finance/v1/accounts/{self.account_id}"
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return None

    async def get_positions(self) -> Optional[List[Dict]]:
        url = f"https://api.arena.finance/v1/accounts/{self.account_id}/positions"
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return None

    async def get_bars(self, symbol: str, timeframe: str, days: int) -> Optional[List[Dict]]:
        url = f"https://api.arena.finance/v1/instruments/{symbol}/bars?timeframe={timeframe}&days={days}"
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return None

    async def place_order(self, symbol: str, side: str, qty: int) -> Optional[Dict]:
        url = f"https://api.arena.finance/v1/accounts/{self.account_id}/orders"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        data = {
            "symbol": symbol,
            "side": side,
            "qty": qty
        }
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 201:
                return await response.json()
            else:
                return None