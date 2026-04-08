from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class OpenEnvHTTPClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        payload = {}
        if task_name:
            payload["task_name"] = task_name
        return self._request("POST", "/reset", payload)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/step", action)

    def state(self) -> Dict[str, Any]:
        return self._request("GET", "/state")

    def close(self) -> None:
        return None

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.request(method, url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
