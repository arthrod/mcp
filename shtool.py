"""Streamlined StagehandTool - Drop-in replacement that makes endpoint calls
Just replace your existing StagehandTool with this class - no other changes needed!
"""
from typing import Any

import requests


class StagehandTool:
    """Drop-in replacement - makes endpoint calls to Node.js bridge."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:3001"
        self.session_id: str | None = None
        # Removed server process management per request

    def _request(self, method: str, endpoint: str, data: dict | None = None) -> dict[str, Any]:
        """Make request to bridge server."""
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=30)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def initialize(self) -> dict[str, Any]:
        """Initialize session."""
        result = self._request("POST", "/init")
        if result.get("success"):
            self.session_id = "bridge_session"
        return result

    async def navigate(self, url: str) -> dict[str, Any]:
        """Navigate to URL."""
        return self._request("POST", "/navigate", {"url": url})

    async def act(self, action: str, dom_settle_timeout_ms: int = 1000) -> dict[str, Any]:
        """Perform action."""
        return self._request("POST", "/act", {"action": action})

    async def observe(self, instruction: str, draw_overlay: bool = True) -> dict[str, Any]:
        """Observe page."""
        return self._request("POST", "/observe", {"instruction": instruction})

    async def extract(self, instruction: str, schema_model) -> dict[str, Any]:
        """Extract data."""
        schema = schema_model.model_json_schema() if hasattr(schema_model, "model_json_schema") else schema_model
        return self._request("POST", "/extract", {"instruction": instruction, "schema": schema})

    async def paste_into_page(self, content: str) -> dict[str, Any]:
        """Paste content - uses agent for complex task."""
        return self._request("POST", "/agent", {
            "instructions": f"Paste or Write the following: {content} and then click add, continue, save or submit (the appropriate button to effectively submit the content). TO PASTE INTO NOTEBOOKLM, YOU CAN DO SO BY CLICKING 'COPIED TEXT' and then paste, within the add/upload sources screen",
            "maxSteps": 5,
        })

    async def close(self) -> dict[str, Any]:
        """Close session."""
        return self._request("POST", "/close")

    # Simple wrapper properties for backward compatibility
    @property
    def stagehand(self):
        """Stagehand wrapper."""
        return self

    @property
    def page(self):
        """Page wrapper."""
        class PageWrapper:
            def __init__(self, tool) -> None:
                self.tool = tool

            async def goto(self, url: str) -> None:
                result = await self.tool.navigate(url)
                if not result.get("success"):
                    raise Exception(result.get("error", "Navigation failed"))

            @property
            def url(self):
                result = self.tool._request("GET", "/url")
                return result.get("url", "unknown")

        return PageWrapper(self)
