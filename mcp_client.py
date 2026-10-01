"""
Live client for Microsoft Dynamics 365 Business Central.
Supports credentials from st.secrets (Streamlit Cloud) or local bc_config.json.
"""

import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BC_MCP_Client")

MCP_SESSION_EXPIRED = ("Session not found", "-32001")


def _body(res: requests.Response) -> Any:
    try:
        return res.json()
    except ValueError:
        return res.text


class BusinessCentralMCPClient:
    def __init__(self, config_path: str = "bc_config.json"):
        self.config: Dict[str, Any] = {}

        # 1. Try reading from Streamlit Secrets first (for Streamlit Cloud)
        if HAS_STREAMLIT and hasattr(st, "secrets") and len(st.secrets) > 0:
            try:
                self.config = {
                    "groq_api_key": st.secrets.get("groq_api_key"),
                    "business_central": dict(st.secrets.get("business_central", {})),
                    "auth": dict(st.secrets.get("auth", {})),
                    "cfo": dict(st.secrets.get("cfo", {})),
                }
            except Exception:
                pass

        # 2. Fall back to local bc_config.json
        if not self.config or not self.config.get("business_central"):
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)

        self.bc_cfg: Dict[str, Any] = self.config.get("business_central", {})
        self.auth_cfg: Dict[str, Any] = self.config.get("auth", {})

        self.mcp_url = self.bc_cfg.get("mcp_server_url") or "https://mcp.businesscentral.dynamics.com/"
        self.api_base = (
            f"https://api.businesscentral.dynamics.com/v2.0/"
            f"{self.bc_cfg.get('tenant_id')}/{self.bc_cfg.get('environment')}/api/v2.0"
        )

        self.access_token: Optional[str] = None
        self._token_expires_at = 0.0
        self.mcp_session_id: Optional[str] = None
        self._company_id: Optional[str] = None
        self._rpc_id = 0

    def authenticate(self) -> Dict[str, Any]:
        url = f"https://login.microsoftonline.com/{self.bc_cfg['tenant_id']}/oauth2/v2.0/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.auth_cfg["client_id"],
            "client_secret": self.auth_cfg["client_secret"],
            "scope": "https://api.businesscentral.dynamics.com/.default",
        }
        try:
            res = requests.post(url, data=payload, timeout=15)
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

        body = _body(res)
        if res.status_code != 200:
            self.access_token = None
            logger.error(f"Entra ID token error {res.status_code}: {res.text[:300]}")
            return {"success": False, "status_code": res.status_code, "error": body}

        self.access_token = body["access_token"]
        self._token_expires_at = time.time() + int(body.get("expires_in", 3600)) - 120
        logger.info("Authenticated with Microsoft Entra ID")
        return {"success": True}

    def _ensure_token(self) -> None:
        if not self.access_token or time.time() >= self._token_expires_at:
            res = self.authenticate()
            if not res.get("success"):
                raise RuntimeError(f"Authentication failed: {res.get('error')}")

    def _rest(self, method: str, path: str, **kw) -> requests.Response:
        self._ensure_token()
        url = path if path.startswith("http") else f"{self.api_base}/{path.lstrip('/')}"
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
            if "json" in kw:
                headers["Content-Type"] = "application/json"
            res = requests.request(method, url, headers=headers, timeout=90, **kw)
            if res.status_code == 401 and attempt == 1:
                self.authenticate()
                continue
            return res
        return res

    def get_company_id(self) -> str:
        if self._company_id:
            return self._company_id
        res = self._rest("GET", "companies")
        if res.status_code != 200:
            raise RuntimeError(f"Could not list companies ({res.status_code}): {res.text[:500]}")
        companies = res.json().get("value", [])
        wanted = self.bc_cfg["company_name"].strip().lower()
        for c in companies:
            if wanted in (str(c.get("name", "")).lower(), str(c.get("displayName", "")).lower()):
                self._company_id = c["id"]
                return self._company_id
        names = [c.get("name") for c in companies]
        raise RuntimeError(f"Company '{self.bc_cfg['company_name']}' not found. Companies in this environment: {names}")

    def _company_path(self, sub: str) -> str:
        return f"companies({self.get_company_id()})/{sub.lstrip('/')}"
