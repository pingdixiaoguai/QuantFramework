"""Tushare token configuration."""

import os

from dotenv import load_dotenv


def get_tushare_token() -> str:
    """Load Tushare API token from env var or .env file."""
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token

    load_dotenv()
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token

    raise RuntimeError("TUSHARE_TOKEN missing; set env var or put it in .env")
