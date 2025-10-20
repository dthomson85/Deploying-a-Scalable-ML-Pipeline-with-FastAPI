"""Reusable client helpers for the tutorial FastAPI server.

This module centralizes simple HTTP calls used by the tutorial examples.
It keeps the request details (base URL, headers, timeouts) in one place
so example scripts remain concise.

Public functions:
    - get_root(timeout=3.0): GET request to the API root (BASE_URL).
    - post_data(data, path='/predict', timeout=10.0): POST JSON to the API.
    - parse_json_safe(response): attempt to decode JSON; return None on failure.

The functions use the requests library. Callers should handle network
exceptions (requests.RequestException) if they need to retry or surface
errors differently.
"""
from typing import Any, Dict, Optional

import requests

BASE_URL = "http://127.0.0.1:8000"


def get_root(timeout: float = 3.0) -> requests.Response:
    """Perform a GET against the configured API root.

    Args:
        timeout: How many seconds to wait for the server to respond.

    Returns:
        requests.Response: the raw response object from requests.

    Notes:
        This function does not catch network errors (e.g. connection errors)
        — those will propagate as requests.RequestException to the caller.
    """
    return requests.get(BASE_URL, timeout=timeout)


def post_data(data: Dict[str, Any], path: str = "/data/", timeout: float = 10.0) -> requests.Response:
    """Send a JSON POST to the API.

    Args:
        data: JSON-serializable dictionary to send in the request body.
        path: URL path appended to BASE_URL (default '/predict').
        timeout: request timeout in seconds.

    Returns:
        requests.Response: response object. Caller can check status_code
        and use parse_json_safe() to extract JSON safely.
    """
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    return requests.post(url, json=data, headers=headers, timeout=timeout)


def parse_json_safe(response: requests.Response) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from a Response and return it, or None.

    This helper intentionally only swallows JSON decoding errors. It will
    not catch network-related exceptions (those should already have
    happened during the request). Use this to avoid raising when a server
    returns plain text or an empty body.

    Returns:
        dict or None: parsed JSON on success, otherwise None.
    """
    try:
        return response.json()
    except ValueError:
        # requests raises ValueError (simplejson JSONDecodeError subclass) on bad JSON
        return None
