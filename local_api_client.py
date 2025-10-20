from typing import Any, Dict, Optional

import requests

BASE_URL = "http://127.0.0.1:8000"


def get_root(timeout: float = 3.0) -> requests.Response:
    """Send a GET to the API root (BASE_URL).

    Args:
        timeout: request timeout in seconds.

    Returns:
        requests.Response
    """
    return requests.get(BASE_URL, timeout=timeout)


def post_data(
    data: Dict[str, Any],
    path: str = "/data/",
    timeout: float = 10.0
) -> requests.Response:
    """Send a JSON POST to the API.

    By default posts to BASE_URL + '/data/'. Adjust `path` if your
    API uses a different route.
    """
    url = BASE_URL + path
    headers = {"Content-Type": "application/json"}
    return requests.post(url, json=data, headers=headers, timeout=timeout)


def parse_json_safe(
    response: requests.Response
) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from a response, return None on failure.

    Only catch the JSON decoding errors from the response to avoid
    swallowing unrelated exceptions.
    """
    try:
        return response.json()
    except ValueError:
        return None
