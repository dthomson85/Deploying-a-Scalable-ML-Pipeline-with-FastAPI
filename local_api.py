from typing import Any, Dict

from local_api_client import get_root, post_data, parse_json_safe


data: Dict[str, Any] = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def main() -> None:
    r = get_root()
    print(r.status_code)
    print(r.text)

    r = post_data(data, path="/data/")
    print(r.status_code)
    print(parse_json_safe(r))


if __name__ == "__main__":
    main()
