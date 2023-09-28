import json
from typing import List, Any

from memoization import cached


def column(matrix, i):
    return [row[i] for row in matrix]


@cached
def get_arg_max(_list: list) -> int:
    return _list.index(max(_list))


@cached
def get_arg_min(_list: list) -> int:
    return _list.index(min(_list))


def get_points_distances_from_file() -> List[List[float]]:
    with open('customer_point_distances.json', 'r') as f:
        return json.load(f)


def save_file(file_name: str, **kwargs) -> None:
    with open(file_name, 'w') as f:
        json.dump(kwargs, f)


def load_file(file_name: str) -> Any:
    with open(file_name, 'r') as f:
        return json.load(f)
