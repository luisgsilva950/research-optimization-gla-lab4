from typing import Optional, List, Collection

import numpy
from memoization import cached

from utils import get_arg_min


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def get_distance(self, point: 'Coordinate'):
        return CoordinatesCalculator.get_distance(self, point)

    def get_closer_point(self, points: Collection['Coordinate'], distances: List[float] = None) -> 'Coordinate':
        points = list(points)
        if not distances:
            distances = [self.get_distance(p) for p in points]
        if len(points) != len(distances):
            distances = [self.get_distance(p) for p in points]
        index_min = get_arg_min(distances)
        return points[index_min]

    def __repr__(self):
        return f'Coordinate(x={self.x}, y={self.y})'

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def __lt__(self, other: 'Coordinate'):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x

    def __gt__(self, other: 'Coordinate'):
        if self.x == other.x:
            return self.y > other.y
        return self.x > other.x

    def __ne__(self, other):
        return not (self == other)


class PA(Coordinate):

    def __init__(self, x: float, y: float, index: Optional[int]):
        super(PA, self).__init__(x, y)
        self.index = index
        self.possible_customers = []

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other: 'PA'):
        return self.index == other.index

    def get_neighbor_indexes(self) -> List[int]:
        return [min(self.index + 1, 10200), self.index - 1, min(self.index + 1000, 10200), self.index - 1000]


class Customer:
    def __init__(self, consume: float, index: int, coordinates: Coordinate):
        self.consume = consume
        self.coordinates = coordinates
        self.index = index
        self.point_idx = None

    def connected(self) -> bool:
        return self.point_idx is not None

    def get_closer_point(self, points: Collection['Coordinate'], distances: List[float] = None) -> 'Coordinate':
        return self.coordinates.get_closer_point(points=points, distances=distances)


class CoordinatesCalculator:
    @staticmethod
    @cached
    def find_pareto_frontier(points: List[Coordinate]) -> List[Coordinate]:
        pareto_great_solutions = []
        points.sort()
        for index, coordinate in enumerate(points):
            is_dominant = False

            if not pareto_great_solutions:
                is_dominant = True
            elif coordinate.x == pareto_great_solutions[-1].x:
                is_dominant = coordinate.y <= pareto_great_solutions[-1].y
            elif coordinate.y == pareto_great_solutions[-1].y:
                is_dominant = coordinate.x <= pareto_great_solutions[-1].x
            elif coordinate.x > pareto_great_solutions[-1].x:
                is_dominant = coordinate.y < pareto_great_solutions[-1].y

            if is_dominant:
                pareto_great_solutions.append(coordinate)

        return pareto_great_solutions

    @staticmethod
    @cached
    def get_distance(p: Coordinate, c: Coordinate):
        a = numpy.array((p.x, p.y, 0))
        b = numpy.array((c.x, c.y, 0))
        return numpy.linalg.norm(a - b)
