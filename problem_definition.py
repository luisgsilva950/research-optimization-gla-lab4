import collections
from typing import Set, List, Optional

import numpy
from memoization import cached
from numpy import random

from graphic_plotter import GraphicPlotter
from models import PA, Customer
from utils import get_arg_min, DISTANCES


@cached
def get_points_with_space_100(ps) -> List[PA]:
    points = []
    for p in ps:
        if p.x % 100 == 0 and p.y % 100 == 0 and p.x >= 100 and p.y >= 100:
            points.append(p)
    return points


class ProblemDefinition:
    k: int
    total_distance: float
    max_distance: float
    max_consumed_capacity: float
    max_active_points: int
    min_customers_attended: int
    active_points: Set[PA]
    customers: List[Customer]
    points: List[PA]
    penal_fitness: float
    fitness: float
    penal: float

    def objective_function(self) -> 'ProblemDefinition':
        ...

    def neighborhood_change(self, y: 'ProblemDefinition') -> 'ProblemDefinition':
        ...

    def shake(self) -> 'ProblemDefinition':
        ...

    def get_initial_solution(self) -> 'ProblemDefinition':
        ...

    def update_active_points(self):
        self.active_points = set()
        for customer in self.customers:
            if customer.connected():
                self.active_points.add(self.points[customer.point_idx])

    # FO methods
    def penalize_distance(self, distance: float):
        if distance > self.max_distance:
            self.penal = self.penal + (distance - self.max_distance) * 2

    def penalize_consumed_capacity(self, consumed_capacity: float):
        if consumed_capacity > self.max_consumed_capacity:
            self.penal = self.penal + 2 * (consumed_capacity - self.max_consumed_capacity)
            print(f"The consumed capacity restriction was affected. Consumed capacity: {consumed_capacity}")

    def penalize_total_active_points(self):
        total_active_points = len(self.active_points)
        if total_active_points > self.max_active_points:
            self.penal = self.penal + 600 * (total_active_points - self.max_active_points)
            print(f"The total active points restriction was affected. Total active points: {total_active_points}")

    def penalize_total_customers(self, customers_attended_count: int):
        if customers_attended_count < self.min_customers_attended:
            self.penal = self.penal + 600 * (self.min_customers_attended - customers_attended_count)
            print(f"The total customers restriction was affected. Total customers: {customers_attended_count}")

    # Shake methods
    def deactivate_point(self, index: int):
        for customer in self.customers:
            if customer.point_idx == index:
                customer.point_idx = None

    def enable_customer_point(self, customer: Customer, point: PA):
        customer.point_idx = point.index

    def disable_customer_point(self, customer: Customer, point: PA):
        if customer.point_idx == point.index:
            customer.point_idx = None

    def enable_random_customers(self, points: List[PA], size: int = 10):
        random_customers: List[Customer] = [
            *numpy.random.choice([c for c in self.customers if not c.connected()], size=size)]
        for customer in random_customers:
            closer_point = customer.get_closer_point(points=points)
            if DISTANCES[customer.index][closer_point.index] < self.max_distance:
                self.enable_customer_point(customer=customer, point=closer_point)

    def deactivate_random_customers(self, size: int = 10):
        random_customers: List[Customer] = [
            *numpy.random.choice([c for c in self.customers if c.connected()], size=size)]
        for customer in random_customers:
            if customer.point_idx is not None:
                self.disable_customer_point(customer=customer, point=self.points[customer.point_idx])

    def deactivate_random_access_points(self, size: int = 2):
        random_points = [*numpy.random.choice([*self.active_points], size=size)]
        for point in random_points:
            self.deactivate_point(index=point.index)

    def deactivate_random_demand_point_and_connect_closer_point(self):
        random_point: PA = numpy.random.choice(list(self.active_points))
        active_indexes: List[int] = [p.index for p in self.active_points]
        possible_indexes: List[int] = random_point.get_neighbor_indexes()
        possible_indexes: List[int] = [i for i in possible_indexes if i not in active_indexes]
        for customer in self.customers:
            if customer.point_idx == random_point.index:
                possible_distances = [DISTANCES[customer.index][i] for i in possible_indexes]
                if possible_distances:
                    closer_index = possible_indexes[get_arg_min(possible_distances)]
                    self.enable_customer_point(customer=customer, point=self.points[closer_index])
        self.deactivate_point(index=random_point.index)

    def connect_random_customers_to_closer_active_access_point(self, points, size: int = 5):
        random_customers: List[Customer] = [*numpy.random.choice(self.customers, size=size)]
        for customer in random_customers:
            closer_point = customer.get_closer_point(points=points, distances=DISTANCES[customer.index])
            if customer.connected() and closer_point.index != customer.point_idx:
                self.enable_customer_point(customer=customer, point=closer_point)
                self.disable_customer_point(customer=customer, point=self.points[customer.point_idx])

    def deactivate_less_demanded_point_and_enable_highest_access_closer_point(self):
        less_demanded_point: PA = self.get_less_demanded_point()
        if less_demanded_point:
            for customer in self.customers:
                if customer.point_idx == less_demanded_point.index:
                    candidates = [p for p in self.active_points if
                                  p.index != less_demanded_point.index and
                                  DISTANCES[customer.index][p.index] < self.max_distance]
                    if candidates:
                        closer_point = random.choice(candidates)
                        self.enable_customer_point(customer=customer, point=self.points[closer_point.index])
            self.deactivate_point(index=less_demanded_point.index)

    def deactivate_less_demanded_access_point(self):
        less_demanded_point: PA = self.get_less_demanded_point()
        if less_demanded_point:
            self.deactivate_point(index=less_demanded_point.index)

    # Utils
    def get_customers_attended_count(self) -> int:
        return len([c for c in self.customers if c.connected()])

    def get_consumed_capacity(self) -> dict:
        consumed_capacity_per_point = collections.defaultdict(float)
        for customer in self.customers:
            if customer.point_idx:
                consumed_capacity_per_point[customer.point_idx] += customer.consume
        return consumed_capacity_per_point

    def get_less_demanded_point(self) -> Optional[PA]:
        consumed_capacity_per_point = self.get_consumed_capacity()
        if not self.active_points:
            return None
        point = next(iter(self.active_points))
        for p in self.active_points:
            if consumed_capacity_per_point[p.index] < consumed_capacity_per_point[point.index]:
                point = p
        return point

    def get_points_with_space_100(self) -> List[PA]:
        return get_points_with_space_100(self.points)

    # Plot methods
    def plot_solution(self):
        plotter = GraphicPlotter(title='Connexions', connexions=self.get_connexions())
        plotter.plot()

    def get_connexions(self):
        connexions_by_customer = collections.defaultdict(list)
        for customer in self.customers:
            if customer.connected():
                connexions_by_customer[customer.point_idx].append(customer)
        return list(connexions_by_customer.items())
