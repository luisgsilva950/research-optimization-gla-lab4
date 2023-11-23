from typing import List

import numpy as np
import numpy.random

from graphic_plotter import GraphicPlotter
from models import Customer, PA, Coordinate
from utils import DISTANCES


class NSGAII:
    min_customers_attended = 570
    max_distance = 85
    max_active_points = 100
    max_consumed_capacity = 150

    def __init__(self, customers: List[Customer], points: List[PA],
                 solution=None, active_points=None):
        self.customers = customers or []
        self.points = points or []
        self.active_points = active_points or set()
        self.solution = solution
        self.total_distance = 0
        self.pas_count = 0
        self.priorities = np.random.permutation(len(self.points))

        print(self.priorities)

    def get_solution(self, priorities: List[int] = None):
        if not priorities:
            priorities = np.random.permutation(len(self.points))
        count_pas, total_distance = 0, 0
        solution = [[] for _ in range(len(self.points))]
        customers_attended = set()
        for p in priorities:
            point = self.points[p]
            consume = 0
            for c in point.possible_customers:
                if c.index in customers_attended:
                    continue

                next_consume = consume + c.consume
                if next_consume > self.max_consumed_capacity:
                    break

                solution[p].append(c.index)
                customers_attended.add(c.index)
                consume = next_consume
                total_distance += DISTANCES[c.index][p]

        customers_count = 0
        for idx, l in enumerate(solution):
            if not l:
                continue
            count_pas += 1
            customers_count += len(l)
            print(idx, l, "\n")

        print("Numero de pontos", count_pas, "Taxa de clientes atendidos", customers_count / len(self.customers),
              "Distância total", total_distance)

        plotter = GraphicPlotter('Teste', connexions=[
            (self.points[idx], [self.customers[cidx].coordinates for cidx in customers_idxs]) for idx, customers_idxs in
            enumerate(solution) if customers_idxs])

        plotter.plot()

        self.pas_count = count_pas
        self.total_distance = total_distance
        self.solution = solution

        return count_pas, total_distance, solution

    @staticmethod
    def from_csv() -> 'NSGAII':

        def get_customers() -> List[Customer]:
            customers = []
            with open('clientes.csv') as file:
                content = file.readlines()
                for index, row in enumerate(content):
                    row = row.split(",")
                    customer = Customer(coordinates=Coordinate(x=float(row[0]), y=float(row[1])),
                                        consume=float(row[2]),
                                        index=index)
                    customers.append(customer)
            return customers

        def get_pas() -> List[PA]:
            points = []
            for x in range(0, 1010, 10):
                for y in range(0, 1010, 10):
                    points.append(PA(x=x, y=y, index=len(points)))
            return points

        return NSGAII(customers=get_customers(), points=get_pas())

    def plot_customers(self):
        import matplotlib.pyplot as plt

        customers = self.customers
        pas = self.points

        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        color = (r, g, b)
        for customer in customers:
            plt.plot(customer.coordinates.x, customer.coordinates.y, 'o', c=color)

        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        color = (r, g, b)
        for pa in pas:
            plt.plot(pa.x, pa.y, '^', c=color)

        plt.grid()
        plt.show()


if __name__ == '__main__':
    solutions = []
    nsga = NSGAII.from_csv()

    # start = time.time()
    #
    # for i in range(50):
    #     s = nsga.get_initial_solution()
    #     solutions.append(s)
    #
    # print("Initial solution", min([m.penal_fitness for m in solutions]), f"{time.time() - start} seconds")
    #
    # start = time.time()
    #
    # for _ in range(500):
    #     size = len(solutions)
    #
    #     for i in range(size):
    #         solution: NSGAII = solutions[i]
    #         solution.k = random.randint(1, 3)
    #         start = time.time()
    #         # print("Shaking...")
    #         solutions.append(solution.shake())
    # print("Shake", f"{time.time() - start} seconds")

    # print("Finish to generate mutations", f"{time.time() - start} seconds")

    # start = time.time()

    # solutions = numpy.random.permutation(solutions)

    # print("Finish to permutate", f"{time.time() - start} seconds")

    # new_solutions = []
    #
    # start = time.time()

    # print("Comparing...")

    # for i in range(0, len(solutions), 2):
    #     s1, s2 = solutions[i], solutions[i + 1]
    #     s1.objective_function()
    #     s2.objective_function()
    #     if s1.penal_fitness <= s2.penal_fitness:
    #         new_solutions.append(s1)
    #     else:
    #         new_solutions.append(s2)
    #
    # solutions = new_solutions
    #
    # print("Final solution", min([m.penal_fitness for m in solutions]), f"{time.time() - start} seconds")
