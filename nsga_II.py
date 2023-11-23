import random
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

        self.pas_count = count_pas
        self.total_distance = total_distance
        self.solution = solution

        return count_pas, total_distance, solution

    def plot(self, idx: int = None):
        plotter = GraphicPlotter(f'NSGA Solution {idx or random.randint(0, 100)}', connexions=[
            (self.points[idx], [self.customers[cidx].coordinates for cidx in customers_idxs]) for idx, customers_idxs in
            enumerate(self.solution) if customers_idxs])

        plotter.plot()

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

    def mutation(self):
        N = len(self.priorities)
        mutate_priorities = self.priorities

        for i in range(N):
            if(random.random() < 0.7): #mutation prob = 0.7
                j = random.sample(range(N),1)
                mutate_priorities[i], mutate_priorities[j] = mutate_priorities[j], mutate_priorities[i]
        
        self.priorities = mutate_priorities


def dominate(solution1, solution2): #verifica se geracao1 domina geracao2

  if(solution1.total_distance < solution2.total_distance and solution1.active_points < solution2.active_points):
    return True

  else:
    return False
 
def crossover(solutionA,solutionB):
    N = len(solutionA.priorities)
    C = [0]*N

    A = solutionA.priorities
    B = solutionB.priorities

    for i in range(1,int(N/2)+1):
        indice = A.index(i)
        C[indice] = A[indice]

    for i in range(int(N/2)+1, N+1):
        indice = B.index(i)
        if(C[indice] == 0):
            C[indice] = B[indice]

        else:
            for j in range(N):
                if(C[j] == 0):
                    C[j] = B[indice]
                    break

    nsga = NSGAII()
    return nsga.get_solution(C)

def get_next_generation_solutions(front): #get the first N/2 front elements to the next generation
    next_gen_solutions = []

    for i in range(len(front)):
        for objeto in front[i]:
            if(len(next_gen_solutions) < 50):
                if(50 - len(next_gen_solutions) > len(front[i])):
                    next_gen_solutions.append(objeto)
                else:
                    next_gen_solutions = next_gen_solutions + (crowding_distance_(front[i], 50 - len(next_gen_solutions)))
    
    return next_gen_solutions

def crowding_distance_(front_component, N):
    points = []

    for i in range(len(front_component)):
        points.append((front_component[i].total_distance, front_component[i].active_points))

    def crowding_distance(point, other_points):
        distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in other_points]
        return sum(sorted(distances)[1:-1])
    
    selected_points = []

    for _ in range(N):
        crowding_dict = {i: crowding_distance(point, points) for i, point in enumerate(points)}
        max_crowding_index = max(crowding_dict, key=crowding_dict.get)
        selected_points.append(points[max_crowding_index])
        del points[max_crowding_index]

    selected_solutions = []

    for i in range(len(selected_points)):
        indice = points.index(selected_points[i])
        selected_solutions.append(front_component[indice])

    return selected_solutions


if __name__ == '__main__':
    solution_number = 50
    nsga = NSGAII.from_csv()
    generation = []
    first_pop = []

    for i in range(solution_number):
        s = NSGAII.from_csv()
        first_pop.append(s)
    
    generation = []

    NSGAII_interactions = 50

    population = first_pop



    for count in range(NSGAII_interactions):
        front = []

        children = []
        for i in range(len(population)):
            j, k = random.sample(range(0, len(population)),2)
            children.append(crossover(population[j], population[k]))


        for i in range(len(population)):
            children[i].mutation()

        population = population + children

        for i in range(len(population)):
            population[i].get_solution()

        while(len(population) > 0):
            for i in range(len(population)):
                front_atual = []

                Sp = []
                np = 0

                for j in range(len(population)):
                    if(dominate(population[i],population[j])):
                        Sp.append(population[i])
                    elif(dominate(population[j], population[i])):
                        np += 1

                if(np == 0):
                    front_atual.append(population[i])
            
            front.append(front_atual)

            population = [elemento for elemento in population if elemento not in front_atual]

        generation.append(front[0])

        population = get_next_generation_solutions(front)



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
