from typing import Tuple, List

import numpy as np

from models import Coordinate


class GraphicPlotter:

    def __init__(self, title: str, connexions: List[Tuple[Coordinate, List[Coordinate]]]):
        self.title = title
        self.connexions = connexions

    def plot(self):
        import matplotlib.pyplot as plt
        for point, connected_points in self.connexions:
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color = (r, g, b)
            plt.plot(point.x, point.y, '^', c=color)
            plt.grid()
            for connected_point in connected_points:
                plt.plot(connected_point.x, connected_point.y, '.', c=color)
            plt.legend(['Access point', 'Demand point'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)
        plt.show()
