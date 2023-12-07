import os
import json

directory_path = './results'


def plot_heuristic(file_name: str):
    title = ""
    f1s = []
    f2s = []
    hipervolumes = []
    with open(f'{directory_path}/{file_name}') as fp:
        line = fp.readline()
        while line:
            if "[" in line:
                pareto_frontier, hipervolume = line.split("],")
                pareto_frontier = pareto_frontier + "]"
                pareto_frontier = pareto_frontier.replace("(", "[")
                pareto_frontier = pareto_frontier.replace(")", "]")
                pareto_frontier = json.loads(pareto_frontier)
                hipervolume = float(hipervolume)
                f1 = [f1v for f1v, f2v in pareto_frontier]
                f2 = [f2v for f1v, f2v in pareto_frontier]
                f1s.append(f1)
                f2s.append(f2)
                hipervolumes.append(hipervolume)
            else:
                title = line

            line = fp.readline()

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(15, 6))

    # Plot the boxplot
    plt.boxplot(f1s, positions=range(1, len(f1s) + 1), bootstrap=5000)
    plt.xticks(rotation=45)
    plt.xlabel('Generation Number')
    plt.ylabel('Values')
    plt.title(title.replace("\n", "") + " - f1")
    plt.show()
    plt.savefig(f'./images/{file_name.replace(".txt", "")}.png')


if __name__ == '__main__':
    for entry in os.listdir(directory_path):
        plot_heuristic(file_name=entry)
