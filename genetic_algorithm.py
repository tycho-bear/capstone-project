# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import numpy as np
from config import seed
from problems import Problem, Solution


np.random.seed(seed)


class GeneticAlgorithm():
    """
    This class implements a genetic algorithm.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 elitism_percent: float,
                 crossover_percent: float,
                 mutation_rate: float,
                 tournament_size: int
                 ):
        """"""

        # what to pass in here besides a problem?
        # population size
        # crossover rate (percentage?)
        # mutation rate

        self.problem = problem
        self.solution_values = []  # for plotting

        # main GA parameters



    # methods...

    def ga_generation(self):
        """"""



    def evolve(self):
        """"""
        # loop, call ga_generation
        # at the end, return self.solution.best_agent


    def get_solution_values(self):
        """"""
        return self.solution_values


def main():
    """"""
    # =========================================
    # |  Hyperparameter combinations:
    # =========================================


    # =========================================
    # |  The actual code to run the algorithm:
    # =========================================




if __name__ == '__main__':
    main()
