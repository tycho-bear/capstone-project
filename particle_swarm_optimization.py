# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
import numpy as np
from config import SEED
from problem import Problem, Solution
import time
from helper_functions import (generate_random_city_population,
                              generate_grid_population,
                              visualize_solution_fitness)
from traveling_salesman_problem import TravelingSalesmanProblem


np.random.seed(SEED)


class ParticleSwarmOptimization:
    """
    This class implements a particle swarm optimizer.
    """

    def __init__(self,

                 ):
        """"""
        # problem
        # population
        # hyperparameters


    def pso_iteration(self):
        """"""
        # performs one iteration of PSO
        # housekeeping
        # for each particle, generate velocity, move, and update current best
            # velocity = calculate_velocity
            # apply_velocity
        # update current global best


    def optimize(self):
        """"""
        # run the algorithm for multiple iterations
        # time it, do housekeeping at the end
        # also print the best solution at the end


    def print_initial_information(self):
        """"""
        # prints hyperparameters


    def print_current_iteration_information(self):
        """"""
        # iteration ####/#####, fitness of best individual


    def get_solution_values(self):
        """"""
        # gets the stored solution fitness values for plotting



def main():
    """"""
    # testing PSO on TSP grid
    # copy some stuff / structure from genetic_algorithm.py main

    # =========================================
    # |  Hyperparameter combinations:
    # =========================================

    # ...

    # =========================================
    # |  The actual code to run the algorithm:
    # =========================================

    # ...


if __name__ == '__main__':
    main()
