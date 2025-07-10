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
                 problem: Problem,
                 initial_population: list,
                 population_size: int,
                 num_iterations: int,
                 alpha: float,
                 beta: float,
                 ):
        """"""
        # problem
        # population
        # hyperparameters

        # what problem, also save solution values like normal
        self.problem = problem
        self.solution_values = []  # for plotting

        # main PSO parameters
        self.POPULATION_SIZE = population_size
        self.NUM_ITERATIONS = num_iterations
        self.ALPHA = alpha
        self.BETA = beta

        # setting things up
        self.population = initial_population
        self.iteration_number = 0
        self.global_best = None
        self.global_best_fitness = None

        # keep track of the best
        # self.gen_best_solution = None  # updated every iteration
        # self.gen_best_fitness = None  # updated every iteration



    def pso_iteration(self):
        """"""
        # performs one iteration of PSO
        # housekeeping
        # for each particle, generate velocity, move, and update current best
            # velocity = calculate_velocity
            # apply_velocity
        # update current global best

        # keep track of the best solution and print stuff
        self.iteration_housekeeping()

        # for each particle, generate velocity, move, and update current best
        for particle in self.population:
            velocity = self.problem.calculate_velocity(particle,
                                                       self.global_best,
                                                       self.ALPHA,
                                                       self.BETA)
            self.problem.apply_velocity(particle, velocity)

        # update global best after the loop completes? or in housekeeping?


    def iteration_housekeeping(self):
        """
        Helper method that does a few things:
            - updates the current global best solution in the population and its
            fitness inside the class
            - keeps track of this solution so we can plot it later
            - every 100 iterations, prints info about the current generation

        Behaves much like the similarly-named method in `genetic_algorithm.py`.

        :return: None
        """

        # don't need to sort this time, but still need to get the best
        # todo - do we update the global best in here?
        #  yes, need a global best to move to during the first iteration
        #  we can just plot this, too

        # update the global best solution and its fitness
        # particles don't store fitness, just current solution and best solution
        # use problem.evaluate_solution

        self.global_best = min(self.population, key=lambda p: self.problem.evaluate_solution(p.current_solution))
        self.global_best_fitness = self.problem.evaluate_solution(self.global_best.current_solution)

        self.solution_values.append(self.global_best_fitness)

        # print stuff every 100 iterations (called again after the main loop)
        if self.iteration_number % 100 == 0:
            self.print_current_iteration_information()



    def optimize(self) -> None:
        """
        Runs the PSO algorithm for the given number of iterations.
        Every so often, housekeeping prints the information about the
        current iteration.

        Also keeps track of elapsed time.

        :return: None
        """
        # run the algorithm for multiple iterations
        # time it, do housekeeping at the end
        # also print the best solution at the end

        start_time = time.time()

        while self.iteration_number < self.NUM_ITERATIONS:
            self.pso_iteration()
            self.iteration_number += 1

        # take care of the final generation here
        self.iteration_housekeeping()

        end_time = time.time()
        # print best solution data
        print(f"Best solution in population at iteration "
              f"{self.iteration_number}: distance "
              f"{self.solution_values[-1]:.3f}")
        print(f"Elapsed time: {(end_time - start_time):.1f} seconds.")


    def print_initial_information(self) -> None:
        """
        Helper function that prints the hyperparameters used in PSO.

        :return: None
        """

        print("Running particle swarm optimization...")
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Number of iterations: {self.NUM_ITERATIONS}")
        print(f"Alpha: {self.ALPHA}")
        print(f"Beta: {self.BETA}")
        print()


    def print_current_iteration_information(self) -> None:
        """
        Helper function that prints out information about the current iteration.
        Something like this:

        `"Generation {current}/{total}: best fitness: {best_fitness}"`

        :return: NOne
        """

        print(f"Iteration {self.iteration_number}/{self.NUM_ITERATIONS}, "
              f"Fitness of best individual: {self.global_best_fitness:.3f}")


    def get_solution_values(self) -> list:
        """
        Getter for the solution fitness values.

        :return: (list) The current list of solution fitness values.
        """

        return self.solution_values



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
