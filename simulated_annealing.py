# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
from helper_functions import (generate_random_cities, generate_square_grid,
                              visualize_solution_fitness)
from config import SEED, X_MIN_WA, X_MAX_WA, Y_MIN_WA, Y_MAX_WA
from helper_classes import Tour
import copy
import numpy as np
from typing import Any
import time
from problem import Problem, Solution
from traveling_salesman_problem import TravelingSalesmanProblem


np.random.seed(SEED)


class SimulatedAnnealing:
    """
    This class implements the simulated annealing algorithm.
    """

    def __init__(self,
                 problem: Problem,
                 initial_guess: Solution,
                 max_iterations: int,
                 initial_temperature: float,
                 cooling_rate: float
                 ):
        """
        Creates a new instance of a simulated annealing algorithm with the given
        hyperparameters. These influence the behavior of the algorithm and the
        quality of the solutions it finds.

        Class structure subject to revisions in the near future.

        :param problem: (Problem) The problem to be optimized, such as the
            traveling salesman problem (represented by the class
            TravelingSalesmanProblem).
        :param max_iterations: (int) The maximum number of iterations the
            algorithm should run for.
        :param initial_temperature: (float) The temperature to start at. This
            influences the acceptance probability.
        :param cooling_rate: (float) For a geometric cooling rate, the
            temperature will be multiplied by this number after every iteration.
        """

        self.problem = problem
        self.solution_values = []

        # main SA parameters
        self.MAX_ITERATIONS = max_iterations
        self.current_iteration = 0
        self.temperature = initial_temperature
        self.COOLING_RATE = cooling_rate

        # set up initial stuff
        # self.solution = problem.generate_initial_guess()
        self.solution = initial_guess
        self.solution_fitness = problem.evaluate_solution(self.solution)
        """Like golf, lower is better (at least for the TSP)."""

        # keep track of the best here, this won't be used in the algorithm
        self.best_solution = copy.deepcopy(self.solution)
        self.best_solution_fitness = copy.deepcopy(self.solution_fitness)
        self.best_solution_iteration_num = copy.deepcopy(self.current_iteration)


    def accept_solution(self, new_solution: Solution) -> None:
        """
        Helper function that accepts the given solution. It sets the current
        solution to ``new_solution``, and saves its fitness value.

        :param new_solution: (Solution) The new solution to accept. For the TSP,
            this would be a Tour object.
        :return: None
        """

        self.solution = new_solution
        self.solution_fitness = self.problem.evaluate_solution(new_solution)


    def update_temperature(self) -> None:
        """
        Updates the current temperature using a geometric cooling schedule.
        Multiplies the temperature by the cooling rate (less than 1).

        :return: None
        """

        # can do something more complicated in the future
        self.temperature *= self.COOLING_RATE


    def sa_step(self) -> None:
        """
        Performs one step of the simulated annealing algorithm.

        First obtains a new solution, then evaluates its fitness compared to the
        current solution. If the new solution's fitness is better, it is
        accepted. If it's worse, then it's accepted with an exponential
        probability based on (a) the current temperature, and (b) how much worse
        it is.

        Also keeps track of the best solution found.

        :return: None
        """

        # get new solution
        new_solution = self.problem.generate_neighbor(self.solution)

        # make comparison
        difference = (self.problem.evaluate_solution(new_solution) -
                      self.solution_fitness)

        # new solution is worse, so we will accept it with a probability
        if difference >= 0:
            r = np.random.random()
            if r < math.exp((-1 * difference) / self.temperature):
                self.accept_solution(new_solution)
            return

        # new solution is better, so accept it
        self.accept_solution(new_solution)

        # keep track of the best solution, not super important, but it's cool
        if self.solution_fitness < self.best_solution_fitness:
            self.best_solution = copy.deepcopy(self.solution)
            self.best_solution_fitness = self.solution_fitness
            self.best_solution_iteration_num = self.current_iteration


    def anneal(self) -> None:
        """
        Runs the whole simulated annealing algorithm, performing the specified
        number of iterations. At each iteration, performs a single sa_step() and
        updates the temperature.

        :return: None
        """

        start_time = time.time()

        # self.print_current_iteration_information()
        while self.current_iteration < self.MAX_ITERATIONS:
            self.sa_step()
            self.update_temperature()

            # would be nice to get a visualization this way, but there's a bug
            # that causes an error after too many plots are generated.
            # self.problem.display_solution(self.solution)

            # keep track of solution fitness for each iteration
            self.solution_values.append(self.solution_fitness)

            # print every 500 iterations
            if self.current_iteration % 500 == 0:
                self.print_current_iteration_information()

            self.current_iteration += 1
        self.print_current_iteration_information()

        end_time = time.time()

        # print best solution data
        # print(f"Found best solution at iteration "
        #       f"{self.best_solution_iteration_num}:  "
        #       f"distance {self.best_solution_fitness:.3f}")
        print(f"Elapsed time: {(end_time - start_time):.1f} seconds.")


    def print_current_iteration_information(self) -> None:
        """
        Helper function that prints out information about the current iteration.
        For the TSP, this includes the iteration number, the current tour, and
        the tour's distance.

        :return: None
        """

        print(f"Iteration {self.current_iteration}: "
              f"temp. {self.temperature:.4f},  "
              f"distance {self.solution_fitness:.3f}")

    def get_solution_values(self) -> list[float]:
        """
        Helper function that returns the stored list of solution fitness values.

        :return: (list[float]) The stored list of solution fitness values.
        """

        return self.solution_values


def main() -> None:
    """
    Driver, runs the simulated annealing algorithm.

    :return: None
    """



    # random 64 cities, distance 44.598
    # max_iterations = 50000
    # initial_temperature = 34
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 42.603
    # max_iterations = 50000
    # initial_temperature = 32
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 39.819
    # max_iterations = 55000
    # initial_temperature = 36
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 42.977 (cool)
    # max_iterations = 50000
    # initial_temperature = 39
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 41.216 (cool)
    # max_iterations = 50000
    # initial_temperature = 40
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 41.493 (cool)
    # max_iterations = 50000
    # initial_temperature = 43
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 41.436 (cool)
    # max_iterations = 50000
    # initial_temperature = 45
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 42.056 (cool)
    # max_iterations = 50000
    # initial_temperature = 49
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 40.882 (cool)
    # max_iterations = 50000
    # initial_temperature = 75
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance 46.573
    # max_iterations = 50000
    # initial_temperature = 1.8
    # cooling_rate = 0.9998
    # shift_max = 32

    # random 64 cities, distance
    # max_iterations = 50000
    # initial_temperature = 2.05
    # cooling_rate = 0.9998
    # shift_max = 32

    # ------------------------------------------------------------

    # 64 city grid, distance 69.136
    # max_iterations = 50000
    # initial_temperature = 29
    # cooling_rate = 0.9998
    # shift_max = 32
    # grid_side_length = 8

    # random 64 cities, distance 38.171 (good)
    max_iterations = 50000
    initial_temperature = 2
    cooling_rate = 0.9998
    shift_max = 32
    grid_side_length = 8

    # max_iterations = 50000
    # initial_temperature = 29
    # cooling_rate = 0.9998
    # shift_max = 32
    # grid_side_length = 8


    # ==========================================================================
    # |  The actual code to run the algorithm:
    # ==========================================================================

    # grid_side_length = 8
    # initial_guess = generate_square_grid(grid_side_length)  # grid
    num_cities = grid_side_length**2
    initial_guess = generate_random_cities(num_cities, X_MIN_WA, X_MAX_WA,
                                           Y_MIN_WA, Y_MAX_WA)

    # define problem here
    problem = TravelingSalesmanProblem(
        # initial_guess=initial_guess,
        num_cities=num_cities,
        shift_max=shift_max
    )

    sa_solver = SimulatedAnnealing(
        problem=problem,
        initial_guess=initial_guess,
        max_iterations=max_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate
    )
    sa_solver.anneal()
    problem.display_solution(sa_solver.solution)  # 74.601?

    print(f"Generating plot of fitness values...")
    visualize_solution_fitness(sa_solver.get_solution_values(),
                               # downsample_factor=50
                               )



if __name__ == '__main__':
    main()

