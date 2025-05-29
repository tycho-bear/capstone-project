# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
from helper_functions import generate_random_cities, generate_square_grid
from config import seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA
from helper_classes import Tour
import copy
import numpy as np
from typing import Any
import time


np.random.seed(seed)


class SimulatedAnnealing:
    """
    This class implements the simulated annealing algorithm.
    """

    def __init__(self, initial_guess: Tour, max_iterations, initial_temperature,
                 cooling_rate, shift_max):
        """
        Creates a new instance of a simulated annealing algorithm with the given
        hyperparameters. These influence the behavior of the algorithm and the
        quality of the solutions it finds.

        Class structure subject to revisions in the near future.

        :param initial_guess:
        :param max_iterations: (int) The maximum number of iterations the
            algorithm should run for.
        :param initial_temperature: (float) The temperature to start at. This
            influences the acceptance probability.
        :param cooling_rate: (float) For a geometric cooling rate, the
            temperature will be multiplied by this number after every iteration.
        :param num_cities: ABC
        :param shift_max:
        """
        # TODO: Update class to be more modular. Pass in these things:
        # TODO:     (1) an initial solution guess, (2) a fitness metric
        # TODO:     function, and (3) a function that generates a new solution.

        # main SA parameters
        self.MAX_ITERATIONS = max_iterations
        self.current_iteration = 0
        self.temperature = initial_temperature
        self.COOLING_RATE = cooling_rate
        self.SHIFT_MAX = shift_max

        # set up initial stuff
        # initial_cities = generate_random_cities(num_cities, x_min_WA, x_max_WA,
        #                                         y_min_WA, y_max_WA)
        # self.solution = Tour(initial_cities)

        self.solution = initial_guess
        self.solution_distance = self.solution.tour_distance
        self.NUM_CITIES = self.solution.num_cities

        # keep track of the best here, this won't be used in the algorithm
        self.best_solution = copy.deepcopy(self.solution)
        self.best_solution_distance = copy.deepcopy(self.solution_distance)
        self.best_solution_iteration_num = copy.deepcopy(self.current_iteration)


    def accept_solution(self, new_solution: Any) -> None:
        """
        Helper function that accepts the given solution. It sets the current
        solution to ``new_solution``, and saves its fitness value.

        :param new_solution: (Any) The new solution to accept. For the TSP, this
            would be a Tour object.
        :return: None
        """

        self.solution = new_solution
        self.solution_distance = new_solution.tour_distance


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
        position = np.random.randint(low=0, high=self.NUM_CITIES)
        shift = np.random.randint(low=1, high=self.SHIFT_MAX+1)
        new_solution = self.solution.swap_cities(position, shift)

        difference = new_solution.tour_distance - self.solution_distance

        # this solution is worse, we will accept it with a probability
        if difference >= 0:
            r = np.random.random()
            if r < math.exp((-1 * difference) / self.temperature):
                self.accept_solution(new_solution)
            return

        # this solution is better, so accept it
        self.accept_solution(new_solution)

        # keep track of the best solution, not super important, but it's cool
        if self.solution_distance < self.best_solution_distance:
            self.best_solution = copy.deepcopy(self.solution)
            self.best_solution_distance = self.solution_distance
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

            # print every 100 iterations
            if self.current_iteration % 500 == 0:
                self.print_current_iteration_information()

            self.current_iteration += 1
        self.print_current_iteration_information()

        end_time = time.time()

        print(f"Found best solution at iteration "
              f"{self.best_solution_iteration_num}:  "
              f"distance {self.best_solution_distance:.3f}")
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
              f"distance {self.solution_distance:.3f}")


    def display_solution(self) -> None:
        """
        Helper function that calls draw_tour() on the current solution. Shows
        a visual representation of the current tour, and also displays its
        length.

        :return: None
        """
        self.solution.draw_tour(include_start_end=False, show_segments=True,
                                plot_title=f"{self.NUM_CITIES} cities, distance"
                                           f" {self.solution_distance:.3f}")


def main() -> None:
    """
    Driver, runs the algorithms.

    :return: None
    """
    # temp 1000
    # cooling rate 0.99
    # max iter 1000

    # start with temp=100, rate=0.9, 20 iterations, just to see if it works


    # see final tour

    # ==========================================================================
    # |  Different combinations of hyperparameters below:
    # ==========================================================================

    # Distance 20.839
    # max_iterations = 8000
    # initial_temperature = 40
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 20.548
    # max_iterations = 8000
    # initial_temperature = 33
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 20.839
    # max_iterations = 10000
    # initial_temperature = 40
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 21.315
    # max_iterations = 10000
    # initial_temperature = 46
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 19.752
    # max_iterations = 10000
    # initial_temperature = 17
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 19.780
    # max_iterations = 10000
    # initial_temperature = 5
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 2

    # Distance 19.782
    max_iterations = 10000
    initial_temperature = 2.5
    cooling_rate = 0.999
    num_cities = 20
    shift_max = 2


    # ==========================================================================
    # |  The actual code to run the algorithm:
    # ==========================================================================

    # 25 cities, distance 26.243
    # max_iterations = 10000
    # initial_temperature = 9
    # cooling_rate = 0.999
    # num_cities = 20
    # shift_max = 10

    # 49 cities, distance 56.055
    # max_iterations = 20000
    # initial_temperature = 12
    # cooling_rate = 0.9995
    # num_cities = 20
    # shift_max = 20

    # 64 cities, distance 73.265
    # max_iterations = 30000
    # initial_temperature = 20
    # cooling_rate = 0.9997
    # num_cities = 20
    # shift_max = 30

    # 64 cities, distance 72.028
    max_iterations = 40000
    initial_temperature = 19
    cooling_rate = 0.9997
    num_cities = 20
    shift_max = 32

    # using the commented out parameter settings
    # initial_guess = generate_random_cities(num_cities, x_min_WA, x_max_WA,
    #                                        y_min_WA, y_max_WA)

    grid_side_length = 8
    initial_guess = generate_square_grid(grid_side_length)

    annealer = SimulatedAnnealing(initial_guess, max_iterations,
                                  initial_temperature, cooling_rate,
                                  shift_max)
    annealer.anneal()
    annealer.display_solution()



if __name__ == '__main__':
    main()

