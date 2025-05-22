import math
from helper_functions import generate_random_cities
from config import seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA
from helper_classes import Tour
import copy
import numpy as np


np.random.seed(seed)


class SimulatedAnnealing:
    """
    ...
    """

    def __init__(self, max_iterations, initial_temperature, cooling_rate,
                 num_cities, shift_max):
        """
        ...

        :param max_iterations:
        :param initial_temperature:
        :param cooling_rate:
        :param num_cities:
        :param shift_max:
        """

        # main SA parameters
        self.MAX_ITERATIONS = max_iterations
        self.current_iteration = 0
        self.temperature = initial_temperature
        self.COOLING_RATE = cooling_rate
        self.NUM_CITIES = num_cities
        self.SHIFT_MAX = shift_max

        # set up initial stuff
        initial_cities = generate_random_cities(num_cities, x_min_WA, x_max_WA,
                                                y_min_WA, y_max_WA)
        self.solution = Tour(initial_cities)
        self.solution_distance = self.solution.tour_distance

        # keep track of the best here, this won't be used in the algorithm
        self.best_solution = copy.deepcopy(self.solution)
        self.best_solution_distance = copy.deepcopy(self.solution_distance)
        self.best_solution_iteration_num = copy.deepcopy(self.current_iteration)

        # self.solution_fitness =

        # set current solution
        # set current solution fitness

        # set best solution  # careful of copying...


        # iteration number (set to 0)   (while this < max {..., this += 1})
        # max iterations
        # temperature (set with initial)
        # cooling rate
        # current solution (set with initial guess)
        # current solution fitness

        # (just check these each iteration when updating the current solution)
        # best solution
        # best solution fitness
        # best solution iteration number


    def accept_solution(self, new_solution):
        """"""
        self.solution = new_solution
        self.solution_distance = new_solution.tour_distance


    def update_temperature(self):
        """"""
        # can do something more complicated in the future
        self.temperature *= self.COOLING_RATE


    def sa_step(self):
        """
        Performs one step of the simulated annealing algorithm.
        :return:
        """
        # get new solution
        position = np.random.randint(low=0, high=self.NUM_CITIES)
        shift = np.random.randint(low=1, high=self.SHIFT_MAX+1)
        new_solution = self.solution.swap_cities(position, shift)

        difference = new_solution.tour_distance - self.solution_distance

        # if new_solution.tour_distance < self.solution_distance:
        # this solution is worse, we will accept it with a probability
        if difference >= 0:
            r = np.random.random()
            if r < math.exp((-1 * difference) / self.temperature):
                self.accept_solution(new_solution)
            return

        # this solution is better, so accept it
        # self.solution = new_solution
        # self.solution_distance = new_solution.tour_distance
        self.accept_solution(new_solution)

        # keep track of the best solution, not super important, but it's cool
        if self.solution_distance < self.best_solution_distance:
            self.best_solution = copy.deepcopy(self.solution)
            self.best_solution_distance = self.solution_distance
            self.best_solution_iteration_num = self.current_iteration


    def anneal(self):
        """run the whole algorithm"""
        self.print_current_iteration_information()
        while self.current_iteration < self.MAX_ITERATIONS:
            self.sa_step()
            self.update_temperature()
            self.current_iteration += 1
            self.print_current_iteration_information()

    def print_current_iteration_information(self):
        """"""
        # another quick helper function
        print(f"Iteration {self.current_iteration}: {self.solution}, distance "
              f"{self.solution_distance}")

    def display_solution(self):
        """"""
        self.solution.draw_tour(include_start_end=True)



def main():
    """"""
    # temp 1000
    # cooling rate 0.99
    # max iter 1000

    # start with temp=100, rate=0.9, 20 iterations, just to see if it works


    # see final tour

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

    annealer = SimulatedAnnealing(max_iterations, initial_temperature,
                                      cooling_rate, num_cities, shift_max)
    annealer.anneal()

    annealer.display_solution()





if __name__ == '__main__':
    main()

