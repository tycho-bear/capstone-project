# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from abc import ABC, abstractmethod
from typing import Any
from helper_classes import Tour
from helper_functions import generate_random_cities, generate_square_grid
import numpy as np
from config import seed
import math
import copy


np.random.seed(seed)
Solution = Any  # abstract methods return this instead of a Tour or something

class Problem(ABC):
    """"""

    # generate random initial solution
    # @abstractmethod
    # def generate_initial_guess(self) -> Solution:
    #     pass

    # generate new solution
    @abstractmethod
    def generate_new_solution(self, current_solution) -> Solution:
        pass

    # evaluate solution
    @abstractmethod
    def evaluate_solution(self, solution) -> float:
        pass


class TravelingSalesmanProblem(Problem):
    """"""

    def __init__(self, num_cities: int, shift_max: int):
        """"""

        # self.initial_guess = initial_guess  # so we can pass in a square grid
        # self.NUM_CITIES = initial_guess.num_cities
        self.NUM_CITIES = num_cities
        self.SHIFT_MAX = shift_max

    # def generate_initial_guess(self) -> Solution:
    #     """(For simulated annealing)"""
    #
    #     return self.initial_guess

    def swap_two_cities(self, current_solution: Tour) -> Solution:
        """"""
        position = np.random.randint(low=0, high=self.NUM_CITIES)
        shift = np.random.randint(low=1, high=self.SHIFT_MAX + 1)
        # new_solution = self.solution.swap_cities(position, shift)
        new_solution = current_solution.swap_cities(position, shift)
        return new_solution


    def generate_new_solution(self, current_solution: Tour) -> Solution:
        """(For simulated annealing)"""

        return self.swap_two_cities(current_solution)


    def evaluate_solution(self, solution: Tour) -> float:
        """(For simulated annealing?)"""

        return solution.tour_distance

    def display_solution(self, solution: Tour) -> None:
        """"""
        solution.draw_tour(include_start_end=False, show_segments=True,
                           plot_title=f"{self.NUM_CITIES} cities, distance "
                                      f"{solution.tour_distance:.3f}")


    # TODO - add new methods to support the genetic algorithm.

    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    def sort_by_fitness(self, population: list[Tour]) -> list[Tour]:
        """
        Simple helper function that sorts a given population according to
        fitness value. Most fit individuals are at the start of the list. For
        the TSP, this sorts in ascending order according to the tour distance.

        :param population: (list[Tour]) The population to sort.
        :return: (list[Tour]) The population, sorted by tour distance in
            ascending order.
        """

        new_population = sorted(population, key=lambda tour: tour.tour_distance)
        return new_population


    def get_elite(self, sorted_population: list[Tour],
                  elitism_percent) -> list[Tour]:
        """
        Simple function to extract

        :param sorted_population: (list[Tour]) A population sorted in ascending
            order by tour distance.
        :param elitism_percent: (float) The percentage of the population to
            retain for the next generation. Ceiling will be used if this does
            not come out to a whole number.
        :return: (list[Tour]) The elite subset of the population.
        """

        population_size = len(sorted_population)
        # count_to_retain = math.ceil(elitism_percent * population_size)
        count_to_retain = round(elitism_percent * population_size)
        elite = sorted_population[:count_to_retain]
        return elite


    def tournament_selection(self, population: list[Tour], num_samples: int) -> Tour:
        """
        Chooses `num_samples` Tours from the population without replacement and
        returns the one with the shortest tour distance.

        :param population: (list[Tour]) The population to choose from.
        :param num_samples: (int) The number of samples, without replacement.
        :return: (Tour) The shortest Tour from the samples.
        """

        # participants = np.random.choice(population, size=num_samples,
        #                                 replace=False)  # works fine?

        pop_size = len(population)
        # indices = np.random.randint(low=0, high=pop_size, size=num_samples)
        indices = np.random.choice(pop_size, size=num_samples, replace=False)
        participants = [population[i] for i in indices]

        ###
        # print("participants:")
        # for tour in participants:
        #     print(tour, tour.tour_distance)
        ###

        best = participants[0]
        for tour in participants:
            if tour.tour_distance < best.tour_distance:
                best = tour

        return best


    def crossover(self, parent1: Tour, parent2: Tour):
        """
        Performs ordered crossover with the two parents. Randomly chooses a
        slice from parent 1, wrapping around as needed, then fills in the
        remaining cities from parent 2.

        :param parent1: (Tour) The first parent.
        :param parent2: (Tour) The second parent.
        :return: A child containing elements of parent 1 and parent 2.
        """

        # just have this one return a single tour?

        num_cities = parent1.num_cities

        # pick random point
        # start = np.random.uniform(low=0, high=num_cities)
        start = np.random.randint(low=0, high=num_cities)

        # slice will be 1/3 to 2/3 the tour size
        # TODO: use random size slices?
        lower_bound = round((1/3) * num_cities)
        upper_bound = round((2/3) * num_cities)
        # slice_size = np.random.uniform(low=lower_bound, high=upper_bound)
        slice_size = np.random.randint(low=lower_bound, high=upper_bound)

        end = (start + slice_size) % num_cities

        # copy that slice from the first parent to the same position in the child
        child_cities = [None] * num_cities
        if start <= end:  # normal slice here
            child_cities[start:end] = copy.deepcopy(parent1.cities[start:end])
        else:
            # wrap around slice here
            child_cities[start:] = copy.deepcopy(parent1.cities[start:])
            child_cities[:end] = copy.deepcopy(parent1.cities[:end])

        # get the cities from parent2 that aren't in the child
        parent2_safe_cities = []
        for city in parent2.cities:
            if city not in child_cities:
                parent2_safe_cities.append(copy.deepcopy(city))

        # fill the rest of the child in order from the second parent
        parent_index = 0
        for i in range(num_cities):
            if child_cities[i] is None:
                # grab one from the second parent
                new_city = copy.deepcopy(parent2_safe_cities[parent_index])
                child_cities[i] = new_city
                parent_index += 1

        child = Tour(child_cities)
        return child


    def mutate_individual(self, individual: Tour) -> Tour:
        """
        Applies a random mutation to the specified Tour. The mutated Tour is
        returned as a new object.

        Mutations can include

        :param individual: (Tour) The Tour to mutate.
        :return: (Tour) The mutated Tour.
        """

        # TODO: add more
        return self.swap_two_cities(individual)



    def apply_mutation(self, population: list[Tour], mutation_prob: float) -> list[Tour]:
        """
        Probabilistically applies mutation to the entire population. Sometimes
        nothing happens, and sometimes several Tours are changed.

        :param population:
        :param mutation_prob:
        :return:
        """

        for i in range(len(population)):
            rand = np.random.random()
            if rand < mutation_prob:
                population[i] = self.mutate_individual(population[i])

        return population


