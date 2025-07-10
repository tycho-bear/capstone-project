# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from abc import ABC, abstractmethod
from typing import Any
from helper_functions import generate_random_cities, generate_square_grid
import numpy as np
from config import SEED
import math
import copy


np.random.seed(SEED)
Solution = Any  # abstract methods return this instead of a Tour or something
Velocity = Any  # can be a list of swaps or something
Particle = Any

# class Particle(ABC):
#     """
#     Class representing a particle for particle swarm optimization. Stores the
#     particle's current solution, plus the best solution it has seen so far.
#     """
#
#     @abstractmethod
#     def __init__(self,
#                  current_solution: Solution,
#                  best_solution: Solution
#                  ):
#         """
#         Initializes this particle with its current solution and the best
#         solution it has seen so far.
#
#         :param current_solution: (Solution) This particle's current solution.
#         :param best_solution: (Solution) The best solution seen by this particle
#             so far.
#         """
#         pass


class Problem(ABC):
    """
    Abstract class representing a problem to be optimized. Metaheuristic
    algorithms will call methods on extensions of this class.
    """

    # ==========================================================================
    # |  General methods
    # ==========================================================================

    @abstractmethod
    def evaluate_solution(self, solution: Solution) -> Any:
        """
        Helper method, mainly used to avoid looking inside the solution class,
        since those will differ based on solution implementation. Returns a
        numerical fitness value for the given solution.

        :param solution: (Solution) The solution whose fitness will be returned.
        :return: (Any) A numerical fitness value for the given solution.
        """
        pass


    @abstractmethod
    def display_solution(self, solution: Solution) -> None:
        """
        Displays the given solution in an understandable format.

        :param solution: (Solution) The solution to display.
        :return: None
        """
        pass


    # ==========================================================================
    # |  Simulated annealing methods
    # ==========================================================================

    @abstractmethod
    def generate_neighbor(self, current_solution: Solution) -> Solution:
        """
        Given the current solution, generates a neighboring solution.

        :param current_solution: (Solution) The current solution.
        :return: (Solution) A neighboring solution to the current solution.
        """
        pass


    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    @abstractmethod
    def sort_by_fitness(self, population: list[Solution]) -> list[Solution]:
        """
        Simple helper function that sorts a given population according to
        fitness value. Most fit individuals are at the start of the list.

        :param population: (list[Solution]) The population to sort.
        :return: (list[Solution]) The population, sorted by fitness in ascending
            order.
        """
        pass


    @abstractmethod
    def get_elite(self, sorted_population: list[Solution],
                  elitism_percent: float) -> list[Solution]:
        """
        Simple function to extract the best few individuals from a population.

        :param sorted_population: (list[Solution]) A population sorted in
            ascending order by fitness value.
        :param elitism_percent: (float) The percentage of the population to keep
            as elite individuals.
        :return: (list[Solution]) A list of elite individuals from the
            sorted population.
        """
        pass


    @abstractmethod
    def tournament_selection(self, population: list[Solution],
                             num_samples: int) -> Solution:
        """
        Selects a single individual from the population using tournament
        selection. This method randomly samples a subset of individuals from the
        population, evaluates their fitness, and returns the individual with the
        highest fitness value.

        :param population: (list[Solution]) The population from which to select
            an individual.
        :param num_samples: (int) The number of individuals to sample from the
            population, without replacement.
        :return: (Solution) The best solution from the samples.
        """
        pass


    @abstractmethod
    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """
        Performs crossover between two parent solutions to produce a new
        individual. The implementation of this function depends on the problem
        being solved.

        :param parent1: (Solution) The first parent.
        :param parent2: (Solution) The second parent.
        :return: (Solution) A child solution containing elements of both
            parents.
        """
        pass


    @abstractmethod
    def mutate_individual(self, individual: Solution) -> Solution:
        """
        Mutates the given individual to produce a new solution. The
        implementation of this function depends on the problem being solved.

        :param individual: (Solution) The individual to mutate.
        :return: (Solution) A new, mutated individual.
        """
        pass


    @abstractmethod
    def apply_mutation(self, population: list[Solution],
                       mutation_prob: float) -> list[Solution]:
        """
        Probabilistically applies mutation to each individual in the
        population. Each individual has a chance of being mutated based on the
        mutation probability.

        :param population: (list[Solution]) The population to mutate.
        :param mutation_prob: (float) The probability of mutating each
            individual in the population.
        :return: (list[Solution]) The mutated population.
        """
        pass


    @abstractmethod
    def generate_new_individual(self, reference_individual: Solution) -> (
            Solution):
        """
        Generates a new individual based on the reference individual. This is
        useful when generating a population for a genetic algorithm.

        :param reference_individual: (Solution) The individual whose properties
            will be used to generate a new individual.
        :return: (Solution) A new, randomly generated individual based on the
            reference individual.
        """
        pass


    # ==========================================================================
    # |  Particle swarm optimization methods
    # ==========================================================================


    # calculate velocity? (current_tour, other_tour)
    # apply velocity?

    # calculate swap sequence (current_tour, other_tour)  (not abstract method)
    # calculate_velocity (particle, global_best)

    @abstractmethod
    def calculate_velocity(self, particle: Particle, global_best: Solution,
                           alpha: float, beta: float):
        """


        :param particle:
        :param global_best:
        :param alpha:
        :param beta:
        :return:
        """
        pass


    @abstractmethod
    def apply_velocity(self, particle: Particle, velocity: Velocity):
        """


        :param particle:
        :param velocity:
        :return:
        """
        pass
