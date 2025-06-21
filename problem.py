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
    """
    Abstract class representing a problem to be optimized. Metaheuristic
    algorithms will call methods on extensions of this class.
    """

    # ==========================================================================
    # |  General methods
    # ==========================================================================

    @abstractmethod
    def evaluate_solution(self, solution) -> float:
        pass

    def display_solution(self, solution):
        pass

    # ==========================================================================
    # |  Simulated annealing methods
    # ==========================================================================

    @abstractmethod
    def generate_neighbor(self, current_solution) -> Solution:
        pass

    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    @abstractmethod
    def sort_by_fitness(self, population: list[Solution]) -> list[Solution]:
        pass

    @abstractmethod
    def get_elite(self, sorted_population: list[Solution],
                  elitism_percent) -> list[Solution]:
        pass

    @abstractmethod
    def tournament_selection(self, population: list[Solution],
                             num_samples: int) -> Solution:
        pass

    @abstractmethod
    def crossover(self, parent1: Solution, parent2: Solution):
        pass

    @abstractmethod
    def mutate_individual(self, individual: Solution) -> Solution:
        pass

    @abstractmethod
    def apply_mutation(self, population: list[Solution],
                       mutation_prob: float) -> list[Solution]:
        pass

    @abstractmethod
    def generate_new_individual(self, reference_individual: Solution) -> (
            Solution):
        pass

    # ==========================================================================
    # |  Future algorithm methods
    # ==========================================================================



