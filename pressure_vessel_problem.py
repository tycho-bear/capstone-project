# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================
from typing import Any

import numpy as np
from config import seed
import copy
from problem import Problem, Solution
from collections import Counter


np.random.seed(seed)
PVSolution = np.ndarray


class PressureVesselProblem(Problem):
    """

    """

    def __init__(self):
        """

        """

    # ==========================================================================
    # |  General methods
    # ==========================================================================

    def evaluate_solution(self, solution: PVSolution) -> float:
        """"""

        # remember


    def display_solution(self, solution: Solution) -> None:
        """"""

        # just print stuff


    # ==========================================================================
    # |  Simulated annealing methods
    # ==========================================================================

    def generate_neighbor(self, current_solution: Solution) -> Solution:
        """"""

        # normal SA


    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    def sort_by_fitness(self, population: list[Solution]) -> list[Solution]:
        pass

    def get_elite(self, sorted_population: list[Solution],
                  elitism_percent: float) -> list[Solution]:
        pass

    def tournament_selection(self, population: list[Solution],
                             num_samples: int) -> Solution:
        pass

    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        pass

    def mutate_individual(self, individual: Solution) -> Solution:
        pass

    def apply_mutation(self, population: list[Solution],
                       mutation_prob: float) -> list[Solution]:
        pass

    def generate_new_individual(self, reference_individual: Solution) -> (
            Solution):
        pass

