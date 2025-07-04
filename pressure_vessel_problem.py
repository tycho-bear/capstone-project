# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================
from typing import Any

import numpy as np
from config import SEED
import copy
from problem import Problem, Solution
from helper_classes import Design
from collections import Counter


np.random.seed(SEED)
# PVSolution = np.ndarray


class PressureVesselProblem(Problem):
    """

    """

    def __init__(self):
        """

        """

    # ==========================================================================
    # |  General methods
    # ==========================================================================

    def evaluate_solution(self, solution: Design) -> float:
        """"""

        # remember
        # print(f"Evaluating solution with thicknesses: "
        #       f"{solution.head_thickness}, {solution.body_thickness}")
        return solution.cost


    def display_solution(self, solution: Design) -> None:
        """"""

        # just print stuff
        print(f"Pressure vessel design variables:\n"
              f"\tHead thickness:\t{solution.head_thickness}\n"
              f"\tBody thickness:\t{solution.body_thickness}\n"
              f"\tInner radius:\t{solution.inner_radius:.4f}\n"
              f"\tCylindrical length:\t{solution.cylindrical_length:.4f}\n"
              f"Total cost: {solution.cost:.3f}")
        print(f"Valid solution? {solution.is_valid_design()}")


    # ==========================================================================
    # |  Simulated annealing methods
    # ==========================================================================

    def generate_neighbor(self, current_solution: Design) -> Design:
        """"""

        # sorta normal SA, need to also remember
        new_solution = current_solution.generate_neighbor()
        return new_solution


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

