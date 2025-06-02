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


np.random.seed(seed)
Solution = Any  # abstract methods return this instead of a Tour or something

class Problem(ABC):
    """"""

    # generate random initial solution
    @abstractmethod
    def generate_initial_guess(self) -> Solution:
        pass

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

    def __init__(self, initial_guess: Tour, shift_max: int):
        """"""

        self.initial_guess = initial_guess  # so we can pass in a square grid
        self.NUM_CITIES = initial_guess.num_cities
        self.SHIFT_MAX = shift_max

    def generate_initial_guess(self) -> Solution:
        """(For simulated annealing)"""

        return self.initial_guess

    def generate_new_solution(self, current_solution: Tour) -> Solution:
        """(For simulated annealing)"""

        position = np.random.randint(low=0, high=self.NUM_CITIES)
        shift = np.random.randint(low=1, high=self.SHIFT_MAX + 1)
        # new_solution = self.solution.swap_cities(position, shift)
        new_solution = current_solution.swap_cities(position, shift)
        return new_solution

    def evaluate_solution(self, solution: Tour) -> float:
        """(For simulated annealing?)"""

        return solution.tour_distance

    def display_solution(self, solution: Tour) -> None:
        """"""
        solution.draw_tour(include_start_end=False, show_segments=True,
                           plot_title=f"{self.NUM_CITIES} cities, distance "
                                      f"{solution.tour_distance:.3f}")


# (SA) generate random initial solution

# (SA) generate new solution, just swap 2 cities

# (SA) evaluate solution, use tour distance

