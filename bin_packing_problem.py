# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from helper_classes import BinConfiguration
import numpy as np
from config import seed
import copy
from problem import Problem, Solution


np.random.seed(seed)


class BinPackingProblem(Problem):
    """"""

    def __init__(self, num_bins: int, bin_capacity: int):
        """"""

        self.NUM_BINS = num_bins
        self.BIN_CAPACITY = bin_capacity


    def evaluate_solution(self, solution: BinConfiguration) -> int:
        """minimize this"""
        return solution.fitness


    # def swap_two_items(self, current_solution: BinConfiguration) \
    #         -> BinConfiguration:
    #     """"""


    def generate_neighbor(self, current_solution: BinConfiguration) \
            -> BinConfiguration:
        """"""

        # can do other stuff here too, like reversing/scrambling segments
        # Unfortunately, bin packing isn't a very interesting problem, so I may
            # just move on and look at something else

        # return self.swap_two_items(current_solution)
        swapped_config = current_solution.swap_two_bins()
        return swapped_config




    # use BinConfiguration as the solution class


