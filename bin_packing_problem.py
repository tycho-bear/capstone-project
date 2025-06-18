# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from helper_classes import Tour
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


    def evaluate_solution(self, solution) -> float:
        """"""

    def generate_new_solution(self, current_solution) -> Solution:
        """"""


