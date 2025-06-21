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

    def __init__(self,
                 # num_bins: int,
                 # bin_capacity: int,
                 ):
        """"""

        # self.NUM_BINS = num_bins
        # self.BIN_CAPACITY = bin_capacity


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


    def display_solution(self, solution: BinConfiguration):
        """"""

        capacity = solution.BIN_CAPACITY
        # bar_length = 30
        bar_length = solution.BIN_CAPACITY
        bin_num = 1
        # block = "█"
        block_chars = ["█", "▒"]
        space = " "
        for bin in solution.bins:
            """"""
            # Bin __: xx/yy (zz.z%) -- [a, b, c...]
            # or print some kind of progress bar with ascii block characters
            # like a progress bar in a terminal
            current_weight = sum(bin)
            percent_filled = (current_weight / capacity) * 100

            # progress bar
            filled_length = round(bar_length * current_weight / capacity)
            remainder = bar_length - filled_length
            # bar = (block * filled_length) + (space * remainder)
            bar = ""
            block_char_index = 0
            for item_weight in bin:
                bar += (item_weight * block_chars[block_char_index])
                # bar += "|"
                block_char_index = (block_char_index + 1) % 2
            bar += (space * remainder)

            # print stuff
            print(f"Bin {bin_num}:\t{current_weight}/{capacity}\t"
                  f"({percent_filled:.1f}%) --\t[{bar}]")
            # print(f"[{bar}]")

            bin_num += 1




    # use BinConfiguration as the solution class

    def apply_mutation(self, population: list[Solution],
                       mutation_prob: float) -> list[Solution]:
        """"""
        pass

    def crossover(self, parent1: Solution, parent2: Solution):
        """"""
        pass

    def generate_new_individual(self, reference_individual: Solution) -> (
            Solution):
        """"""
        pass

    def get_elite(self, sorted_population: list[Solution],
                  elitism_percent) -> list[Solution]:
        """"""
        pass

    def mutate_individual(self, individual: Solution) -> Solution:
        """"""
        pass

    def sort_by_fitness(self, population: list[Solution]) -> list[Solution]:
        """"""
        pass

    def tournament_selection(self, population: list[Solution],
                             num_samples: int) -> Solution:
        """"""
        pass


