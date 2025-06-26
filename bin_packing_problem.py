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
    """
    Implementation of the Problem class for the bin packing problem. The
    methods are designed to work with BinConfiguration objects, since those
    represent solutions to this problem.
    """

    def __init__(self):
        """
        Creates a new problem instance. This object will be passed into one of
        the metaheuristic algorithms, where the overridden abstract methods will
        be called.
        """


    def evaluate_solution(self, solution: BinConfiguration) -> int:
        """
        Helper method, mainly used to avoid looking inside the solution class,
        since those will differ based on solution implementation. Returns a
        numerical fitness value for the given solution.

        In this case, the fitness is simply the number of bins used in the
        configuration.

        :param solution: (BinConfiguration) The bin configuration whose fitness
            (number of bins used) will be returned.
        :return: (int) The number of bins used in the given configuration.
        """

        return solution.fitness


    def generate_neighbor(self, current_solution: BinConfiguration) \
            -> BinConfiguration:
        """
        Given a current BinConfiguration, generates a neighboring solution by
        swapping two items in the internal list representation. Since first-fit
        packing is used, this may not change the fitness of the solution.

        :param current_solution: (BinConfiguration) The current solution to
            generate a neighbor for.
        :return: (BinConfiguration) The neighboring BinConfiguration object.
        """

        # can do other stuff here too, like reversing/scrambling segments
        # Unfortunately, bin packing isn't a very interesting problem, so I may
            # just move on and look at something else

        swapped_config = current_solution.swap_two_bins()
        return swapped_config


    def display_solution(self, solution: BinConfiguration) -> None:
        """
        Displays the given BinConfiguration in an understandable format. This is
        done by printing a visual representation of the bins and their contents.

        :param solution: (BinConfiguration) The bin configuration to display.
        :return: None
        """

        capacity = solution.BIN_CAPACITY
        bar_length = solution.BIN_CAPACITY
        bin_num = 1
        block_chars = ["█", "▒"]
        space = " "

        for filled_bin in solution.bins:
            # Bin __: xx/yy (zz.z%) -- [a, b, c...]
            # or print some kind of progress bar with ascii block characters
            # like a progress bar in a terminal
            current_weight = sum(filled_bin)
            percent_filled = (current_weight / capacity) * 100

            # progress bar
            filled_length = round(bar_length * current_weight / capacity)
            remainder = bar_length - filled_length

            bar = ""
            block_char_index = 0

            # fill the bar with alternating blocks
            for item_weight in filled_bin:
                bar += (item_weight * block_chars[block_char_index])
                block_char_index = (block_char_index + 1) % 2
            bar += (space * remainder)

            # print stuff
            print(f"Bin {bin_num}:\t{current_weight}/{capacity}\t"
                  f"({percent_filled:.1f}%) --\t[{bar}]")

            bin_num += 1


    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    def sort_by_fitness(self, population: list[BinConfiguration]) \
            -> list[BinConfiguration]:
        """
        Simple

        :param population:
        :return:
        """

        # can be the same

        pass


    def get_elite(self, sorted_population: list[BinConfiguration],
                  elitism_percent: float) -> list[BinConfiguration]:
        """"""
        # can be the same


        pass


    def tournament_selection(self, population: list[BinConfiguration],
                             num_samples: int) -> BinConfiguration:
        """"""

        # can be the same


        pass



    def crossover(self, parent1: BinConfiguration, parent2: BinConfiguration) \
            -> BinConfiguration:
        """"""

        # need to think more about this, it will be a little different
        # like TSP, must use the same weights
        # unlike TSP, weights are not unique



        pass


    def mutate_individual(self, individual: BinConfiguration) \
            -> BinConfiguration:
        """"""

        # can be the same

        pass


    def apply_mutation(self, population: list[BinConfiguration],
                       mutation_prob: float) -> list[BinConfiguration]:
        """"""
        # can be the same

        pass

    def generate_new_individual(self, reference_individual: BinConfiguration) \
            -> BinConfiguration:
        """"""

        # can be the same

        pass









