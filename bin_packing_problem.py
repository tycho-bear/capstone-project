# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from helper_classes import BinConfiguration
import numpy as np
from config import SEED
import copy
from problem import Problem, Solution
from collections import Counter


np.random.seed(SEED)


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
        # block_chars = ["█", "▓"]  # not as good
        # block_chars = ["▓", "▒"]  # better?
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
        Simple helper function that sorts a given population according to
        fitness value. Most fit individuals are at the start of the list. For
        bin packing, this sorts in ascending order according to the number
        of bins used.

        :param population: (list[BinConfiguration]) The population to sort.
        :return: (list[BinConfiguration]) The population, sorted by number of
            bins used in ascending order.
        """

        new_population = sorted(population, key=lambda config: config.fitness)
        return new_population


    def get_elite(self, sorted_population: list[BinConfiguration],
                  elitism_percent: float) -> list[BinConfiguration]:
        """
        Simple function to extract the best few individuals from a population.

        :param sorted_population: (list[BinConfiguration]) A population sorted
            in ascending order by number of bins used.
        :param elitism_percent: (float) The percentage of the population to
            retain for the next generation. Ceiling will be used if this does
            not come out to a whole number.
        :return: (list[BinConfiguration]) The elite subset of the population.
        """
        # can be the same
        population_size = len(sorted_population)
        count_to_retain = round(elitism_percent * population_size)
        elite = sorted_population[:count_to_retain]
        return elite


    def tournament_selection(self, population: list[BinConfiguration],
                             num_samples: int) -> BinConfiguration:
        """
        Chooses `num_samples` BinConfigurations from the population without
        replacement and returns the one with the fewest bins used.

        :param population: (list[BinConfiguration]) The population to choose
            from.
        :param num_samples: (int) The number of samples, without replacement.
        :return: (BinConfiguration) The BinConfiguration with the fewest bins.
        """

        # choose a few unique participants from the population
        pop_size = len(population)
        indices = np.random.choice(pop_size, size=num_samples, replace=False)
        participants = [population[i] for i in indices]

        # get the best participant
        best = participants[0]
        for configuration in participants:
            if configuration.fitness < best.fitness:
                best = configuration

        return best


    def crossover(self, parent1: BinConfiguration, parent2: BinConfiguration) \
            -> BinConfiguration:
        """
        Performs crossover between two parent BinConfigurations to create a new
        individual. The crossover is done by randomly selecting half of the
        items from parent1 and filling the rest with items from parent2.

        We also make sure that the resulting configuration is valid. This means
        that the weights of the items in the child configuration do not exceed
        the bin capacity.

        :param parent1: (BinConfiguration) The first parent.
        :param parent2: (BinConfiguration) The second parent.
        :return: (BinConfiguration) A new BinConfiguration containing elements
            of parent 1 and parent 2.
        """

        num_items = parent1.num_weights
        child_weights = [None] * parent1.num_weights

        # get crossover positions
        crossover_positions = sorted(np.random.choice(num_items,
                                                      size=num_items // 2,
                                                      replace=False))

        # copy in the weights from parent1 at the crossover positions
        for pos in crossover_positions:
            child_weights[pos] = parent1.ITEM_WEIGHTS[pos]

        # calculate what weights we sitll need from parent2
        remaining_weights = Counter(parent1.ITEM_WEIGHTS)
        for weight in child_weights:
            if weight is not None:
                remaining_weights[weight] -= 1

        # fill the rest with stuff from parent2
        parent2_weights = iter(parent2.ITEM_WEIGHTS)
        for i in range(num_items):
            if child_weights[i] is None:
                # find next available weight
                while True:
                    candidate_weight = next(parent2_weights)
                    if remaining_weights[candidate_weight] > 0:
                        child_weights[i] = candidate_weight
                        remaining_weights[candidate_weight] -= 1
                        break

        child = BinConfiguration(child_weights, parent1.BIN_CAPACITY)
        return child


    def mutate_individual(self, individual: BinConfiguration) \
            -> BinConfiguration:
        """
        Applies a random mutation to the given BinConfiguration. The mutated
        configuration is returned as a new object.

        Mutations include swapping the position of two items in the internal
        list, reversing a random segment, or scrambling a random segment.

        :param individual: (BinConfiguration) The bin configuration to mutate.
        :return: (BinConfiguration) The mutated BinConfiguration.
        """

        operation = np.random.randint(low=0, high=3)

        # 0 = swap two items, already have a method for this
        if operation == 0:
            return individual.swap_two_bins()

        # pick segment (not wrapping around, but easier this way)
        num_weights = individual.num_weights
        start, end = sorted(np.random.choice(num_weights, size=2,
                                             replace=False))
        new_weights = copy.deepcopy(individual.ITEM_WEIGHTS)
        segment = new_weights[start:end]

        # 1 = reverse a segment
        if operation == 1:
            segment = reversed(segment)

        # 2 = scramble a segment
        elif operation == 2:
            np.random.shuffle(segment)

        # put it back together
        new_weights[start:end] = segment
        mutated_configuration = BinConfiguration(new_weights,
                                                 individual.BIN_CAPACITY)
        return mutated_configuration


    def apply_mutation(self, population: list[BinConfiguration],
                       mutation_prob: float) -> list[BinConfiguration]:
        """
        Probabilistically applies mutation to each individual in the population.
        Sometimes nothing happens, and sometimes several BinConfigurations are
        changed.

        :param population: (list[BinConfiguration]) The population to mutate.
        :param mutation_prob: (float) The probability of mutating a given
            individual in the population.
        :return: (list[BinConfiguration]) The mutated population.
        """

        for i in range(len(population)):
            rand = np.random.random()
            if rand < mutation_prob:
                population[i] = self.mutate_individual(population[i])

        return population


    def generate_new_individual(self, reference_individual: BinConfiguration) \
            -> BinConfiguration:
        """
        Generates a new individual with the same weights as the reference
        configuration. This is useful when generating a population for a genetic
        algorithm.

        :param reference_individual: (BinConfiguration) The individual whose
            weights will be used to generate a new individual.
        :return: (BinConfiguration) A new, shuffled BinConfiguration.
        """

        # can be the same

        new_config = reference_individual.shuffle_bins()
        return new_config
