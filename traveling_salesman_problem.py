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


class TravelingSalesmanProblem(Problem):
    """
    Implementation of the Problem class for the traveling salesman problem.
    The methods are designed to work with Tour objects, since those represent
    solutions to this problem.
    """

    def __init__(self):
        """
        Creates a new problem instance. This object will be passed into one of
        the metaheuristic algorithms, where the overridden abstract methods will
        be called.
        """


    def swap_two_cities(self, current_solution: Tour) -> Tour:
        """
        Helper method that swaps two cities at random in the given Tour.

        :param current_solution: (Tour) The current solution, where two cities
            will be swapped.
        :return: (Tour) A new Tour object where the two cities are swapped.
        """

        shift_max = round(current_solution.num_cities / 2)

        position = np.random.randint(low=0, high=current_solution.num_cities)
        shift = np.random.randint(low=1, high=shift_max + 1)
        new_solution = current_solution.swap_cities(position, shift)
        return new_solution


    def generate_neighbor(self, current_solution: Tour) -> Tour:
        """
        Given a current Tour, generates a neighboring solution by swapping two
        cities.

        :param current_solution: (Tour) The current solution to generate a
            neighbor for.
        :return: (Tour) The neighboring Tour object.
        """

        return self.swap_two_cities(current_solution)


    def evaluate_solution(self, solution: Tour) -> float:
        """
        Helper method, mainly used to avoid looking inside the solution class,
        since those will differ based on solution implementation.

        :param solution: (Tour) The Tour whose fitness will be returned.
        :return: (float) The total distance of the given Tour.
        """

        return solution.tour_distance


    def display_solution(self, solution: Tour) -> None:
        """
        Displays the given Tour in a plot. Cities are shown as points, and lines
        connect them to show the Tour's path.

        :param solution: (Tour) The solution to display.
        :return: None
        """

        solution.draw_tour(include_start_end=False, show_segments=True,
                           plot_title=f"{solution.num_cities} cities, distance "
                                      f"{solution.tour_distance:.3f}")


    # ==========================================================================
    # |  Genetic algorithm methods
    # ==========================================================================

    def sort_by_fitness(self, population: list[Tour]) -> list[Tour]:
        """
        Simple helper function that sorts a given population according to
        fitness value. Most fit individuals are at the start of the list. For
        the TSP, this sorts in ascending order according to the tour distance.

        :param population: (list[Tour]) The population to sort.
        :return: (list[Tour]) The population, sorted by tour distance in
            ascending order.
        """

        new_population = sorted(population, key=lambda tour: tour.tour_distance)
        return new_population


    def get_elite(self, sorted_population: list[Tour], elitism_percent: float) \
            -> list[Tour]:
        """
        Simple function to extract the best few individuals from a population.

        :param sorted_population: (list[Tour]) A population sorted in ascending
            order by tour distance.
        :param elitism_percent: (float) The percentage of the population to
            retain for the next generation. Ceiling will be used if this does
            not come out to a whole number.
        :return: (list[Tour]) The elite subset of the population.
        """

        population_size = len(sorted_population)
        count_to_retain = round(elitism_percent * population_size)
        elite = sorted_population[:count_to_retain]
        return elite


    def tournament_selection(self, population: list[Tour], num_samples: int) \
            -> Tour:
        """
        Chooses `num_samples` Tours from the population without replacement and
        returns the one with the shortest tour distance.

        :param population: (list[Tour]) The population to choose from.
        :param num_samples: (int) The number of samples, without replacement.
        :return: (Tour) The shortest Tour from the samples.
        """

        # choose a few unique participants from the population
        pop_size = len(population)
        indices = np.random.choice(pop_size, size=num_samples, replace=False)
        participants = [population[i] for i in indices]

        # get the best participant
        best = participants[0]
        for tour in participants:
            if tour.tour_distance < best.tour_distance:
                best = tour

        return best


    def crossover(self, parent1: Tour, parent2: Tour) -> Tour:
        """
        Performs ordered crossover with the two parents. Randomly chooses a
        slice from parent 1, wrapping around as needed, then fills in the
        remaining cities from parent 2.

        :param parent1: (Tour) The first parent.
        :param parent2: (Tour) The second parent.
        :return: (Tour) A child containing elements of parent 1 and parent 2.
        """

        num_cities = parent1.num_cities

        # pick random point
        start = np.random.randint(low=0, high=num_cities)

        # slice will be 1/3 to 2/3 the tour size    TODO: use random slices?
        lower_bound = round((1/3) * num_cities)
        upper_bound = round((2/3) * num_cities)
        slice_size = np.random.randint(low=lower_bound, high=upper_bound)

        end = (start + slice_size) % num_cities

        # copy the slice from the first parent to the same position in the child
        child_cities = [None] * num_cities
        if start <= end:  # normal slice here
            child_cities[start:end] = copy.deepcopy(parent1.cities[start:end])
        else:
            # wrap around slice here
            child_cities[start:] = copy.deepcopy(parent1.cities[start:])
            child_cities[:end] = copy.deepcopy(parent1.cities[:end])

        # get the cities from parent2 that aren't in the child
        parent2_safe_cities = []
        for city in parent2.cities:
            if city not in child_cities:
                parent2_safe_cities.append(copy.deepcopy(city))

        # fill the rest of the child in order from the second parent
        parent_index = 0
        for i in range(num_cities):
            if child_cities[i] is None:
                # grab one from the second parent
                new_city = copy.deepcopy(parent2_safe_cities[parent_index])
                child_cities[i] = new_city
                parent_index += 1

        child = Tour(child_cities)
        return child


    def mutate_individual(self, individual: Tour) -> Tour:
        """
        Applies a random mutation to the specified Tour. The mutated Tour is
        returned as a new object.

        Mutations include swapping two cities, reversing a random segment, or
        scrambling a random segment.

        :param individual: (Tour) The Tour to mutate.
        :return: (Tour) The mutated Tour.
        """

        operation = np.random.randint(low=0, high=3)

        # 0 = swap two cities, already have a method for this
        if operation == 0:
            return self.swap_two_cities(individual)

        # pick segment (not wrapping around, but easier this way)
        num_cities = individual.num_cities
        start, end = sorted(np.random.choice(num_cities, size=2, replace=False))
        new_cities = copy.deepcopy(individual.cities)
        segment = new_cities[start:end]

        # 1 = reverse a segment
        if operation == 1:
            segment = reversed(segment)

        # 2 = scramble a segment
        elif operation == 2:
            np.random.shuffle(segment)  # this works, ignore pycharm highlight

        new_cities[start:end] = segment
        mutated_tour = Tour(new_cities)
        return mutated_tour


    def apply_mutation(self, population: list[Tour], mutation_prob: float) ->\
            list[Tour]:
        """
        Probabilistically applies mutation to the entire population. Sometimes
        nothing happens, and sometimes several Tours are changed.

        :param population: (list[Tour]) The population to mutate.
        :param mutation_prob: (float) The probability of mutating a given
            individual in the population.
        :return: (list[Tour]) The mutated population.
        """

        for i in range(len(population)):
            rand = np.random.random()
            if rand < mutation_prob:
                population[i] = self.mutate_individual(population[i])

        return population


    def generate_new_individual(self, reference_individual: Tour) -> Tour:
        """
        Generates a new individual on the same cities as the reference tour.
        This is useful when generating a population for a genetic algorithm.

        :param reference_individual: (Tour) The tour whose cities will be
            referenced.
        :return: (Tour) A new, shuffled tour.
        """

        new_tour = reference_individual.shuffle_tour()
        return new_tour
