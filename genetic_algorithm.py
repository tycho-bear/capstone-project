# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
import numpy as np
from config import seed
from problem import Problem, Solution
import time
from helper_functions import (generate_random_city_population,
                              generate_grid_population,
                              visualize_solution_fitness)
from traveling_salesman_problem import TravelingSalesmanProblem


np.random.seed(seed)


class GeneticAlgorithm:
    """
    This class implements a genetic algorithm.
    """

    def __init__(self,
                 problem: Problem,
                 initial_population: list,
                 population_size: int,
                 num_generations: int,
                 elitism_percent: float,
                 crossover_percent: float,
                 mutation_rate: float,
                 tournament_size: int
                 ):
        """
        Creates a new instance of a genetic algorithm solver with the given
        hyperparameters. These influence the behavior of the algorithm, its
        overall running time, and the quality of the solutions it finds.

        :param problem: (Problem) The problem to be optimized, such as the
            traveling salesman problem (as a TravelingSalesmanProblem class).
        :param initial_population: (list) The initial population to run the
            algorithm on. This is configured by the user. For the TSP, this is
            a list of Tour objects over the same set of cities.
        :param population_size: (int) The number of individuals in the
            population.
        :param num_generations: (int) The number of generations (or iterations)
            to run the algorithm for.
        :param elitism_percent: (float) The percent of most fit individuals to
            retain for the next generation.
        :param crossover_percent: (float) The percentage of the population that
            should be generated through selection and crossover.
        :param mutation_rate: (float) The probability that any given individual
            in the population is mutated randomly in each generation.
        :param tournament_size: (int) The number of participants in each
            tournament during tournament selection.
        """

        # what problem do we have? also save the solution values for plotting
        self.problem = problem
        self.solution_values = []  # for plotting

        # main GA parameters
        self.POPULATION_SIZE = population_size
        self.NUM_GENERATIONS = num_generations
        self.ELITISM_PERCENT = elitism_percent
        self.CROSSOVER_PERCENT = crossover_percent
        self.MUTATION_RATE = mutation_rate
        self.TOURNAMENT_SIZE = tournament_size

        # setting things up
        self.population = initial_population
        self.generation_number = 0

        self.gen_best_solution = None  # updated every generation
        self.gen_best_fitness = None  # updated every generation


    def ga_generation(self) -> None:
        """
        Simulates one generation in the genetic algorithm.

        First, elitism is carried out. This preserves the most fit solutions
        from the current generation for use in the next generation.

        Next is crossover. First, tournament selection is used to select pairs
        of parents. After that, crossover uses each pair of parents to create a
        new solution.

        Then comes mutation. Each solution in the population is mutated with a
        small probability.

        Finally, the new population is filled with random solutions until the
        population size is restored.

        :return: None
        """

        # elitism
        sorted_population = self.problem.sort_by_fitness(self.population)
        elite = self.problem.get_elite(sorted_population, self.ELITISM_PERCENT)
        new_population = elite

        # keep track of the best solution and print stuff
        self.generation_housekeeping(sorted_population)

        # apply crossover: use tournament selection to choose pairs of parents,
        # then create a child via some kind of crossover.
        crossover_generated = self.crossover()
        new_population += crossover_generated  # [1,2] + [3,4,5] = [1,2,3,4,5]

        # mutation
        new_population = self.problem.apply_mutation(new_population,
                                                     self.MUTATION_RATE)

        # fill with random solutions until we hit the population size
        while len(new_population) < self.POPULATION_SIZE:
            reference = self.population[0]
            new_individual = self.problem.generate_new_individual(reference)
            new_population.append(new_individual)

        # this generation is done, onto the next
        self.population = new_population


    def generation_housekeeping(self, sorted_population: list) -> None:
        """
        Helper method that does a few things:
            - updates the best solution and its fitness inside the class
            - keeps track of this solution so we can plot it later
            - every 10 generations, prints info about the current generation

        :param sorted_population: (list) the population, sorted by fitness
            value.
        :return: None
        """

        # keep track of the best solution for plotting later
        # this is really for the previous generation (or the initial population)
        # we get the best individual in the final population later
        self.gen_best_solution = sorted_population[0]
        self.gen_best_fitness = self.problem.evaluate_solution(
            self.gen_best_solution)
        self.solution_values.append(self.gen_best_fitness)

        # print stuff every 10 generations (called again after the main loop)
        if self.generation_number % 10 == 0:
            self.print_current_generation_information()


    def crossover(self) -> list:
        """
        Uses tournament selection to choose two parents, then creates a child
        solution through crossover. This process is repeated in a loop to create
        the desired number of children.

        :return: (list) The individuals generated by crossover.
        """

        crossover_generated = []

        # (tournament) selection and crossover
        num_children = round(self.CROSSOVER_PERCENT * self.POPULATION_SIZE)
        for i in range(num_children):
            parent1 = self.problem.tournament_selection(self.population,
                                                        self.TOURNAMENT_SIZE)
            parent2 = self.problem.tournament_selection(self.population,
                                                        self.TOURNAMENT_SIZE)
            child = self.problem.crossover(parent1, parent2)
            crossover_generated.append(child)

        return crossover_generated


    def evolve(self) -> None:
        """
        Runs the whole genetic algorithm for the given number of generations.
        Every 10 generations, prints the information about the current
        generation.

        Also keeps track of elapsed time.

        :return: None
        """

        start_time = time.time()

        while self.generation_number < self.NUM_GENERATIONS:
            self.ga_generation()
            self.generation_number += 1

        # take care of the final generation here
        sorted_population = self.problem.sort_by_fitness(self.population)
        self.generation_housekeeping(sorted_population)

        end_time = time.time()

        # print best solution data
        print(f"Best solution in population at generation "
              f"{self.generation_number}: distance "
              f"{self.solution_values[-1]:.3f}")
        print(f"Elapsed time: {(end_time - start_time):.1f} seconds.")


    def print_initial_information(self) -> None:
        """
        Helper function that prints the hyperparameters used in the genetic
        algorithm.

        :return: None
        """

        print("Running genetic algorithm...")
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Generations to complete: {self.NUM_GENERATIONS}")
        print(f"Elitism percentage: {self.ELITISM_PERCENT}")
        print(f"Crossover percentage: {self.CROSSOVER_PERCENT}")
        print(f"Mutation rate: {self.MUTATION_RATE}")
        print(f"Tournament size: {self.TOURNAMENT_SIZE}")
        print()


    def print_current_generation_information(self) -> None:
        """
        Helper function that prints out information about the current
        generation. Something like this:

        `"Generation {current}/{total}: best fitness: {best_fitness}"`

        :return: None
        """

        print(f"Generation {self.generation_number}/{self.NUM_GENERATIONS}: "
              f"Fitness of best individual: {self.gen_best_fitness:.3f}")


    def get_solution_values(self) -> list:
        """
        Getter for the solution fitness values.

        :return: (list) The current list of solution fitness values.
        """

        return self.solution_values


def main():
    """"""
    # =========================================
    # |  Hyperparameter combinations:
    # =========================================

    # # 64 city grid, distance 68.307
    # # 228 --> 175 --> 170 --> 151 (30)
    pop_size = 500
    num_generations = 400  # 400
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.10
    tournament_size = 3

    # # 64 city grid, distance 70.614 (looks cool though, no loops)
    # # converges pretty fast
    # pop_size = 500
    # num_generations = 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.07
    # tournament_size = 3

    # 64 city grid, distance 71.621
    # (too much randomness, jumps around after 175 generations)
    # pop_size = 500
    # num_generations = 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.15
    # tournament_size = 3

    # # 64 city grid, distance ???
    # pop_size = 500
    # num_generations = 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.10
    # tournament_size = 3

    # =========================================
    # |  The actual code to run the algorithm:
    # =========================================

    grid_side_length = 8
    num_cities = grid_side_length ** 2
    problem = TravelingSalesmanProblem(num_cities, shift_max=num_cities)

    print("------------------------------------------")
    print(f"Solving {num_cities}-city TSP with GA.")
    print(f"If grid, optimal solution distance is {num_cities}.")
    print("------------------------------------------")

    initial_population = generate_grid_population(pop_size, grid_side_length)

    ga_solver = GeneticAlgorithm(
        problem=problem,
        initial_population=initial_population,
        population_size=pop_size,
        num_generations=num_generations,
        elitism_percent=elitism_percent,
        crossover_percent=crossover_percent,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size
    )
    ga_solver.print_initial_information()
    ga_solver.evolve()

    best_individual = ga_solver.gen_best_solution
    problem.display_solution(best_individual)
    visualize_solution_fitness(ga_solver.get_solution_values(),
                               xlabel="Generation",
                               ylabel="Current Tour Distance",
                               title="Tour Distance Over Generations")



if __name__ == '__main__':
    main()
