# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================
import math

import numpy as np
from config import seed
from problems import Problem, Solution
import time
from helper_functions import (generate_random_population,
                              generate_grid_population,
                              visualize_solution_fitness)
from problems import TravelingSalesmanProblem


np.random.seed(seed)


class GeneticAlgorithm():
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
        """"""

        # what to pass in here besides a problem?
        # population size
        # crossover rate (percentage?)
        # mutation rate

        self.problem = problem
        self.solution_values = []  # for plotting

        # main GA parameters
        self.POPULATION_SIZE = population_size
        self.NUM_GENERATIONS = num_generations
        self.ELITISM_PERCENT = elitism_percent
        self.CROSSOVER_PERCENT = crossover_percent
        self.MUTATION_RATE = mutation_rate
        self.TOURNAMENT_SIZE = tournament_size

        self.population = initial_population
        self.generation_number = 0
        self.current_best_fitness = problem.evaluate_solution(
            self.population[0])

        self.best_solution = None
        self.best_solution_fitness = None





    # methods...

    def ga_generation(self):
        """"""

        new_population = []

        # elitism
        sorted_population = self.problem.sort_by_fitness(self.population)
        elite = self.problem.get_elite(sorted_population, self.ELITISM_PERCENT)
        # new_population.append(elite)
        new_population = elite

        # todo: add best solution to list here
        # sorted_population = self.problem.sort_by_fitness(self.population)
        current_best_fitness = self.problem.evaluate_solution(sorted_population[0])
        self.solution_values.append(current_best_fitness)


        # (tournament) selection and crossover
        num_children = round(self.CROSSOVER_PERCENT * self.POPULATION_SIZE)
        for i in range(num_children):
            parent1 = self.problem.tournament_selection(self.population,
                                                        self.TOURNAMENT_SIZE)
            parent2 = self.problem.tournament_selection(self.population,
                                                        self.TOURNAMENT_SIZE)
            child = self.problem.crossover(parent1, parent2)
            new_population.append(child)

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


    def evolve(self):
        """"""

        start_time = time.time()

        # loop, call ga_generation
        # at the end, return the most fit individual in the solution

        # save the best in this generation for plotting later
        # TODO - comment after timing
        # sorted_population = self.problem.sort_by_fitness(self.population)
        # self.solution_values.append(self.problem.evaluate_solution(
        #     sorted_population[0]))



        while self.generation_number < self.NUM_GENERATIONS:
            self.ga_generation()

            # save the best in this generation for plotting later
            # TODO - comment after timing
            # sorted_population = self.problem.sort_by_fitness(self.population)
            # self.solution_values.append(self.problem.evaluate_solution(
            #     sorted_population[0]))

            if self.generation_number % 10 == 0:
                self.print_current_generation_information()

            self.generation_number += 1

        # get the last one into the list
        # TODO - uncomment after timing
        sorted_population = self.problem.sort_by_fitness(self.population)
        final_best = self.problem.evaluate_solution(sorted_population[0])
        self.solution_values.append(final_best)

        self.print_current_generation_information()
        end_time = time.time()

        # print best solution data
        print(f"Best solution in population at generation "
              f"{self.generation_number}: distance "
              # f"{self.problem.evaluate_solution(self.solution_values[-1])}")
              f"{self.solution_values[-1]:.3f}")
        print(f"Elapsed time: {(end_time - start_time):.1f} seconds.")


    def print_initial_information(self) -> None:
        """"""
        print("Running genetic algorithm...")
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Generations to complete: {self.NUM_GENERATIONS}")
        print(f"Elitism percentage: {self.ELITISM_PERCENT}")
        print(f"Crossover percentage: {self.CROSSOVER_PERCENT}")
        print(f"Mutation rate: {self.MUTATION_RATE}")
        print(f"Tournament size: {self.TOURNAMENT_SIZE}")
        print()


    def print_current_generation_information(self) -> None:
        """"""
        index_0 = self.problem.evaluate_solution(self.population[0])
        print(f"Generation {self.generation_number}/{self.NUM_GENERATIONS}: "
              f"Population index 0 fitness (possibly mutated): {index_0:.3f}")


    def get_solution_values(self):
        """"""
        return self.solution_values


def main():
    """"""
    # =========================================
    # |  Hyperparameter combinations:
    # =========================================

    pop_size = 400
    num_generations = 400
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.10
    # tournament_size = round(math.sqrt(pop_size))
    # tournament_size = round(pop_size**(1/3))
    tournament_size = 3

    # =========================================
    # |  The actual code to run the algorithm:
    # =========================================

    grid_side_length = 7
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
    # problem.display_solution(ga_solver.solution_values[-1])
    # problem.display_solution(ga_solver.)
    best_individual = min(ga_solver.population,
                          key=lambda individual:
                          problem.evaluate_solution(individual))
    problem.display_solution(best_individual)
    visualize_solution_fitness(ga_solver.get_solution_values())



    # initial_guess = generate_grid_population()




if __name__ == '__main__':
    main()
