# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
from helper_functions import (generate_random_cities, generate_square_grid,
                              visualize_solution_fitness,
                              generate_grid_population,
                              generate_random_bin_config)
from config import seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA
from helper_classes import Tour
import copy
import numpy as np
from typing import Any
import time
from problem import Problem, Solution
from traveling_salesman_problem import TravelingSalesmanProblem
from bin_packing_problem import BinPackingProblem
from simulated_annealing import SimulatedAnnealing
from genetic_algorithm import GeneticAlgorithm


np.random.seed(seed)


# SA with TSP
def sa_with_tsp():
    """"""

    # 64 city grid, distance 69.136
    max_iterations = 50000
    initial_temperature = 29
    cooling_rate = 0.9998
    shift_max = 32
    grid_side_length = 8

    # random 64 cities, distance 38.171 (good)
    # max_iterations = 50000
    # initial_temperature = 2
    # cooling_rate = 0.9998
    # shift_max = 32
    # grid_side_length = 8

    # ==========================================================================
    # |  The actual code to run the algorithm:
    # ==========================================================================

    # grid_side_length = 8
    initial_guess = generate_square_grid(grid_side_length)  # grid
    num_cities = grid_side_length ** 2
    # initial_guess = generate_random_cities(num_cities, x_min_WA, x_max_WA,
    #                                        y_min_WA, y_max_WA)

    # define problem here
    problem = TravelingSalesmanProblem(
        # initial_guess=initial_guess,
        num_cities=num_cities,
        shift_max=shift_max
    )

    sa_solver = SimulatedAnnealing(
        problem=problem,
        initial_guess=initial_guess,
        max_iterations=max_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate
    )
    sa_solver.anneal()
    problem.display_solution(sa_solver.solution)  # 74.601?

    print(f"Generating plot of fitness values...")
    visualize_solution_fitness(sa_solver.get_solution_values(),
                               # downsample_factor=50
                               )




# GA with TSP
def ga_with_tsp():
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



# SA with bin packing
def sa_with_bin_packing():
    """"""
    #
    max_iterations = 10000
    initial_temperature = 10
    cooling_rate = 0.999
    num_items = 50
    bin_capacity = 25
    weights_min = 1
    # weights_max = bin_capacity
    # weights_max = bin_capacity - 1
    # weights_max = round(bin_capacity / 2)
    weights_max = round(bin_capacity**(5/6))


    # ==========================================================================
    # |  The actual code to run the algorithm:
    # ==========================================================================

    initial_guess = generate_random_bin_config(num_items, weights_min,
                                               weights_max, bin_capacity)

    problem = BinPackingProblem()

    sa_solver = SimulatedAnnealing(
        problem=problem,
        initial_guess=initial_guess,
        max_iterations=max_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate
    )
    sa_solver.anneal()
    problem.display_solution(sa_solver.solution)

    print(f"Displaying bin configuration...")
    visualize_solution_fitness(sa_solver.get_solution_values())



# GA with bin packing
def ga_with_bin_packing():
    """"""





def main():
    """"""

    # future: choose algorithm/problem combination to run

    # sa_with_tsp()
    # ga_with_tsp()
    sa_with_bin_packing()







if __name__ == '__main__':
    main()
