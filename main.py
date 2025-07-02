# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
from helper_functions import (generate_random_cities,
                              generate_square_grid,
                              visualize_solution_fitness,
                              generate_grid_population,
                              generate_random_bin_config,
                              generate_random_city_population,
                              generate_random_bin_population,
                              generate_pressure_vessel_solution)
from config import (seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA,
                    thickness_min, thickness_max, thickness_scalar, radius_min, radius_max, length_min, length_max)
from helper_classes import Tour
import copy
import numpy as np
from typing import Any
import time
from problem import Problem, Solution
from traveling_salesman_problem import TravelingSalesmanProblem
from bin_packing_problem import BinPackingProblem
from pressure_vessel_problem import PressureVesselProblem
from simulated_annealing import SimulatedAnnealing
from genetic_algorithm import GeneticAlgorithm


np.random.seed(seed)


def sa_with_tsp():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # 64 city grid, distance 69.136
    # 263 --> 271 --> 246 --> 250
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

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    # grid_side_length = 8
    initial_guess = generate_square_grid(grid_side_length)  # grid
    num_cities = grid_side_length ** 2

    # -----------------------------
    # For random cities, do this:
    # -----------------------------
    # initial_guess = generate_random_cities(num_cities, x_min_WA, x_max_WA,
    #                                        y_min_WA, y_max_WA)

    # define problem here
    problem = TravelingSalesmanProblem(
        # initial_guess=initial_guess,
        # num_cities=num_cities,
        # shift_max=shift_max
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

# ==============================================================================

def ga_with_tsp():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # # 64 city grid, distance 68.307
    # # 228 --> 175 --> 170 --> 151 (30)
    # pop_size = 500
    # num_generations = 400  # 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.10
    # tournament_size = 3

    # 64 city grid, distance 66.243
    # 228 --> 191 --> 163 --> 147
    pop_size = 600
    num_generations = 400  # 400
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.10
    tournament_size = 4

    # # 64 city grid, distance 70.614 (looks cool though, no loops)
    # # converges pretty fast
    # pop_size = 500
    # num_generations = 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.07
    # tournament_size = 3

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    grid_side_length = 8
    num_cities = grid_side_length ** 2
    problem = TravelingSalesmanProblem(
        # num_cities,
        # shift_max=num_cities
    )

    print("------------------------------------------")
    print(f"Solving {num_cities}-city TSP with GA.")
    print(f"If grid, optimal solution distance is {num_cities}.")
    print("------------------------------------------")

    initial_population = generate_random_city_population(pop_size, num_cities,
                                                         x_min=x_min_WA,
                                                         x_max=x_max_WA,
                                                         y_min=y_min_WA,
                                                         y_max=y_max_WA)
    # initial_population = generate_grid_population(pop_size, grid_side_length)

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

# ==============================================================================

def sa_with_bin_packing():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    #
    max_iterations = 10000
    initial_temperature = 10
    cooling_rate = 0.999
    num_items = 200
    bin_capacity = 50
    weights_min = 1
    # weights_max = bin_capacity
    weights_max = bin_capacity - 1
    # weights_max = round(bin_capacity / 2)
    # weights_max = round(bin_capacity**(5/6))
    # weights_max = 29
    # weights_max = round(bin_capacity**(2/3))

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = BinPackingProblem()
    initial_guess = generate_random_bin_config(num_items, weights_min,
                                               weights_max, bin_capacity)
    # print("Initial guess, for reference:")
    # problem.display_solution(initial_guess)

    sa_solver = SimulatedAnnealing(
        problem=problem,
        initial_guess=initial_guess,
        max_iterations=max_iterations,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate
    )
    sa_solver.anneal()

    print(f"Displaying bin configuration...")
    problem.display_solution(sa_solver.solution)

    lower_bound = math.ceil(sum(initial_guess.ITEM_WEIGHTS) / bin_capacity)
    print(f"Theoretical minimum number of bins, maybe impossible: {lower_bound}")

    print("Displaying solution fitness over time...")
    visualize_solution_fitness(sa_solver.get_solution_values(),
                               ylabel="Number of bins in solution",
                               title="Number of bins over iterations")

    # current # bins used
    # print(f"Current number of bins used: {}")
    # theoretical minimum

# ==============================================================================

def ga_with_bin_packing():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    #
    #
    pop_size = 150
    num_generations = 300
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.10
    tournament_size = 4

    num_items = 150
    bin_capacity = 40
    weights_min = 1
    weights_max = bin_capacity - 1

    # weights_max = bin_capacity
    # weights_max = round(bin_capacity / 2)
    # weights_max = round(bin_capacity**(5/6))
    # weights_max = 29
    # weights_max = round(bin_capacity**(2/3))

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = BinPackingProblem()

    print("-----------------------------------------------")
    print(f"Solving {num_items}-item bin packing with GA.")
    print("-----------------------------------------------")

    initial_population = generate_random_bin_population(pop_size, num_items,
                                                        weights_min=weights_min,
                                                        weights_max=weights_max,
                                                    bin_capacity=bin_capacity)

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

    print(f"Displaying bin configuration...")
    best_individual = ga_solver.gen_best_solution
    problem.display_solution(best_individual)

    lower_bound = math.ceil(sum(best_individual.ITEM_WEIGHTS) / bin_capacity)
    print(f"Theoretical minimum number of bins, maybe impossible: {lower_bound}")

    print("Displaying solution fitness over time...")
    visualize_solution_fitness(ga_solver.get_solution_values(),
                               xlabel="Generation",
                               ylabel="Number of Bins Used",
                               title="Bins Used Over Generations")

# ==============================================================================

def sa_with_pressure_vessel_design():
    """"""

    # TODO d_1 and d_2 are actually discrete...
    # d_min = 1
    # d_max = 100
    # d_scalar = 0.0625
    # r_min = 10
    # r_max = 100  # arbitrary, but not too high
    # L_min = 10   # arbitrary, but not too low
    # L_max = 200



    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # ...
    max_iterations = 5000
    initial_temperature = 100
    cooling_rate = 0.99

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    # initial guess, need a helper method
    initial_guess = generate_pressure_vessel_solution()

    problem = PressureVesselProblem()

    # set up and run solver
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





def main():
    """"""

    # sa_with_tsp()
    # ga_with_tsp()
    # sa_with_bin_packing()
    # ga_with_bin_packing()

    sa_with_pressure_vessel_design()



if __name__ == '__main__':
    main()
