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
                              generate_pressure_vessel_solution,
                              generate_grid_swarm,
                              generate_random_city_swarm,
                              generate_random_bin_swarm,
                              generate_pressure_vessel_swarm,
                              generate_pressure_vessel_population)
from config import (SEED, X_MIN_WA, X_MAX_WA, Y_MIN_WA, Y_MAX_WA,
                    PVD_PLOT_Y_MAX, PVD_PLOT_Y_MIN,
                    THICKNESS_MIN, THICKNESS_MAX, THICKNESS_SCALAR,
                    RADIUS_MIN, RADIUS_MAX, LENGTH_MIN, LENGTH_MAX,
                    Y_AXIS_SA_TSP_GRID, Y_AXIS_SA_TSP_RANDOM,
                    Y_AXIS_SA_BPP, Y_AXIS_SA_PVD,
                    Y_AXIS_POP_TSP_GRID, Y_AXIS_POP_TSP_RANDOM,
                    Y_AXIS_POP_BPP, Y_AXIS_POP_PVD,
                    X_AXIS_SA, X_AXIS_GA, X_AXIS_PSO,
                    LEGEND_TSP, LEGEND_BPP, LEGEND_PVD,
                    COLOR_SA, COLOR_GA, COLOR_PSO,
                    TITLE_SA_TSP, TITLE_GA_TSP, TITLE_PSO_TSP,
                    TITLE_SA_BPP, TITLE_GA_BPP, TITLE_PSO_BPP,
                    TITLE_SA_PVD, TITLE_GA_PVD, TITLE_PSO_PVD
                    )
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
from particle_swarm_optimization import ParticleSwarmOptimization


np.random.seed(SEED)


def sa_with_tsp():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # # 64 city grid, distance 69.136
    # # 263 --> 271 --> 246 --> 250
    # max_iterations = 50000
    # initial_temperature = 29
    # cooling_rate = 0.9998
    # shift_max = 32
    # grid_side_length = 8

    # 64 city grid, distance 72.075  # TODO using this
    max_iterations = 50000
    initial_temperature = 25
    cooling_rate = 0.9998
    grid_side_length = 8

    # 64 city grid, distance 68.485 (very good)
    # max_iterations = 50000
    # initial_temperature = 10
    # cooling_rate = 0.9998
    # grid_side_length = 8

    # random 64 cities, distance 40.952  # TODO using this
    # max_iterations = 1000  # 50000
    # max_iterations = 50000
    # initial_temperature = 5
    # cooling_rate = 0.9998
    # grid_side_length = 8

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
    # initial_guess = generate_random_cities(num_cities, X_MIN_WA, X_MAX_WA,
    #                                        Y_MIN_WA, Y_MAX_WA)

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
    # visualize_solution_fitness(sa_solver.get_solution_values(),
    #                            # downsample_factor=50
    #                            )
    visualize_solution_fitness(sa_solver.get_solution_values(),
                                 xlabel=X_AXIS_SA,
                                 # ylabel=Y_AXIS_SA_TSP_RANDOM,
                                 ylabel=Y_AXIS_SA_TSP_GRID,
                                 title=TITLE_SA_TSP,
                                 legend=LEGEND_TSP,
                                 linecolor=COLOR_SA,
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

    # 64 city grid, distance 65.657  # TODO using this one
    # 228 --> 188 --> 148 --> 134
    pop_size = 600  # 600
    num_generations = 500  # 400
    elitism_percent = 0.06
    crossover_percent = 0.75
    mutation_rate = 0.10
    tournament_size = 4

    # 64 city grid, distance 68.142 (real)
    # 228 --> 193 --> 173 --> 157
    # pop_size = 600
    # num_generations = 400  # 400
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.10
    # tournament_size = 4

    # 64 random cities, distance 35.833 (real)  # TODO using this one
    # 141 --> 115 --> 97 --> 75
    # pop_size = 600
    # num_generations = 500
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.10
    # tournament_size = 4

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

    initial_population = generate_grid_population(pop_size, grid_side_length)

    # initial_population = generate_random_city_population(pop_size, num_cities,
    #                                                      x_min=X_MIN_WA,
    #                                                      x_max=X_MAX_WA,
    #                                                      y_min=Y_MIN_WA,
    #                                                      y_max=Y_MAX_WA)

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
    # visualize_solution_fitness(ga_solver.get_solution_values(),
    #                            xlabel="Generation",
    #                            ylabel="Current Tour Distance",
    #                            title="Tour Distance Over Generations")
    visualize_solution_fitness(ga_solver.get_solution_values(),
                               xlabel=X_AXIS_GA,
                               ylabel=Y_AXIS_POP_TSP_GRID,
                               # ylabel=Y_AXIS_POP_TSP_RANDOM,
                               title=TITLE_GA_TSP,
                               legend=LEGEND_TSP,
                               linecolor=COLOR_GA,
                               )

# ==============================================================================


def pso_with_tsp():
    """"""
    # testing PSO on TSP grid
    # copy some stuff / structure from genetic_algorithm.py main
    inertia_weight = -999  # not used here

    # =========================================
    # |  Hyperparameter combinations:
    # =========================================

    # problem is this algorithm is really slow
    # should keep track of all hyperparameter combinations and their results
    # 500 pop / 500 iterations seems good? 500 iterations low, maybe do 1000

    # 64 city grid, plateaus a bit, distance ___
    # pop_size = 500
    # num_iterations = 500
    # alpha = 0.3
    # beta = 0.3
    # mutation_rate = 0.1

    # 64 city grid, plateaus a bit, distance 73.282, time 760.0 seconds
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.3
    # beta = 0.3
    # mutation_rate = 0.1

    # todo - using this one for the report
    # 64 city grid, distance 70.844, time 719.0 seconds
    # 228 --> 192 --> 158 --> 136
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.4
    # beta = 0.4
    # mutation_rate = 0.1

    # 64 city grid, distance 82.407, time 801.1 seconds
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.5
    # beta = 0.5
    # mutation_rate = 0.1

    # ------------------------------------------------------
    # random cities parameters

    # 64 random cities, distance 46.149, time 732.1 seconds
    # 141 --> 109 --> 92 --> 88
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.4
    # beta = 0.4
    # mutation_rate = 0.1

    # 64 random cities, distance 47.879, time 769.7 seconds
    # 141 --> 123 --> 109 --> 96
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.3
    # beta = 0.3
    # mutation_rate = 0.1

    # todo - using this one for the report
    # 64 random cities, distance 43.494, time 777.9 seconds
    # 141 --> 125 --> 114 --> 102
    pop_size = 500
    num_iterations = 1000
    alpha = 0.2
    beta = 0.2
    mutation_rate = 0.1


    # =========================================
    # |  The actual code to run the algorithm:
    # =========================================

    grid_side_length = 8
    # grid_side_length = 4
    num_cities = grid_side_length ** 2
    problem = TravelingSalesmanProblem()

    print("------------------------------------------")
    print(f"Solving {num_cities}-city TSP with PSO.")
    print(f"If grid, optimal solution distance is {num_cities}.")
    print("------------------------------------------")

    # initial_population = generate_grid_swarm(pop_size, grid_side_length)

    initial_population = generate_random_city_swarm(pop_size, num_cities,
                                                    x_min=X_MIN_WA,
                                                    x_max=X_MAX_WA,
                                                    y_min=Y_MIN_WA,
                                                    y_max=Y_MAX_WA)


    pso_solver = ParticleSwarmOptimization(
        problem=problem,
        initial_population=initial_population,
        population_size=pop_size,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        inertia_weight=inertia_weight,
        mutation_rate=mutation_rate
    )
    pso_solver.print_initial_information()
    pso_solver.swarm()

    best_individual = pso_solver.global_best
    problem.display_solution(best_individual.current_solution)
    # visualize_solution_fitness(pso_solver.get_solution_values(),
    #                            xlabel="Iteration",
    #                            ylabel="Current Tour Distance",
    #                            title="Tour Distance Over Iterations")
    visualize_solution_fitness(pso_solver.get_solution_values(),
                               xlabel=X_AXIS_PSO,
                               ylabel=Y_AXIS_POP_TSP_RANDOM,
                               # ylabel=Y_AXIS_POP_TSP_GRID,
                               title=TITLE_PSO_TSP,
                               legend=LEGEND_TSP,
                               linecolor=COLOR_PSO,
                               )


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
    weights_max = bin_capacity
    # weights_max = bin_capacity - 1
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
    # visualize_solution_fitness(sa_solver.get_solution_values(),
    #                            ylabel="Number of Bins in Solution",
    #                            title="Number of Bins Over Iterations")
    visualize_solution_fitness(sa_solver.get_solution_values(),
                               xlabel=X_AXIS_SA,
                               ylabel=Y_AXIS_SA_BPP,
                               title=TITLE_SA_BPP,
                               legend=LEGEND_BPP,
                               linecolor=COLOR_SA,
                               )

    # current # bins used
    # print(f"Current number of bins used: {}")
    # theoretical minimum

# ==============================================================================

def ga_with_bin_packing():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # max_iterations = 10000
    # initial_temperature = 10
    # cooling_rate = 0.999

    # num_items = 200
    # bin_capacity = 50
    # weights_min = 1
    # weights_max = bin_capacity


    #
    #
    pop_size = 150
    num_generations = 200
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.10
    tournament_size = 4

    num_items = 200
    bin_capacity = 50
    weights_min = 1
    weights_max = bin_capacity

    # weights_max = bin_capacity - 1
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
    # visualize_solution_fitness(ga_solver.get_solution_values(),
    #                            xlabel="Generation",
    #                            ylabel="Number of Bins Used",
    #                            title="Bins Used Over Generations")
    visualize_solution_fitness(ga_solver.get_solution_values(),
                               xlabel=X_AXIS_GA,
                               ylabel=Y_AXIS_POP_BPP,
                               title=TITLE_GA_BPP,
                               legend=LEGEND_BPP,
                               linecolor=COLOR_GA,
                               )

# ==============================================================================

def pso_with_bin_packing():
    """"""

    num_items = 200
    bin_capacity = 50
    weights_min = 1
    weights_max = bin_capacity
    inertia_weight = -999  # not used here

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # let's see what this does
    pop_size = 100
    num_iterations = 100
    alpha = 0.4
    beta = 0.4
    # mutation_rate = 0.03  # turn this way down to 0.02 maybe?
    mutation_rate = 0.00  # turn this way down to 0.02 maybe?


    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = BinPackingProblem()

    print("------------------------------------------------")
    print(f"Solving {num_items}-item bin packing with PSO.")
    print("------------------------------------------------")

    initial_population = generate_random_bin_swarm(pop_size, num_items,
                                                   weights_min=weights_min,
                                                   weights_max=weights_max,
                                                   bin_capacity=bin_capacity)

    pso_solver = ParticleSwarmOptimization(
        problem=problem,
        initial_population=initial_population,
        population_size=pop_size,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        inertia_weight=inertia_weight,
        mutation_rate=mutation_rate
    )
    pso_solver.print_initial_information()
    pso_solver.swarm()

    print(f"Displaying bin configuration...")
    best_individual = pso_solver.global_best.current_solution
    # problem.display_solution(best_individual)

    lower_bound = math.ceil(sum(best_individual.ITEM_WEIGHTS) / bin_capacity)
    print(
        f"Theoretical minimum number of bins, maybe impossible: {lower_bound}")


    problem.display_solution(best_individual)
    # visualize_solution_fitness(pso_solver.get_solution_values(),
    #                            xlabel="Iteration",
    #                            ylabel="Number of Bins Used",
    #                            title="Bins Used Over Iterations",
    #                            legend="Num. bins used")

    visualize_solution_fitness(pso_solver.get_solution_values(),
                               xlabel=X_AXIS_PSO,
                               ylabel=Y_AXIS_POP_BPP,
                               title=TITLE_PSO_BPP,
                               legend=LEGEND_BPP,
                               linecolor=COLOR_PSO,
                               )


# ==============================================================================

def sa_with_pressure_vessel_design():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # Cost $6096.407
    # 114683 --> 6711 --> 6990 --> 7270
    # use this one for the report
    max_iterations = 15000
    initial_temperature = 400
    cooling_rate = 0.9997
    radius_step_size = 1
    length_step_size = 2

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = PressureVesselProblem()

    # initial guess, need a helper method
    initial_guess = generate_pressure_vessel_solution(radius_step_size,
                                                      length_step_size)
    print("Initial solution:")
    problem.display_solution(initial_guess)

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
                               xlabel=X_AXIS_SA,
                               ylabel=Y_AXIS_SA_PVD,
                               title=TITLE_SA_PVD,
                               legend=LEGEND_PVD,
                               linecolor=COLOR_SA,
                               y_min=PVD_PLOT_Y_MIN,
                               y_max=PVD_PLOT_Y_MAX
                               )

# ==============================================================================

def ga_with_pressure_vessel_design():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # Cost $6094.163
    # 14322 --> 11596 --> 7017 --> 6228
    # todo can Use this
    pop_size = 300
    num_generations = 100
    elitism_percent = 0.05
    crossover_percent = 0.75
    mutation_rate = 0.15
    tournament_size = 4
    mutation_radius_step_size = 2
    mutation_length_step_size = 6

    # pop_size = 300
    # num_generations = 150
    # elitism_percent = 0.05
    # crossover_percent = 0.75
    # mutation_rate = 0.2
    # tournament_size = 4
    # mutation_radius_step_size = 3
    # mutation_length_step_size = 10


    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = PressureVesselProblem()
    initial_population = generate_pressure_vessel_population(pop_size,
                                                             mutation_radius_step_size,
                                                             mutation_length_step_size)
    # GA solver
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

    print(f"Printing found solution...")
    best_individual = ga_solver.gen_best_solution
    problem.display_solution(best_individual)
    visualize_solution_fitness(ga_solver.get_solution_values(),
                               # xlabel="Generation",
                               # ylabel="Current Pressure Vessel Cost ($)",
                               # title="Pressure Vessel Design Cost Over GA Generations",
                               # legend="Cost of design",
                               xlabel=X_AXIS_GA,
                               ylabel=Y_AXIS_POP_PVD,
                               title=TITLE_GA_PVD,
                               legend=LEGEND_PVD,
                               linecolor=COLOR_GA,
                               y_min=PVD_PLOT_Y_MIN,
                               y_max=PVD_PLOT_Y_MAX,
                               )

# ==============================================================================

def pso_with_pressure_vessel_design():
    """"""

    # -----------------------------------
    # |  Hyperparameter combinations:
    # -----------------------------------

    # # Cost $6063.565
    # # 14322 --> 7061 --> 6289 --> 6258
    # # TODO can using this one
    pop_size = 300
    num_iterations = 100
    alpha = 1.3
    beta = 1.2
    inertia_weight = 0.7
    mutation_radius_step_size = 1.5
    mutation_length_step_size = 3
    mutation_rate = 0.15

    # pop_size = 300
    # num_iterations = 100
    # alpha = 1.3
    # beta = 1.3
    # inertia_weight = 0.7
    # mutation_radius_step_size = 1.5
    # mutation_length_step_size = 3
    # mutation_rate = 0.15

    # ------------------------------------------
    # |  The actual code to run the algorithm:
    # ------------------------------------------

    problem = PressureVesselProblem()
    initial_population = generate_pressure_vessel_swarm(pop_size,
                                                        mutation_radius_step_size,
                                                        mutation_length_step_size)

    pso_solver = ParticleSwarmOptimization(
        problem=problem,
        initial_population=initial_population,
        population_size=pop_size,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        inertia_weight=inertia_weight,
        mutation_rate=mutation_rate
    )
    pso_solver.print_initial_information()
    pso_solver.swarm()

    print(f"Printing found solution...")
    best_individual = pso_solver.global_best.current_solution
    # problem.display_solution(best_individual)

    problem.display_solution(best_individual)
    visualize_solution_fitness(pso_solver.get_solution_values(),
                               # xlabel="Iteration",
                               # ylabel="Current Pressure Vessel Cost ($)",
                               # title="Pressure Vessel Design Cost Over Iterations",
                               # legend="Cost of design",
                               xlabel=X_AXIS_PSO,
                               ylabel=Y_AXIS_POP_PVD,
                               title=TITLE_PSO_PVD,
                               legend=LEGEND_PVD,
                               linecolor=COLOR_PSO,
                               y_min=PVD_PLOT_Y_MIN,
                               y_max=PVD_PLOT_Y_MAX,
                               )



def main():
    """"""

    # sa_with_tsp()  # grid collected, random collected
    # ga_with_tsp()  # grid collected, random collected
    # pso_with_tsp()  # grid collected, random collected
    # sa_with_bin_packing()  # collected
    # ga_with_bin_packing()  # collected
    # pso_with_bin_packing()  # collected
    # sa_with_pressure_vessel_design()  # collected
    # ga_with_pressure_vessel_design()  # collected
    pso_with_pressure_vessel_design()  # collected

    # sa tsp    (done)
    # sa bpp    (done)
    # sa pvd    (done)
    # ga tsp    (done)
    # ga bpp    (done)
    # ga pvd    (need this)
    # pso tsp   (done)
    # pso bpp   (done)
    # pso pvd   (done)


if __name__ == '__main__':
    main()
