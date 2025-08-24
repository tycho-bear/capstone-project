# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import math
import numpy as np
from config import SEED
from problem import Problem
import time
from helper_classes import Particle
from helper_functions import (generate_random_city_population,
                              generate_grid_population,
                              visualize_solution_fitness,
                              generate_grid_swarm)
from traveling_salesman_problem import TravelingSalesmanProblem
import copy


np.random.seed(SEED)


class ParticleSwarmOptimization:
    """
    This class implements a particle swarm optimizer.
    """

    def __init__(self,
                 problem: Problem,
                 initial_population: list[Particle],
                 population_size: int,
                 num_iterations: int,
                 alpha: float,
                 beta: float,
                 inertia_weight: float,
                 mutation_rate: float,
                 ):
        """"""
        # problem
        # population
        # hyperparameters

        # what problem, also save solution values like normal
        self.problem = problem
        self.solution_values = []  # for plotting

        # main PSO parameters
        self.POPULATION_SIZE = population_size
        self.NUM_ITERATIONS = num_iterations
        self.ALPHA = alpha
        self.BETA = beta
        self.INERTIA_WEIGHT = inertia_weight
        self.MUTATION_RATE = mutation_rate

        # setting things up
        self.population = initial_population
        self.iteration_number = 0
        self.global_best = None  # particle
        self.global_best_fitness = None

        # keep track of the best
        # self.gen_best_solution = None  # updated every iteration
        # self.gen_best_fitness = None  # updated every iteration



    def pso_iteration(self):
        """"""
        # performs one iteration of PSO
        # housekeeping
        # for each particle, generate velocity, move, and update current best
            # velocity = calculate_velocity
            # apply_velocity
        # update current global best

        # keep track of the best solution and print stuff
        self.iteration_housekeeping()

        # for each particle, generate velocity, move, and update current best
        for particle in self.population:
            # velocity = self.problem.calculate_velocity(particle,
            #                                            self.global_best,
            #                                            self.ALPHA,
            #                                            self.BETA,
            #                                            self.INERTIA_WEIGHT)

            self.problem.calculate_velocity(particle, self.global_best,
                                            self.ALPHA, self.BETA,
                                            self.INERTIA_WEIGHT)

            # self.problem.apply_velocity(particle, velocity)
            self.problem.apply_velocity(particle)

        # self.problem.update_particle_bests(self.population)
        self.update_particle_bests()  # do these two work the same?

        # can comment this to turn off mutation and see the crap solutions
        self.problem.apply_mutation_to_swarm(self.population,
                                             self.MUTATION_RATE)


        # apply mutation to swarm
        # update bests

        # update global best after the loop completes? or in housekeeping?


    def update_particle_bests(self) -> None:
        """
        Updates the best solutions seen by any particle in the swarm.

        :return: None
        """

        for particle in self.population:
            if (self.problem.evaluate_solution(particle.current_solution) <
                    self.problem.evaluate_solution(particle.best_solution)):
                particle.best_solution = copy.deepcopy(
                    particle.current_solution)


    def iteration_housekeeping(self):
        """
        Helper method that does a few things:
            - updates the current global best solution in the population and its
            fitness inside the class
            - keeps track of this solution so we can plot it later
            - every 100 iterations, prints info about the current generation

        Behaves much like the similarly-named method in `genetic_algorithm.py`.

        :return: None
        """

        # don't need to sort this time, but still need to get the best
        # todo - do we update the global best in here?
        #  yes, need a global best to move to during the first iteration
        #  we can just plot this, too

        # update the global best solution and its fitness
        # particles don't store fitness, just current solution and best solution
        # use problem.evaluate_solution

        self.global_best = copy.deepcopy(min(self.population, key=lambda p: self.problem.evaluate_solution(p.current_solution)))
        self.global_best_fitness = self.problem.evaluate_solution(self.global_best.current_solution)

        self.solution_values.append(self.global_best_fitness)

        # print stuff every 100 iterations (called again after the main loop)
        if self.iteration_number % 10 == 0:
            self.print_current_iteration_information()


    def swarm(self) -> None:
        """
        Runs the PSO algorithm for the given number of iterations.
        Every so often, housekeeping prints the information about the
        current iteration.

        Also keeps track of elapsed time.

        :return: None
        """
        # run the algorithm for multiple iterations
        # time it, do housekeeping at the end
        # also print the best solution at the end

        start_time = time.time()

        while self.iteration_number < self.NUM_ITERATIONS:
            self.pso_iteration()
            self.iteration_number += 1

        # take care of the final generation here
        self.iteration_housekeeping()

        end_time = time.time()
        # print best solution data
        print(f"Best solution in population at iteration "
              f"{self.iteration_number}: fitness "
              f"{self.solution_values[-1]:.3f}")
        print(f"Elapsed time: {(end_time - start_time):.1f} seconds.")


    def print_initial_information(self) -> None:
        """
        Helper function that prints the hyperparameters used in PSO.

        :return: None
        """

        print("Running particle swarm optimization...")
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Number of iterations: {self.NUM_ITERATIONS}")
        print(f"Alpha: {self.ALPHA}")
        print(f"Beta: {self.BETA}")
        print(f"Mutation rate: {self.MUTATION_RATE}")
        print()


    def print_current_iteration_information(self) -> None:
        """
        Helper function that prints out information about the current iteration.
        Something like this:

        `"Generation {current}/{total}: best fitness: {best_fitness}"`

        :return: NOne
        """

        print(f"Iteration {self.iteration_number}/{self.NUM_ITERATIONS}, "
              f"Fitness of best individual: {self.global_best_fitness:.3f}")


    def get_solution_values(self) -> list:
        """
        Getter for the solution fitness values.

        :return: (list) The current list of solution fitness values.
        """

        return self.solution_values



def main():
    """"""
    # testing PSO on TSP grid
    # copy some stuff / structure from genetic_algorithm.py main

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

    # running this now
    # use this one for the report
    # 64 city grid, distance 70.844, time 719.0 seconds
    # pop_size = 500
    # num_iterations = 1000
    # alpha = 0.4
    # beta = 0.4
    # mutation_rate = 0.1

    # 64 city grid, distance 82.407, time 801.1 seconds
    pop_size = 500
    num_iterations = 1000
    alpha = 0.5
    beta = 0.5
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

    initial_population = generate_grid_swarm(pop_size, grid_side_length)

    pso_solver = ParticleSwarmOptimization(
        problem=problem,
        initial_population=initial_population,
        population_size=pop_size,
        num_iterations=num_iterations,
        alpha=alpha,
        beta=beta,
        mutation_rate=mutation_rate
    )
    pso_solver.print_initial_information()
    pso_solver.swarm()

    best_individual = pso_solver.global_best
    problem.display_solution(best_individual.current_solution)
    visualize_solution_fitness(pso_solver.get_solution_values(),
                               xlabel="Iteration",
                               ylabel="Current Tour Distance",
                               title="Tour Distance Over Iterations")






if __name__ == '__main__':
    main()
