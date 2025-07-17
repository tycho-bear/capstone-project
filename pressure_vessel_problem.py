# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================
from typing import Any

import numpy as np
from config import SEED, NUM_DESIGN_VARIABLES, THICKNESS_SCALAR
import copy
from problem import Problem, Solution, Velocity
from helper_classes import Design, Particle, PVDParticle
from collections import Counter
from helper_functions import generate_pressure_vessel_solution


np.random.seed(SEED)
# PVSolution = np.ndarray


class PressureVesselProblem(Problem):
    """

    """

    def __init__(self):
        """

        """

    # ==========================================================================
    # |  General methods
    # ==========================================================================

    def evaluate_solution(self, solution: Design) -> float:
        """"""

        # remember
        # print(f"Evaluating solution with thicknesses: "
        #       f"{solution.head_thickness}, {solution.body_thickness}")
        return solution.cost


    def display_solution(self, solution: Design) -> None:
        """"""

        # just print stuff
        print(f"Pressure vessel design variables:\n"
              f"\tHead thickness:\t{solution.head_thickness}\n"
              f"\tBody thickness:\t{solution.body_thickness}\n"
              f"\tInner radius:\t{solution.inner_radius:.4f}\n"
              f"\tCylindrical length:\t{solution.cylindrical_length:.4f}\n"
              f"Total cost: ${solution.cost:.3f}")
        print(f"Valid solution? {solution.is_valid_design()}")


    # ==========================================================================
    # |  Simulated annealing methods
    # ==========================================================================

    def generate_neighbor(self, current_solution: Design) -> Design:
        """"""

        # sorta normal SA, need to also remember
        new_solution = current_solution.generate_neighbor()
        return new_solution


    # ==========================================================================
    # |  TODO - Genetic algorithm methods
    # ==========================================================================

    def sort_by_fitness(self, population: list[Design]) -> list[Design]:
        """
        Sorts the given population by fitness in ascending order. Most fit
        individuals are at the start of the list. For the pressure vessel
        problem, this sorts in ascending order according to the cost.

        :param population: (list[Design]) The population to sort.
        :return: (list[Design]) The population, sorted by design cost in
            ascending order.
        """

        new_population = sorted(population, key=lambda design: design.cost)
        return new_population



    def get_elite(self, sorted_population: list[Design],
                  elitism_percent: float) -> list[Design]:
        """
        Simple function to extract the best few individuals from a population.

        :param sorted_population: (list[Design]) A population sorted in
            ascending order by cost.
        :param elitism_percent: (float) The percentage of the population to
            retain for the next generation. If the number of individuals does
            not come out to a whole number, it will be rounded.
        :return: (list[Design]) The elite subset of the population.
        """

        population_size = len(sorted_population)
        count_to_retain = round(elitism_percent * population_size)
        elite = sorted_population[:count_to_retain]
        return elite


    def tournament_selection(self, population: list[Design], num_samples: int) \
            -> Design:
        """
        Chooses `num_samples` Designs from the population without replacement
        and returns the one with the lowest cost.

        :param population: (list[Design]) The population to choose from.
        :param num_samples: (int) The number of samples, without replacement.
        :return: (Design) The lowest cost Design from the samples.
        """

        # choose a few unique participants
        pop_size = len(population)
        indices = np.random.choice(pop_size, size=num_samples, replace=False)
        participants = [population[i] for i in indices]

        # get the best
        best = participants[0]
        for design in participants:
            if design.cost < best.cost:
                best = design

        return best


    def crossover(self, parent1: Design, parent2: Design) -> Design:
        """
        Performs crossover with two parents.


        TODO

        :param parent1: (Design) The first parent.
        :param parent2: (Design) The second parent.
        :return: (Design) A child containing TODO.
        """

        head_thickness_average = (parent1.head_thickness +
                                  parent2.head_thickness) / 2
        body_thickness_average = (parent1.body_thickness +
                                  parent2.body_thickness) / 2

        new_head_thickness = Design.round_thickness_value(head_thickness_average)
        new_body_thickness = Design.round_thickness_value(body_thickness_average)

        def blx_crossover(value1: float, value2: float, alpha=0.3) -> float:
            """


            :param value1:
            :param value2:
            :param alpha:
            :return:
            """

            lower = min(value1, value2)
            upper = max(value1, value2)
            difference = upper - lower
            new_value = np.random.uniform(lower - alpha*difference,
                                          upper + alpha*difference)
            return new_value

        new_inner_radius = blx_crossover(parent1.inner_radius,
                                         parent2.inner_radius)
        new_cylindrical_length = blx_crossover(parent1.cylindrical_length,
                                               parent2.cylindrical_length)

        new_design = Design(
            head_thickness=new_head_thickness,
            body_thickness=new_body_thickness,
            inner_radius=new_inner_radius,
            cylindrical_length=new_cylindrical_length,
            radius_step_size=parent1.RADIUS_STEP_SIZE,
            length_step_size=parent1.LENGTH_STEP_SIZE
        )
        return new_design





    def mutate_individual(self, individual: Design) -> Design:
        """
        Applies a random mutation to the given Design. The mutated Design is
        returned as a new object.

        Mutation involves a simple vector perturbation similar to neighbor
        generation in simulated annealing.

        :param individual: (Design) The Design to mutate.
        :return: (Design) The mutated Design.
        """

        new_design = individual.generate_neighbor()
        return new_design


    def apply_mutation(self, population: list[Design], mutation_prob: float) \
            -> list[Design]:
        """
        Probabilistically applies mutation to the entire population.

        :param population: (list[Design]) The population to mutate.
        :param mutation_prob: (float) The probability of mutating each
            individual in the population.
        :return: (list[Design]) The mutated population.
        """

        for i in range(len(population)):
            rand = np.random.rand()
            if rand < mutation_prob:
                population[i] = self.mutate_individual(population[i])

        return population



    def generate_new_individual(self, reference_individual: Design) -> Design:
        """
        Generates a new Design. The reference individual is not used in this
        implementation. This method is used when generating a population for a
        genetic algorithm.

        :param reference_individual: (Design) Not used in this implementation
            for the pressure vessel problem.
        :return: (Design) A new, randomly generated Design.
        """

        # use generate_pressure_vessel_solution
        new_design = generate_pressure_vessel_solution(
            reference_individual.RADIUS_STEP_SIZE,
            reference_individual.LENGTH_STEP_SIZE
        )
        return new_design


    # ==========================================================================
    # |  Particle swarm optimization methods
    # ==========================================================================


    def calculate_velocity(self, particle: PVDParticle,
                           global_best: PVDParticle,
                           alpha: float, beta: float, inertia_weight: float):
        """


        :param particle:
        :param global_best:
        :param alpha:
        :param beta:
        :param inertia_weight:
        :return:
        """

        # L = 0.7  # inertia weight
        # # generate new velocity vi using equation (7.1)
        # v[i] = L * v[i] + alpha * e1 * (g_star - x[i]) + beta * e2 * (x_bests[i] - x[i])

        # TODO - need copy.deepcopy in here?

        current_velocity = copy.deepcopy(particle.velocity)

        e1 = np.random.uniform(size=NUM_DESIGN_VARIABLES)
        e2 = np.random.uniform(size=NUM_DESIGN_VARIABLES)

        # might not be able to one-line it like above
        # current_values = particle.current_solution.get_values()
        current_values = copy.deepcopy(particle.current_solution.get_values())
        global_best_values = copy.deepcopy(global_best.current_solution.get_values())
        personal_best_values = copy.deepcopy(particle.best_solution.get_values())

        difference_to_global_best = global_best_values - current_values
        difference_to_personal_best = personal_best_values - current_values

        new_velocity = (inertia_weight * current_velocity) + \
                       (alpha * e1 * difference_to_global_best) + \
                       (beta * e2 * difference_to_personal_best)

        particle.velocity = new_velocity




    def apply_velocity(self, particle: PVDParticle):
        """


        :param particle:
        :param velocity:
        :return:
        """


        # x[i] = x[i] + v[i]

        # so get a new set of values
        # clip to bounds here
        # then create a new Design object
        # and do particle.current_solution = new_design

        current_values = copy.deepcopy(particle.current_solution.get_values())
        new_values = current_values + particle.velocity

        # round the thicknesses to nearest 0.0625
        # TODO - put the rounding stuff in a static method in Design?
        # head_thickness = new_values[0]
        # body_thickness = new_values[1]
        # rounded_head_thickness = (round(head_thickness / THICKNESS_SCALAR) *
        #                           THICKNESS_SCALAR)
        # rounded_body_thickness = (round(body_thickness / THICKNESS_SCALAR) *
        #                           THICKNESS_SCALAR)
        # new_values[0] = rounded_head_thickness
        # new_values[1] = rounded_body_thickness
        new_values[0] = Design.round_thickness_value(new_values[0])  # works, ignore warning
        new_values[1] = Design.round_thickness_value(new_values[1])

        # clip to bounds
        new_values = Design.clip_values_to_bounds(new_values)

        new_design = Design(head_thickness=new_values[0],
                            body_thickness=new_values[1],
                            inner_radius=new_values[2],
                            cylindrical_length=new_values[3],
                            radius_step_size=particle.current_solution.
                                                            RADIUS_STEP_SIZE,
                            length_step_size=particle.current_solution.
                                                            LENGTH_STEP_SIZE)

        particle.current_solution = new_design





    def apply_mutation_to_swarm(self, population: list[PVDParticle],
                                mutation_prob: float) -> None:
        """
        Applies mutation to each particle's current solution with the given
        probability.

        :param population: (list[PVDParticle]) The swarm of particles.
        :param mutation_prob: (float) The probability of mutating a given
            particle's current solution.
        :return: None
        """

        # leaving this blank for now, may turn it on later
        # pass

        for particle in population:
            rand = np.random.rand()
            if rand < mutation_prob:
                particle.current_solution = self.mutate_individual(
                    particle.current_solution)


