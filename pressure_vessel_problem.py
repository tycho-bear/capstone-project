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

    def sort_by_fitness(self, population: list[Solution]) -> list[Solution]:
        pass

    def get_elite(self, sorted_population: list[Solution],
                  elitism_percent: float) -> list[Solution]:
        pass

    def tournament_selection(self, population: list[Solution],
                             num_samples: int) -> Solution:
        pass

    def crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        pass

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


    def apply_mutation(self, population: list[Solution],
                       mutation_prob: float) -> list[Solution]:
        pass

    def generate_new_individual(self, reference_individual: Solution) -> (
            Solution):
        pass


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
        head_thickness = new_values[0]
        body_thickness = new_values[1]
        rounded_head_thickness = (round(head_thickness / THICKNESS_SCALAR) *
                                  THICKNESS_SCALAR)
        rounded_body_thickness = (round(body_thickness / THICKNESS_SCALAR) *
                                  THICKNESS_SCALAR)
        new_values[0] = rounded_head_thickness
        new_values[1] = rounded_body_thickness

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


