# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from config import seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA
import numpy as np
from helper_functions import generate_random_cities
from problems import TravelingSalesmanProblem


np.random.seed(seed)


def test_stuff():
    """"""
    # first create a random tour of 10 cities
    pop = []
    num_cities = 6
    population_size = 6
    for i in range(population_size):
        pop.append(generate_random_cities(num_cities, x_min_WA, x_max_WA,
                                           y_min_WA, y_max_WA))
    problem = TravelingSalesmanProblem(num_cities, 32)

    def print_population(pop):
        print("Population:")
        for tour in pop:
            print(f"\t{tour}, distance {tour.tour_distance:.3f}")

    print_population(pop)

    # sorting by fitness (easy) - works
    print()
    sorted_pop = problem.sort_by_fitness(pop)
    print_population(sorted_pop)


    # get elite (easy) - works
    print()
    elitism_percent = 0.15
    elite = problem.get_elite(sorted_pop, elitism_percent)
    print_population(elite)

    # tournament selection (medium) - works
    print()
    num_samples = 3
    champion = problem.tournament_selection(sorted_pop, num_samples)
    print(f"best: {champion}, {champion.tour_distance:.3f}")

    # crossover (medium)

    # mutate_individual (easy/medium)

    # apply_random_mutation (easy)


def main():
    """"""
    test_stuff()


if __name__ == '__main__':
    main()

