# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from config import SEED, X_MIN_WA, X_MAX_WA, Y_MIN_WA, Y_MAX_WA
import numpy as np
from helper_functions import (generate_random_cities, generate_square_grid,
                              generate_random_city_population,
                              generate_grid_population)
from traveling_salesman_problem import TravelingSalesmanProblem


np.random.seed(SEED)


def test_stuff():
    """"""
    # first create a random tour of 10 cities
    pop = []
    num_cities = 8
    population_size = 10
    for i in range(population_size):
        # pop.append(generate_random_cities(num_cities, x_min_WA, x_max_WA,
        #                                    y_min_WA, y_max_WA))
        pop.append(generate_square_grid(side_length=3))
    problem = TravelingSalesmanProblem(num_cities, 32)

    def print_population(pop):
        print("Population:")
        for tour in pop:
            print(f"\t{tour}, distance {tour.tour_distance:.3f}")

    print_population(pop)

    # sorting by fitness (easy) - works
    print()
    print("sorting by fitness:")
    sorted_pop = problem.sort_by_fitness(pop)
    print_population(sorted_pop)


    # get elite (easy) - works
    print()
    print("get elite:")
    elitism_percent = 0.15
    elite = problem.get_elite(sorted_pop, elitism_percent)
    print_population(elite)

    # tournament selection (medium) - works
    print()
    print("tournament selection:")
    num_samples = 2
    champion = problem.tournament_selection(sorted_pop, num_samples)
    print(f"best: {champion}, {champion.tour_distance:.3f}")

    # crossover (medium) - works? think so
    print()
    print("ordered crossover:")
    parent1 = problem.tournament_selection(pop, 3)
    parent2 = problem.tournament_selection(pop, 3)
    child = problem.crossover(parent1, parent2)

    print(f"parent 1: {parent1}, distance {parent1.tour_distance}")
    print(f"parent 2: {parent2}, distance {parent2.tour_distance}")
    print(f"child:    {child}, distance {child.tour_distance}")


    # mutate_individual (easy/medium)

    # apply_random_mutation (easy)
    print()
    print("mutation:")
    print_population(sorted_pop)
    mutation_prob = 0.10
    mutated_pop = problem.apply_mutation(sorted_pop, mutation_prob)
    print("Applied mutation.")
    print_population(mutated_pop)

    # generate random population
    print()
    print("generating random population:")
    new_pop_size = 5
    new_num_cities = 5
    new_pop = generate_random_city_population(new_pop_size, new_num_cities, X_MIN_WA,
                                              X_MAX_WA, Y_MIN_WA, Y_MAX_WA)
    print_population(new_pop)

    new_grid = generate_grid_population(new_pop_size, 3)
    print_population(new_grid)



def main():
    """"""
    test_stuff()


if __name__ == '__main__':
    main()

