# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import numpy as np
from helper_classes import (City, Tour, BinConfiguration, Design, TSPParticle,
                            BPPParticle)
from config import (SEED, THICKNESS_MIN, THICKNESS_MAX, THICKNESS_SCALAR,
                    RADIUS_MIN, RADIUS_MAX, LENGTH_MIN, LENGTH_MAX,
                    THICKNESS_MIN_INT, THICKNESS_MAX_INT)
import matplotlib.pyplot as plot
import math
from matplotlib.ticker import MaxNLocator

np.random.seed(SEED)


def generate_random_cities(num_cities: int, x_min: float, x_max: float,
                           y_min: float, y_max: float) -> Tour:
    """
    Generates a specific number of cities in random positions.

    :param num_cities: (int) The number of cities to generate.
    :param x_min: (double) Lower bound on the x coordinates.
    :param x_max: (double) Upper bound on the x coordinates.
    :param y_min: (double) Lower bound on the y coordinates.
    :param y_max: (double) Upper bound on the y coordinates.
    :return: (Tour) A Tour containing the City objects with random x and y
        coordinates.
    """

    cities = []
    for i in range(num_cities):
        # name = f"city{i}"
        name = f"{i}"
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        city = City(name, x, y)
        cities.append(city)

    return Tour(cities)


def generate_random_bin_config(num_items, weights_min, weights_max,
                               bin_capacity) -> BinConfiguration:
    """SA with bin packing, weights_max is inclusive"""

    weights = []
    for i in range(num_items):


        # weight = np.random.randint(low=weights_min, high=weights_max)
        # weight = round(weights_min + np.random.beta(a=2, b=7) * (weights_max - weights_min))
        weight = round(weights_min + np.random.beta(a=2, b=3) * (weights_max - weights_min))
        #

        # weight = round(weights_min + np.random.beta(2, 8) * (weights_max - weights_min))
        # weight = round(weights_min + np.random.beta(2, 5) * (weights_max - weights_min))
        # weight = round(weights_min + np.random.beta(0.3, 0.3) * (weights_max - weights_min))
        weights.append(weight)

    config = BinConfiguration(weights, bin_capacity)
    return config



def generate_random_city_population(pop_size: int, num_cities: int,
                                    x_min: float, x_max: float, y_min: float,
                                    y_max: float) -> list[Tour]:
    """
    Generates a random population of Tours for use in a genetic algorithm.
    Each individual in the population is a different Tour over the same cities.

    :param pop_size:
    :param num_cities:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return:
    """

    # make sure we use the same cities for all tours in the population
    pop = [generate_random_cities(num_cities, x_min, x_max, y_min, y_max)]
    pop = pop * pop_size

    # shuffle each tour
    for i in range(pop_size):
        pop[i] = pop[i].shuffle_tour()

    return pop


def generate_random_city_swarm(pop_size: int, num_cities: int,
                               x_min: float, x_max: float, y_min: float,
                               y_max: float) -> list[TSPParticle]:
    """
    Generates a random swarm of particles for use in a particle swarm
    optimization algorithm. Each particle in the swarm is a different Tour over
    the same cities.

    :param pop_size:
    :param num_cities:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return:
    """

    # make sure we use the same cities for all tours in the population
    swarm = [generate_random_cities(num_cities, x_min, x_max, y_min, y_max)]
    swarm = swarm * pop_size
    # now have a bunch of tours over the same cities

    # shuffle each tour
    particles = []
    for i in range(pop_size):
        swarm[i] = swarm[i].shuffle_tour()

    # convert to particles
    for tour in swarm:
        # each particle is a tour
        # TODO - this and the same method below may need to use copy.deepcopy
        #  for best_solution.
        particle = TSPParticle(current_solution=tour, best_solution=tour)
        particles.append(particle)

    return particles



def generate_random_bin_population(pop_size: int, num_items: int,
                                   weights_min: int, weights_max: int,
                                   bin_capacity: int) -> list[BinConfiguration]:
     """
     Generates a random population of BinConfigurations for use in a genetic
     algorithm. Each individual in the population is a different
     BinConfiguration over the same items.

     :param pop_size: (int) The size of the population to generate.
     :param num_items: (int) The number of items in each configuration.
     :param weights_min: (int) The minimum weight of an item.
     :param weights_max: (int) The maximum weight of an item.
     :param bin_capacity: (int) The capacity of each bin.
     :return: (list[BinConfiguration]) A list of BinConfiguration objects.
     """

     # make sure we use the same items for all configurations in the population
     pop = [generate_random_bin_config(num_items, weights_min, weights_max,
                                          bin_capacity)]
     pop = pop * pop_size

     return pop


def generate_random_bin_swarm(pop_size: int, num_items: int,
                              weights_min: int, weights_max: int,
                              bin_capacity: int) -> list[BPPParticle]:
    """
    Generates a random swarm of particles for use in a particle swarm
    optimization algorithm. Each particle in the swarm is a different
    BinConfiguration over the same items.

    :param pop_size:
    :param num_items:
    :param weights_min:
    :param weights_max:
    :param bin_capacity:
    :return:
    """

    # make sure we use the same items for all configurations in the population
    swarm = [generate_random_bin_config(num_items, weights_min, weights_max,
                                        bin_capacity)]
    swarm = swarm * pop_size

    # shuffle bins
    for i in range(pop_size):
        swarm[i] = swarm[i].shuffle_bins()

    # convert to particles
    particles = []
    for config in swarm:
        # each particle is a configuration
        # TODO - this and the same method above may need to use copy.deepcopy
        #  for best_solution.
        particle = BPPParticle(current_solution=config, best_solution=config)
        particles.append(particle)

    return particles




def generate_square_grid(side_length: int) -> Tour:
    """
    Generates a Tour whose cities are arranged in a completely square grid. The
    positions of each City are shuffled to avoid starting with the optimal
    solution.
    A square grid is useful for benchmarking the algorithms, since we know the
    shortest path beforehand.

    :param side_length: (int) The length of each side. Setting this to `5`
        would result in a Tour of `25` cities.
    :return: (Tour) A Tour object containing a grid of cities.
    """

    cities = []
    city_number = 1
    for x in range(1, side_length + 1):
        for y in range(1, side_length + 1):
            city = City(f"{city_number}", x, y)
            cities.append(city)
            city_number += 1

    # print(f"Before shuffling: {Tour(cities)}")
    np.random.shuffle(cities)  # works fine
    # print(f"After shuffling: {Tour(cities)}")

    return Tour(cities)


def generate_grid_population(pop_size: int, side_length: int):
    """"""
    # make sure we use the same cities for all tours in the population
    pop = [generate_square_grid(side_length)]
    pop = pop * pop_size

    # shuffle each tour
    for i in range(pop_size):
        pop[i] = pop[i].shuffle_tour()

    return pop



def generate_grid_swarm(pop_size: int, side_length: int) -> list[TSPParticle]:
    """
    Generates a swarm of particles that will optimize a square grid of cities.

    :param pop_size:
    :param side_length:
    :return:
    """

    # make sure we use the same cities for all tours in the population

    swarm = [generate_square_grid(side_length)]
    swarm = swarm * pop_size
    # now have a bunch of tours over the same square grid

    # shuffle each tour
    for i in range(pop_size):
        swarm[i] = swarm[i].shuffle_tour()

    # convert to particles
    particles = []
    for tour in swarm:
        # each particle is a tour
        # TODO - this and the same method above may need to use copy.deepcopy
        #  for best_solution.
        particle = TSPParticle(current_solution=tour, best_solution=tour)
        particles.append(particle)

    return particles






def generate_random_solution_in_bounds(radius_step_size, length_step_size) \
        -> Design:
    """"""

    # thicknesses are discrete
    head_thickness = THICKNESS_SCALAR * np.random.randint(low=THICKNESS_MIN_INT,
                                                          high=THICKNESS_MAX_INT
                                                          )
    body_thickness = THICKNESS_SCALAR * np.random.randint(low=THICKNESS_MIN_INT,
                                                          high=THICKNESS_MAX_INT
                                                          )

    # head_thickness = np.random.uniform(low=THICKNESS_MIN, high=THICKNESS_MAX)
    # body_thickness = np.random.uniform(low=THICKNESS_MIN, high=THICKNESS_MAX)

    inner_radius = np.random.uniform(low=RADIUS_MIN, high=RADIUS_MAX)
    cylindrical_length = np.random.uniform(low=LENGTH_MIN, high=LENGTH_MAX)

    random_solution = Design(head_thickness=head_thickness,
                             body_thickness=body_thickness,
                             inner_radius=inner_radius,
                             cylindrical_length=cylindrical_length,
                             radius_step_size=radius_step_size,
                             length_step_size=length_step_size)

    # return np.array([head_thickness, body_thickness, inner_radius,
    #                  cylindrical_length])

    return random_solution









def generate_pressure_vessel_solution(radius_step_size, length_step_size):
    """"""

    while True:
        potential_solution = generate_random_solution_in_bounds(
            radius_step_size,
            length_step_size
        )
        # if is_valid_pressure_vessel_solution(potential_solution):
        if potential_solution.is_valid_design():
            return potential_solution




    # # thicknesses are discrete
    # head_thickness = thickness_scalar * np.random.randint(low=thickness_min,
    #                                                       high=thickness_max)
    # body_thickness = thickness_scalar * np.random.randint(low=thickness_min,
    #                                                       high=thickness_max)
    #
    # inner_radius = np.random.uniform(low=radius_min, high=radius_max)
    # cylindrical_length = np.random.uniform(low=length_min, high=length_max)


    # generate values within the simple bounds
    # check for validity
    # repeatedly generate until we have a valid solution




def visualize_solution_fitness(fitness_values: list[float],
                               xlabel: str="Iteration",
                               ylabel: str="Current Tour Distance",
                               title: str="Tour Distance Over Iterations",
                               y_min: float=None,
                               y_max: float=None) \
        -> None:
    """
    Helper function that creates a plot showing fitness values over iterations.

    :param fitness_values: (list) The fitness values to show.
    :param xlabel: (str) The label for the x-axis.
    :param ylabel: (str) The label for the y-axis.
    :param title: (str) The plot title.
    :param y_min: (float) The minimum value for the y-axis.
    :param y_max: (float) The maximum value for the y-axis. Useful for filtering
        extremely high costs in the pressure vessel design problem when the
        constraints are violated.
    :return: None
    """

    iteration_numbers = range(1, len(fitness_values) + 1)  # for x-axis

    # making the plot size smaller basically scales up the text and axis numbers
    # this will be useful for presentations and reports
    # plot.figure(figsize=(6, 4))
    plot.figure(figsize=(12, 8))
    plot.rcParams.update({"font.size": 22})
    plot.plot(iteration_numbers, fitness_values,
              # marker="o", markersize=2,
              linestyle="-",
              # linewidth=3,
              linewidth=5,
              # linewidth=1,
              color="b",
              # color="mediumslateblue",
              # color="royalblue",  # good color
              label="Distance")

    plot.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # only ints

    # plot.xlabel(xlabel)  # sets label font size but not axis #s
    # plot.ylabel(ylabel)
    # plot.title(title)

    plot.xlabel(xlabel, fontsize=26)  # sets label font size but not axis #s
    plot.ylabel(ylabel, fontsize=26)
    plot.title(title, fontsize=30)

    if y_min:
        plot.ylim(bottom=y_min)
    if y_max:
        plot.ylim(top=y_max)

    plot.legend()
    plot.grid(True)
    plot.show()


