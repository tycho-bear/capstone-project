# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import numpy as np
from helper_classes import City, Tour
from config import seed
import matplotlib.pyplot as plot

np.random.seed(seed)


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


def visualize_solution_fitness(fitness_values: list[float],
                               downsample_factor: int = 1) -> None:
    """
    Helper function that creates a plot showing fitness values over iterations.

    :param fitness_values: (list) The fitness values to show.
    :param downsample_factor: (int) Factor to downsample the data for plotting.
    :return: None
    """

    # downsampling also affects the x-axis values... :(
    if downsample_factor > 1:
        fitness_values = fitness_values[::downsample_factor]

    iteration_numbers = range(1, len(fitness_values) + 1)  # for x-axis

    # making the plot size smaller basically scales up the text and axis numbers
    # this will be useful for presentations and reports
    plot.figure(figsize=(6, 4))
    plot.plot(iteration_numbers, fitness_values,
              # marker="o", markersize=2,
              linestyle="-", linewidth=3,
              color="b",
              # color="mediumslateblue",
              # color="royalblue",  # good color
              label="Distance")

    plot.xlabel("Iteration")
    plot.ylabel("Current Tour Distance")  # TODO: need to pass axis labels
    plot.title("Tour Distance Over Iterations")

    plot.legend()
    plot.grid(True)
    plot.show()


