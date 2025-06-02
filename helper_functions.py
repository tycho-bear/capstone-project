# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

import numpy as np
from helper_classes import City, Tour
from config import seed

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
        name = f"c{i}"
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
            city = City(f"c{city_number}", x, y)
            cities.append(city)
            city_number += 1

    # print(f"Before shuffling: {Tour(cities)}")
    np.random.shuffle(cities)  # works fine
    # print(f"After shuffling: {Tour(cities)}")

    return Tour(cities)

