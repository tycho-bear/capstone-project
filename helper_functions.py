import numpy as np
from helper_classes import City
from config import seed

np.random.seed(seed)


def generate_random_cities(num_cities: int, x_min: float, x_max: float,
                           y_min: float, y_max: float) -> list[City]:
    """
    Generates a specific number of cities in random positions.

    :param num_cities: (int) The number of cities to generate.
    :param x_min: (double) Lower bound on the x coordinates.
    :param x_max: (double) Upper bound on the x coordinates.
    :param y_min: (double) Lower bound on the y coordinates.
    :param y_max: (double) Upper bound on the y coordinates.
    :return: (list) A list of City objects with random x and y coordinates.
    """

    cities = []
    for i in range(num_cities):
        # name = f"city{i}"
        name = f"c{i}"
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        city = City(name, x, y)
        cities.append(city)

    return cities

