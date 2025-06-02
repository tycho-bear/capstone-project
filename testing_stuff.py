# ===============================
# Tycho Bear
# CSCI.788.01C1
# Capstone Project
# Summer 2025
# ===============================

from helper_classes import City, Tour
from helper_functions import generate_random_cities
from config import seed, x_min_WA, x_max_WA, y_min_WA, y_max_WA
import numpy as np

np.random.seed(seed)


def test_stuff():
    """"""
    num_cities = 7
    # intuitive x/y view in WA
    # this is reversed from lat/lon


    city1 = City("Seattle", 2, 5)
    city2 = City("Boston", 5, 4)
    city3 = City("Austin", 3, 3)
    print()

    # two city distance - works
    distance = city1.distance_to(city2)
    print(f"Euclidean distance from {city1.name} to {city2.name}: {distance}")
    print()

    # tour distance - works
    test_tour = Tour([city1, city2, city3])
    tour_distance = test_tour.calculate_tour_distance()
    print(f"Euclidean distance of the entire tour: {tour_distance}")
    print()

    # random city generation - works
    random_cities = generate_random_cities(num_cities, x_min_WA, x_max_WA,
                                           y_min_WA, y_max_WA)
    random_tour = Tour(random_cities)
    print("Randomly generated cities:")
    for i in range(random_tour.num_cities):
        print(f"\t{random_tour.cities[i]}")
    print()

    random_tour.draw_tour(plot_title="random_tour", include_start_end=False)

    # swapping two cities - works
    # swap city1 and city3
    pos1 = 1
    shift1 = 2  # 1 + 2 = 3
    print(f"Starting tour:\t{random_tour}")
    new_tour = random_tour.swap_cities(pos1, shift1)
    print(f"After swapping:\t{new_tour}")

    new_tour.draw_tour(plot_title="new_tour", include_start_end=False)

    print()
    pos2 = 3
    shift2 = 2
    print(f"New tour:\t\t{new_tour}")
    new_new_tour = new_tour.swap_cities(pos2, shift2)
    print(f"After swapping:\t{new_new_tour}")
    print(f"new_tour:\t\t{new_tour}")

    new_new_tour.draw_tour(plot_title="new_new_tour", include_start_end=False)


def main():
    """"""
    test_stuff()


if __name__ == '__main__':
    main()

