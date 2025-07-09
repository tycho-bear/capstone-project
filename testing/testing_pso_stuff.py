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
from helper_classes import Tour, City


np.random.seed(SEED)


def test_stuff():
    """"""

    # city1 = City("Seattle", 2, 5)
    # city2 = City("Boston", 5, 4)
    # city3 = City("Austin", 3, 3)
    # city4 = City("SanFrancisco", 2, 4)
    # city5 = City("FlyoverCity", 3, 5)

    city1 = City("S", 2, 5)
    city2 = City("B", 5, 4)
    city3 = City("A", 3, 3)
    city4 = City("SF", 2, 4)
    city5 = City("FC", 3, 5)

    # test_tour = Tour([city1, city2, city3, city4, city5])
    test_tour = Tour([city3, city1, city2, city5, city4])
    # test_tour = Tour([city1, city2, city5, city3, city4])
    # target_tour = test_tour.shuffle_tour()
    target_tour = Tour([city2, city4, city5, city1, city3])

    print(f"test_tour:\t\t{test_tour}")
    print(f"target_tour:\t{target_tour}")

    swap_sequence = test_tour.calculate_swap_sequence(target_tour)
    print(f"Swap sequence: {swap_sequence}")
    print("Applying swap sequence...")

    for swap in swap_sequence:
        pos1, pos2 = swap
        test_tour = test_tour.swap_cities(pos1, pos2)
        print(f"\tIntermediate tour: {test_tour}")

    print(f"test_tour now:\t{test_tour}")










def main():
    """"""
    test_stuff()


if __name__ == '__main__':
    main()
