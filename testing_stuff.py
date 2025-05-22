from helper_classes import City, Tour
from helper_functions import generate_random_cities

def test_stuff():
    """"""
    city1 = City("Seattle", 2, 5)
    city2 = City("Boston", 5, 4)
    city3 = City("Austin", 3, 2)

    distance = city1.distance_to(city2)
    print(f"Euclidean distance from {city1.name} to {city2.name}: {distance}")

    test_tour = Tour([city1, city2, city3])
    tour_distance = test_tour.calculate_tour_distance()
    print(f"Euclidean distance of the entire tour: {tour_distance}")

    random_cities = generate_random_cities(5, 1, 10, 1, 10)
    random_tour = Tour(random_cities)
    print("Randomly generated cities:")
    for i in range(random_tour.num_cities):
        print(random_tour.cities[i])



def main():
    """"""
    test_stuff()


if __name__ == '__main__':
    main()

