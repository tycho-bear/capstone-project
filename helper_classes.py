import math
import numpy
import numpy as np


# tour class



# city class
# name, position
# maybe indicate whether these are lat/long coordinates?

class City:
    """
    Class representing a city. It has a name and two x/y or lat/lon coordinates.
    """
    def __init__(self, name, x, y):
        """
        Creates a new City.

        :param name: (str) The name of this City, such as "Seattle".
        :param x: (double) The x coordinate (alternatively, latitude) of this
            City.
        :param y: (double) The y coordinate (alternatively, longitude) of this
            City.
        """
        self.name = name
        self.x = x
        self.y = y

    def distance_to(self, other_city, use_geographical_distance=False):
        """
        Calculates the distance between this City and another City. Set
        `use_geographical_distance` = `True` to calculate the distance along
        Earth's surface.

        :param other_city: (City) The other City object to calculate the
            distance to.
        :return: (double) the distance to the other City.
        """
        if not use_geographical_distance:
            # use Euclidean distance
            sum_term = (self.x - other_city.x)**2 + (self.y - other_city.y)**2
            return math.sqrt(sum_term)

        # TODO: implement geographical distance
        # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
        earth_radius = 6371.009  # kilometers

    def __str__(self):
        """
        This is used for printing.

        :return: (str) A string representation of this City object.
        """
        return f"City \"{self.name}\" at ({self.x}, {self.y})"


class Tour:
    """
    Class representing a tour of cities in the traveling salesman problem.
    It is essentially a collection of City objects with auxiliary functions to
    calculate the total distance or swap two cities.
    """
    def __init__(self, cities: list[City]):
        """
        Create a new Tour.

        :param cities: (list) A list of all the City objects in this tour.
        """
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_tour_distance(self):
        """"""
        # distance from city 1 to city 2, then city 2 to city 3, and so on
        # then distance from last city to first city
        total_distance = 0
        for i in range(self.num_cities - 1):
            this_distance = self.cities[i].distance_to(self.cities[i + 1])




    def swap_cities(self):
        """"""
        # need current position and a shift value
        # example: current tour [0, 1, 2, 3, 4]
        # length = 5
        # random position = 3 (index)
        # shift = 2
        # (this can be 1 or 2 for a lower dimensional graph)
        # (higher dimensional graph may improve the ability to break out of
        # local optima)
        # swap index 3 with index (3 + 2) mod 5, so swap index 3 with index 0


    def update_tour(self):
        """"""
        # just call swap_cities for now
        # want to re-calculate the tour distance after doing this

    def draw_tour(self, include_segments=False):
        """"""
        # get a visualization of the tour and the points in it


    # TODO: __str__?

